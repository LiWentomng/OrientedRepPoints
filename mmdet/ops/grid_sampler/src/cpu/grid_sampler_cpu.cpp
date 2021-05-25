// Modified from https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/GridSampler.cpp

#include <torch/extension.h>
#include "grid_sampler_cpu.h"
#include <ATen/ATen.h>
#include <ATen/Device.h>
#include <ATen/NativeFunctions.h>
#include <c10/core/Layout.h>
#include <c10/util/Exception.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mmdetection {

using namespace at;
using mmdetection::detail::GridSamplerInterpolation;
using mmdetection::detail::GridSamplerPadding;

namespace {

  template<typename scalar_t>
  Tensor grid_sampler_2d_forward_cpu_impl(const Tensor& input, const Tensor& grid,
                                          GridSamplerInterpolation interpolation_mode,
                                          GridSamplerPadding padding_mode,
                                          bool align_corners) {
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t inp_H = input.size(2);
    int64_t inp_W = input.size(3);
    int64_t out_H = grid.size(1);
    int64_t out_W = grid.size(2);
    auto output = at::empty({N, C, out_H, out_W}, input.options());
    int64_t inp_sN = input.stride(0);
    int64_t inp_sC = input.stride(1);
    int64_t inp_sH = input.stride(2);
    int64_t inp_sW = input.stride(3);
    int64_t grid_sN = grid.stride(0);
    int64_t grid_sH = grid.stride(1);
    int64_t grid_sW = grid.stride(2);
    int64_t grid_sCoor = grid.stride(3);
    int64_t out_sN = output.stride(0);
    int64_t out_sC = output.stride(1);
    int64_t out_sH = output.stride(2);
    int64_t out_sW = output.stride(3);
    scalar_t *inp_ptr = input.data<scalar_t>();
    scalar_t *out_ptr = output.data<scalar_t>();
    scalar_t *grid_ptr = grid.data<scalar_t>();
    // loop over each output pixel
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int64_t n = 0; n < N; ++n) {
        scalar_t *grid_ptr_N = grid_ptr + n * grid_sN;
        scalar_t *inp_ptr_N = inp_ptr + n * inp_sN;
          for (int64_t h = 0; h < out_H; ++h) {
            for (int64_t w = 0; w < out_W; ++w) {
              // get the corresponding input x, y, z co-ordinates from grid
              scalar_t *grid_ptr_NHW = grid_ptr_N + h * grid_sH + w * grid_sW;
              scalar_t ix = *grid_ptr_NHW;
              scalar_t iy = grid_ptr_NHW[grid_sCoor];

              ix = grid_sampler_compute_source_index(ix, inp_W, padding_mode, align_corners);
              iy = grid_sampler_compute_source_index(iy, inp_H, padding_mode, align_corners);

              if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
                // get corner pixel values from (x, y, z)
                // for 4d, we used north-east-south-west
                // for 5d, we add top-bottom
                int64_t ix_nw = static_cast<int64_t>(std::floor(ix));
                int64_t iy_nw = static_cast<int64_t>(std::floor(iy));

                int64_t ix_ne = ix_nw + 1;
                int64_t iy_ne = iy_nw;

                int64_t ix_sw = ix_nw;
                int64_t iy_sw = iy_nw + 1;

                int64_t ix_se = ix_nw + 1;
                int64_t iy_se = iy_nw + 1;

                // get surfaces to each neighbor:
                scalar_t nw = (ix_se - ix)    * (iy_se - iy)   ;
                scalar_t ne = (ix    - ix_sw) * (iy_sw - iy)   ;
                scalar_t sw = (ix_ne - ix)    * (iy    - iy_ne);
                scalar_t se = (ix    - ix_nw) * (iy    - iy_nw);

                // calculate bilinear weighted pixel value and set output pixel
                scalar_t *out_ptr_NCHW = out_ptr + n * out_sN + h * out_sH + w * out_sW;
                scalar_t *inp_ptr_NC = inp_ptr_N;
                for (int c = 0; c < C; ++c, out_ptr_NCHW += out_sC, inp_ptr_NC += inp_sC) {
                  //   (c, iz_tnw, iy_tnw, ix_tnw) * tnw + (c, iz_tne, iy_tne, ix_tne) * tne
                  // + (c, iz_tsw, iy_tsw, ix_tsw) * tsw + (c, iz_tse, iy_tse, ix_tse) * tse
                  // + (c, iz_bnw, iy_bnw, ix_bnw) * bnw + (c, iz_bne, iy_bne, ix_bne) * bne
                  // + (c, iz_bsw, iy_bsw, ix_bsw) * bsw + (c, iz_bse, iy_bse, ix_bse) * bse
                  *out_ptr_NCHW = static_cast<scalar_t>(0);
                  if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
                    *out_ptr_NCHW += inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW] * nw;
                  }
                  if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
                    *out_ptr_NCHW += inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW] * ne;
                  }
                  if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
                    *out_ptr_NCHW += inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW] * sw;
                  }
                  if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
                    *out_ptr_NCHW += inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW] * se;
                  }
                }
              } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
                int64_t ix_nearest = static_cast<int64_t>(std::round(ix));
                int64_t iy_nearest = static_cast<int64_t>(std::round(iy));

                // assign nearest neighor pixel value to output pixel
                scalar_t *out_ptr_NCHW = out_ptr + n * out_sN + h * out_sH + w * out_sW;
                scalar_t *inp_ptr_NC = inp_ptr_N;
                for (int c = 0; c < C; ++c, out_ptr_NCHW += out_sC, inp_ptr_NC += inp_sC) {
                  if (within_bounds_2d(iy_nearest, ix_nearest, inp_H, inp_W)) {
                    *out_ptr_NCHW = inp_ptr_NC[iy_nearest * inp_sH + ix_nearest * inp_sW];
                  } else {
                    *out_ptr_NCHW = static_cast<scalar_t>(0);
                  }
                }
              }
            }
          }
      }

    return output;
  }

  template<typename scalar_t>
  Tensor grid_sampler_3d_forward_cpu_impl(const Tensor& input, const Tensor& grid,
                                  GridSamplerInterpolation interpolation_mode,
                                  GridSamplerPadding padding_mode,
                                  bool align_corners) {
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t inp_D = input.size(2);
    int64_t inp_H = input.size(3);
    int64_t inp_W = input.size(4);
    int64_t out_D = grid.size(1);
    int64_t out_H = grid.size(2);
    int64_t out_W = grid.size(3);
    auto output = at::empty({N, C, out_D, out_H, out_W}, input.options());
    int64_t inp_sN = input.stride(0);
    int64_t inp_sC = input.stride(1);
    int64_t inp_sD = input.stride(2);
    int64_t inp_sH = input.stride(3);
    int64_t inp_sW = input.stride(4);
    int64_t grid_sN = grid.stride(0);
    int64_t grid_sD = grid.stride(1);
    int64_t grid_sH = grid.stride(2);
    int64_t grid_sW = grid.stride(3);
    int64_t grid_sCoor = grid.stride(4);
    int64_t out_sN = output.stride(0);
    int64_t out_sC = output.stride(1);
    int64_t out_sD = output.stride(2);
    int64_t out_sH = output.stride(3);
    int64_t out_sW = output.stride(4);
    scalar_t *inp_ptr = input.data<scalar_t>();
    scalar_t *out_ptr = output.data<scalar_t>();
    scalar_t *grid_ptr = grid.data<scalar_t>();
    // loop over each output pixel
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int64_t n = 0; n < N; ++n) {
        scalar_t *grid_ptr_N = grid_ptr + n * grid_sN;
        scalar_t *inp_ptr_N = inp_ptr + n * inp_sN;
        for (int64_t d = 0; d < out_D; ++d) {
          for (int64_t h = 0; h < out_H; ++h) {
            for (int64_t w = 0; w < out_W; ++w) {
              // get the corresponding input x, y, z co-ordinates from grid
              scalar_t *grid_ptr_NDHW = grid_ptr_N + d * grid_sD + h * grid_sH + w * grid_sW;
              scalar_t ix = *grid_ptr_NDHW;
              scalar_t iy = grid_ptr_NDHW[grid_sCoor];
              scalar_t iz = grid_ptr_NDHW[2 * grid_sCoor];

              ix = grid_sampler_compute_source_index(ix, inp_W, padding_mode, align_corners);
              iy = grid_sampler_compute_source_index(iy, inp_H, padding_mode, align_corners);
              iz = grid_sampler_compute_source_index(iz, inp_D, padding_mode, align_corners);

              if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
                // get corner pixel values from (x, y, z)
                // for 4d, we used north-east-south-west
                // for 5d, we add top-bottom
                int64_t ix_tnw = static_cast<int64_t>(std::floor(ix));
                int64_t iy_tnw = static_cast<int64_t>(std::floor(iy));
                int64_t iz_tnw = static_cast<int64_t>(std::floor(iz));

                int64_t ix_tne = ix_tnw + 1;
                int64_t iy_tne = iy_tnw;
                int64_t iz_tne = iz_tnw;

                int64_t ix_tsw = ix_tnw;
                int64_t iy_tsw = iy_tnw + 1;
                int64_t iz_tsw = iz_tnw;

                int64_t ix_tse = ix_tnw + 1;
                int64_t iy_tse = iy_tnw + 1;
                int64_t iz_tse = iz_tnw;

                int64_t ix_bnw = ix_tnw;
                int64_t iy_bnw = iy_tnw;
                int64_t iz_bnw = iz_tnw + 1;

                int64_t ix_bne = ix_tnw + 1;
                int64_t iy_bne = iy_tnw;
                int64_t iz_bne = iz_tnw + 1;

                int64_t ix_bsw = ix_tnw;
                int64_t iy_bsw = iy_tnw + 1;
                int64_t iz_bsw = iz_tnw + 1;

                int64_t ix_bse = ix_tnw + 1;
                int64_t iy_bse = iy_tnw + 1;
                int64_t iz_bse = iz_tnw + 1;

                // get surfaces to each neighbor:
                scalar_t tnw = (ix_bse - ix)    * (iy_bse - iy)    * (iz_bse - iz);
                scalar_t tne = (ix    - ix_bsw) * (iy_bsw - iy)    * (iz_bsw - iz);
                scalar_t tsw = (ix_bne - ix)    * (iy    - iy_bne) * (iz_bne - iz);
                scalar_t tse = (ix    - ix_bnw) * (iy    - iy_bnw) * (iz_bnw - iz);
                scalar_t bnw = (ix_tse - ix)    * (iy_tse - iy)    * (iz - iz_tse);
                scalar_t bne = (ix    - ix_tsw) * (iy_tsw - iy)    * (iz - iz_tsw);
                scalar_t bsw = (ix_tne - ix)    * (iy    - iy_tne) * (iz - iz_tne);
                scalar_t bse = (ix    - ix_tnw) * (iy    - iy_tnw) * (iz - iz_tnw);

                // calculate bilinear weighted pixel value and set output pixel
                scalar_t *out_ptr_NCDHW = out_ptr + n * out_sN + d * out_sD + h * out_sH + w * out_sW;
                scalar_t *inp_ptr_NC = inp_ptr_N;
                for (int c = 0; c < C; ++c, out_ptr_NCDHW += out_sC, inp_ptr_NC += inp_sC) {
                  //   (c, iz_tnw, iy_tnw, ix_tnw) * tnw + (c, iz_tne, iy_tne, ix_tne) * tne
                  // + (c, iz_tsw, iy_tsw, ix_tsw) * tsw + (c, iz_tse, iy_tse, ix_tse) * tse
                  // + (c, iz_bnw, iy_bnw, ix_bnw) * bnw + (c, iz_bne, iy_bne, ix_bne) * bne
                  // + (c, iz_bsw, iy_bsw, ix_bsw) * bsw + (c, iz_bse, iy_bse, ix_bse) * bse
                  *out_ptr_NCDHW = static_cast<scalar_t>(0);
                  if (within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, inp_D, inp_H, inp_W)) {
                    *out_ptr_NCDHW += inp_ptr_NC[iz_tnw * inp_sD + iy_tnw * inp_sH + ix_tnw * inp_sW] * tnw;
                  }
                  if (within_bounds_3d(iz_tne, iy_tne, ix_tne, inp_D, inp_H, inp_W)) {
                    *out_ptr_NCDHW += inp_ptr_NC[iz_tne * inp_sD + iy_tne * inp_sH + ix_tne * inp_sW] * tne;
                  }
                  if (within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, inp_D, inp_H, inp_W)) {
                    *out_ptr_NCDHW += inp_ptr_NC[iz_tsw * inp_sD + iy_tsw * inp_sH + ix_tsw * inp_sW] * tsw;
                  }
                  if (within_bounds_3d(iz_tse, iy_tse, ix_tse, inp_D, inp_H, inp_W)) {
                    *out_ptr_NCDHW += inp_ptr_NC[iz_tse * inp_sD + iy_tse * inp_sH + ix_tse * inp_sW] * tse;
                  }
                  if (within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, inp_D, inp_H, inp_W)) {
                    *out_ptr_NCDHW += inp_ptr_NC[iz_bnw * inp_sD + iy_bnw * inp_sH + ix_bnw * inp_sW] * bnw;
                  }
                  if (within_bounds_3d(iz_bne, iy_bne, ix_bne, inp_D, inp_H, inp_W)) {
                    *out_ptr_NCDHW += inp_ptr_NC[iz_bne * inp_sD + iy_bne * inp_sH + ix_bne * inp_sW] * bne;
                  }
                  if (within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, inp_D, inp_H, inp_W)) {
                    *out_ptr_NCDHW += inp_ptr_NC[iz_bsw * inp_sD + iy_bsw * inp_sH + ix_bsw * inp_sW] * bsw;
                  }
                  if (within_bounds_3d(iz_bse, iy_bse, ix_bse, inp_D, inp_H, inp_W)) {
                    *out_ptr_NCDHW += inp_ptr_NC[iz_bse * inp_sD + iy_bse * inp_sH + ix_bse * inp_sW] * bse;
                  }
                }
              } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
                int64_t ix_nearest = static_cast<int64_t>(std::round(ix));
                int64_t iy_nearest = static_cast<int64_t>(std::round(iy));
                int64_t iz_nearest = static_cast<int64_t>(std::round(iz));

                // assign nearest neighor pixel value to output pixel
                scalar_t *out_ptr_NCDHW = out_ptr + n * out_sN + d * out_sD + h * out_sH + w * out_sW;
                scalar_t *inp_ptr_NC = inp_ptr_N;
                for (int c = 0; c < C; ++c, out_ptr_NCDHW += out_sC, inp_ptr_NC += inp_sC) {
                  if (within_bounds_3d(iz_nearest, iy_nearest, ix_nearest, inp_D, inp_H, inp_W)) {
                    *out_ptr_NCDHW = inp_ptr_NC[iz_nearest * inp_sD + iy_nearest * inp_sH + ix_nearest * inp_sW];
                  } else {
                    *out_ptr_NCDHW = static_cast<scalar_t>(0);
                  }
                }
              }
            }
          }
        }
      }
    return output;
  }

  template<typename scalar_t>
  std::tuple<Tensor, Tensor>
  grid_sampler_2d_backward_cpu_impl(const Tensor& grad_output,
                                    const Tensor& input, const Tensor& grid,
                                    GridSamplerInterpolation interpolation_mode,
                                    GridSamplerPadding padding_mode,
                                    bool align_corners) {
    auto grad_input = at::zeros_like(input);
    auto grad_grid = at::empty_like(grid);
    // If interpolation mode is Nearest, then grad_grid is not filled in the
    // loop below.
    if (interpolation_mode == GridSamplerInterpolation::Nearest) {
      grad_grid.zero_();
    }
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t inp_H = input.size(2);
    int64_t inp_W = input.size(3);
    int64_t out_H = grid.size(1);
    int64_t out_W = grid.size(2);
    int64_t inp_sN = input.stride(0);
    int64_t inp_sC = input.stride(1);
    int64_t inp_sH = input.stride(2);
    int64_t inp_sW = input.stride(3);
    int64_t grid_sN = grid.stride(0);
    int64_t grid_sH = grid.stride(1);
    int64_t grid_sW = grid.stride(2);
    int64_t grid_sCoor = grid.stride(3);
    int64_t gOut_sN = grad_output.stride(0);
    int64_t gOut_sC = grad_output.stride(1);
    int64_t gOut_sH = grad_output.stride(2);
    int64_t gOut_sW = grad_output.stride(3);
    int64_t gInp_sN = grad_input.stride(0);
    int64_t gInp_sC = grad_input.stride(1);
    int64_t gInp_sH = grad_input.stride(2);
    int64_t gInp_sW = grad_input.stride(3);
    int64_t gGrid_sN = grad_grid.stride(0);
    int64_t gGrid_sW = grad_grid.stride(2);
    scalar_t *inp_ptr = input.data<scalar_t>();
    scalar_t *grid_ptr = grid.data<scalar_t>();
    scalar_t *gOut_ptr = grad_output.data<scalar_t>();
    scalar_t *gInp_ptr = grad_input.data<scalar_t>();
    scalar_t *gGrid_ptr = grad_grid.data<scalar_t>();
    // loop over each output pixel
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int64_t n = 0; n < N; ++n) {
        scalar_t *grid_ptr_N = grid_ptr + n * grid_sN;
        scalar_t *inp_ptr_N = inp_ptr + n * inp_sN;
        scalar_t *gGrid_ptr_NHW = gGrid_ptr + n * gGrid_sN;
          for (int64_t h = 0; h < out_H; ++h) {
            for (int64_t w = 0; w < out_W; ++w, gGrid_ptr_NHW += gGrid_sW /* grad_grid is contiguous */ ) {
              // get the corresponding input x, y, z co-ordinates from grid
              scalar_t *grid_ptr_NHW = grid_ptr_N + h * grid_sH + w * grid_sW;
              scalar_t ix = *grid_ptr_NHW;
              scalar_t iy = grid_ptr_NHW[grid_sCoor];

              // multipliers for gradients on ix, iy, and iz
              scalar_t gix_mult, giy_mult;
              ix = grid_sampler_compute_source_index_set_grad(ix, inp_W, padding_mode, align_corners, &gix_mult);
              iy = grid_sampler_compute_source_index_set_grad(iy, inp_H, padding_mode, align_corners, &giy_mult);

              if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
                // get corner pixel values from (x, y, z)
                // for 4d, we used north-east-south-west
                // for 5d, we add top-bottom
                int64_t ix_nw = static_cast<int64_t>(std::floor(ix));
                int64_t iy_nw = static_cast<int64_t>(std::floor(iy));

                int64_t ix_ne = ix_nw + 1;
                int64_t iy_ne = iy_nw;

                int64_t ix_sw = ix_nw;
                int64_t iy_sw = iy_nw + 1;

                int64_t ix_se = ix_nw + 1;
                int64_t iy_se = iy_nw + 1;

                // get surfaces to each neighbor:
                scalar_t nw = (ix_se - ix)    * (iy_se - iy)   ;
                scalar_t ne = (ix    - ix_sw) * (iy_sw - iy)   ;
                scalar_t sw = (ix_ne - ix)    * (iy    - iy_ne);
                scalar_t se = (ix    - ix_nw) * (iy    - iy_nw);

                scalar_t gix = static_cast<scalar_t>(0), giy = static_cast<scalar_t>(0);
                scalar_t *gOut_ptr_NCHW = gOut_ptr + n * gOut_sN + h * gOut_sH + w * gOut_sW;
                scalar_t *gInp_ptr_NC = gInp_ptr + n * gInp_sN;
                scalar_t *inp_ptr_NC = inp_ptr_N;
                // calculate bilinear weighted pixel value and set output pixel
                for (int c = 0; c < C; ++c, gOut_ptr_NCHW += gOut_sC, gInp_ptr_NC += gInp_sC, inp_ptr_NC += inp_sC) {
                  scalar_t gOut = *gOut_ptr_NCHW;

                  // calculate and set grad_input
                  safe_add_2d(gInp_ptr_NC, iy_nw, ix_nw, gInp_sH, gInp_sW, inp_H, inp_W, nw * gOut);
                  safe_add_2d(gInp_ptr_NC, iy_ne, ix_ne, gInp_sH, gInp_sW, inp_H, inp_W, ne * gOut);
                  safe_add_2d(gInp_ptr_NC, iy_sw, ix_sw, gInp_sH, gInp_sW, inp_H, inp_W, sw * gOut);
                  safe_add_2d(gInp_ptr_NC, iy_se, ix_se, gInp_sH, gInp_sW, inp_H, inp_W, se * gOut);

                  // calculate grad_grid
                  if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
                    scalar_t nw_val = inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW];
                    gix -= nw_val * (iy_se - iy) * gOut;
                    giy -= nw_val * (ix_se - ix) * gOut;
                  }
                  if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
                    scalar_t ne_val = inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW];
                    gix += ne_val * (iy_sw - iy)    * gOut;
                    giy -= ne_val * (ix    - ix_sw) * gOut;
                  }
                  if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
                    scalar_t sw_val = inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW];
                    gix -= sw_val * (iy - iy_ne) * gOut;
                    giy += sw_val * (ix_ne - ix) * gOut;
                  }
                  if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
                    scalar_t se_val = inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW];
                    gix += se_val * (iy - iy_nw)    * gOut;
                    giy += se_val * (ix    - ix_nw) * gOut;
                  }
                }

                // assuming grad_grid is contiguous
                gGrid_ptr_NHW[0] = gix_mult * gix;
                gGrid_ptr_NHW[1] = giy_mult * giy;
              } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
                int64_t ix_nearest = static_cast<int64_t>(std::round(ix));
                int64_t iy_nearest = static_cast<int64_t>(std::round(iy));

                // assign nearest neighor pixel value to output pixel
                scalar_t *gOut_ptr_NCHW = gOut_ptr + n * gOut_sN + h * gOut_sH + w * gOut_sW;
                scalar_t *gInp_ptr_NC = gInp_ptr + n * gInp_sN;
                for (int c = 0; c < C; ++c, gOut_ptr_NCHW += gOut_sC, gInp_ptr_NC += gInp_sC) {
                  // calculate and set grad_input
                  safe_add_2d(gInp_ptr_NC, iy_nearest, ix_nearest,
                              gInp_sH, gInp_sW, inp_H, inp_W, *gOut_ptr_NCHW);
                }
              }
            }
          }
      }
    return std::make_tuple(grad_input, grad_grid);
  }

  template<typename scalar_t>
  std::tuple<Tensor, Tensor>
  grid_sampler_3d_backward_cpu_impl(const Tensor& grad_output,
                                    const Tensor& input, const Tensor& grid,
                                    GridSamplerInterpolation interpolation_mode,
                                    GridSamplerPadding padding_mode,
                                    bool align_corners) {
    auto grad_input = at::zeros_like(input);
    auto grad_grid = at::empty_like(grid);
    // If interpolation mode is Nearest, then grad_grid is not filled in the
    // loop below.
    if (interpolation_mode == GridSamplerInterpolation::Nearest) {
      grad_grid.zero_();
    }
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t inp_D = input.size(2);
    int64_t inp_H = input.size(3);
    int64_t inp_W = input.size(4);
    int64_t out_D = grid.size(1);
    int64_t out_H = grid.size(2);
    int64_t out_W = grid.size(3);
    int64_t inp_sN = input.stride(0);
    int64_t inp_sC = input.stride(1);
    int64_t inp_sD = input.stride(2);
    int64_t inp_sH = input.stride(3);
    int64_t inp_sW = input.stride(4);
    int64_t grid_sN = grid.stride(0);
    int64_t grid_sD = grid.stride(1);
    int64_t grid_sH = grid.stride(2);
    int64_t grid_sW = grid.stride(3);
    int64_t grid_sCoor = grid.stride(4);
    int64_t gOut_sN = grad_output.stride(0);
    int64_t gOut_sC = grad_output.stride(1);
    int64_t gOut_sD = grad_output.stride(2);
    int64_t gOut_sH = grad_output.stride(3);
    int64_t gOut_sW = grad_output.stride(4);
    int64_t gInp_sN = grad_input.stride(0);
    int64_t gInp_sC = grad_input.stride(1);
    int64_t gInp_sD = grad_input.stride(2);
    int64_t gInp_sH = grad_input.stride(3);
    int64_t gInp_sW = grad_input.stride(4);
    int64_t gGrid_sN = grad_grid.stride(0);
    int64_t gGrid_sW = grad_grid.stride(3);
    scalar_t *inp_ptr = input.data<scalar_t>();
    scalar_t *grid_ptr = grid.data<scalar_t>();
    scalar_t *gOut_ptr = grad_output.data<scalar_t>();
    scalar_t *gInp_ptr = grad_input.data<scalar_t>();
    scalar_t *gGrid_ptr = grad_grid.data<scalar_t>();
    // loop over each output pixel
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int64_t n = 0; n < N; ++n) {
        scalar_t *grid_ptr_N = grid_ptr + n * grid_sN;
        scalar_t *inp_ptr_N = inp_ptr + n * inp_sN;
        scalar_t *gGrid_ptr_NDHW = gGrid_ptr + n * gGrid_sN;
        for (int64_t d = 0; d < out_D; ++d) {
          for (int64_t h = 0; h < out_H; ++h) {
            for (int64_t w = 0; w < out_W; ++w, gGrid_ptr_NDHW += gGrid_sW /* grad_grid is contiguous */ ) {
              // get the corresponding input x, y, z co-ordinates from grid
              scalar_t *grid_ptr_NDHW = grid_ptr_N + d * grid_sD + h * grid_sH + w * grid_sW;
              scalar_t ix = *grid_ptr_NDHW;
              scalar_t iy = grid_ptr_NDHW[grid_sCoor];
              scalar_t iz = grid_ptr_NDHW[2 * grid_sCoor];

              // multipliers for gradients on ix, iy, and iz
              scalar_t gix_mult, giy_mult, giz_mult;
              ix = grid_sampler_compute_source_index_set_grad(ix, inp_W, padding_mode, align_corners, &gix_mult);
              iy = grid_sampler_compute_source_index_set_grad(iy, inp_H, padding_mode, align_corners, &giy_mult);
              iz = grid_sampler_compute_source_index_set_grad(iz, inp_D, padding_mode, align_corners, &giz_mult);

              if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
                // get corner pixel values from (x, y, z)
                // for 4d, we used north-east-south-west
                // for 5d, we add top-bottom
                int64_t ix_tnw = static_cast<int64_t>(std::floor(ix));
                int64_t iy_tnw = static_cast<int64_t>(std::floor(iy));
                int64_t iz_tnw = static_cast<int64_t>(std::floor(iz));

                int64_t ix_tne = ix_tnw + 1;
                int64_t iy_tne = iy_tnw;
                int64_t iz_tne = iz_tnw;

                int64_t ix_tsw = ix_tnw;
                int64_t iy_tsw = iy_tnw + 1;
                int64_t iz_tsw = iz_tnw;

                int64_t ix_tse = ix_tnw + 1;
                int64_t iy_tse = iy_tnw + 1;
                int64_t iz_tse = iz_tnw;

                int64_t ix_bnw = ix_tnw;
                int64_t iy_bnw = iy_tnw;
                int64_t iz_bnw = iz_tnw + 1;

                int64_t ix_bne = ix_tnw + 1;
                int64_t iy_bne = iy_tnw;
                int64_t iz_bne = iz_tnw + 1;

                int64_t ix_bsw = ix_tnw;
                int64_t iy_bsw = iy_tnw + 1;
                int64_t iz_bsw = iz_tnw + 1;

                int64_t ix_bse = ix_tnw + 1;
                int64_t iy_bse = iy_tnw + 1;
                int64_t iz_bse = iz_tnw + 1;

                // get surfaces to each neighbor:
                scalar_t tnw = (ix_bse - ix)    * (iy_bse - iy)    * (iz_bse - iz);
                scalar_t tne = (ix    - ix_bsw) * (iy_bsw - iy)    * (iz_bsw - iz);
                scalar_t tsw = (ix_bne - ix)    * (iy    - iy_bne) * (iz_bne - iz);
                scalar_t tse = (ix    - ix_bnw) * (iy    - iy_bnw) * (iz_bnw - iz);
                scalar_t bnw = (ix_tse - ix)    * (iy_tse - iy)    * (iz - iz_tse);
                scalar_t bne = (ix    - ix_tsw) * (iy_tsw - iy)    * (iz - iz_tsw);
                scalar_t bsw = (ix_tne - ix)    * (iy    - iy_tne) * (iz - iz_tne);
                scalar_t bse = (ix    - ix_tnw) * (iy    - iy_tnw) * (iz - iz_tnw);

                scalar_t gix = static_cast<scalar_t>(0), giy = static_cast<scalar_t>(0), giz = static_cast<scalar_t>(0);
                scalar_t *gOut_ptr_NCDHW = gOut_ptr + n * gOut_sN + d * gOut_sD + h * gOut_sH + w * gOut_sW;
                scalar_t *gInp_ptr_NC = gInp_ptr + n * gInp_sN;
                scalar_t *inp_ptr_NC = inp_ptr_N;
                // calculate bilinear weighted pixel value and set output pixel
                for (int c = 0; c < C; ++c, gOut_ptr_NCDHW += gOut_sC, gInp_ptr_NC += gInp_sC, inp_ptr_NC += inp_sC) {
                  scalar_t gOut = *gOut_ptr_NCDHW;

                  // calculate and set grad_input
                  safe_add_3d(gInp_ptr_NC, iz_tnw, iy_tnw, ix_tnw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tnw * gOut);
                  safe_add_3d(gInp_ptr_NC, iz_tne, iy_tne, ix_tne, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tne * gOut);
                  safe_add_3d(gInp_ptr_NC, iz_tsw, iy_tsw, ix_tsw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tsw * gOut);
                  safe_add_3d(gInp_ptr_NC, iz_tse, iy_tse, ix_tse, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tse * gOut);
                  safe_add_3d(gInp_ptr_NC, iz_bnw, iy_bnw, ix_bnw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bnw * gOut);
                  safe_add_3d(gInp_ptr_NC, iz_bne, iy_bne, ix_bne, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bne * gOut);
                  safe_add_3d(gInp_ptr_NC, iz_bsw, iy_bsw, ix_bsw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bsw * gOut);
                  safe_add_3d(gInp_ptr_NC, iz_bse, iy_bse, ix_bse, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bse * gOut);

                  // calculate grad_grid
                  if (within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, inp_D, inp_H, inp_W)) {
                    scalar_t tnw_val = inp_ptr_NC[iz_tnw * inp_sD + iy_tnw * inp_sH + ix_tnw * inp_sW];
                    gix -= tnw_val * (iy_bse - iy)    * (iz_bse - iz)    * gOut;
                    giy -= tnw_val * (ix_bse - ix)    * (iz_bse - iz)    * gOut;
                    giz -= tnw_val * (ix_bse - ix)    * (iy_bse - iy)    * gOut;
                  }
                  if (within_bounds_3d(iz_tne, iy_tne, ix_tne, inp_D, inp_H, inp_W)) {
                    scalar_t tne_val = inp_ptr_NC[iz_tne * inp_sD + iy_tne * inp_sH + ix_tne * inp_sW];
                    gix += tne_val * (iy_bsw - iy)    * (iz_bsw - iz)    * gOut;
                    giy -= tne_val * (ix    - ix_bsw) * (iz_bsw - iz)    * gOut;
                    giz -= tne_val * (ix    - ix_bsw) * (iy_bsw - iy)    * gOut;
                  }
                  if (within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, inp_D, inp_H, inp_W)) {
                    scalar_t tsw_val = inp_ptr_NC[iz_tsw * inp_sD + iy_tsw * inp_sH + ix_tsw * inp_sW];
                    gix -= tsw_val * (iy - iy_bne)    * (iz_bne - iz)    * gOut;
                    giy += tsw_val * (ix_bne - ix)    * (iz_bne - iz)    * gOut;
                    giz -= tsw_val * (ix_bne - ix)    * (iy    - iy_bne) * gOut;
                  }
                  if (within_bounds_3d(iz_tse, iy_tse, ix_tse, inp_D, inp_H, inp_W)) {
                    scalar_t tse_val = inp_ptr_NC[iz_tse * inp_sD + iy_tse * inp_sH + ix_tse * inp_sW];
                    gix += tse_val * (iy - iy_bnw)    * (iz_bnw - iz)    * gOut;
                    giy += tse_val * (ix    - ix_bnw) * (iz_bnw - iz)    * gOut;
                    giz -= tse_val * (ix    - ix_bnw) * (iy    - iy_bnw) * gOut;
                  }
                  if (within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, inp_D, inp_H, inp_W)) {
                    scalar_t bnw_val = inp_ptr_NC[iz_bnw * inp_sD + iy_bnw * inp_sH + ix_bnw * inp_sW];
                    gix -= bnw_val * (iy_tse - iy)    * (iz - iz_tse)    * gOut;
                    giy -= bnw_val * (ix_tse - ix)    * (iz - iz_tse)    * gOut;
                    giz += bnw_val * (ix_tse - ix)    * (iy_tse - iy)    * gOut;
                  }
                  if (within_bounds_3d(iz_bne, iy_bne, ix_bne, inp_D, inp_H, inp_W)) {
                    scalar_t bne_val = inp_ptr_NC[iz_bne * inp_sD + iy_bne * inp_sH + ix_bne * inp_sW];
                    gix += bne_val * (iy_tsw - iy)    * (iz - iz_tsw)    * gOut;
                    giy -= bne_val * (ix    - ix_tsw) * (iz - iz_tsw)    * gOut;
                    giz += bne_val * (ix    - ix_tsw) * (iy_tsw - iy)    * gOut;
                  }
                  if (within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, inp_D, inp_H, inp_W)) {
                    scalar_t bsw_val = inp_ptr_NC[iz_bsw * inp_sD + iy_bsw * inp_sH + ix_bsw * inp_sW];
                    gix -= bsw_val * (iy - iy_tne)    * (iz - iz_tne)    * gOut;
                    giy += bsw_val * (ix_tne - ix)    * (iz - iz_tne)    * gOut;
                    giz += bsw_val * (ix_tne - ix)    * (iy    - iy_tne) * gOut;
                  }
                  if (within_bounds_3d(iz_bse, iy_bse, ix_bse, inp_D, inp_H, inp_W)) {
                    scalar_t bse_val = inp_ptr_NC[iz_bse * inp_sD + iy_bse * inp_sH + ix_bse * inp_sW];
                    gix += bse_val * (iy - iy_tnw)    * (iz - iz_tnw)    * gOut;
                    giy += bse_val * (ix    - ix_tnw) * (iz - iz_tnw)    * gOut;
                    giz += bse_val * (ix    - ix_tnw) * (iy    - iy_tnw) * gOut;
                  }
                }

                // assuming grad_grid is contiguous
                gGrid_ptr_NDHW[0] = gix_mult * gix;
                gGrid_ptr_NDHW[1] = giy_mult * giy;
                gGrid_ptr_NDHW[2] = giz_mult * giz;
              } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
                int64_t ix_nearest = static_cast<int64_t>(std::round(ix));
                int64_t iy_nearest = static_cast<int64_t>(std::round(iy));
                int64_t iz_nearest = static_cast<int64_t>(std::round(iz));

                // assign nearest neighor pixel value to output pixel
                scalar_t *gOut_ptr_NCDHW = gOut_ptr + n * gOut_sN + d * gOut_sD + h * gOut_sH + w * gOut_sW;
                scalar_t *gInp_ptr_NC = gInp_ptr + n * gInp_sN;
                for (int c = 0; c < C; ++c, gOut_ptr_NCDHW += gOut_sC, gInp_ptr_NC += gInp_sC) {
                  // calculate and set grad_input
                  safe_add_3d(gInp_ptr_NC, iz_nearest, iy_nearest, ix_nearest,
                              gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, *gOut_ptr_NCDHW);
                }
              }
            }
          }
        }
      }
    return std::make_tuple(grad_input, grad_grid);
  }

}  // namespace

// No shape checking needed here. See # NOTE [ grid_sampler Native Functions ].
Tensor grid_sampler_2d_forward_cpu(const Tensor& input, const Tensor& grid,
                                   int64_t interpolation_mode, int64_t padding_mode,
                                   bool align_corners) {
    return AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "grid_sampler_2d_forward_cpu", [&] {
        return grid_sampler_2d_forward_cpu_impl<scalar_t>(
                input, grid, static_cast<GridSamplerInterpolation>(interpolation_mode),
                static_cast<GridSamplerPadding>(padding_mode), align_corners);
    });
}

// No shape checking needed here. See # NOTE [ grid_sampler Native Functions ].
Tensor grid_sampler_3d_forward_cpu(const Tensor& input, const Tensor& grid,
                           int64_t interpolation_mode, int64_t padding_mode,
                           bool align_corners) {
  return AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "grid_sampler_3d_forward_cpu", [&] {
    return grid_sampler_3d_forward_cpu_impl<scalar_t>(
      input, grid, static_cast<GridSamplerInterpolation>(interpolation_mode),
      static_cast<GridSamplerPadding>(padding_mode), align_corners);
  });
}

// No shape checking needed here. See # NOTE [ grid_sampler Native Functions ].
std::tuple<Tensor, Tensor>
grid_sampler_2d_backward_cpu(const Tensor& grad_output, const Tensor& input, const Tensor& grid,
                             int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
    return AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "grid_sampler_2d_backward_cpu", [&] {
        return grid_sampler_2d_backward_cpu_impl<scalar_t>(
                grad_output, input, grid,
                static_cast<GridSamplerInterpolation>(interpolation_mode),
                static_cast<GridSamplerPadding>(padding_mode), align_corners);
    });
}

// No shape checking needed here. See # NOTE [ grid_sampler Native Functions ].
std::tuple<Tensor, Tensor>
grid_sampler_3d_backward_cpu(const Tensor& grad_output, const Tensor& input, const Tensor& grid,
                             int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  return AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "grid_sampler_3d_backward_cpu", [&] {
    return grid_sampler_3d_backward_cpu_impl<scalar_t>(
      grad_output, input, grid,
      static_cast<GridSamplerInterpolation>(interpolation_mode),
      static_cast<GridSamplerPadding>(padding_mode), align_corners);
  });
}

}  // namespace mmdetection
