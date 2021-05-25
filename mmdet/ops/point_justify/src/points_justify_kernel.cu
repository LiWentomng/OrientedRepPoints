#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>
# include <math.h>
# define EPS 1e-8


#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 1024

inline int GET_BLOCKS(const int N) {
    int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int max_block_num = 65000;
    return min(optimal_block_num, max_block_num);
}

struct point
{
    float x,y;
};

template <typename scalar_t>
__global__ void PointsJF(const int nthreads, const scalar_t *vertex1, const scalar_t *vertex2, 
                    const int rows, const int cols, scalar_t *inside_flag) {
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        //确定现在计算第row个点 和 第col个多边形间的关系
        int row = index / cols;
        int col = index % cols;

        const scalar_t *offset_vertex1 = vertex1 + row * 2;
        const scalar_t *offset_vertex2 = vertex2 + col * 8;

        point point_[1];
        point polygon[4];

        point_[0].x = offset_vertex1[0];
        point_[0].y = offset_vertex1[1];
        
        polygon[0].x = offset_vertex2[0];
        polygon[0].y = offset_vertex2[1];
        polygon[1].x = offset_vertex2[2];
        polygon[1].y = offset_vertex2[3];
        polygon[2].x = offset_vertex2[4];
        polygon[2].y = offset_vertex2[5];
        polygon[3].x = offset_vertex2[6];
        polygon[3].y = offset_vertex2[7];

        // 指示当前点与多边形的所属关系
        int nCross = 0;
        int i, j;
        float sx, sy, tx, ty, px, py, x;
        for(i=0,j=3;i<4;j=i,i++)    // i,j分别是当前节点和下一个点
        {
            sx = polygon[i].x;
            sy = polygon[i].y;
            tx = polygon[j].x;
            ty = polygon[j].y;

            px = point_[0].x;
            py = point_[0].y;

            if ( py < min(sy, ty)) //如果目标点低于这个线段，跳过
                continue; 
            if ( py > max(sy, ty)) //如果目标点高于这个线段，跳过
                continue; 
                
            // 顶点情况
            if((sx == px && sy == py) || (tx == px && ty == py))
            {
                break;
            }
            else //射线法
            {
                if((sy < py && ty >= py) || (sy >= py && ty < py)) 
                //如果过p1画水平线，过p2画水平线，目标点在这两条线中间
                {
                    x = sx + (py - sy) * (tx - sx) / (ty - sy);
                    // 在边界线上
                    if(x == px)
                    {
                        break;
                    }
                    if(x > px)
                    {  
                        nCross++;
                    }
                }
            }
        }
        if (nCross % 2 == 1) {

            inside_flag[index] = 1.0; //如果是奇数，说明在多边形里
        }
        else {
    
            inside_flag[index] = 0.0; //否则在多边形外 或 边上
        }
        return;
    }
}

int PointsJFLaucher(const at::Tensor points, const at::Tensor polygons,
                const int rows, const int cols, at::Tensor output) {
    const int output_size = rows * cols;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        points.type(), "PointsJFLaucher", ([&] {
        const scalar_t *vertex1 = points.data<scalar_t>();
        const scalar_t *vertex2 = polygons.data<scalar_t>();
        scalar_t *inside_flag = output.data<scalar_t>(); 

        PointsJF<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, vertex1, vertex2, rows, cols, inside_flag);
        }));
    THCudaCheck(cudaGetLastError());
    return 1;
}