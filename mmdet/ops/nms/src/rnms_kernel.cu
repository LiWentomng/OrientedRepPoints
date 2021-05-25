// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCDeviceUtils.cuh>

#include <vector>
#include <iostream>
#include <algorithm>

#define maxn 510
const float eps=1E-8;
int const threadsPerBlock = sizeof(unsigned long long) * 8;


__device__ int sig(float d){
    return(d>eps)-(d<-eps);
}

struct Point{
    float x,y;
    __device__ Point(){}
    __device__ Point(float x,float y):x(x),y(y){}
};

__device__ bool point_same(Point& a, Point& b){
    return sig(a.x - b.x) == 0 && sig(a.y - b.y) == 0;
}

__device__ void swap1(Point* a, Point* b){
    Point temp;
    temp.x = a->x;
    temp.y = a->y;

    a->x = b->x;
    a->y = b->y;

    b->x = temp.x;
    b->y = temp.y;
}

__device__ void reverse1(Point* a, const int n){
    Point temp[maxn];
    for(int i = 0; i < n; i++){
        temp[i].x = a[i].x;
        temp[i].y = a[i].y;
    }
    for(int i = 0; i < n; i++){
        a[i].x = temp[n - 1 - i].x;
        a[i].y = temp[n - 1 - i].y;
    }
}

__device__ float cross(Point o,Point a,Point b){  
    return(a.x-o.x)*(b.y-o.y)-(b.x-o.x)*(a.y-o.y);
}

__device__ float area(Point* ps,int n){
    ps[n]=ps[0];
    float res=0;
    for(int i=0;i<n;i++){
        res+=ps[i].x*ps[i+1].y-ps[i].y*ps[i+1].x;
    }
    return res/2.0;
}
__device__ int lineCross(Point a,Point b,Point c,Point d,Point&p){
    float s1,s2;
    s1=cross(a,b,c);
    s2=cross(a,b,d);
    if(sig(s1)==0&&sig(s2)==0) return 2;
    if(sig(s2-s1)==0) return 0;
    p.x=(c.x*s2-d.x*s1)/(s2-s1);
    p.y=(c.y*s2-d.y*s1)/(s2-s1);
    return 1;
}
__device__ void polygon_cut(Point*p,int&n,Point a,Point b){
    Point pp[maxn];//static Point pp[maxn];
    int m=0;p[n]=p[0];
    for(int i=0;i<n;i++){
        if(sig(cross(a,b,p[i]))>0) pp[m++]=p[i];
        if(sig(cross(a,b,p[i]))!=sig(cross(a,b,p[i+1])))
            lineCross(a,b,p[i],p[i+1],pp[m++]);
    }
    n=0;
    for(int i=0;i<m;i++)
//        if(!i||!(pp[i]==pp[i-1]))
          if(!i || !(point_same(pp[i], pp[i-1])))
    		p[n++]=pp[i];
//    while(n>1&&p[n-1]==p[0])n--;
    while(n > 1 && point_same(p[n-1], p[0]))n--;
}
__device__ float intersectArea(Point a,Point b,Point c,Point d){
    Point o(0,0);
    int s1=sig(cross(o,a,b));
    int s2=sig(cross(o,c,d));
    if(s1==0||s2==0)return 0.0;
    if(s1==-1){
    	Point* i = &a;
    	Point* j = &b;
    	swap1(i, j);
    }
    if(s2==-1){
    	Point* i = &c;
    	Point* j = &d;
    	swap1(i, j);
    }
    Point p[10]={o,a,b};
    int n=3;
    polygon_cut(p,n,o,c);
    polygon_cut(p,n,c,d);
    polygon_cut(p,n,d,o);
    float res=fabs(area(p,n));
    if(s1*s2==-1) res=-res;return res;
}
__device__ float intersectAreaO(Point*ps1,int n1,Point*ps2,int n2){
    if(area(ps1,n1)<0) reverse1(ps1,n1);
    if(area(ps2,n2)<0) reverse1(ps2,n2);
    ps1[n1]=ps1[0];
    ps2[n2]=ps2[0];
    float res=0;
    for(int i=0;i<n1;i++){
        for(int j=0;j<n2;j++){
            res+=intersectArea(ps1[i],ps1[i+1],ps2[j],ps2[j+1]);
        }
    }
    return res;
}


__device__ inline float devrIoU(float const * const p, float const * const q) {
    Point ps1[maxn],ps2[maxn];
    int n1 = 4;
    int n2 = 4;
    for (int i = 0; i < 4; i++) {
        ps1[i].x = p[i * 2];
        ps1[i].y = p[i * 2 + 1];

        ps2[i].x = q[i * 2];
        ps2[i].y = q[i * 2 + 1];
    }
    float inter_area = intersectAreaO(ps1, n1, ps2, n2);
    float union_area = fabs(area(ps1, n1)) + fabs(area(ps2, n2)) - inter_area;
    float iou = inter_area / union_area;
    return iou;

}

__global__ void rnms_kernel(const int n_boxes, const float nms_overlap_thresh,
                           const float *dev_boxes, unsigned long long *dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  // if (row_start > col_start) return;

  const int row_size =
        min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ float block_boxes[threadsPerBlock * 9];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 9 + 0] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 9 + 0];
    block_boxes[threadIdx.x * 9 + 1] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 9 + 1];
    block_boxes[threadIdx.x * 9 + 2] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 9 + 2];
    block_boxes[threadIdx.x * 9 + 3] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 9 + 3];
    block_boxes[threadIdx.x * 9 + 4] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 9 + 4];
    block_boxes[threadIdx.x * 9 + 5] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 9 + 5];
    block_boxes[threadIdx.x * 9 + 6] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 9 + 6];
    block_boxes[threadIdx.x * 9 + 7] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 9 + 7];
    block_boxes[threadIdx.x * 9 + 8] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 9 + 8];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const float *cur_box = dev_boxes + cur_box_idx * 9;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devrIoU(cur_box, block_boxes + i * 9) > nms_overlap_thresh) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = THCCeilDiv(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

// boxes is a N x 9 tensor (boxes(8 tuple), score)
at::Tensor rnms_cuda(const at::Tensor boxes, float nms_overlap_thresh) {
  using scalar_t = float;
  AT_ASSERTM(boxes.type().is_cuda(), "boxes must be a CUDA tensor");
  auto scores = boxes.select(1, 8);
  auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));
  auto boxes_sorted = boxes.index_select(0, order_t);

  int boxes_num = boxes.size(0);

  const int col_blocks = THCCeilDiv(boxes_num, threadsPerBlock);

  scalar_t* boxes_dev = boxes_sorted.data<scalar_t>();

  THCState *state = at::globalContext().lazyInitCUDA(); // TODO replace with getTHCState

  unsigned long long* mask_dev = NULL;
  //THCudaCheck(THCudaMalloc(state, (void**) &mask_dev,
  //                      boxes_num * col_blocks * sizeof(unsigned long long)));

  mask_dev = (unsigned long long*) THCudaMalloc(state, boxes_num * col_blocks * sizeof(unsigned long long));

  dim3 blocks(THCCeilDiv(boxes_num, threadsPerBlock),
              THCCeilDiv(boxes_num, threadsPerBlock));
  dim3 threads(threadsPerBlock);
  rnms_kernel<<<blocks, threads>>>(boxes_num,
                                  nms_overlap_thresh,
                                  boxes_dev,
                                  mask_dev);

  std::vector<unsigned long long> mask_host(boxes_num * col_blocks);
  THCudaCheck(cudaMemcpy(&mask_host[0],
                        mask_dev,
                        sizeof(unsigned long long) * boxes_num * col_blocks,
                        cudaMemcpyDeviceToHost));

  std::vector<unsigned long long> remv(col_blocks);
  memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

  at::Tensor keep = at::empty({boxes_num}, boxes.options().dtype(at::kLong).device(at::kCPU));
  int64_t* keep_out = keep.data<int64_t>();

  int num_to_keep = 0;
  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    if (!(remv[nblock] & (1ULL << inblock))) {
      keep_out[num_to_keep++] = i;
      unsigned long long *p = &mask_host[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }

  THCudaFree(state, mask_dev);
  // TODO improve this part
  return std::get<0>(order_t.index({
                       keep.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep).to(
                         order_t.device(), keep.scalar_type())
                     }).sort(0, false));
}
