// Modified from https://github.com/bharatsingh430/soft-nms/blob/master/lib/nms/cpu_nms.pyx, Soft-NMS is added
// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <torch/extension.h>

#define maxn 510
const float eps=1E-8;

int sig(float d){
    return(d>eps)-(d<-eps);
}

struct Point{
    float x,y;
    Point(){}
    Point(float x,float y):x(x),y(y){}
};

bool point_same(Point& a, Point& b){
    return sig(a.x - b.x) == 0 && sig(a.y - b.y) == 0;
}

void swap1(Point* a, Point* b){
    Point temp;
    temp.x = a->x;
    temp.y = a->y;

    a->x = b->x;
    a->y = b->y;

    b->x = temp.x;
    b->y = temp.y;
}

void reverse1(Point* a, const int n){
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

float cross(Point o,Point a,Point b){  
    return(a.x-o.x)*(b.y-o.y)-(b.x-o.x)*(a.y-o.y);
}

float area(Point* ps,int n){
    ps[n]=ps[0];
    float res=0;
    for(int i=0;i<n;i++){
        res+=ps[i].x*ps[i+1].y-ps[i].y*ps[i+1].x;
    }
    return res/2.0;
}
int lineCross(Point a,Point b,Point c,Point d,Point&p){
    float s1,s2;
    s1=cross(a,b,c);
    s2=cross(a,b,d);
    if(sig(s1)==0&&sig(s2)==0) return 2;
    if(sig(s2-s1)==0) return 0;
    p.x=(c.x*s2-d.x*s1)/(s2-s1);
    p.y=(c.y*s2-d.y*s1)/(s2-s1);
    return 1;
}
void polygon_cut(Point*p,int&n,Point a,Point b){
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
float intersectArea(Point a,Point b,Point c,Point d){
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
float intersectAreaO(Point*ps1,int n1,Point*ps2,int n2){
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

float rotate_iou(float const x11, float const y11, 
                float const x12, float const y12,
                float const x13, float const y13,
                float const x14, float const y14,
                 
                float const x21, float const y21,
                float const x22, float const y22,
                float const x23, float const y23,
                float const x24, float const y24){
    
    Point ps1[maxn],ps2[maxn];
    int n1 = 4;
    int n2 = 4;
//     for (int i = 0; i < 4; i++) {
//         ps1[i].x = p[i * 2];
//         ps1[i].y = p[i * 2 + 1];

//         ps2[i].x = q[i * 2];
//         ps2[i].y = q[i * 2 + 1];
//     }
    ps1[0].x = x11;
    ps1[0].y = y11;
    ps1[1].x = x12;
    ps1[1].y = y12;
    ps1[2].x = x13;
    ps1[2].y = y13;
    ps1[3].x = x14;
    ps1[3].y = y14;
    
    ps2[0].x = x21;
    ps2[0].y = y21;
    ps2[1].x = x22;
    ps2[1].y = y22;
    ps2[2].x = x23;
    ps2[2].y = y23;
    ps2[3].x = x24;
    ps2[3].y = y24;
    
    float inter_area = intersectAreaO(ps1, n1, ps2, n2);
    float union_area = fabs(area(ps1, n1)) + fabs(area(ps2, n2)) - inter_area;
    float iou = inter_area / union_area;
    return iou;
}

template <typename scalar_t>
at::Tensor soft_rnms_cpu_kernel(const at::Tensor& dets, const float threshold,
                               const unsigned char method, const float sigma, const float min_score) {
  AT_ASSERTM(!dets.type().is_cuda(), "dets must be a CPU tensor");

  if (dets.numel() == 0) {
    return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
  }
    
  auto x1_t = dets.select(1, 0).contiguous();
  auto y1_t = dets.select(1, 1).contiguous();
  auto x2_t = dets.select(1, 2).contiguous();
  auto y2_t = dets.select(1, 3).contiguous();
  auto x3_t = dets.select(1, 4).contiguous();
  auto y3_t = dets.select(1, 5).contiguous();
  auto x4_t = dets.select(1, 6).contiguous();
  auto y4_t = dets.select(1, 7).contiguous();
  auto scores_t = dets.select(1, 8).contiguous();

  auto ndets = dets.size(0);
  auto x1 = x1_t.data<scalar_t>();
  auto y1 = y1_t.data<scalar_t>();
  auto x2 = x2_t.data<scalar_t>();
  auto y2 = y2_t.data<scalar_t>();
  auto x3 = x3_t.data<scalar_t>();
  auto y3 = y3_t.data<scalar_t>();
  auto x4 = x4_t.data<scalar_t>();
  auto y4 = y4_t.data<scalar_t>();    
  auto scores = scores_t.data<scalar_t>();

  int64_t pos = 0;
  at::Tensor inds_t = at::arange(ndets, dets.options());
  auto inds = inds_t.data<scalar_t>();

  for (int64_t i = 0; i < ndets; i++) {
    auto max_score = scores[i];
    auto max_pos = i;

//     auto irbox = rbboxes[i];
    auto ix1 = x1[i];
    auto iy1 = y1[i];
    auto ix2 = x2[i];
    auto iy2 = y2[i];
    auto ix3 = x3[i];
    auto iy3 = y3[i];
    auto ix4 = x4[i];
    auto iy4 = y4[i];
      
    auto iscore = scores[i];
    auto iind = inds[i];

    pos = i + 1;
    // get max box
    while (pos < ndets){
        if (max_score < scores[pos]) {
            max_score = scores[pos];
            max_pos = pos;
        }
        pos = pos + 1;
    }
    // add max box as a detection
//     rbboxes[i] = rbboxes[max_pos];
    x1[i] = x1[max_pos];
    y1[i] = y1[max_pos];
    x2[i] = x2[max_pos];
    y2[i] = y2[max_pos];
    x3[i] = x3[max_pos];
    y3[i] = y3[max_pos];
    x4[i] = x4[max_pos];
    y4[i] = y4[max_pos];
    scores[i] = scores[max_pos];
    inds[i] = inds[max_pos];

    // swap ith box with position of max box
    x1[max_pos] =  ix1;
    y1[max_pos] =  iy1;
    x2[max_pos] =  ix2;
    y2[max_pos] =  iy2;
    x3[max_pos] =  ix3;
    y3[max_pos] =  iy3;
    x4[max_pos] =  ix4;
    y4[max_pos] =  iy4;
    scores[max_pos] = iscore;
    inds[max_pos] = iind;

//     irbox = rbboxes[i];
    ix1 = x1[i];
    iy1 = y1[i];
    ix2 = x2[i];
    iy2 = y2[i];    
    ix3 = x3[i];
    iy3 = y3[i];
    ix4 = x4[i];
    iy4 = y4[i];
    iscore = scores[i];

    pos = i + 1;
    // NMS iterations, note that N changes if detection boxes fall below threshold
    while (pos < ndets) {
//       float ovr = rotate_iou(irbox, rbboxes[pos]); 
      float ovr = rotate_iou(ix1, iy1, ix2, iy2, ix3, iy3, ix4, iy4, 
                             x1[pos], y1[pos], x2[pos], y2[pos], 
                             x3[pos], y3[pos], x4[pos], y4[pos]);      
      scalar_t weight = 1.;
      if (method == 1) {
        if (ovr > threshold) weight = 1 - ovr;
      }
      else if (method == 2) {
        weight = std::exp(-(ovr * ovr) / sigma);
      }
      else {
        // original NMS
        if (ovr > threshold) {
            weight = 0;
        }
        else {
            weight = 1;
        }
      }
      scores[pos] = weight * scores[pos];

      // if box score falls below threshold, discard the box by
      // swapping with last box update N
      if (scores[pos] < min_score) {
//         rbboxes[pos] = rbboxes[ndets - 1];
        x1[pos] = x1[ndets - 1];
        y1[pos] = y1[ndets - 1];
        x2[pos] = x2[ndets - 1];
        y2[pos] = y2[ndets - 1];
        x3[pos] = x3[ndets - 1];
        y3[pos] = y3[ndets - 1];
        x4[pos] = x4[ndets - 1];
        y4[pos] = y4[ndets - 1];
        scores[pos] = scores[ndets - 1];
        inds[pos] = inds[ndets - 1];
        ndets = ndets -1;
        pos = pos - 1;
      }
      pos = pos + 1;
    }
  }
  at::Tensor result = at::zeros({10, ndets}, dets.options());
  result[0] = x1_t.slice(0, 0, ndets);
  result[1] = y1_t.slice(0, 0, ndets);
  result[2] = x2_t.slice(0, 0, ndets);
  result[3] = y2_t.slice(0, 0, ndets);
  result[4] = x3_t.slice(0, 0, ndets);
  result[5] = y3_t.slice(0, 0, ndets);
  result[6] = x4_t.slice(0, 0, ndets);
  result[7] = y4_t.slice(0, 0, ndets);
  result[8] = scores_t.slice(0, 0, ndets);
  result[9] = inds_t.slice(0, 0, ndets);

  result =result.t().contiguous();
  return result;
}

at::Tensor soft_rnms(const at::Tensor& dets, const float threshold,
                    const unsigned char method, const float sigma, const float min_score) {
  at::Tensor result;
  AT_DISPATCH_FLOATING_TYPES(dets.scalar_type(), "soft_rnms", [&] {
    result = soft_rnms_cpu_kernel<scalar_t>(dets, threshold, method, sigma, min_score);
  });
  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("soft_rnms", &soft_rnms, "soft rotate non-maximum suppression");
}
