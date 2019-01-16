#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <math.h>
#include <float.h>
#include "flow_align_cuda_kernel.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
    i += blockDim.x * gridDim.x)


__global__ void FlowAlignForward_kernel(const int nthreads,const  int height, const int width, const int channels, const double* bottom, const double* flow,double* top)
{

  CUDA_1D_KERNEL_LOOP(index, nthreads)
  {
      // get feature position
      int w = index % width;
      int h = (index / width) %  height;
      int c  = (index / width /  height) % channels;
      int n  = index / width /  height / channels;
      // get flow
      int ind_flow_x = w+h*width+n*height*width*2;
      int ind_flow_y = w+h*width+width*height+n*height*width*2;
      double flo_x = flow[ind_flow_x];
      double flo_y = flow[ind_flow_y];
      // get bilinear positions
      double w_flo = w+flo_x;
      double h_flo = h+flo_y;
      if ( h_flo<0 || h_flo>height-1 || w_flo<0 || w_flo>width-1 )
      {
        //outof image, we set it to 0.
        top[index] = 0;
      }
      else{
        int h_start = floor(h_flo);
        int w_start = floor(w_flo);
        int nc_start = n*height*width*channels+c*height*width;
        double h_ratio = h_flo - (double)h_start;
        double w_ratio = w_flo - (double)w_start;
        int upleft = nc_start+w_start+width*h_start;
        int upright = upleft+1;
        int downleft = upleft+width;
        int downright = downleft+1;

        top[index] = bottom[upleft] * (1.-h_ratio) * (1.-w_ratio)
                         + bottom[upright] * (1.-h_ratio) * (w_ratio)
                         + bottom[downleft] * (h_ratio) * (1.-w_ratio)
                         + bottom[downright] * (h_ratio) * (w_ratio);
      }
  }
}

__global__ void FlowAlignBackward_kernel(const int nthreads, const int height, const int width, const int channels, const double* topdiff,const double* bottom, const double*flow, double* bottomdiff, double* flowdiff)
{
  CUDA_1D_KERNEL_LOOP(index, nthreads)
  {
      // get feature position
      int w = index % width;
      int h = (index / width) %  height;
      int c  = (index / width /  height) % channels;
      int n  = index / width /  height / channels;
      // get flow
      int ind_flow_x = w+h*width+n*height*width*2;
      int ind_flow_y = w+h*width+width*height+n*height*width*2;
      double flo_x = flow[ind_flow_x];
      double flo_y = flow[ind_flow_y];
      // get bilinear positions
      double w_flo = w+flo_x;
      double h_flo = h+flo_y;
      if ( h_flo<0 || h_flo>height-1 || w_flo<0 || w_flo>width-1 )
      {
        //outof image, no grad propagated.
        continue;
      }
      else
      {
        int h_start = floor(h_flo);
        int w_start = floor(w_flo);
        int nc_start = n*height*width*channels+c*height*width;
        double h_ratio = h_flo - (double)h_start;
        double w_ratio = w_flo - (double)w_start;
        int upleft = nc_start+w_start+width*h_start;
        int upright = upleft+1;
        int downleft = upleft+width;
        int downright = downleft+1;

        //bottom diff
        atomicAdd(bottomdiff+upleft, topdiff[index] * (1.-h_ratio) * (1.-w_ratio));
        atomicAdd(bottomdiff+upright, topdiff[index] *  (1.-h_ratio) * (w_ratio));
        atomicAdd(bottomdiff+downleft, topdiff[index] * (h_ratio) * (1.-w_ratio));
        atomicAdd(bottomdiff+downright, topdiff[index] * (h_ratio) * (w_ratio));

        //approximate flow diff
        double f1 = bottom[upleft];
        double f2 = bottom[upright];
        double f3 = bottom[downleft];
        double f4 = bottom[downright];
        //recalculate top rather than saving it temporarily in the layer.
        /*
        top[index] = bottom[upleft] * (1.-h_ratio) * (1.-w_ratio)
                         + bottom[upright] * (1.-h_ratio) * (w_ratio)
                         + bottom[downleft] * (h_ratio) * (1.-w_ratio)
                         + bottom[downright] * (h_ratio) * (w_ratio);
                         */        
        //calculate dx
        double dx = -f1*(1.-h_ratio) + f2*(1.-h_ratio) - f3*(h_ratio) + f4*(h_ratio);
        //calculate dy
        double dy = -f1*(1.-w_ratio) - f2*(w_ratio) + f3*(1.-w_ratio) + f4*(w_ratio);
        atomicAdd(flowdiff+ind_flow_x, topdiff[index]*dx);
        atomicAdd(flowdiff+ind_flow_y, topdiff[index]*dy);
      }
  }
}

int FlowAlignForward(const int batches, const int height, const int width, const int channels, const double* bottom, const double* flow,double* top, cudaStream_t stream)
{
  const int kThreadPerBlock = 512;
  const int nthreads = batches*height*width*channels;
  cudaError_t err;

  FlowAlignForward_kernel<<<(nthreads+kThreadPerBlock-1)/kThreadPerBlock, kThreadPerBlock, 0, stream>>>(
    nthreads, height, width, channels, bottom, flow, top);

  err = cudaGetLastError();
  if(cudaSuccess != err) {
      fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
      exit( -1 );
  }

  return 1;
}


int FlowAlignBackward(const int batches, const int height, const int width, const int channels, const double* topdiff, const double*bottom, const double*flow, double* bottomdiff, double* flowdiff, cudaStream_t stream)
{
  const int kThreadPerBlock = 512;
  const int nthreads = batches*height*width*channels;
  cudaError_t err;
  FlowAlignBackward_kernel<<<(nthreads+kThreadPerBlock-1)/kThreadPerBlock, kThreadPerBlock, 0, stream>>>(
    nthreads, height, width, channels, topdiff, bottom, flow, bottomdiff, flowdiff);

  err = cudaGetLastError();
  if(cudaSuccess != err) {
      fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
      exit( -1 );
  }

  return 1;
}


#ifdef __cplusplus
}
#endif
