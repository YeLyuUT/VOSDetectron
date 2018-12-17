#ifndef _FLOW_ALIGN_KERNEL
#define _FLOW_ALIGN_KERNEL

#ifdef __cplusplus
extern "C"{
#endif

__global__ void FlowAlignForward_kernel(const int nthreads,const  int height, const int width, const int channels, const float* bottom, const float* flow,float* top);

int FlowAlignForward(const int batches,const  int height,const  int width,const  int channels, const float* bottom, const float* flow,float* top, cudaStream_t stream);

__global__ void FlowAlignBackward_kernel(const int nthreads, const int height, const int width, const int channels, const float* topdiff, const float*bottom,const float*flow, float* bottomdiff, float* flowdiff);

int FlowAlignBackward(const int batches, const int height, const int width, const int channels, const float* topdiff, const float*bottom,const float*flow, float* bottomdiff, float* flowdiff, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
#endif