#ifndef _FLOW_ALIGN_KERNEL
#define _FLOW_ALIGN_KERNEL

#ifdef __cplusplus
extern "C"{
#endif

__global__ void FlowAlignForward_kernel(const int nthreads, int height, int width, int channels, const float* bottom, const float* flow,float* top);

int FlowAlignForward(const int batches, int height, int width, int channels, const float* bottom, const float* flow,float* top, cudaStream_t stream);

__global__ void FlowAlignBackward_kernel(const int nthreads, int height, int width, int channels, const float* topdiff,const float*flow, float* bottomdiff);

int FlowAlignBackward(const int batches, int height, int width, int channels, const float* topdiff,const float*flow, float* bottomdiff, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
#endif