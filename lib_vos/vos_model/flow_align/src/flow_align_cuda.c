#include <THC/THC.h>
#include <math.h>
#include "flow_align_cuda_kernel.h"

extern THCState *state;

int flow_align_forward_cuda(THCudaTensor* bottom, THCudaTensor* flow, THCudaTensor* top)
{
  // Grab the input tensor
  float* bottom_flat = THCudaTensor_data(state, bottom);
  float* flow_flat = THCudaTensor_data(state, flow);
  float* top_flat = THCudaTensor_data(state, top);

  int batches = THCudaTensor_size(state,bottom,0);
  int channels = THCudaTensor_size(state,bottom,1);
  int height = THCudaTensor_size(state,bottom,2);
  int width = THCudaTensor_size(state,bottom,3);

  cudaStream_t stream = THCState_getCurrentStream(state);

  return FlowAlignForward(batches, height, width, channels, bottom_flat, flow_flat, top_flat, stream);
}

int flow_align_backward_cuda(THCudaTensor* top_grad, THCudaTensor* bottom, THCudaTensor* flow, THCudaTensor* bottom_grad, THCudaTensor* flow_grad)
{
  // Grab the input tensor
  float* top_grad_flat = THCudaTensor_data(state, top_grad);
  float* bottom_flat = THCudaTensor_data(state, bottom);
  float* flow_flat = THCudaTensor_data(state, flow);
  float* bottom_grad_flat = THCudaTensor_data(state, bottom_grad);
  float* flow_grad_flat = THCudaTensor_data(state, flow_grad);

  int batches = THCudaTensor_size(state,top_grad,0);
  int channels = THCudaTensor_size(state,top_grad,1);
  int height = THCudaTensor_size(state,top_grad,2);
  int width = THCudaTensor_size(state,top_grad,3);

  cudaStream_t stream = THCState_getCurrentStream(state);

  return FlowAlignBackward(batches, height, width, channels, top_grad_flat, bottom_flat, flow_flat, bottom_grad_flat, flow_grad_flat, stream);
}
