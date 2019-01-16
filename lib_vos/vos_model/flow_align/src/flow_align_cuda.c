#include <THC/THC.h>
#include <math.h>
#include "flow_align_cuda_kernel.h"

extern THCState *state;

int flow_align_forward_cuda(THCudaDoubleTensor* bottom, THCudaDoubleTensor* flow, THCudaDoubleTensor* top)
{
  // Grab the input tensor
  double* bottom_flat = THCudaDoubleTensor_data(state, bottom);
  double* flow_flat = THCudaDoubleTensor_data(state, flow);
  double* top_flat = THCudaDoubleTensor_data(state, top);

  int batches = THCudaDoubleTensor_size(state,bottom,0);
  int channels = THCudaDoubleTensor_size(state,bottom,1);
  int height = THCudaDoubleTensor_size(state,bottom,2);
  int width = THCudaDoubleTensor_size(state,bottom,3);

  cudaStream_t stream = THCState_getCurrentStream(state);

  return FlowAlignForward(batches, height, width, channels, bottom_flat, flow_flat, top_flat, stream);
}

int flow_align_backward_cuda(THCudaDoubleTensor* top_grad, THCudaDoubleTensor* bottom, THCudaDoubleTensor* flow, THCudaDoubleTensor* bottom_grad, THCudaDoubleTensor* flow_grad)
{
  // Grab the input tensor
  double* top_grad_flat = THCudaDoubleTensor_data(state, top_grad);
  double* bottom_flat = THCudaDoubleTensor_data(state, bottom);
  double* flow_flat = THCudaDoubleTensor_data(state, flow);
  double* bottom_grad_flat = THCudaDoubleTensor_data(state, bottom_grad);
  double* flow_grad_flat = THCudaDoubleTensor_data(state, flow_grad);

  int batches = THCudaDoubleTensor_size(state,top_grad,0);
  int channels = THCudaDoubleTensor_size(state,top_grad,1);
  int height = THCudaDoubleTensor_size(state,top_grad,2);
  int width = THCudaDoubleTensor_size(state,top_grad,3);

  cudaStream_t stream = THCState_getCurrentStream(state);

  return FlowAlignBackward(batches, height, width, channels, top_grad_flat, bottom_flat, flow_flat, bottom_grad_flat, flow_grad_flat, stream);
}


