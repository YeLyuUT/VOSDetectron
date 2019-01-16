int flow_align_forward_cuda(THCudaDoubleTensor* bottom, THCudaDoubleTensor* flow, THCudaDoubleTensor* top);
int flow_align_backward_cuda(THCudaDoubleTensor* top_grad, THCudaDoubleTensor* bottom, THCudaDoubleTensor* flow, THCudaDoubleTensor* bottom_grad, THCudaDoubleTensor* flow_grad);

