int flow_align_forward_cuda(THCudaTensor* bottom, THCudaTensor* flow, THCudaTensor* top);
int flow_align_backward_cuda(THCudaTensor* top_grad, THCudaTensor* flow, THCudaTensor* bottom_grad);