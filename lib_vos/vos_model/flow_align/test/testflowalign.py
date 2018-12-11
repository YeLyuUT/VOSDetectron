import torch
import torch.nn as nn

def _flow_downsample_convolutional_layer(self,spatial_scale):
  assert(spatial_scale<=1)
  inv_scale = int(1.0/spatial_scale)
  kernel_size = inv_scale
  stride = inv_scale
  conv_flow_downsample = nn.Conv2d(2,2,kernel_size = kernel_size,stride = stride,padding = 0, dilation = 1,groups = 1,bias = False)
  init_weights = torch.zeros(self.conv_flow_downsample.weight.shape)
  for idx in range(init_weights.shape[0]):
    init_weights[idx,idx,:,:] = spatial_scale**3
  torch.nn.init.constant(self.conv_flow_downsample,init_weights)
  return conv_flow_downsample



if __name__=='__main__':
  conv_flow_downsample = _flow_downsample_convolutional_layer(spatial_scale)
  
