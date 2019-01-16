import torch
import torch.nn as nn
from ..functions.flow_align import FlowAlignFunction

class FlowAlign(nn.Module):
  def __init__(self,spatial_scale):
    super(FlowAlign,self).__init__()
    self.spatial_scale = spatial_scale
    self.feature_size = None
    self.conv_flow_downsample = self._flow_downsample_convolutional_layer(spatial_scale)

  def _flow_downsample_convolutional_layer(self,spatial_scale):
    assert(spatial_scale<=1.0 and spatial_scale in [1.0,0.5,0.25,0.125,0.0625,0.03125])
    inv_scale = int(1.0/spatial_scale)
    kernel_size = inv_scale
    stride = inv_scale
    conv_flow_downsample = nn.Conv2d(2,2,kernel_size = kernel_size,stride = stride,padding = 0, dilation = 1,groups = 1,bias = False)
    init_weights = torch.zeros(conv_flow_downsample.weight.shape)
    for idx in range(init_weights.shape[0]):
      init_weights[idx,idx,:,:] = spatial_scale**3
    conv_flow_downsample.weight = torch.nn.Parameter(init_weights)
    #This layer should not be trained.
    for p in conv_flow_downsample.parameters():
      p.requires_grad = False
    return conv_flow_downsample

  def forward(self,features,flows):
    _flows = None
    if self.spatial_scale!=1.0:
      #scale the flow for align operation.
      _flows = self.conv_flow_downsample(flows)
    else:
      _flows = flows
    return FlowAlignFunction()(features, _flows)
