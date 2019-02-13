import torch
from torch.autograd import Function
import torch.nn as nn
from .._ext import flow_align

class FlowAlignFunction(Function):
  """
  Args:
  spatial scale marks the difference between down sampled features and original sized flows.
  """
  @staticmethod
  def forward(ctx, features, flows):
    #self._flows = torch.tensor(flows)
    #self._features = torch.tensor(features)
    #self.feature_size = features.size()
    feature_size = features.size()
    ctx.feature_size = feature_size
    if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
      ctx.save_for_backward(features, flows)
    
    batch_size_flo, num_channels_flo, data_height_flo, data_width_flo = flows.size()
    batch_size, num_channels, data_height, data_width = features.size()
    assert(batch_size_flo==batch_size_flo and data_height_flo==data_height and data_width_flo==data_width)
    
    output = features.new(batch_size, num_channels, data_height, data_width)
    if features.is_cuda:
      flow_align.flow_align_forward_cuda(features, flows, output)
    else:
      raise NotImplementedError
    return output

  @staticmethod
  def backward(ctx, grad_output):
    if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
      features, flows = ctx.saved_variables
      feature_size = ctx.feature_size
      assert(feature_size is not None and grad_output.is_cuda)
      batch_size, num_channels, data_height, data_width = feature_size
      grad_feature = features.new(batch_size, num_channels, data_height,data_width).zero_()
      #hard code the channel to 2
      grad_flow = flows.new(batch_size, 2, data_height,data_width).zero_()
      flow_align.flow_align_backward_cuda(grad_output, features, flows, grad_feature, grad_flow)
      #self._flows = None
      #self._features = None

      return grad_feature, grad_flow
    else:
      return None, None
