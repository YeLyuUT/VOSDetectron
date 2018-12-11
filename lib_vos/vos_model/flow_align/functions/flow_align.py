import torch
from torch.autograd import Function
import torch.nn as nn
from ..ext import flow_align

class FlowAlignFunction(Function):
  """
  Args:
  spatial scale marks the difference between down sampled features and original sized flows.
  """
  def __init__(self):
    super(FlowAlignFunction,self).__init__()
    
  def forward(self,features,flows):
    self._flows = flows
    self.feature_size = features.size()
    batch_size_flo, num_channels_flo, data_height_flo, data_width_flo = self._flows.size()
    batch_size, num_channels, data_height, data_width = features.size()
    assert(batch_size_flo==batch_size_flo and data_height_flo==data_height and data_width_flo==data_width)
    
    output = features.new(batch_size, num_channels, data_height, data_width)
    if features.is_cuda:
      flow_align.flow_align_forward_cuda(features, self._flows, output)
    else:
      raise NotImplementedError
    return output

  def backward(self, grad_output):
    assert(self.feature_size is not None and grad_output.is_cuda)
    batch_size, num_channels, data_height, data_width = self.feature_size
    grad_input = self.rois.new(batch_size, num_channels, data_height,data_width).zero_()

    flow_align.flow_align_backward_cuda(grad_output, self._flows, grad_input)
    #delete the saved _flows
    del self._flows
    self._flows = None

    return grad_input, None