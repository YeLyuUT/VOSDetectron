import torch
import torch.nn as nn

def metric_pixel_accurary_2d(y_pred, y_gt, ignore_last_label = True):
  ''' Pixel-wise comparison.
  Args:
  ignore_last_label: if the last label is dummy for training, then set it to True. Default True.
  '''
  assert(len(y_pred.shape)==len(y_gt.shape))
  assert(3==len(y_gt.shape))
  out_cmp = torch.eq(y_pred,y_gt)
  numerator = torch.sum(out_cmp,dtype=torch.float)
  if not ignore_last_label:
    denominator = torch.prod(torch.Tensor(list(y_pred.shape)).cuda())
  else:
    denominator = torch.sum(y_gt<(y_pred.shape[1]-1),dtype=torch.float)
  return torch.div(numerator,denominator)
  
