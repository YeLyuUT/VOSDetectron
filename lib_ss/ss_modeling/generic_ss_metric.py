import torch
import torch.nn as nn

def metric_pixel_accurary_2d(y_pred, y_gt):
  ''' Pixel-wise comparison.
  '''
  assert(len(y_pred.shape)-1==len(y_gt.shape))
  assert(len(y_pred.shape)==4)
  y_pred = torch.argmax(y_pred, dim=1, keepdim=False)
  out_cmp = torch.eq(y_pred,y_gt)
  return torch.sum(out_cmp)/torch.prod(torch.Tensor(list(x.shape)))
