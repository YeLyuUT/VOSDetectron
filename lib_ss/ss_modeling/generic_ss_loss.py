import torch

def ss_loss_2d_single_scale(weight=None, ignore_index=-100, reduction='elementwise_mean'):
  # size_average and reduce are deprecated.
  return torch.nn.CrossEntropyLoss(weight=weight, size_average=None, ignore_index=ignore_index, reduce=None, reduction=reduction)