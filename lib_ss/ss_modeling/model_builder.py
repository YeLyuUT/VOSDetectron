import torch
import torch.nn as nn
from .IDA_ss_head import IDA_ss_outputs
from .IDA import IDA_ResNet101_conv5_body, IDA_ResNet50_conv5_body
from .generic_ss_loss import ss_loss_2d_single_scale
from .generic_ss_metric import metric_pixel_accurary_2d
import os
import sys
import os.path as osp


class Generic_SS_Model(nn.Module):
  def __init__(self, class_num, ignore_index = -100, weight = None):
    super(Generic_SS_Model,self).__init__()
    self.backbone = self._getBackBone()
    self.head = self._getHead(class_num)
    self.loss_func = self._loss_func(weight=weight, ignore_index=ignore_index, reduction='elementwise_mean')
    self.return_dict = {}
    self.return_dict['loss'] = None
    self.return_dict['metric'] = None
    self.mapping_to_detectron = None

  def forward(self, x,y):
    outputs_backbone = self.backbone(x)
    output_pred = self.head(outputs_backbone)
    loss = self.loss_func(output_pred, y)
    accuracy = metric_pixel_accurary_2d(output_pred, y)
    self.return_dict['loss'] = loss
    self.return_dict['metric'] = accuracy
    return self.return_dict

  def _getBackBone(self):
    return IDA_ResNet50_conv5_body()
    #return IDA_ResNet101_conv5_body()

  def _getHead(self,class_num):
    return IDA_ss_outputs(class_num)

  def _loss_func(self, weight, ignore_index, reduction):
    return ss_loss_2d_single_scale(weight=weight, ignore_index=ignore_index, reduction=reduction)

  @property
  def detectron_weight_mapping(self):
    conv_body_mapping = {}
    if self.mapping_to_detectron is None:
      self.mapping_to_detectron = {}
      d_wmap = {}  # detectron_weight_mapping
      d_orphan = []  # detectron orphan weight list
      for name, m_child in self.named_children():
        if hasattr(m_child, 'detectron_weight_mapping'):
          print(name+' has detectron_weight_mapping')
          if list(m_child.parameters()):  # if module has any parameter
            child_map, child_orphan = m_child.detectron_weight_mapping()
            d_orphan.extend(child_orphan)
            for key, value in child_map.items():
              new_key = name + '.' + key
              d_wmap[new_key] = value
      conv_body_mapping = d_wmap
      self.orphans_in_detectron = d_orphan

      for key, value in conv_body_mapping.items():
        self.mapping_to_detectron['conv_body.'+key] = value

    return self.mapping_to_detectron, self.orphans_in_detectron


    