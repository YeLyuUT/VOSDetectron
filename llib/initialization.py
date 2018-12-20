import os.path as osp
import os
import sys
dir_path = osp.dirname(osp.realpath(__file__))
sys.path.append(osp.abspath(osp.join(dir_path,'../lib')))
import nn as mynn
import torch.nn as nn

DEBUG = False

def init_func(m):
  if DEBUG:
    print(m)
  if isinstance(m, nn.Conv2d):
    mynn.init.XavierFill(m.weight)
    if DEBUG:
      print('XavierFill')
    if m.bias is not None:
      init.constant_(m.bias, 0)
  elif isinstance(m,nn.Sequential):
    for mm in m.children():
      init_func(mm)
  elif isinstance(m,nn.ModuleList):
    for mm in m.children():
      init_func(mm)
  else:
    if DEBUG:
      print('No init')
