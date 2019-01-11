import os.path as osp
import os
import sys
dir_path = osp.dirname(osp.realpath(__file__))
sys.path.append(osp.abspath(osp.join(dir_path,'../lib')))
import nn as mynn
import torch.nn as nn

DEBUG = False

def init_module(m, recursively = False):
  for child_m in self.upsampleModules.children():
    init_func(child_m, recursively)

def init_func(m, recursively = False):
  if DEBUG:
    print(m)
  if isinstance(m, nn.Conv2d):
    mynn.init.XavierFill(m.weight)
    if DEBUG:
      print('XavierFill')
    if m.bias is not None:
      init.constant_(m.bias, 0)
  elif recursively is True and (isinstance(m,nn.Sequential) or isinstance(m,nn.ModuleList)):
      init_func(mm)
  else:
      pass
