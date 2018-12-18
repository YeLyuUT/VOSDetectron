from lib import nn as mynn
import torch.nn as nn

DEBUG = True

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
