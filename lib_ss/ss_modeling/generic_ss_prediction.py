import torch
import torch.nn as nn
from PIL import Image
import numpy as np
  
def predict_pixel_2d(y_pred):
  assert(len(y_pred.shape)==4)
  y_pred = torch.argmax(y_pred, dim=1, keepdim=False)
  return y_pred
