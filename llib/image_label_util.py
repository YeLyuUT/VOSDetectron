import numpy as np
from copy import copy
def blend_img_with_instance_label(img,label,w1,w2):
  assert(w1>=0 and w1<=1)
  assert(w2>=0 and w2<=1)
  img_cpy = copy(img)
  label = np.array(np.all(gt[:,:,:]==0,axis=2,keepdims=True),dtype=np.uint8)
  unmasked_img = np.array(img_cpy*label,dtype=np.uint8)
  mask_img = img-unmasked_img
  blend = unmasked_img+np.array(mask_img*w1+gt*w2,dtype=np.uint8)
  return blend

def blend_img_with_label(img,label,w1,w2):
  blend = img*w1+label*w2
  return blend