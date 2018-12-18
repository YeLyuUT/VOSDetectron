import torch
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
import PIL
from easydict import EasyDict


class Transformer():
  def __init__(self, 
                expected_blob_size, 
                brightness_factor_min = 0.9, brightness_factor_max = 1.1,
                contrast_factor_min = 0.9, contrast_factor_max = 1.1,
                gamma_min = 0.9, gamma_max = 1.1,
                hue_factor_min = -0.05, hue_factor_max = 0.05,
                saturation_factor_min = 0.9, saturation_factor_max = 1.1,
                angle_min = -15, angle_max = 15,
                shear_min = -15, shear_max = 15,
                scale_min = 0.8, scale_max = 1.5,
                hflip = True):
    """
    expected_blob_size: (h,w)
    hflip: if use random hflip
    """
    assert(brightness_factor_max>=brightness_factor_min)
    assert(contrast_factor_max>=contrast_factor_min)
    assert(gamma_max>=gamma_min)
    assert(hue_factor_max>=hue_factor_min)
    assert(saturation_factor_max>=saturation_factor_min)
    assert(angle_max>=angle_min)
    assert(shear_max>=shear_min)
    assert(scale_max>=scale_min)
    assert(hflip is True or hflip is False)
    self.brightness_factor_min = brightness_factor_min
    self.brightness_factor_max = brightness_factor_max
    self.contrast_factor_min = contrast_factor_min
    self.contrast_factor_max = contrast_factor_max
    self.gamma_min = gamma_min
    self.gamma_max = gamma_max
    self.hue_factor_min = hue_factor_min
    self.hue_factor_max = hue_factor_max
    self.saturation_factor_min = saturation_factor_min
    self.saturation_factor_max = saturation_factor_max
    self.angle_min = angle_min
    self.angle_max = angle_max
    self.shear_min = shear_min
    self.shear_max = shear_max
    self.scale_min = scale_min
    self.scale_max = scale_max
    self.hflip = hflip
    self.expected_blob_size = expected_blob_size
    self.TF_params = self._init_TF_dict(expected_blob_size)
    self.getNewRandomTransformForImage()

  def _init_TF_dict(self,expected_blob_size):
    TF_params = EasyDict()
    TF_params.brightness_factor = 1.0
    TF_params.contrast_factor = 1.0
    TF_params.gamma = 1.0
    TF_params.hue_factor = 0.0
    TF_params.saturation_factor = 1.0
    TF_params.angle = 0
    TF_params.shear = 0
    TF_params.scale = 1.0
    TF_params.size = self.expected_blob_size
    TF_params.hflip = False
    return TF_params

  def getNewRandomTransformForImage(self):
    brightness_factor = np.random.uniform(self.brightness_factor_min, self.brightness_factor_max)
    self.TF_params.brightness_factor = brightness_factor

    contrast_factor = np.random.uniform(self.contrast_factor_min,self.contrast_factor_max)
    self.TF_params.contrast_factor = contrast_factor

    gamma = np.random.uniform(self.gamma_min,self.gamma_max)
    self.TF_params.gamma = gamma

    hue_factor = np.random.uniform(self.hue_factor_min,self.hue_factor_max)
    self.TF_params.hue_factor = hue_factor

    saturation_factor = np.random.uniform(self.saturation_factor_min,self.saturation_factor_max)
    self.TF_params.saturation_factor = saturation_factor

    angle = np.random.uniform(self.angle_min,self.angle_max)
    self.TF_params.angle = angle

    shear = np.random.uniform(self.shear_min,self.shear_max)
    self.TF_params.shear = shear

    scale = np.random.uniform(self.scale_min,self.scale_max)
    self.TF_params.scale = scale

    if self.hflip:
      self.TF_params.hflip = True if np.random.randint(0,1) else False

  def getNewRandomCrop(self, img):
    img_w, img_h = img.size
    #set crop w, h
    w = int(np.around(self.TF_params.size[1]/self.TF_params.scale,decimals = 0))
    h = int(np.around(self.TF_params.size[0]/self.TF_params.scale,decimals = 0))
    i_max = img_h-h
    i_min = 0
    j_max = img_w-w
    j_min = 0
    i = np.random.randint(i_min,i_max)
    j = np.random.randint(j_min,j_max)
    self.crop_tuple = (i,j,h,w)

  def __call__(self,img,gt = None, batch_size = 1):
    """
    Args:
     blob: blob to be transformed.
    """
    #color
    img = TF.adjust_brightness(img,self.TF_params.brightness_factor)
    img = TF.adjust_contrast(img,self.TF_params.contrast_factor)
    img = TF.adjust_gamma(img,self.TF_params.gamma,gain=1)
    img = TF.adjust_hue(img,self.TF_params.hue_factor)
    img = TF.adjust_saturation(img,self.TF_params.saturation_factor)
    #affine
    #here we do not use translate and scale in affine function.
    scale=1.0
    translate=(0,0)
    #resample =  PIL.Image.BICUBIC or PIL.Image.NEAREST or PIL.Image.BILINEAR
    img = TF.affine(img, self.TF_params.angle, translate, scale, self.TF_params.shear, PIL.Image.BICUBIC, fillcolor=None)
    if gt is not None:
      gt = TF.affine(img, self.TF_params.angle, translate, scale, self.TF_params.shear, PIL.Image.NEAREST, fillcolor=None)

    if self.TF_params.hflip:
      img = TF.hflip(img)
      if gt is not None:
        gt = TF.hflip(gt)
    img_crops = []
    if gt is not None:
      gt_crops = []

    for b in range(batch_size):
      self.getNewRandomCrop(img)
      img_crop = TF.resized_crop(img,*self.crop_tuple,self.TF_params.size,interpolation = PIL.Image.BICUBIC)
      img_crops.append(img_crop)
      if gt is not None:
        gt_crop = TF.resized_crop(gt,*self.crop_tuple,self.TF_params.size,interpolation = PIL.Image.NEAREST)
        gt_crops.append(gt_crop)
    if gt is not None: 
        return img_crops, gt_crops
    else:
        return img_crops

