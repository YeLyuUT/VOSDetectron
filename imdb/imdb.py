# --------------------------------------------------------
# Panoptic Segmentation CNN
# Copyright (c) 2019 Ye Lyu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import os.path as osp
from PIL import Image
import numpy as np
import scipy.sparse

class imdb():
  """ 
  Image database for segmentation. 
  Different segments in input labels are of different colors.
  """
  def __init__(self,name):
    self._name = name
    self._classes = []
    self._image_index = []
    self._ps_db = None
    self._ps_db_handler = self.default_ps_db_handler
    self.config = {}

  @property
  def name(self):
    return self._name

  @property
  def num_classes(self):
    return len(self._classes)

  @property
  def image_index(self):
    return self._image_index

  @property
  def num_images(self):
    return len(self._image_index)

  def get_image_path_at(self,idx):
    raise NotImplementedError

  '''
  get corresponding label path of an image 
  at path of 'get_image_path_at(idx)'.
  '''
  def get_label_path_at(self,idx):
    raise NotImplementedError

  @property
  def ps_db_handler(self):
    return self._ps_db_handler

  @ps_db_handler.setter
  def ps_db_handler(self,val):
    self._ps_db_handler = val

  def default_ps_db_handler(self):
    raise NotImplementedError

  @property
  def ps_db(self):
    if self._ps_db is None:
      self._ps_db = self._ps_db_handler()
    return self._ps_db

