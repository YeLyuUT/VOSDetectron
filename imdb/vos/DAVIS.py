from vos_imdb import vos_imdb
import sys
import os
from os import path as osp

from .config import cfg

home_path = osp.join(cfg.DAVIS.HOME)
if not home_path in sys.path:
  sys.path.append(home_path)

from davis import cfg as cfg_davis

splits = ['trainval','test-dev']
class DAVIS_imdb(vos_imdb):
  def __init__(self,db_name="DAVIS",split = 'trainval'):
    super().__init__(db_name)
    self.split = split

    cfg = 






