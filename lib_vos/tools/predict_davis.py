import sys
import os
from os import path as osp
import pprint
import subprocess
from collections import defaultdict
def addPath(path):
    if path not in sys.path:
        sys.path.append(path)      
imdbPath = osp.abspath('../../imdb')
home = osp.abspath('../..')
lib_vos_path = osp.abspath('../../lib_vos')
lib_path = osp.abspath('../../lib')
tool_path = osp.abspath('../../tools')
addPath(lib_vos_path)
addPath(lib_path)
addPath(home)
addPath(imdbPath)
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from vos_modeling import vos_model_builder
from vos import davis_db
from utils.timer import Timer
import torch
import torch.nn as nn
import nn as mynn
from torch.autograd import Variable
from core.test import im_detect_all
import distutils.util
import utils.misc as misc_utils
import utils.net as net_utils
import utils.vis as vis_utils
import datasets.dummy_datasets as datasets
import argparse

def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Demonstrate mask-rcnn results')
    parser.add_argument(
        '--dataset', required=True,
        help='training dataset')

    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='optional config file')
    parser.add_argument(
        '--set', dest='set_cfgs',
        help='set config keys, will overwrite config in the cfg_file',
        default=[], nargs='+')

    parser.add_argument(
        '--no_cuda', dest='cuda', help='whether use CUDA', action='store_false')

    parser.add_argument('--load_ckpt', help='path of checkpoint to load')

    parser.add_argument('--no_overwrite',help='not overwrite output', action='store_false')
    
    parser.add_argument(
        '--image_dir',
        help='directory to load images for demo')
    parser.add_argument(
        '--images', nargs='+',
        help='images to infer. Must not use with --image_dir')
    parser.add_argument(
        '--output_dir',
        help='directory to save demo results',
        default="./Output/")
    parser.add_argument(
        '--merge_pdfs', type=distutils.util.strtobool, default=True)

    args = parser.parse_args()

    return args

if __name__=='__main__':
  args = parse_args()
  print('Called with args:')
  print(args)

  if not torch.cuda.is_available():
      sys.exit("Need a CUDA device to run the code.")

  if args.cuda or cfg.NUM_GPUS > 0:
      cfg.CUDA = True
  else:
      raise ValueError("Need Cuda device to run !")

  if args.dataset.startswith("coco"):
      dataset = datasets.get_coco_dataset()
      cfg.MODEL.NUM_CLASSES = len(dataset.classes)
      print('cfg.MODEL.NUM_CLASSES:',cfg.MODEL.NUM_CLASSES)
  else:
      raise ValueError('Unexpected dataset name: {}'.format(args.dataset))

  #Add unknow class type if necessary.
  if cfg.MODEL.ADD_UNKNOWN_CLASS is True:
      cfg.MODEL.NUM_CLASSES +=1

  cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
      cfg_from_list(args.set_cfgs)
  assert bool(args.load_ckpt)
  assert_and_infer_cfg()
  maskRCNN = vos_model_builder.Generalized_VOS_RCNN()
  
  if args.cuda:
      maskRCNN.cuda()
  
  if args.load_ckpt:
      load_name = args.load_ckpt
      print("loading checkpoint %s" % (load_name))
      checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
      net_utils.load_ckpt_no_mapping(maskRCNN, checkpoint['model'])
  
  maskRCNN = mynn.DataParallel(maskRCNN, cpu_keywords=['im_info', 'roidb'],minibatch=True, device_ids=[0])
  
  maskRCNN.eval()
  db = davis_db.DAVIS_imdb(db_name="DAVIS", split = 'train', cls_mapper = None)
  
  for seq_idx in range(db.get_num_sequence()):
    db.set_to_sequence(seq_idx)
    seq_name = db.get_current_seq_name()
    cur_output_dir = osp.join(args.output_dir,seq_name)
    if args.no_overwrite is True and osp.exist(osp.join(cur_output_dir,'results.pdf')):
      continue
    if not osp.isdir(cur_output_dir):
      os.makedirs(cur_output_dir)
      assert(cur_output_dir)
    for idx in range(db.get_current_seq_length()):
      im = db.get_image_cv2(idx)
      assert im is not None
      timers = defaultdict(Timer)
      cls_boxes, cls_segms, cls_keyps = im_detect_all(maskRCNN, im, timers=timers)
      im_name = '%03d-%03d'%(seq_idx,idx)
      print(osp.join(seq_name,im_name))
      vis_utils.vis_one_image(
          im[:, :, ::-1],  # BGR -> RGB for visualization
          im_name,
          cur_output_dir,
          cls_boxes,
          cls_segms,
          cls_keyps,
          dataset=dataset,
          box_alpha=0.3,
          show_class=True,
          thresh=0.7,
          kp_thresh=2
      )

    if args.merge_pdfs:
        merge_out_path = '{}/results.pdf'.format(cur_output_dir)
        if os.path.exists(merge_out_path):
            os.remove(merge_out_path)
        command = "pdfunite {}/*.pdf {}".format(cur_output_dir,
                                                merge_out_path)
        subprocess.call(command, shell=True)
    
