""" Training script for steps_with_decay policy"""
import argparse
import os
from os import path as osp
import sys
import pickle
import resource
import traceback
import logging
from collections import defaultdict

import numpy as np
import yaml
import torch
from torch.autograd import Variable
import torch.nn as nn
import cv2
cv2.setNumThreads(0)  # pytorch issue 1355: possible deadlock in dataloader

def addPath(path):
    if path not in sys.path:
        sys.path.append(path)

home = osp.abspath('../..')
lib_vos_path = osp.abspath('../../lib_vos')
lib_path = osp.abspath('../../lib')
imdb_path = osp.abspath(osp.join(osp.dirname(__file__),'../../imdb/'))
addPath(lib_vos_path)
addPath(lib_path)
addPath(home)
addPath(imdb_path)
#import _init_paths  # pylint: disable=unused-import
import nn as mynn
import utils.net as net_utils
import utils.misc as misc_utils
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from roi_data.minibatch import get_minibatch
from vos_datasets.sequence_roidb import sequenced_roidb_for_training
from vos_roi_data.vos_loader import RoiDataLoader, MinibatchSampler, BatchSampler, collate_minibatch
#from roi_data.loader import RoiDataLoader, MinibatchSampler, BatchSampler, collate_minibatch
#from modeling.model_builder import Generalized_RCNN
from vos_modeling import vos_model_builder
from vos_datasets.sequence_roidb import sequenced_roidb_for_training_from_db
import utils.blob as blob_utils
from utils.detectron_weight_helper import load_detectron_weight
from utils.logging import setup_logging
from utils.timer import Timer
from utils.training_stats import TrainingStats
from vos_test import im_detect_all
import distutils.util
import utils.misc as misc_utils
import utils.net as net_utils
import utils.vis as vis_utils
from vos.davis_db import DAVIS_imdb, image_saver
import subprocess
from numpy import random as npr
# Set up logging and load config options
logger = setup_logging(__name__)
logging.getLogger('roi_data.loader').setLevel(logging.INFO)

# RuntimeError: received 0 items of ancdata. Issue: pytorch/pytorch#973
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Train a X-RCNN network')

    parser.add_argument(
        '--dataset', dest='dataset', required=True,
        help='Dataset to use')
    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='Config file for training (and optionally testing)')
    parser.add_argument(
        '--set', dest='set_cfgs',
        help='Set config keys. Key value sequence seperate by whitespace.'
             'e.g. [key] [value] [key] [value]',
        default=[], nargs='+')

    parser.add_argument(
        '--disp_interval',
        help='Display training info every N iterations',
        default=20, type=int)
    parser.add_argument(
        '--no_cuda', dest='cuda', help='Do not use CUDA device', action='store_false')

    # Optimization
    # These options has the highest prioity and can overwrite the values in config file
    # or values set by set_cfgs. `None` means do not overwrite.
    parser.add_argument(
        '--bs', dest='batch_size',
        help='Explicitly specify to overwrite the value comed from cfg_file.',
        type=int)
    parser.add_argument(
        '--nw', dest='num_workers',
        help='Explicitly specify to overwrite number of workers to load data. Defaults to 4',
        type=int)
    parser.add_argument(
        '--iter_size',
        help='Update once every iter_size steps, as in Caffe.',
        default=1, type=int)

    parser.add_argument(
        '--o', dest='optimizer', help='Training optimizer.',
        default=None)
    parser.add_argument(
        '--lr', help='Base learning rate.',
        default=None, type=float)
    parser.add_argument(
        '--lr_decay_gamma',
        help='Learning rate decay rate.',
        default=None, type=float)

    # Epoch
    parser.add_argument(
        '--start_step',
        help='Starting step count for training epoch. 0-indexed.',
        default=0, type=int)

    # Resume training: requires same iterations per epoch
    parser.add_argument(
        '--resume',
        help='resume to training on a checkpoint',
        action='store_true')

    parser.add_argument(
        '--no_save', help='do not save anything', action='store_true')

    parser.add_argument(
        '--load_ckpt', help='checkpoint path to load')
    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument(
        '--use_tfboard', help='Use tensorflow tensorboard to log training info',
        action='store_true')

    parser.add_argument('--no_overwrite',help='not overwrite output', action='store_true')

    parser.add_argument(
        '--output_dir',
        help='directory to save demo results',
        default="./Output/")
    parser.add_argument(
        '--merge_pdfs', type=distutils.util.strtobool, default=True)

    return parser.parse_args()


def save_ckpt(output_dir, args, step, train_size, model, optimizer):
    """Save checkpoint"""
    if args.no_save:
        return
    ckpt_dir = os.path.join(output_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    save_name = os.path.join(ckpt_dir, 'model_step{}.pth'.format(step))
    if isinstance(model, mynn.DataParallel):
        model = model.module
    model_state_dict = model.state_dict()
    torch.save({
        'step': step,
        'train_size': train_size,
        'batch_size': args.batch_size,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()}, save_name)
    logger.info('save model: %s', save_name)

def load_ckpt(args, model):
    ### Load checkpoint
    load_name = args.load_ckpt
    logging.info("loading checkpoint %s", load_name)
    checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
    net_utils.load_ckpt_no_mapping(model, checkpoint['model'])    
    del checkpoint
    torch.cuda.empty_cache()

    ### Optimizer ###
    gn_param_nameset = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.GroupNorm):
            gn_param_nameset.add(name+'.weight')
            gn_param_nameset.add(name+'.bias')
    gn_params = []
    gn_param_names = []
    bias_params = []
    bias_param_names = []
    nonbias_params = []
    nonbias_param_names = []
    nograd_param_names = []
    for key, value in model.named_parameters():
        if value.requires_grad:
            if 'bias' in key:
                bias_params.append(value)
                bias_param_names.append(key)
            elif key in gn_param_nameset:
                gn_params.append(value)
                gn_param_names.append(key)
            else:
                nonbias_params.append(value)
                nonbias_param_names.append(key)
        else:
            nograd_param_names.append(key)
    assert (gn_param_nameset - set(nograd_param_names) - set(bias_param_names)) == set(gn_param_names)

    # Learning rate of 0 is a dummy value to be set properly at the start of training
    params = [
        {'params': nonbias_params,
         'lr': 0,
         'weight_decay': cfg.SOLVER.WEIGHT_DECAY},
        {'params': bias_params,
         'lr': 0 * (cfg.SOLVER.BIAS_DOUBLE_LR + 1),
         'weight_decay': cfg.SOLVER.WEIGHT_DECAY if cfg.SOLVER.BIAS_WEIGHT_DECAY else 0},
        {'params': gn_params,
         'lr': 0,
         'weight_decay': cfg.SOLVER.WEIGHT_DECAY_GN}
    ]
    # names of paramerters for each paramter
    param_names = [nonbias_param_names, bias_param_names, gn_param_names]

    if cfg.SOLVER.TYPE == "SGD":
        optimizer = torch.optim.SGD(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.TYPE == "Adam":
        optimizer = torch.optim.Adam(params)
    return optimizer    

def davis_saver(path, img):
    return image_saver(path, img)

def main():
    """Main function"""

    args = parse_args()
    print('Called with args:')
    print(args)

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    if args.cuda or cfg.NUM_GPUS > 0:
        cfg.CUDA = True
    else:
        raise ValueError("Need Cuda device to run !")

    if args.dataset == "davis2017":
        cfg.TRAIN.DATASETS = ('davis_val',)
        #For davis, coco category is used.
        cfg.MODEL.NUM_CLASSES = 81 #80 foreground + 1 background
    else:
        raise ValueError("Unexpected args.dataset: {}".format(args.dataset))

    cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    if cfg.MODEL.IDENTITY_TRAINING and cfg.MODEL.IDENTITY_REPLACE_CLASS:
        cfg.MODEL.NUM_CLASSES = 145
        cfg.MODEL.IDENTITY_TRAINING = False
        cfg.MODEL.ADD_UNKNOWN_CLASS = False

    #Add unknow class type if necessary.
    if cfg.MODEL.ADD_UNKNOWN_CLASS is True:
        cfg.MODEL.NUM_CLASSES +=1

    ### Adaptively adjust some configs ###
    original_batch_size = cfg.NUM_GPUS * cfg.TRAIN.IMS_PER_BATCH
    original_ims_per_batch = cfg.TRAIN.IMS_PER_BATCH
    original_num_gpus = cfg.NUM_GPUS
    if args.batch_size is None:
        args.batch_size = original_batch_size
    cfg.NUM_GPUS = torch.cuda.device_count()
    assert (args.batch_size % cfg.NUM_GPUS) == 0, \
        'batch_size: %d, NUM_GPUS: %d' % (args.batch_size, cfg.NUM_GPUS)
    cfg.TRAIN.IMS_PER_BATCH = args.batch_size // cfg.NUM_GPUS
    effective_batch_size = args.iter_size * args.batch_size
    
    print('effective_batch_size = batch_size * iter_size = %d * %d' % (args.batch_size, args.iter_size))
    print('Adaptive config changes:')
    print('    effective_batch_size: %d --> %d' % (original_batch_size, effective_batch_size))
    print('    NUM_GPUS:             %d --> %d' % (original_num_gpus, cfg.NUM_GPUS))
    print('    IMS_PER_BATCH:        %d --> %d' % (original_ims_per_batch, cfg.TRAIN.IMS_PER_BATCH))

    ### Adjust learning based on batch size change linearly
    # For iter_size > 1, gradients are `accumulated`, so lr is scaled based
    # on batch_size instead of effective_batch_size
    old_base_lr = cfg.SOLVER.BASE_LR
    cfg.SOLVER.BASE_LR *= args.batch_size / original_batch_size
    print('Adjust BASE_LR linearly according to batch_size change:\n'
          '    BASE_LR: {} --> {}'.format(old_base_lr, cfg.SOLVER.BASE_LR))

    ### Adjust solver steps
    step_scale = original_batch_size / effective_batch_size
    old_solver_steps = cfg.SOLVER.STEPS
    old_max_iter = cfg.SOLVER.MAX_ITER
    cfg.SOLVER.STEPS = list(map(lambda x: int(x * step_scale + 0.5), cfg.SOLVER.STEPS))
    cfg.SOLVER.MAX_ITER = int(cfg.SOLVER.MAX_ITER * step_scale + 0.5)
    print('Adjust SOLVER.STEPS and SOLVER.MAX_ITER linearly based on effective_batch_size change:\n'
          '    SOLVER.STEPS: {} --> {}\n'
          '    SOLVER.MAX_ITER: {} --> {}'.format(old_solver_steps, cfg.SOLVER.STEPS,
                                                  old_max_iter, cfg.SOLVER.MAX_ITER))

    # Scale FPN rpn_proposals collect size (post_nms_topN) in `collect` function
    # of `collect_and_distribute_fpn_rpn_proposals.py`
    #
    # post_nms_topN = int(cfg[cfg_key].RPN_POST_NMS_TOP_N * cfg.FPN.RPN_COLLECT_SCALE + 0.5)
    if cfg.FPN.FPN_ON and cfg.MODEL.FASTER_RCNN:
        cfg.FPN.RPN_COLLECT_SCALE = cfg.TRAIN.IMS_PER_BATCH / original_ims_per_batch
        print('Scale FPN rpn_proposals collect size directly propotional to the change of IMS_PER_BATCH:\n'
              '    cfg.FPN.RPN_COLLECT_SCALE: {}'.format(cfg.FPN.RPN_COLLECT_SCALE))

    if args.num_workers is not None:
        cfg.DATA_LOADER.NUM_THREADS = args.num_workers
    print('Number of data loading threads: %d' % cfg.DATA_LOADER.NUM_THREADS)

    ### Overwrite some solver settings from command line arguments
    if args.optimizer is not None:
        cfg.SOLVER.TYPE = args.optimizer
    if args.lr is not None:
        cfg.SOLVER.BASE_LR = args.lr
    if args.lr_decay_gamma is not None:
        cfg.SOLVER.GAMMA = args.lr_decay_gamma
    assert_and_infer_cfg()

    timers = defaultdict(Timer)

    ### Dataset ###
    timers['roidb'].tic()

    assert len(cfg.TRAIN.DATASETS)==1
    name, split = cfg.TRAIN.DATASETS[0].split('_')
    db = DAVIS_imdb(db_name=name, split = split, cls_mapper = None, load_flow=cfg.MODEL.LOAD_FLOW_FILE)
    merged_roidb, seq_num, seq_start_end = sequenced_roidb_for_training_from_db([db] , cfg.TRAIN.PROPOSAL_FILES)

    timers['roidb'].toc()
    roidb_size = len(merged_roidb)
    logger.info('{:d} roidbs sequences.'.format(roidb_size))
    logger.info('Takes %.2f sec(s) to construct roidbs', timers['roidb'].average_time)

    # Effective training sample size for one epoch, number of sequences.
    train_size = roidb_size // args.batch_size * args.batch_size

    ### Model ###
    maskRCNN = vos_model_builder.Generalized_VOS_RCNN()

    if cfg.CUDA:
        maskRCNN.cuda()


    ### Training Setups ###
    args.run_name = misc_utils.get_run_name() + '_step'
    output_dir = misc_utils.get_output_dir(args, args.run_name)
    args.cfg_filename = os.path.basename(args.cfg_file)

    if not args.no_save:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        blob = {'cfg': yaml.dump(cfg), 'args': args}
        with open(os.path.join(output_dir, 'config_and_args.pkl'), 'wb') as f:
            pickle.dump(blob, f, pickle.HIGHEST_PROTOCOL)

        if args.use_tfboard:
            from tensorboardX import SummaryWriter
            # Set the Tensorboard logger
            tblogger = SummaryWriter(output_dir)

    for seq_idx in range(len(seq_start_end)):
        # TODO: remove following.
        #if seq_idx!=10:
          #  continue
        db.set_to_sequence(seq_idx)
        seq_name = db.get_current_seq_name()
        cur_output_dir = osp.join(args.output_dir,seq_name)
        if args.no_overwrite is True and osp.exists(osp.join(cur_output_dir,'results.pdf')):
          continue
        # first frame for finetuning.
        start_end = seq_start_end[seq_idx,:]
        roidbs = merged_roidb[start_end[0]:start_end[1]]
        # For every new seq, we reload the ckpt and reset optimizer.
        
        cfg.immutable(False)
        cfg.MODEL.NUM_CLASSES = db.instance_number_of_seq[seq_idx]+1
        maskRCNN = vos_model_builder.Generalized_VOS_RCNN()
        optimizer = load_ckpt(args, maskRCNN)
        lr = optimizer.param_groups[0]['lr']  # lr of non-bias parameters, for commmand line outputs.
        maskRCNN = mynn.DataParallel(maskRCNN, cpu_keywords=['im_info', 'roidb'], minibatch=True)
        for idx in range(len(roidbs)):
            if idx == 0:                    
                roidb = roidbs[idx:idx+1]
                maskRCNN.module.clean_hidden_states()
                ### Training Loop ###
                maskRCNN.train()
                # Set index for decay steps
                decay_steps_ind = None
                for i in range(1, len(cfg.SOLVER.STEPS)):
                    if cfg.SOLVER.STEPS[i] >= args.start_step:
                        decay_steps_ind = i
                        break
                if decay_steps_ind is None:
                    decay_steps_ind = len(cfg.SOLVER.STEPS)

                training_stats = TrainingStats(
                    args,
                    args.disp_interval,
                    tblogger if args.use_tfboard and not args.no_save else None)

                blobs, valid = get_minibatch(roidb, target_scale = cfg.TRAIN.SCALES[npr.randint(0, len(cfg.TRAIN.SCALES))])
                for key in blobs:
                    if key != 'roidb' and key != 'data_flow':
                        blobs[key] = blobs[key].squeeze(axis=0)
                blobs['roidb'] = blob_utils.serialize(blobs['roidb'])
                batch = collate_minibatch([blobs])
                input_data = {}
                for key in batch.keys():
                    input_data[key] = batch[key][:1]
                for key in input_data:
                    if key != 'roidb' and key != 'data_flow': # roidb is a list of ndarrays with inconsistent length
                        input_data[key] = list(map(Variable, input_data[key]))
                    if key == 'data_flow':
                        if idx != 0 and input_data[key][0][0][0] is not None: # flow is not None.
                            input_data[key] = [Variable(torch.tensor(np.expand_dims(np.squeeze(np.array(input_data[key],dtype=np.float32)),0), device=input_data['data'][0].device))]
                            assert input_data['data'][0].shape[-2:]==input_data[key][0].shape[-2:], "Spatial shape of image and flow are not equal."
                        else:
                            input_data[key] = [None]
                try:
                    logger.info('Training starts !')
                    step = 0
                    for step in range(0, cfg.SOLVER.MAX_ITER):
                        # Warm up
                        if step < cfg.SOLVER.WARM_UP_ITERS:
                            method = cfg.SOLVER.WARM_UP_METHOD
                            if method == 'constant':
                                warmup_factor = cfg.SOLVER.WARM_UP_FACTOR
                            elif method == 'linear':
                                alpha = step / cfg.SOLVER.WARM_UP_ITERS
                                warmup_factor = cfg.SOLVER.WARM_UP_FACTOR * (1 - alpha) + alpha
                            else:
                                raise KeyError('Unknown SOLVER.WARM_UP_METHOD: {}'.format(method))
                            lr_new = cfg.SOLVER.BASE_LR * warmup_factor
                            net_utils.update_learning_rate(optimizer, lr, lr_new)
                            lr = optimizer.param_groups[0]['lr']
                            assert lr == lr_new
                        elif step == cfg.SOLVER.WARM_UP_ITERS:
                            net_utils.update_learning_rate(optimizer, lr, cfg.SOLVER.BASE_LR)
                            lr = optimizer.param_groups[0]['lr']
                            assert lr == cfg.SOLVER.BASE_LR

                        # Learning rate decay
                        if decay_steps_ind < len(cfg.SOLVER.STEPS) and \
                                step == cfg.SOLVER.STEPS[decay_steps_ind]:
                            logger.info('Decay the learning on step %d', step)
                            lr_new = lr * cfg.SOLVER.GAMMA
                            net_utils.update_learning_rate(optimizer, lr, lr_new)
                            lr = optimizer.param_groups[0]['lr']
                            assert lr == lr_new
                            decay_steps_ind += 1

                        training_stats.IterTic()
                        optimizer.zero_grad()
              
                        net_outputs = maskRCNN(**input_data)
                        training_stats.UpdateIterStats(net_outputs)                        
                        loss = net_outputs['total_loss']
                        # online training early stop criteria.
                        stop_online_training = None
                        if stop_online_training is None and (cfg.TRAIN.SC_CLS_LOSS_TH>0 or cfg.TRAIN.SC_BBOX_LOSS_TH>0 or cfg.TRAIN.SC_MASK_LOSS_TH):
                            stop_online_training = True
                        if cfg.TRAIN.SC_CLS_LOSS_TH>0 and net_outputs['losses']['loss_cls']>cfg.TRAIN.SC_CLS_LOSS_TH:                            
                            stop_online_training = stop_online_training and False
                        if cfg.TRAIN.SC_BBOX_LOSS_TH>0 and net_outputs['losses']['loss_bbox']>cfg.TRAIN.SC_BBOX_LOSS_TH:
                            stop_online_training = stop_online_training and False
                        if cfg.TRAIN.SC_MASK_LOSS_TH>0 and net_outputs['losses']['loss_mask']>cfg.TRAIN.SC_MASK_LOSS_TH:
                            stop_online_training = stop_online_training and False
                        if stop_online_training is True:
                            break
                        loss.backward()
                        optimizer.step()
                        training_stats.IterToc()
                        training_stats.LogIterStats(step, lr)      
                        maskRCNN.module.clean_hidden_states()
                    # ---- Training ends ----
                    # Save last checkpoint
                    #save_ckpt(output_dir, args, step, train_size, maskRCNN, optimizer)

                except (RuntimeError, KeyboardInterrupt):
                    #del dataiterator
                    #logger.info('Save ckpt on exception ...')
                    #save_ckpt(output_dir, args, step, train_size, maskRCNN, optimizer)
                    #logger.info('Save ckpt done.')
                    stack_trace = traceback.format_exc()
                    print(stack_trace)

                finally:
                    if args.use_tfboard and not args.no_save:
                        tblogger.close()
                    # Clean hidden states as the hidden states are modified by multi-scale training.
                    maskRCNN.module.clean_hidden_states()
                        
            
            if not osp.isdir(cur_output_dir):
              os.makedirs(cur_output_dir)
              assert(cur_output_dir)
            maskRCNN.eval()

            im = db.get_image_cv2(idx)
            flo = None
            assert im is not None
            if idx != 0:
                if cfg.MODEL.LOAD_FLOW_FILE:
                    flo = db.get_flow(idx)
                    assert flo is not None
            timers = defaultdict(Timer)
            cls_boxes, cls_segms, cls_keyps = im_detect_all(maskRCNN, im, flo, timers=timers)

            im_name = '%05d'%(idx)
            cls_mapper = [0]+[db.local_id_to_global_id(i, seq_idx) for i in range(1, cfg.MODEL.NUM_CLASSES)]
            thresh = 0.1
            vis_utils.viz_mask_result(im, im_name, cur_output_dir, cls_boxes, segms=cls_segms, thresh=thresh,box_alpha=0.3, dataset=db, ext='png', img_saver = davis_saver)
            print(osp.join(seq_name,im_name))
            im_name = '%03d-%03d'%(seq_idx,idx)
            cur_output_pdf_dir = osp.join(cur_output_dir,'pdf')
            vis_utils.vis_one_image(
              im[:, :, ::-1],  # BGR -> RGB for visualization
              im_name,
              cur_output_pdf_dir,
              cls_boxes,
              cls_segms,
              cls_keyps,
              dataset=db,
              box_alpha=0.3,
              show_class=True,
              thresh=thresh,
              kp_thresh=2,
              cls_mapper = cls_mapper,
              replace_mask_color_id_with_cls_id = True)

        if args.merge_pdfs:
            merge_out_path = '{}/results.pdf'.format(cur_output_pdf_dir)
            if os.path.exists(merge_out_path):
                os.remove(merge_out_path)
            command = "pdfunite {}/*.pdf {}".format(cur_output_pdf_dir,
                                                    merge_out_path)
            subprocess.call(command, shell=True)
        #TODO remove break
        #break



if __name__ == '__main__':
    main()
