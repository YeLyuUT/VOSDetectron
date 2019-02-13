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
from numpy import random as npr
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
addPath(lib_vos_path)
addPath(lib_path)
addPath(home)
#import _init_paths  # pylint: disable=unused-import
import nn as mynn
import utils.net as net_utils
import utils.misc as misc_utils
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from vos_datasets.sequence_roidb import sequenced_roidb_for_training
from vos_roi_data.vos_loader import RoiDataLoader, MinibatchSampler, BatchSampler, collate_minibatch
#from roi_data.loader import RoiDataLoader, MinibatchSampler, BatchSampler, collate_minibatch
#from modeling.model_builder import Generalized_RCNN
from vos_modeling import vos_model_builder
from utils.detectron_weight_helper import load_detectron_weight
from utils.logging import setup_logging
from utils.timer import Timer
from utils.training_stats import TrainingStats
from copy import copy
#from vos_model.flow_align.functions.flow_align import FlowAlignFunction
# Set up logging and load config options
logger = setup_logging(__name__)
logging.getLogger('roi_data.loader').setLevel(logging.INFO)

# RuntimeError: received 0 items of ancdata. Issue: pytorch/pytorch#973
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096*20, rlimit[1]))

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

def gen_sequence_data_sampler(merged_roidb, seq_num, seq_start_end, use_seq_warmup = False):
    if use_seq_warmup:
        warmup_length = int(npr.randint(
            low = cfg.TRAIN.SEQUENCE_WARMUP_LENGTH_RANGE[0],
            high = cfg.TRAIN.SEQUENCE_WARMUP_LENGTH_RANGE[1],
            size = 1)[0])
    else:
        warmup_length = 0
    dataloader = None 
    dataiterator = None
    batch_size = cfg.MODEL.SEQUENCE_LENGTH+warmup_length
    batchSampler = BatchSampler(
        sampler=MinibatchSampler(seq_num, seq_start_end, batch_size = batch_size),
        batch_size=batch_size,
        drop_last=False
        )
    dataset = RoiDataLoader(
        merged_roidb,
        cfg.MODEL.NUM_CLASSES,
        training=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batchSampler,
        num_workers=cfg.DATA_LOADER.NUM_THREADS,
        collate_fn=collate_minibatch)
    dataiterator = iter(dataloader)
    return dataloader, dataiterator, warmup_length

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
        cfg.TRAIN.DATASETS = ('davis_train',)
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
    cfg.SOLVER.BASE_LR *= 1.0/cfg.MODEL.SEQUENCE_LENGTH
    print('Adjust BASE_LR linearly according to batch_size change and sequence length change:\n'
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
    merged_roidb, seq_num, seq_start_end = sequenced_roidb_for_training(
        cfg.TRAIN.DATASETS, cfg.TRAIN.PROPOSAL_FILES, load_inv_db = True)

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

    ### Optimizer ###
    gn_param_nameset = set()
    for name, module in maskRCNN.named_modules():
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
    for key, value in maskRCNN.named_parameters():
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

    ### Load checkpoint
    if args.load_ckpt:
        load_name = args.load_ckpt
        logging.info("loading checkpoint %s", load_name)
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt_no_mapping(maskRCNN, checkpoint['model'])
        if args.resume:
            args.start_step = checkpoint['step'] + 1
            if 'train_size' in checkpoint:  # For backward compatibility
                if checkpoint['train_size'] != train_size:
                    print('train_size value: %d different from the one in checkpoint: %d'
                          % (train_size, checkpoint['train_size']))

            # reorder the params in optimizer checkpoint's params_groups if needed
            # misc_utils.ensure_optimizer_ckpt_params_order(param_names, checkpoint)

            # There is a bug in optimizer.load_state_dict on Pytorch 0.3.1.
            # However it's fixed on master.
            optimizer.load_state_dict(checkpoint['optimizer'])
            # misc_utils.load_optimizer_state_dict(optimizer, checkpoint['optimizer'])
        del checkpoint
        torch.cuda.empty_cache()

    if args.load_detectron:  #TODO resume for detectron weights (load sgd momentum values)
        logging.info("loading Detectron weights %s", args.load_detectron)
        load_detectron_weight(maskRCNN, args.load_detectron,force_load_all=False)

    lr = optimizer.param_groups[0]['lr']  # lr of non-bias parameters, for commmand line outputs.

    maskRCNN = mynn.DataParallel(maskRCNN, cpu_keywords=['im_info', 'roidb'],
                                 minibatch=True)

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

    ### Training Loop ###
    maskRCNN.train()

    CHECKPOINT_PERIOD = int(cfg.TRAIN.SNAPSHOT_ITERS / cfg.NUM_GPUS)

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
    dataloader, dataiterator, warmup_length = gen_sequence_data_sampler(merged_roidb, seq_num, seq_start_end, use_seq_warmup=False)
    try:
        logger.info('Training starts !')
        step = args.start_step
        for step in range(args.start_step, cfg.SOLVER.MAX_ITER):            
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
            
            if cfg.TRAIN.USE_SEQ_WARMUP is True and step>=cfg.TRAIN.ITER_BEFORE_USE_SEQ_WARMUP and ((step-cfg.TRAIN.ITER_BEFORE_USE_SEQ_WARMUP)%cfg.TRAIN.WARMUP_LENGTH_CHANGE_STEP)==0:                                
                #TODO better dataiterator creator.
                raise NotImplementedError('not able to delete.')
                dataloader, dataiterator, warmup_length = gen_sequence_data_sampler(merged_roidb, seq_num, seq_start_end, use_seq_warmup=True)
                print('update warmup length:', warmup_length)
            try:
                input_data_sequence = next(dataiterator)
            except StopIteration:
                dataiterator = iter(dataloader)
                input_data_sequence = next(dataiterator)
            
            # clean hidden states before training.
            maskRCNN.module.clean_hidden_states()
            maskRCNN.module.clean_flow_features()
            if cfg.TRAIN.ALTERNATE_TRAINING:
                saved_input_data = []
                saved_hidden_states = None

            assert len(input_data_sequence['data']) == cfg.MODEL.SEQUENCE_LENGTH+warmup_length, print(len(input_data_sequence['data']), '!=' ,cfg.MODEL.SEQUENCE_LENGTH+warmup_length)
            # this is used for longer sequence training.
            # when reach maximum trainable length, detach the hidden states.
            cnter_for_detach_hidden_states = 0
            for inner_iter in range(cfg.MODEL.SEQUENCE_LENGTH+warmup_length):
                #get input_data
                input_data = {}
                for key in input_data_sequence.keys():
                    input_data[key] = input_data_sequence[key][inner_iter:inner_iter+1]                  
                for key in input_data:
                    if key != 'roidb' and key != 'data_flow': # roidb is a list of ndarrays with inconsistent length
                        input_data[key] = list(map(Variable, input_data[key]))
                    if key == 'data_flow':
                        if inner_iter != 0 and input_data[key][0][0][0] is not None: # flow is not None.
                            input_data[key] = [Variable(torch.tensor(np.expand_dims(np.squeeze(np.array(input_data[key],dtype=np.float32)),0), device=input_data['data'][0].device))]
                            assert input_data['data'][0].shape[-2:]==input_data[key][0].shape[-2:], "Spatial shape of image and flow are not equal."
                        else:
                            input_data[key] = [None]
                if cfg.TRAIN.USE_SEQ_WARMUP and inner_iter < warmup_length:
                    maskRCNN.module.set_stop_after_hidden_states(stop=True)
                    net_outputs = maskRCNN(**input_data)
                    assert net_outputs is None
                    maskRCNN.module.detach_hidden_states()
                    maskRCNN.module.set_stop_after_hidden_states(stop=False)
                    continue
 
                #save input_data for alternative training.
                if cfg.TRAIN.ALTERNATE_TRAINING:
                    saved_hidden_states = maskRCNN.module.clone_detach_hidden_states()
                    saved_input_data.append(copy(input_data))

                net_outputs = maskRCNN(**input_data)
                training_stats.UpdateIterStats(net_outputs, inner_iter)
                loss = net_outputs['total_loss']

                if  cnter_for_detach_hidden_states>= cfg.TRAIN.MAX_TRAINABLE_SEQ_LENGTH or inner_iter == cfg.MODEL.SEQUENCE_LENGTH+warmup_length-1:
                    # if reach the max trainable length or end of the sequence. free the graph and detach the hidden states.
                    loss.backward()
                    maskRCNN.module.detach_hidden_states()
                    cnter_for_detach_hidden_states = 0                    
                else:
                    loss.backward(retain_graph=True)
                    cnter_for_detach_hidden_states+=1
            optimizer.step()
            optimizer.zero_grad()
            # Train the backbone only.
            if cfg.TRAIN.ALTERNATE_TRAINING and len(saved_input_data)>0:
                maskRCNN.module.train_conv_body_only()
                maskRCNN.module.set_hidden_states(saved_hidden_states)
                for input_data in saved_input_data:
                    maskRCNN.module.detach_hidden_states()
                    net_outputs = maskRCNN(**input_data)
                    training_stats.UpdateIterStats(net_outputs, inner_iter)
                    loss = net_outputs['total_loss']                    
                    loss.backward()
                maskRCNN.module.freeze_conv_body_only()
                saved_input_data = []
                saved_hidden_states = None

            optimizer.step()
            optimizer.zero_grad()
            training_stats.IterToc()
            training_stats.LogIterStats(step, lr)
            
            if (step+1) % CHECKPOINT_PERIOD == 0:
                save_ckpt(output_dir, args, step, train_size, maskRCNN, optimizer)
            
        # ---- Training ends ----
        # Save last checkpoint
        save_ckpt(output_dir, args, step, train_size, maskRCNN, optimizer)

    except (RuntimeError, KeyboardInterrupt):
        del dataiterator
        #logger.info('Save ckpt on exception ...')
        #save_ckpt(output_dir, args, step, train_size, maskRCNN, optimizer)
        #logger.info('Save ckpt done.')
        stack_trace = traceback.format_exc()
        print(stack_trace)

    finally:
        if args.use_tfboard and not args.no_save:
            tblogger.close()


if __name__ == '__main__':
    main()
