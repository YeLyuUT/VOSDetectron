""" Training Script """
import argparse
import distutils.util
import os
import os.path as osp
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
from torch.utils.data import DataLoader

sys.path.append('/media/yelyu/18339a64-762e-4258-a609-c0851cd8163e/YeLyu/Pytorch/VOSDetectron/lib')
sys.path.append('/media/yelyu/18339a64-762e-4258-a609-c0851cd8163e/YeLyu/Pytorch/VOSDetectron/lib_ss')

import nn as mynn
import utils.misc as misc
import utils.net as net_utils
import utils.misc as misc_utils
from PIL import Image

from utils.detectron_weight_helper import load_detectron_weight
from utils.logging import log_stats
from utils.timer import Timer
from utils.training_stats import TrainingStats

from ss_core.ss_config import cfg as cfg
from ss_dataloader.ss_loader import SemanticSegmentationDataset, Mode

from ss_modeling.model_builder import Generic_SS_Model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RuntimeError: received 0 items of ancdata. Issue: pytorch/pytorch#973
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

class default_gt_mapper:
    def __init__(self, class_num_valid):
        self._class_num_valid = class_num_valid
    def __call__(self, gt):
        '''
        Args: gt numpy array
        class_num_valid: number of classes for training
        '''
        gt[gt>self._class_num_valid] = self._class_num_valid
        gt[gt<0] = self._class_num_valid
        return gt

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Train a semantic segmentation network')

    parser.add_argument(
        '--txtfile', dest='txtfile', required=True,
        help='txtfile that contains the useful data paths')

    parser.add_argument(
        '--mode',dest='mode', required=True,
        help='train, val or test'
        )

    #parser.add_argument(
     #   '--cfg', dest='cfg_file', required=True,
      #  help='Config file)')

    parser.add_argument(
        '--disp_interval',
        help='Display training info every N iterations',
        default=100, type=int)
    parser.add_argument(
        '--no_cuda', dest='cuda', help='Do not use CUDA device', action='store_false')

    # Optimization
    # These options has the highest prioity and can overwrite the values in config file
    # or values set by set_cfgs. `None` means do not overwrite.

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
    parser.add_argument(
        '--lr_decay_epochs',
        help='Epochs to decay the learning rate on. '
             'Decay happens on the beginning of a epoch. '
             'Epoch is 0-indexed.',
        default=[4, 5], nargs='+', type=int)

    # Epoch
    parser.add_argument(
        '--start_iter',
        help='Starting iteration for first training epoch. 0-indexed.',
        default=0, type=int)
    parser.add_argument(
        '--start_epoch',
        help='Starting epoch count. Epoch is 0-indexed.',
        default=0, type=int)
    parser.add_argument(
        '--epochs', dest='num_epochs',
        help='Number of epochs to train',
        default=10, type=int)

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
        '--ckpt_num_per_epoch',
        help='number of checkpoints to save in each epoch. '
             'Not include the one at the end of an epoch.',
        default=3, type=int)

    parser.add_argument(
        '--load_ckpt', help='checkpoint path to load')
    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument(
        '--use_tfboard', help='Use tensorflow tensorboard to log training info',
        action='store_true')

    return parser.parse_args()

def get_output_dir(run_name):
    outdir_home = '/media/yelyu/18339a64-762e-4258-a609-c0851cd8163e/YeLyu/Pytorch/VOSDetectron/lib_ss/Output'
    outdir = osp.join(outdir_home,run_name)
    return outdir

#Save label as uint8 image.
def save_prediction(y_pred,savePath):
  assert(len(y_pred.shape)==2)
  print('####')
  print(savePath)
  print('####')
  Image.fromarray(y_pred).save(savePath)

def save_ckpt(output_dir, args, step, model, optimizer):
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
        'batch_size': args.batch_size,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()}, save_name)
    logger.info('save model: %s', save_name)

if __name__=='__main__':
    #TODO move class_num to other place
    class_num = 20
    args = parse_args()
    print('Called with args:')
    print(args)

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    if args.cuda or cfg.NUM_GPUS > 0:
        cfg.CUDA = True
    else:
        raise ValueError("Need Cuda device to run !")

    txtfile = args.txtfile
    mode_str = args.mode.lower()
    #'val' is the same with 'val_no_pred'
    if not mode_str in ['train','val','val_pred','val_no_pred','test']:
        raise ValueError('Unexpected mode: {}').format(args.mode_str)

    mode = Mode.TRAIN
    if mode_str == 'train':
        mode = Mode.TRAIN
    elif mode_str == 'val' or mode_str == 'val_no_pred':
        mode = Mode.VAL_NO_PRED
    elif mode_str == 'val_pred':
        mode = Mode.VAL_PRED
    elif mode_str == 'test':
        mode = Mode.TEST
    else:
        raise Exception('Mode is not valid.')
    #Here can insert multi-GPU training config changes.
    #Not implemented yet
    
    #Load config to args
    args.batch_size = cfg.TRAIN.BATCH_SIZE
    cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR/args.batch_size

    use_data_augmentation = False
    if mode == Mode.TRAIN:
        use_data_augmentation = True
        gt_mapper = default_gt_mapper(class_num_valid = class_num-1)
        dataset = SemanticSegmentationDataset(txtfile, mode, use_data_augmentation, assertNum = None, gtMapper = gt_mapper, batch_size = args.batch_size)
        dataloader = DataLoader(dataset, batch_size = 1,shuffle = True)
        dataiterator = iter(dataloader)
    else:
        use_data_augmentation = False
        dataset = SemanticSegmentationDataset(txtfile, mode, use_data_augmentation, assertNum = None, gtMapper = None)
        dataloader = DataLoader(dataset, batch_size = 1,shuffle = False)
        dataiterator = iter(dataloader)

    IDA_Net = Generic_SS_Model(class_num = class_num, ignore_index = class_num-1, weight = None)
    #print(IDA_Net.__dict__.keys())
    assert(IDA_Net.detectron_weight_mapping)
    #Use cuda on compulsory
    IDA_Net = IDA_Net.cuda()
    torch.device('cuda')
  
    if mode == Mode.TEST:
        ### Load checkpoint
        load_name = args.load_ckpt
        logging.info("loading checkpoint %s", load_name)
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt_no_mapping(IDA_Net, checkpoint['model'])
        del checkpoint
        torch.cuda.empty_cache()
        run_name = misc.get_run_name()+'_test'
        output_dir = get_output_dir(run_name)
        IDA_Net.eval()
        IDA_Net.use_logits(True)
        try:
            logger.info('Testing starts !')
            while True:
                try:
                    input_data = next(dataiterator)
                except StopIteration:                    
                    break                               
                
                TF = dataset.get_Transformer()
                input_imgs = input_data['img_PIL'].view(-1,*input_data['img_PIL'].shape[2:])              
                iter_size = input_imgs.shape[0]
                print('iter_size:',iter_size)
                out_logits = []                
                for inner_iter in range(iter_size):
                    input_img = input_imgs[inner_iter:inner_iter+1,:,:,:]
                    input_img = input_img.cuda()         
                    net_outputs = IDA_Net(input_img, None)
                    index = input_data['index']
                    #training_stats.UpdateIterStats(net_outputs, inner_iter)
                    #loss = net_outputs['loss']
                    #metric = net_outputs['metric']
                    logits = net_outputs['logits']
                    logits = logits.numpy()
                    logits = np.squeeze(logits,axis=0)
                    logits = np.transpose(logits,(1,2,0))
                    out_logits.append(logits)
                print('index:',input_data['index'])                   
                big_logits = TF.inverse_transform(out_logits, (1024,2048,class_num),np.float32)
                prediction = np.array(np.argmax(big_logits,axis = -1),dtype=np.uint8)
                save_prediction(prediction,input_data['pred_path'][0])
        except (RuntimeError, KeyboardInterrupt):
            exc_type, exc_obj, tb = sys.exc_info()
            f = tb.tb_frame
            lineno = tb.tb_lineno
            filename = f.f_code.co_filename
            print('EXCEPTION IN ({}, LINE {}): {}'.format(filename, lineno, exc_obj))

            del dataiterator
            logger.info('Stop prediction ...')
        finally:
            if args.use_tfboard and not args.no_save:
                tblogger.close()
        
  
    if mode == Mode.TRAIN:
        ### Optimizer ###
        gn_param_nameset = set()
        for name, module in IDA_Net.named_modules():
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
        for key, value in IDA_Net.named_parameters():
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


        if cfg.SOLVER.TYPE == 'SGD':
            optimizer = torch.optim.SGD(params, momentum=cfg.SOLVER.MOMENTUM)
        elif cfg.SOLVER.TYPE == "Adam":
            optimizer = torch.optim.Adam(params)

        ### Load checkpoint
        if args.load_ckpt:
            load_name = args.load_ckpt
            logging.info("loading checkpoint %s", load_name)
            checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
            net_utils.load_ckpt(IDA_Net, checkpoint['model'])
            if args.resume:
                args.start_step = checkpoint['step'] + 1

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
            load_detectron_weight(IDA_Net, args.load_detectron,force_load_all = False)


        lr = optimizer.param_groups[0]['lr']
        run_name = misc.get_run_name()+'_step'
        output_dir = get_output_dir(run_name)

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

        #train loop
        IDA_Net.train()
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
                try:
                    input_data = next(dataiterator)    
                except StopIteration:
                    dataiterator = iter(dataloader)
                    input_data = next(dataiterator)
                input_data['img_PIL'] = input_data['img_PIL'].view(-1,*input_data['img_PIL'].shape[2:])
                input_data['gt_PIL'] = input_data['gt_PIL'].view(-1,*input_data['gt_PIL'].shape[2:])
                iter_size = input_data['img_PIL'].shape[0]
                for inner_iter in range(iter_size):                                        
                    input_img = input_data['img_PIL'][inner_iter:inner_iter+1].cuda()
                    input_gt = input_data['gt_PIL'][inner_iter:inner_iter+1].cuda()
                    net_outputs = IDA_Net(input_img, input_gt)
                    index = input_data['index']
                    #training_stats.UpdateIterStats(net_outputs, inner_iter)
                    loss = net_outputs['loss']
                    metric = net_outputs['metric']
                    print('loss:%f, metric:%f, index:%d, inner_iter:%d, step:%d'%(loss, metric, index, inner_iter, step))
                    loss.backward()
                optimizer.step()
                training_stats.IterToc()
                #training_stats.LogIterStats(step, lr)

                if (step+1) % CHECKPOINT_PERIOD == 0:
                    save_ckpt(output_dir, args, step, IDA_Net, optimizer)

            # ---- Training ends ----
            # Save last checkpoint
            save_ckpt(output_dir, args, step, IDA_Net, optimizer)

        except (RuntimeError, KeyboardInterrupt):
            del dataiterator
            logger.info('Save ckpt on exception ...')
            save_ckpt(output_dir, args, step, IDA_Net, optimizer)
            logger.info('Save ckpt done.')
            stack_trace = traceback.format_exc()
            print(stack_trace)
        finally:
            if args.use_tfboard and not args.no_save:
                tblogger.close()
