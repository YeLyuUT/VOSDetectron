from functools import wraps
import importlib
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from core.config import cfg
from model.roi_pooling.functions.roi_pool import RoIPoolFunction
from model.roi_crop.functions.roi_crop import RoICropFunction
from modeling.roi_xfrom.roi_align.functions.roi_align import RoIAlignFunction
import modeling.rpn_heads as rpn_heads
import modeling.fast_rcnn_heads as fast_rcnn_heads
import modeling.mask_rcnn_heads as mask_rcnn_heads
import modeling.keypoint_rcnn_heads as keypoint_rcnn_heads
import utils.blob as blob_utils
import utils.net as net_utils
import utils.resnet_weights_helper as resnet_utils
from lib_vos.vos_nn.convgrucell import ConvGRUCell2d
from vos_model.flow_align.modules.flow_align import FlowAlign
import numpy as np

logger = logging.getLogger(__name__)


def get_func(func_name):
    """Helper to return a function object by name. func_name must identify a
    function in this module or the path to a function relative to the base
    'modeling' module.
    """
    if func_name == '':
        return None
    try:
        parts = func_name.split('.')
        # Refers to a function in this module
        if len(parts) == 1:
            return globals()[parts[0]]
        # Otherwise, assume we're referencing a module under modeling
        module_name = 'modeling.' + '.'.join(parts[:-1])
        module = importlib.import_module(module_name)
        return getattr(module, parts[-1])
    except Exception:
        logger.error('Failed to find function: %s', func_name)
        raise

def compare_state_dict(sa, sb):
    if sa.keys() != sb.keys():
        return False
    for k, va in sa.items():
        if not torch.equal(va, sb[k]):
            return False
    return True

def check_inference(net_func):
    @wraps(net_func)
    def wrapper(self, *args, **kwargs):
        if not self.training:
            if cfg.PYTORCH_VERSION_LESS_THAN_040:
                return net_func(self, *args, **kwargs)
            else:
                with torch.no_grad():
                    return net_func(self, *args, **kwargs)
        else:
            raise ValueError('You should call this function only on inference.'
                              'Set the network in inference mode by net.eval().')

    return wrapper


class Generalized_VOS_RCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Stop forwarding from Conv-GRU module.
        self.stop_after_hidden_states = False
        # For cache
        self.mapping_to_detectron = None
        self.orphans_in_detectron = None

        self.update_hidden_states = True
        # Backbone for feature extraction
        self.Conv_Body = get_func(cfg.MODEL.CONV_BODY)()

        assert(cfg.FPN.FPN_ON)
        #insert temporal module for video object segmentation.
        #As we use resNet, dims for 5 levels are hard coded.
        #fpn_dims=(2048, 1024, 512, 256, 64)
        fpn_dims = [cfg.FPN.DIM]*5
        h_channels = cfg.CONVGRU.HIDDEN_STATE_CHANNELS
        self.ConvGRUs = nn.ModuleList()
        for i in range(len(fpn_dims)):
            self.ConvGRUs.append(ConvGRUCell2d(fpn_dims[i], h_channels[i],kernel_size=3, stride=1, dilation=1, groups=1, use_GN=True,GN_groups = 32))

        self.flow_features = None
        if cfg.MODEL.USE_DELTA_FLOW:
            self.flow_features = [None]*5
            self.Conv_Delta_Flows_Features = nn.ModuleList()
            #TODO add initialization.
            for i in range(len(fpn_dims)):
                self.Conv_Delta_Flows_Features.append(
                    nn.Sequential(
                    nn.Conv2d(fpn_dims[i], 2, kernel_size=3, stride=1, padding= 1, bias = False),
                    nn.Tanh()))
                

        self.hidden_states = [None,None,None,None,None]
        # Region Proposal Network
        if cfg.RPN.RPN_ON:
            self.RPN = rpn_heads.generic_rpn_outputs(
                self.Conv_Body.dim_out, self.Conv_Body.spatial_scale)

        if cfg.FPN.FPN_ON:
            # Only supports case when RPN and ROI min levels are the same
            assert cfg.FPN.RPN_MIN_LEVEL == cfg.FPN.ROI_MIN_LEVEL
            # RPN max level can be >= to ROI max level
            assert cfg.FPN.RPN_MAX_LEVEL >= cfg.FPN.ROI_MAX_LEVEL
            # FPN RPN max level might be > FPN ROI max level in which case we
            # need to discard some leading conv blobs (blobs are ordered from
            # max/coarsest level to min/finest level)
            self.num_roi_levels = cfg.FPN.ROI_MAX_LEVEL - cfg.FPN.ROI_MIN_LEVEL + 1

            # Retain only the spatial scales that will be used for RoI heads. `Conv_Body.spatial_scale`
            # may include extra scales that are used for RPN proposals, but not for RoI heads.
            self.Conv_Body.spatial_scale = self.Conv_Body.spatial_scale[-self.num_roi_levels:]

        # BBOX Branch
        if not cfg.MODEL.RPN_ONLY:          
            self.Box_Head = get_func(cfg.FAST_RCNN.ROI_BOX_HEAD)(
                self.RPN.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale)
            if not cfg.MODEL.IDENTITY_TRAINING:
              self.Box_Outs = fast_rcnn_heads.fast_rcnn_outputs(
                  self.Box_Head.dim_out)
            else:
              self.Box_Outs = fast_rcnn_heads.fast_rcnn_outputs(
                  self.Box_Head.dim_out)

        # Mask Branch
        if cfg.MODEL.MASK_ON:
            self.Mask_Head = get_func(cfg.MRCNN.ROI_MASK_HEAD)(
                self.RPN.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale)
            if getattr(self.Mask_Head, 'SHARE_RES5', False):
                self.Mask_Head.share_res5_module(self.Box_Head.res5)
            self.Mask_Outs = mask_rcnn_heads.mask_rcnn_outputs(self.Mask_Head.dim_out, cfg.MODEL.NUM_CLASSES)

        # Keypoints Branch
        if cfg.MODEL.KEYPOINTS_ON:
            self.Keypoint_Head = get_func(cfg.KRCNN.ROI_KEYPOINTS_HEAD)(
                self.RPN.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale)
            if getattr(self.Keypoint_Head, 'SHARE_RES5', False):
                self.Keypoint_Head.share_res5_module(self.Box_Head.res5)
            self.Keypoint_Outs = keypoint_rcnn_heads.keypoint_outputs(self.Keypoint_Head.dim_out)

        if cfg.CONVGRU.DYNAMIC_MODEL:
            self.FlowAligns = nn.ModuleList()
            self.fpn_scales = [1./64., 1./32., 1./16., 1./8., 1./4.]
            for i in range(len(self.fpn_scales)):
                self.FlowAligns.append(FlowAlign(self.fpn_scales[i]))

        self._init_modules()

    def _init_modules(self):
        if cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS:
            resnet_utils.load_pretrained_imagenet_weights(self)
            # Check if shared weights are equaled
            if cfg.MODEL.MASK_ON and getattr(self.Mask_Head, 'SHARE_RES5', False):
                assert compare_state_dict(self.Mask_Head.res5.state_dict(), self.Box_Head.res5.state_dict())
            if cfg.MODEL.KEYPOINTS_ON and getattr(self.Keypoint_Head, 'SHARE_RES5', False):
                assert compare_state_dict(self.Keypoint_Head.res5.state_dict(), self.Box_Head.res5.state_dict())

        if cfg.TRAIN.FREEZE_CONV_BODY:
            for p in self.Conv_Body.parameters():
                p.requires_grad = False
        if cfg.TRAIN.FREEZE_CONV_GRU:
            for i in range(len(self.ConvGRUs)):
                for p in self.ConvGRUs[i].parameters():
                    p.requires_grad = False
        if cfg.TRAIN.FREEZE_MASK:
            for p in self.Mask_Outs.parameters():
                p.requires_grad = False
            for p in self.Mask_Head.parameters():
                p.requires_grad = False
        if cfg.TRAIN.FREEZE_RPN:
            for p in self.RPN.parameters():
                p.requires_grad = False
        if cfg.TRAIN.FREEZE_FAST_RCNN:
            for p in self.Box_Outs.parameters():
                p.requires_grad = False
            for p in self.Box_Head.parameters():
                p.requires_grad = False
        if cfg.TRAIN.FREEZE_BOX_HEAD:
            for p in self.Box_Head.parameters():
                p.requires_grad = False

    def set_require_param_grad_for_module(self, module, requires_grad):
        for p in module.parameters():
                p.requires_grad = requires_grad

    def freeze_all_layers(self):
        self.set_require_param_grad_for_module(self.Conv_Body, False)
        for i in range(len(self.ConvGRUs)):
            self.set_require_param_grad_for_module(self.ConvGRUs[i], False)
        self.set_require_param_grad_for_module(self.Mask_Outs, False)
        self.set_require_param_grad_for_module(self.Mask_Head, False)
        self.set_require_param_grad_for_module(self.Box_Outs, False)
        self.set_require_param_grad_for_module(self.Box_Head, False)
        self.set_require_param_grad_for_module(self.RPN, False)

    def unfreeze_all_layers(self):
        self.set_require_param_grad_for_module(self.Conv_Body, True)
        for i in range(len(self.ConvGRUs)):
            self.set_require_param_grad_for_module(self.ConvGRUs[i], True)
        self.set_require_param_grad_for_module(self.Mask_Outs, True)
        self.set_require_param_grad_for_module(self.Mask_Head, True)
        self.set_require_param_grad_for_module(self.Box_Outs, True)
        self.set_require_param_grad_for_module(self.Box_Head, True)
        self.set_require_param_grad_for_module(self.RPN, True)

    def freeze_conv_body_only(self):
        self.unfreeze_all_layers()
        self.set_require_param_grad_for_module(self.Conv_Body, False)

    def train_conv_body_only(self):
        self.freeze_all_layers()
        self.set_require_param_grad_for_module(self.Conv_Body, True)

    def train_cls_branch_only(self):
        self.freeze_all_layers()
        self.set_require_param_grad_for_module(self.Box_Outs.cls_score, True)
        #self.set_require_param_grad_for_module(self.Box_Outs, True)

    def _create_hidden_state(self, idx, ref_blob):
        h_c = cfg.CONVGRU.HIDDEN_STATE_CHANNELS[idx]
        self.hidden_states[idx] = torch.zeros([ref_blob.shape[0], h_c, ref_blob.shape[2], ref_blob.shape[3]], device = ref_blob.device)

    def create_hidden_states(self, ref_blobs):
        for i in range(5):
            if self.hidden_states[i] is None:
                self._create_hidden_state(i,ref_blobs[i])
            else:
                self.hidden_states[i] = None
                self._create_hidden_state(i,ref_blobs[i])

    def check_exist_hidden_states(self, ref_blobs):
        assert(len(ref_blobs)==5)
        for i in range(5):
            if self.hidden_states[i] is None:
                self._create_hidden_state(i,ref_blobs[i])

    def reset_hidden_states_to_zero(self):        
        for i in range(len(self.hidden_states)):
            self.hidden_states[i].data.zero_()

    def clean_hidden_states(self):
        for i in range(len(self.hidden_states)):
            self.hidden_states[i] = None

    def detach_hidden_states(self):
        for i in range(len(self.hidden_states)):
            if self.hidden_states[i] is not None:
                self.hidden_states[i].detach_()

    def clone_detach_hidden_states(self):
        return [self.hidden_states[i].clone().detach() if self.hidden_states[i] is not None else None for i in range(len(self.hidden_states))]

    def set_hidden_states(self, hidden_states):
        assert len(self.hidden_states) == len(hidden_states)
        for i in range(len(hidden_states)):
            self.hidden_states[i] = hidden_states[i]

    def clean_flow_features(self):
        if not self.flow_features is None:
            for i in range(5):
                self.flow_features[i] = None

    def set_stop_after_hidden_states(self, stop ):
        self.stop_after_hidden_states = stop

    def set_update_hidden_states(self, update = True):
        self.update_hidden_states = update

    def forward(self, data, im_info, roidb=None, data_flow = None, **rpn_kwargs):
        if cfg.PYTORCH_VERSION_LESS_THAN_040:
            return self._forward(data, im_info, roidb, data_flow, **rpn_kwargs)
        else:
            with torch.set_grad_enabled(self.training):
                return self._forward(data, im_info, roidb, data_flow, **rpn_kwargs)

    def _forward(self, data, im_info, roidb=None, data_flow = None, **rpn_kwargs):
        im_data = data
        if self.training:
            roidb = list(map(lambda x: blob_utils.deserialize(x)[0], roidb))

        device_id = im_data.get_device()

        return_dict = {}  # A dict to collect return variables

        blob_conv = self.Conv_Body(im_data)
        assert(len(blob_conv)==5)

        # if gru is for dynamic model, we train gru only.
        #if cfg.CONVGRU.DYNAMIC_MODEL:
         #   for i in range(5):
          #      blob_conv[i].detach_()
        
        # hidden states management.
        if not cfg.CONVGRU.DYNAMIC_MODEL:
            #Every time, the image size may be different, so we create new hidden states.
            self.create_hidden_states(blob_conv)
        else:
            #Caller needs to manually clean the hidden states.
            self.check_exist_hidden_states(blob_conv)
       
        if cfg.MODEL.USE_DELTA_FLOW:
            #overwrite dataflow.                
            data_flow = torch.zeros(im_data.shape[0],2,im_data.shape[2],im_data.shape[3], device = im_data.device)
            for i in range(5):          
                delta_flow_feature = self.Conv_Delta_Flows_Features[i](blob_conv[i].detach())
                delta_flow_lvl =None                    
                if not self.flow_features[i] is None:
                    delta_flow_lvl = delta_flow_feature - self.flow_features[i]
                self.flow_features[i] = delta_flow_feature
                if not delta_flow_lvl is None:
                    delta_flow_lvl = nn.functional.upsample(delta_flow_lvl, scale_factor = 1.0/self.fpn_scales[i], mode = 'bilinear')
                    delta_flow_lvl = delta_flow_lvl/self.fpn_scales[i]
                    data_flow = data_flow+delta_flow_lvl
        if cfg.CONVGRU.DYNAMIC_MODEL is True:
            #if dynamic model, hidden_states need to be updated.
            warped_hidden_states = [None]*5
            for i in range(5):
                if not data_flow is None:
                    #assert not np.any(np.isnan(data_flow.data.cpu().numpy())), 'data_flow has nan.'
                    warped_hidden_states[i] = self.FlowAligns[i](self.hidden_states[i], data_flow)
                else:
                    warped_hidden_states[i] = self.hidden_states[i]

        for i in range(4, -1, -1):
            if cfg.CONVGRU.DYNAMIC_MODEL is True:
                blob_conv[i] = self.ConvGRUs[i]( (blob_conv[i], warped_hidden_states[i]) )
            else:
                blob_conv[i] = self.ConvGRUs[i]( (blob_conv[i], self.hidden_states[i]) )
            if i<4:
                blob_conv[i] = blob_conv[i]/2.0+nn.functional.upsample(blob_conv[i+1],scale_factor=0.5,mode='bilinear')/2.0

            if self.update_hidden_states and cfg.CONVGRU.DYNAMIC_MODEL:
                self.hidden_states[i] = blob_conv[i]

        if self.stop_after_hidden_states is True:
           # print('stop after hidden states.')
            return None

        rpn_ret = self.RPN(blob_conv, im_info, roidb)

        # Handle blob with no GT boxes.
        HEAD_TRAIN = True
        if cfg.CONVGRU.DYNAMIC_MODEL and not roidb is None:
            assert(len(roidb)==1)
            if not len(rpn_ret['bbox_targets'])>0:
                HEAD_TRAIN = False
        # if self.training:
        #     # can be used to infer fg/bg ratio
        #     return_dict['rois_label'] = rpn_ret['labels_int32']

        if cfg.FPN.FPN_ON:
            # Retain only the blobs that will be used for RoI heads. `blob_conv` may include
            # extra blobs that are used for RPN proposals, but not for RoI heads.
            blob_conv = blob_conv[-self.num_roi_levels:]

        if not self.training:
            return_dict['blob_conv'] = blob_conv

        if not cfg.MODEL.RPN_ONLY:
            if HEAD_TRAIN:
                if cfg.MODEL.SHARE_RES5 and self.training:
                    box_feat, res5_feat = self.Box_Head(blob_conv, rpn_ret)
                else:
                    box_feat = self.Box_Head(blob_conv, rpn_ret)
                if not cfg.MODEL.IDENTITY_TRAINING:
                  cls_score, bbox_pred = self.Box_Outs(box_feat)
                else:
                  cls_score, bbox_pred, id_score = self.Box_Outs(box_feat)
        else:
            # TODO: complete the returns for RPN only situation
            pass

        if self.training:
            return_dict['losses'] = {}
            return_dict['metrics'] = {}
            # rpn loss
            rpn_kwargs.update(dict(
                (k, rpn_ret[k]) for k in rpn_ret.keys()
                if (k.startswith('rpn_cls_logits') or k.startswith('rpn_bbox_pred'))
            ))
            loss_rpn_cls, loss_rpn_bbox = rpn_heads.generic_rpn_losses(**rpn_kwargs)
            if cfg.FPN.FPN_ON:
                for i, lvl in enumerate(range(cfg.FPN.RPN_MIN_LEVEL, cfg.FPN.RPN_MAX_LEVEL + 1)):
                    return_dict['losses']['loss_rpn_cls_fpn%d' % lvl] = loss_rpn_cls[i]
                    return_dict['losses']['loss_rpn_bbox_fpn%d' % lvl] = loss_rpn_bbox[i]
            else:
                return_dict['losses']['loss_rpn_cls'] = loss_rpn_cls
                return_dict['losses']['loss_rpn_bbox'] = loss_rpn_bbox

            # bbox loss
            if cfg.MODEL.ADD_UNKNOWN_CLASS is True:
              cls_weights = torch.tensor([1.]*(cfg.MODEL.NUM_CLASSES-1)+[0.], device=cls_score.device)
            else:
                cls_weights = None

            if HEAD_TRAIN:
                if not cfg.MODEL.IDENTITY_TRAINING:
                  loss_cls, loss_bbox, accuracy_cls = fast_rcnn_heads.fast_rcnn_losses(
                      cls_score, bbox_pred, rpn_ret['labels_int32'], rpn_ret['bbox_targets'],
                      rpn_ret['bbox_inside_weights'], rpn_ret['bbox_outside_weights'], cls_weights = cls_weights)
                else:
                  loss_cls, loss_bbox, accuracy_cls, loss_id, accuracy_id = fast_rcnn_heads.fast_rcnn_losses(cls_score, bbox_pred, rpn_ret['labels_int32'], rpn_ret['bbox_targets'],rpn_ret['bbox_inside_weights'], rpn_ret['bbox_outside_weights'], cls_weights = cls_weights, id_score = id_score, id_int32 = rpn_ret['global_id_int32'])
                  return_dict['losses']['loss_id'] = loss_id
                  return_dict['metrics']['accuracy_id'] = accuracy_id
                return_dict['losses']['loss_cls'] = loss_cls
                return_dict['losses']['loss_bbox'] = loss_bbox
                return_dict['metrics']['accuracy_cls'] = accuracy_cls

                if cfg.MODEL.MASK_ON:
                    if getattr(self.Mask_Head, 'SHARE_RES5', False):
                        mask_feat = self.Mask_Head(res5_feat, rpn_ret,
                                                   roi_has_mask_int32=rpn_ret['roi_has_mask_int32'])
                    else:
                        mask_feat = self.Mask_Head(blob_conv, rpn_ret)
                    mask_pred = self.Mask_Outs(mask_feat)
                    # return_dict['mask_pred'] = mask_pred
                    # mask loss
                    loss_mask = mask_rcnn_heads.mask_rcnn_losses(mask_pred, rpn_ret['masks_int32'])
                    return_dict['losses']['loss_mask'] = loss_mask

            # pytorch0.4 bug on gathering scalar(0-dim) tensors
            for k, v in return_dict['losses'].items():
                return_dict['losses'][k] = v.unsqueeze(0)
            for k, v in return_dict['metrics'].items():
                return_dict['metrics'][k] = v.unsqueeze(0)
        else:
            # Testing
            return_dict['rois'] = rpn_ret['rois']
            return_dict['cls_score'] = cls_score
            return_dict['bbox_pred'] = bbox_pred
            if cfg.MODEL.IDENTITY_TRAINING:
              return_dict['id_score'] = id_score

        return return_dict

    def roi_feature_transform(self, blobs_in, rpn_ret, blob_rois='rois', method='RoIPoolF',
                              resolution=7, spatial_scale=1. / 16., sampling_ratio=0):
        """Add the specified RoI pooling method. The sampling_ratio argument
        is supported for some, but not all, RoI transform methods.

        RoIFeatureTransform abstracts away:
          - Use of FPN or not
          - Specifics of the transform method
        """
        assert method in {'RoIPoolF', 'RoICrop', 'RoIAlign'}, \
            'Unknown pooling method: {}'.format(method)

        if isinstance(blobs_in, list):
            # FPN case: add RoIFeatureTransform to each FPN level
            device_id = blobs_in[0].get_device()
            k_max = cfg.FPN.ROI_MAX_LEVEL  # coarsest level of pyramid
            k_min = cfg.FPN.ROI_MIN_LEVEL  # finest level of pyramid
            assert len(blobs_in) == k_max - k_min + 1
            bl_out_list = []
            for lvl in range(k_min, k_max + 1):
                bl_in = blobs_in[k_max - lvl]  # blobs_in is in reversed order
                sc = spatial_scale[k_max - lvl]  # in reversed order
                bl_rois = blob_rois + '_fpn' + str(lvl)
                if len(rpn_ret[bl_rois]):
                    rois = Variable(torch.from_numpy(rpn_ret[bl_rois])).cuda(device_id)
                    if method == 'RoIPoolF':
                        # Warning!: Not check if implementation matches Detectron
                        xform_out = RoIPoolFunction(resolution, resolution, sc)(bl_in, rois)
                    elif method == 'RoICrop':
                        # Warning!: Not check if implementation matches Detectron
                        grid_xy = net_utils.affine_grid_gen(
                            rois, bl_in.size()[2:], self.grid_size)
                        grid_yx = torch.stack(
                            [grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]], 3).contiguous()
                        xform_out = RoICropFunction()(bl_in, Variable(grid_yx).detach())
                        if cfg.CROP_RESIZE_WITH_MAX_POOL:
                            xform_out = F.max_pool2d(xform_out, 2, 2)
                    elif method == 'RoIAlign':
                        xform_out = RoIAlignFunction(
                            resolution, resolution, sc, sampling_ratio)(bl_in, rois)
                    bl_out_list.append(xform_out)

            # The pooled features from all levels are concatenated along the
            # batch dimension into a single 4D tensor.
            xform_shuffled = torch.cat(bl_out_list, dim=0)

            # Unshuffle to match rois from dataloader
            device_id = xform_shuffled.get_device()
            restore_bl = rpn_ret[blob_rois + '_idx_restore_int32']
            restore_bl = Variable(
                torch.from_numpy(restore_bl.astype('int64', copy=False))).cuda(device_id)
            xform_out = xform_shuffled[restore_bl]
        else:
            # Single feature level
            # rois: holds R regions of interest, each is a 5-tuple
            # (batch_idx, x1, y1, x2, y2) specifying an image batch index and a
            # rectangle (x1, y1, x2, y2)
            device_id = blobs_in.get_device()
            rois = Variable(torch.from_numpy(rpn_ret[blob_rois])).cuda(device_id)
            if method == 'RoIPoolF':
                xform_out = RoIPoolFunction(resolution, resolution, spatial_scale)(blobs_in, rois)
            elif method == 'RoICrop':
                grid_xy = net_utils.affine_grid_gen(rois, blobs_in.size()[2:], self.grid_size)
                grid_yx = torch.stack(
                    [grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]], 3).contiguous()
                xform_out = RoICropFunction()(blobs_in, Variable(grid_yx).detach())
                if cfg.CROP_RESIZE_WITH_MAX_POOL:
                    xform_out = F.max_pool2d(xform_out, 2, 2)
            elif method == 'RoIAlign':
                xform_out = RoIAlignFunction(
                    resolution, resolution, spatial_scale, sampling_ratio)(blobs_in, rois)

        return xform_out

    @check_inference
    def convbody_net(self, data):
        """For inference. Run Conv Body only"""
        blob_conv = self.Conv_Body(data)
        if cfg.FPN.FPN_ON:
            # Retain only the blobs that will be used for RoI heads. `blob_conv` may include
            # extra blobs that are used for RPN proposals, but not for RoI heads.
            blob_conv = blob_conv[-self.num_roi_levels:]
        return blob_conv

    @check_inference
    def mask_net(self, blob_conv, rpn_blob):
        """For inference"""
        mask_feat = self.Mask_Head(blob_conv, rpn_blob)
        mask_pred = self.Mask_Outs(mask_feat)
        return mask_pred

    @check_inference
    def keypoint_net(self, blob_conv, rpn_blob):
        """For inference"""
        kps_feat = self.Keypoint_Head(blob_conv, rpn_blob)
        kps_pred = self.Keypoint_Outs(kps_feat)
        return kps_pred

    @property
    def detectron_weight_mapping(self):
        if self.mapping_to_detectron is None:
            d_wmap = {}  # detectron_weight_mapping
            d_orphan = []  # detectron orphan weight list
            for name, m_child in self.named_children():
                if list(m_child.parameters()) and hasattr(m_child, 'detectron_weight_mapping'):  # if module has any parameter                
                    child_map, child_orphan = m_child.detectron_weight_mapping()
                    d_orphan.extend(child_orphan)
                    for key, value in child_map.items():
                        new_key = name + '.' + key
                        d_wmap[new_key] = value
            self.mapping_to_detectron = d_wmap
            self.orphans_in_detectron = d_orphan

        return self.mapping_to_detectron, self.orphans_in_detectron

    def _add_loss(self, return_dict, key, value):
        """Add loss tensor to returned dictionary"""
        return_dict['losses'][key] = value
