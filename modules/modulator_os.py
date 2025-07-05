import torch.nn as nn
import torch
from mmdet.models.roi_extractors import SingleRoIExtractor, MultiRoIExtractor
from mmdet.core import bbox2roi
from mmcv.cnn import normal_init
import torch.nn.functional as F
from .CBAM_Fusion import CBAMFusion
__all__ = ['Joint_Modulator']

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Joint_Modulator(nn.Module):
    def __init__(self,
                 roi_out_size=7,
                 roi_sample_num=2,
                 channels=256,
                 strides=[4, 8, 16, 32],
                 featmap_num=5):
        super(Joint_Modulator, self).__init__()

        self.roi_extractor = MultiRoIExtractor(
            roi_layer={
                'type': 'RoIAlign',
                'out_size': roi_out_size,
                'sample_num': roi_sample_num},
            out_channels=channels,
            featmap_strides=strides,
            finest_scale=28)

        # self.proj_modulator = nn.ModuleList([
        #     nn.Conv2d(channels, channels, roi_out_size, padding=0)
        #     for _ in range(featmap_num)])
        self.proj_modulator = nn.ModuleList([
            CBAMFusion(in_channels=256)
            for _ in range(featmap_num)])

    def forward(self, feats_z, feats_x, gt_bboxes_z):

        self.gt_bboxes_z = gt_bboxes_z
        ''' for instance speicific modlator  for sot tracking '''
        modulator = self.learn_rpn_modulator(feats_z, gt_bboxes_z)

        return self.inference(
            feats_x,
            query_feats=self.learn(feats_z, gt_bboxes_z), modulator = modulator)

    def inference(self, feats_x, query_feats, modulator):
        n_imgs = len(feats_x[0])
        for i in range(n_imgs):
            n_instances = len(query_feats[i])
            for j in range(n_instances):
                gallary = [f[i:i + 1] for f in feats_x]
                # gallary = feats_x = gallary_cls ,query_modulator = modulator
                query_modulator = modulator[i][j:j + 1]
                gallary_cls_ = [f[i:i + 1] for f in gallary]
                # out_ij = [self.proj_modulator[k](query_modulator) * gallary_cls_[k]
                #           for k in range(len(gallary_cls_))]
                out_ij = [self.proj_modulator[k](query_modulator , gallary_cls_[k])
                          for k in range(len(gallary_cls_))]
                yield out_ij, i, j

    def learn(self, feats_z, gt_bboxes_z):
        rois = bbox2roi(gt_bboxes_z)

        bbox_feats_multi, bbox_feats_single = self.roi_extractor(
            feats_z[:self.roi_extractor.num_inputs], rois)
        query_feats_multi = [bbox_feats_multi[:, rois[:, 0] == j, ...].permute(1, 0, 2, 3, 4)
                             for j in range(len(gt_bboxes_z))]
        query_feats_single = [bbox_feats_single[rois[:, 0] == j]
                              for j in range(len(gt_bboxes_z))]
        return query_feats_multi, query_feats_single

    def learn_rpn_modulator(self, feats_z, gt_bboxes_z):
        rois = bbox2roi(gt_bboxes_z)
        _, bbox_feats = self.roi_extractor(
            feats_z[:self.roi_extractor.num_inputs], rois)

        modulator = [bbox_feats[rois[:, 0] == j]
                     for j in range(len(gt_bboxes_z))]
        return modulator

    # def init_weights(self):
    #     ''' instance specific modulation '''
    #     for m in self.proj_modulator:
    #         normal_init(m, std=0.01)




