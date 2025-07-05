import numpy as np
import torch.nn as nn
from mmdet.models.registry import DETECTORS
from mmdet.models.detectors.base import BaseDetector
from mmdet.core import auto_fp16, get_classes, tensor2imgs, \
    bbox2result, bbox2roi, build_assigner, build_sampler
from .modulator_os import Joint_Modulator
from mmdet.models import builder
from .mfmsa import MultiFrequencyChannelAttention
from .CBAM_Fusion import CBAMFusion
import torch
__all__ = ['QG_OS']

@DETECTORS.register_module
class QG_OS(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(QG_OS, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

        # build modulators

        self.modulator = Joint_Modulator(strides=[8, 16, 32, 64],featmap_num=5)

        # build attention block
        self.scattention = MultiFrequencyChannelAttention()
        self.proj_modulator = CBAMFusion(in_channels=256)

        # initialize weights
        #self.modulator.init_weights()
        self.mask_loss = nn.BCELoss() #nn.L1Loss()  #
        self.finest_scale = 28


    def init_weights(self, pretrained=None):
        super(QG_OS, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    @auto_fp16(apply_to=('img_z', 'img_x'))
    def forward(self,
                img_z,
                img_x,
                img_meta_z,
                img_meta_x,
                return_loss=True,
                **kwargs):
        if return_loss:
            return self.forward_train(
                img_z, img_x, img_meta_z, img_meta_x, **kwargs)
        else:
            return self.forward_test(
                img_z, img_x, img_meta_z, img_meta_x, **kwargs)

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmedetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        x_new = []
        device = "cuda:0"
        a = torch.randn(1, 256, 7, 7).to(device)
        for x_single in x:
            x_single = self.proj_modulator(a,x_single)
            x_single = self.scattention(x_single)
            x_new.append(x_single)
        x = x_new
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img_z,
                      img_x,
                      img_meta_z,
                      img_meta_x,
                      gt_bboxes_z,
                      gt_bboxes_x,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        z = self.extract_feat(img_z)
        x = self.extract_feat(img_x)

        losses = {}
        total = 0.

        for x_ij, i, j in self.modulator(z, x, gt_bboxes_z):   #　x_ij是一个特征层数个元素的列表,

            losses_ij = {}

            ''' select the j-th bbox/meta/label of the i-th image '''
            gt_bboxes_ij = gt_bboxes_x[i:i + 1]
            gt_bboxes_ij[0] = gt_bboxes_ij[0][j:j + 1]
            gt_labels_ij = gt_labels[i:i + 1]
            gt_labels_ij[0] = gt_labels_ij[0][j:j + 1]
            img_meta_xi = img_meta_x[i:i + 1]
            'x_ij是包含模板信息的多尺度特征'

            '添加注意力模块'
            x_new = []
            for x_single in x_ij:
                x_single = self.scattention(x_single)
                x_new.append(x_single)
            x_ij = x_new
            'end'

            atss_outs = self.bbox_head(x_ij)   # out 是 cls_score, bbox_pred
            atss_loss_inputs = atss_outs + (
                gt_bboxes_ij, gt_labels_ij, img_meta_xi, self.train_cfg)
            atss_losses_ij = self.bbox_head.loss(
                *atss_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses_ij.update(atss_losses_ij)

            # update losses  就是将loss_ij添加到loss的字典中
            for k, v in losses_ij.items():
                if k in losses:
                    if isinstance(v, (tuple, list)):
                        for u in range(len(v)):
                            losses[k][u] += v[u]
                    else:
                        losses[k] += v
                else:
                    losses[k] = v
            total += 1.

        # average the losses over instances
        for k, v in losses.items():
            if isinstance(v, (tuple, list)):
                for u in range(len(v)):
                    losses[k][u] /= total
            else:
                losses[k] /= total

        return losses

    def forward_test(self,
                     img_z,
                     img_x,
                     img_meta_z,
                     img_meta_x,
                     gt_bboxes_z,
                     rescale=False,
                     **kwargs):
        # assume one image and one instance only
        return self.simple_test(
            img_z, img_x, img_meta_z, img_meta_x,
            gt_bboxes_z,rescale, **kwargs)

    def simple_test(self,
                    img_z,
                    img_x,
                    img_meta_z,
                    img_meta_x,
                    gt_bboxes_z,
                    rescale=False,
                    **kwargs):
        # assume one image and one instance only
        assert len(img_z) == 1
        z = self.extract_feat(img_z)
        x = self.extract_feat(img_x)

        atss_feats = next(self.modulator(z, x, gt_bboxes_z))[0]

        # get predictions
        img_shape = img_meta_x[0]['img_shape']
        scale_factor = img_meta_x[0]['scale_factor']
        img_metas = [{'img_shape': img_shape, 'scale_factor': scale_factor}]

        outs = self.bbox_head(atss_feats)
        bbox_inputs = outs + (img_metas, self.test_cfg,rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)  # 在anchorhead里面，atss head是继承于anchorhead

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)[0]
            for det_bboxes, det_labels in bbox_list
        ]
        return np.array(bbox_results)

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError

    def _process_query(self, img_z, gt_bboxes_z):
        self._query = self.extract_feat(img_z)
        self._gt_bboxes_z = gt_bboxes_z

    def _process_gallary(self, img_x, img_meta_x,rescale = False,online=False, **kwargs):  # for one-shot

        # begin = time.time()
        x = self.extract_feat(img_x)

        atss_feats,_ = next(self.modulator(
            self._query, x, self._gt_bboxes_z))[0:2]

        # box head forward
        # get predictions
        img_shape = img_meta_x[0]['img_shape']
        scale_factor = img_meta_x[0]['scale_factor']
        img_metas = [{'img_shape': img_shape, 'scale_factor': scale_factor}]

        '添加注意力模块'
        x_new = []
        for x_single in atss_feats:
            x_single = self.scattention(x_single)
            x_new.append(x_single)
        atss_feats = x_new
        'end'

        outs = self.bbox_head(atss_feats)

        # begin = time.time()
        bbox_inputs = outs + (img_metas, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)[0]
            for det_bboxes, det_labels in bbox_list
        ]

        return np.array(bbox_results)

