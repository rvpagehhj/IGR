import math
import numpy as np
import numpy.random as npr

import torch
import torch.utils.data as data
import torch.utils.data.sampler as torch_sampler
# from torch.utils.data.dataloader import default_collate
#from torch._six import int_classes as _int_classes
int_classes = int
string_classes = str
# from core.config import cfg
# from roi_data.minibatch import get_minibatch
# import utils.blob as blob_utils
# from copy import deepcopy
import math
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
from .boxes import *
np.random.seed(666)

from PIL import Image

CLASSES=('too many', )

class RoiDataLoader(data.Dataset):
    def __init__(self, roidb, num_classes, full_info_list, ratio_list, training=True):
        self._roidb = roidb
        self._num_classes = num_classes
        self.training = training
        self.DATA_SIZE = len(self._roidb)   # 121034
        self.full_info_list = full_info_list # roidb_index, cls_list, image_id_list, roidb_index is useless.
        self.now_info_list = full_info_list
        self.ratio_list = ratio_list
        self.data_dict = {'ratio_index': self.full_info_list[:, 0], 'cls_ls': self.full_info_list[:, 1], 'img_ls': self.full_info_list[:, 2]}
        self.index_pd = pd.DataFrame.from_dict(self.data_dict)
        self.index_pd = self.index_pd.reset_index()
        self.name = 'fsod_train'

    '''from neuron/data/ops/image.py'''
    def read_image(self, filename, color_fmt='RGB'):
        assert color_fmt in Image.MODES
        try:
            img = Image.open(filename)
        except IOError:
            import piexif
            piexif.remove(filename)
        # img = Image.open(filename)
        if not img.mode == color_fmt:
            img = img.convert(color_fmt)
        return np.asarray(img)

    def __getitem__(self, index):
        ''' query image(search image)'''''
        index = index # this index is just the index of roidb, not the roidb_index
        ratio = self.ratio_list[index]
        # Get query roidb
        query_cls = self.index_pd.loc[self.index_pd['index']==index, 'cls_ls'].tolist()[0]
        query_img = self.index_pd.loc[self.index_pd['index']==index, 'img_ls'].tolist()[0]
        all_cls = self.index_pd.loc[self.index_pd['img_ls']==query_img, 'cls_ls'].tolist()
        single_db = [self._roidb[index]]
        img_z = self.read_image(single_db[0]['image'])
        bboxes_z = single_db[0]['boxes']
        # crop img
        if single_db[0]['need_crop']:
            img_z, bboxes_z = self.crop_data(single_db,img_z,bboxes_z, ratio)
            # Check bounding box
            entry = single_db[0]
            boxes = entry['boxes']
            invalid = (boxes[:, 0] == boxes[:, 2]) | (boxes[:, 1] == boxes[:, 3])
            valid_inds = np.nonzero(~ invalid)[0]
            if len(valid_inds) < len(boxes):
                for key in ['boxes', 'gt_classes', 'seg_areas', 'gt_overlaps', 'is_crowd',
                            'box_to_gt_ind_map', 'gt_keypoints']:
                    if key in entry:
                        entry[key] = entry[key][valid_inds]
                entry['segms'] = [entry['segms'][ind] for ind in valid_inds]
                bboxes_z = entry['boxes']

        ''' exampler image(support image)'''
        # Get support roidb, support cls is same with query cls, and support image is different from query image.
        used_img_ls = [query_img]
        used_index_ls = [index]
        support_index = self.index_pd.loc[
            (self.index_pd['cls_ls'] == query_cls) & (~self.index_pd['img_ls'].isin(used_img_ls)) & (
                ~self.index_pd['index'].isin(used_index_ls)), 'index'].sample(random_state=index).tolist()[0]
        support_cls = self.index_pd.loc[self.index_pd['index'] == support_index, 'cls_ls'].tolist()[0]
        assert support_cls==query_cls

        ratio_x = self.ratio_list[support_index]
        support_img = self.index_pd.loc[self.index_pd['index'] == support_index, 'img_ls'].tolist()[0]
        used_index_ls.append(support_index)
        used_img_ls.append(support_img)
        support_db = [self._roidb[support_index]]
        img_x = self.read_image(support_db[0]['image'])
        bboxes_x = support_db[0]['boxes']

        # crop img
        if support_db[0]['need_crop']:
            img_x, bboxes_x = self.crop_data(support_db, img_x, bboxes_x, ratio_x)
            # Check bounding box
            entry = support_db[0]
            boxes = entry['boxes']
            invalid = (boxes[:, 0] == boxes[:, 2]) | (boxes[:, 1] == boxes[:, 3])
            valid_inds = np.nonzero(~ invalid)[0]
            if len(valid_inds) < len(boxes):
                for key in ['boxes', 'gt_classes', 'seg_areas', 'gt_overlaps', 'is_crowd',
                            'box_to_gt_ind_map', 'gt_keypoints']:
                    if key in entry:
                        entry[key] = entry[key][valid_inds]
                entry['segms'] = [entry['segms'][ind] for ind in valid_inds]
                bboxes_x = entry['boxes']

        ''' exemplar image(support image negative)'''
        while True:
            support_index_ne = self.index_pd.loc[
                (self.index_pd['cls_ls'] != query_cls) & (~self.index_pd['img_ls'].isin(used_img_ls)) & (
                    ~self.index_pd['index'].isin(used_index_ls)), 'index'].sample(random_state=index).tolist()[0]
            support_cls_ne = self.index_pd.loc[self.index_pd['index'] == support_index_ne, 'cls_ls'].tolist()[0]
            assert support_cls_ne != query_cls

            cls_list = self.index_pd.loc[(self.index_pd['index'] == support_index_ne),'cls_ls'].tolist()
            if query_cls not in cls_list:
                break
            else:
                continue
        ratio_x_ne = self.ratio_list[support_index_ne]
        support_img_ne = self.index_pd.loc[self.index_pd['index'] == support_index_ne, 'img_ls'].tolist()[0]
        used_index_ls.append(support_index_ne)
        used_img_ls.append(support_img_ne)
        support_db_ne = [self._roidb[support_index_ne]]
        img_x_ne = self.read_image(support_db_ne[0]['image'])
        bboxes_x_ne = support_db_ne[0]['boxes']

        # crop img
        if support_db_ne[0]['need_crop']:
            img_x_ne, bboxes_x_ne = self.crop_data(support_db_ne, img_x_ne, bboxes_x_ne, ratio_x_ne)
            # Check bounding box
            entry_ne = support_db_ne[0]
            boxes_ne = entry_ne['boxes']
            invalid = (boxes_ne[:, 0] == boxes_ne[:, 2]) | (boxes_ne[:, 1] == boxes_ne[:, 3])
            valid_inds = np.nonzero(~ invalid)[0]
            if len(valid_inds) < len(boxes_ne):
                for key in ['boxes', 'gt_classes', 'seg_areas', 'gt_overlaps', 'is_crowd',
                            'box_to_gt_ind_map', 'gt_keypoints']:
                    if key in entry_ne:
                        entry_ne[key] = entry_ne[key][valid_inds]
                entry_ne['segms'] = [entry_ne['segms'][ind] for ind in valid_inds]
                bboxes_x_ne = entry_ne['boxes']

        target_z = {}
        # bboxes_z = xyxy_to_xywh(bboxes_z)
        target_z['bboxes'] = bboxes_z
        target_x = {}
        # bboxes_x = xyxy_to_xywh(bboxes_x)
        target_x['bboxes'] = bboxes_x

        ''' exemplar image(support image negative) '''
        target_z_ne = {}
        # bboxes_x = xyxy_to_xywh(bboxes_x)
        target_z_ne['bboxes'] = bboxes_x_ne

        cls = np.array([query_cls-1])  # cls is provided, np.array([1])  0-799 not 1-800

        return img_z, img_x, target_z, target_x, img_x_ne, target_z_ne, cls
        # return img_z, img_x, target_z, target_x, cls

        # '''for fsod classifier test'''
        # single_db = [self._roidb[index]]
        # img_z = self.read_image(single_db[0]['image'])
        # bboxes_z = single_db[0]['boxes']
        # bboxes_z = xyxy_to_xywh(bboxes_z)
        # query_cls = self.index_pd.loc[self.index_pd['index'] == index, 'cls_ls'].tolist()[0]
        # target = {'bboxes':bboxes_z, 'labels':query_cls}
        # return img_z, target

    def crop_data(self, roidb, img, bboxes, ratio):  # x1,y1,x2,y2 format
        data_height, data_width = roidb[0]['height'], roidb['width']
        boxes = bboxes
        if ratio < 1:  # width << height, crop height
            size_crop = math.ceil(data_width / ratio)  # size after crop
            min_y = math.floor(np.min(boxes[:, 1]))
            max_y = math.floor(np.max(boxes[:, 3]))
            box_region = max_y - min_y + 1
            if min_y == 0:
                y_s = 0
            else:
                if (box_region - size_crop) < 0:
                    y_s_min = max(max_y - size_crop, 0)
                    y_s_max = min(min_y, data_height - size_crop)
                    y_s = y_s_min if y_s_min == y_s_max else \
                        npr.choice(range(y_s_min, y_s_max + 1))
                else:
                    # CHECK: rethinking the mechnism for the case box_region > size_crop
                    # Now, the crop is biased on the lower part of box_region caused by
                    # // 2 for y_s_add
                    y_s_add = (box_region - size_crop) // 2
                    y_s = min_y if y_s_add == 0 else \
                        npr.choice(range(min_y, min_y + y_s_add + 1))
            # Crop the image
            img = img[y_s:(y_s + size_crop), :,:] # h,w,3
            # Shift and clamp boxes ground truth
            boxes[:, 1] -= y_s
            boxes[:, 3] -= y_s
            np.clip(boxes[:, 1], 0, size_crop - 1, out=boxes[:, 1])
            np.clip(boxes[:, 3], 0, size_crop - 1, out=boxes[:, 3])
            bboxes = boxes
            return img, bboxes
        else:  # width >> height, crop width
            size_crop = math.ceil(data_height * ratio)
            min_x = math.floor(np.min(boxes[:, 0]))
            max_x = math.floor(np.max(boxes[:, 2]))
            box_region = max_x - min_x + 1
            if min_x == 0:
                x_s = 0
            else:
                if (box_region - size_crop) < 0:
                    x_s_min = max(max_x - size_crop, 0)
                    x_s_max = min(min_x, data_width - size_crop)
                    print(x_s_min, x_s_max)
                    x_s = x_s_min if x_s_min == x_s_max else \
                        npr.choice(range(x_s_min, x_s_max + 1))
                else:
                    x_s_add = (box_region - size_crop) // 2
                    x_s = min_x if x_s_add == 0 else \
                        npr.choice(range(min_x, min_x + x_s_add + 1))
            # Crop the image
            img = img[:, x_s:(x_s + size_crop),:]
            # Shift and clamp boxes ground truth
            boxes[:, 0] -= x_s
            boxes[:, 2] -= x_s
            np.clip(boxes[:, 0], 0, size_crop - 1, out=boxes[:, 0])
            np.clip(boxes[:, 2], 0, size_crop - 1, out=boxes[:, 2])
            bboxes = boxes
            return img, bboxes


    def vis_image(self, im, bbox, im_name, output_dir):
        dpi = 300
        fig, ax = plt.subplots()
        ax.imshow(im, aspect='equal')
        plt.axis('off')
        height, width, channels = im.shape
        fig.set_size_inches(width/100.0/3.0, height/100.0/3.0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
        plt.margins(0,0)
        # Show box (off by default, box_alpha=0.0)
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1],
                          fill=False, edgecolor='r',
                          linewidth=0.5, alpha=1))
        output_name = os.path.basename(im_name)
        plt.savefig(os.path.join(output_dir, '{}'.format(output_name)), dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close('all')

    def __len__(self):
        return self.DATA_SIZE


