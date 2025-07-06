# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Representation of the standard COCO json dataset format.

When working with a new dataset, we strongly suggest to convert the dataset into
the COCO json format and use the existing code; it is not recommended to write
code to support new dataset formats.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
from six.moves import cPickle as pickle
import logging
import numpy as np
import os
import scipy.sparse
import pandas as pd
# Must happen before importing COCO API (which imports matplotlib)
"""Set matplotlib up."""
import matplotlib
# Use a non-interactive backend
matplotlib.use('Agg')

# COCO API
from pycocotools import mask as COCOmask
from pycocotools.coco import COCO

from .boxes import *
from .config import cfg
import time

logger = logging.getLogger(__name__)
np.random.seed(666)

class Timer(object):
  """A simple timer."""

  def __init__(self):
    self.reset()

  def tic(self):
    # using time.time instead of time.clock because time time.clock
    # does not normalize for multithreading
    self.start_time = time.time()

  def toc(self, average=True):
    self.diff = time.time() - self.start_time
    self.total_time += self.diff
    self.calls += 1
    self.average_time = self.total_time / self.calls
    if average:
      return self.average_time
    else:
      return self.diff

  def reset(self):
    self.total_time = 0.
    self.calls = 0
    self.start_time = 0.
    self.diff = 0.
    self.average_time = 0.


class JsonDataset(object):  # a littile modification
    """A class representing a COCO json dataset."""

    def __init__(self, root_dir=None, subset='train',cache_dir='cache'):
        self.name = 'fsod_' + subset
        self.image_directory = root_dir + '/fsod/images'
        self.annotations = root_dir + '/fsod/annotations/' + self.name +'.json'

        assert os.path.exists(self.image_directory), \
            'Image directory \'{}\' not found'.format(self.image_directory)
        assert os.path.exists(self.annotations), \
            'Annotation file \'{}\' not found'.format(self.annotations)
        logger.debug('Creating: {}'.format(self.name))

        self.image_prefix = ''

        self.COCO = COCO(self.annotations)
        self.debug_timer = Timer()
        # Set up dataset classes
        category_ids = self.COCO.getCatIds()
        categories = [c['name'] for c in self.COCO.loadCats(category_ids)]
        self.category_to_id_map = dict(zip(categories, category_ids))
        self.id_to_category_map = dict(zip(category_ids, categories))
        self.classes = ['__background__'] + categories
        self.num_classes = len(self.classes)
        self.json_category_id_to_contiguous_id = {
            v: i + 1
            for i, v in enumerate(self.COCO.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k
            for k, v in self.json_category_id_to_contiguous_id.items()
        }
        '''cache reconstruct by lc'''
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, self.name + '.pkl')   # fsod_train.pkl

        '''keypoints to be none'''
        self.keypoints = None

    @property
    def valid_cached_keys(self):
        """ Can load following key-ed values from the cached roidb file

        'image'(image path) and 'flipped' values are already filled on _prep_roidb_entry,
        so we don't need to overwrite it again.
        """
        keys = ['boxes', 'segms', 'gt_classes', 'seg_areas', 'gt_overlaps',
                'is_crowd', 'box_to_gt_ind_map']
        if self.keypoints is not None:
            keys += ['gt_keypoints', 'has_visible_keypoints']
        return keys

    def get_roidb(
            self,
            gt=False,
            crowd_filter_thresh=0,
            test_flag=False
        ):
        """Return an roidb corresponding to the json dataset. Optionally:
           - include ground truth boxes in the roidb
           - add proposals specified in a proposals file
           - filter proposals based on a minimum side length
           - filter proposals that intersect with crowd regions
        """
        assert gt is True or crowd_filter_thresh == 0, \
            'Crowd filter threshold must be 0 if ground-truth annotations ' \
            'are not included.'
        image_ids = self.COCO.getImgIds()
        image_ids.sort()
        if cfg.DEBUG:
            roidb = copy.deepcopy(self.COCO.loadImgs(image_ids))[:100]
        else:
            roidb = copy.deepcopy(self.COCO.loadImgs(image_ids))
        for entry in roidb:
            self._prep_roidb_entry(entry)
        if gt:
            # Include ground-truth object annotations
            if os.path.exists(self.cache_file) and not cfg.DEBUG:
                self.debug_timer.tic()
                self._add_gt_from_cache(roidb, self.cache_file)
                logger.debug(
                    '_add_gt_from_cache took {:.3f}s'.
                    format(self.debug_timer.toc(average=False))
                )
            else:
                self.debug_timer.tic()
                for entry in roidb:
                    self._add_gt_annotations(entry)
                logger.debug(
                    '_add_gt_annotations took {:.3f}s'.
                    format(self.debug_timer.toc(average=False))
                )
                if not cfg.DEBUG:
                    with open(self.cache_file, 'wb') as fp:
                        pickle.dump(roidb, fp, pickle.HIGHEST_PROTOCOL)
                    logger.info('Cache ground truth roidb to %s', self.cache_file)
        _add_class_assignments(roidb)
        # contrust query roidb for the groundtruth and query image in the evaluation.
        if test_flag:
            # new dataset split according to class
            episode_num = 600 #600 #500
            way_num = 5
            shot_num = 10
            new_roidb = []
            cnt = 0
            print('len', len(roidb))  # 14152
            # contrust index_pd for picking query image in every episode.
            for item_id, item in enumerate(roidb):
                gt_classes = list(set(item['gt_classes']))
                for cls in gt_classes:
                    item_new = item.copy()
                    #item_new['id'] = item_new['id'] * 1000 + int(cls)
                    item_new['target_cls'] = int(cls)
                    all_cls = item['gt_classes']
                    target_idx = np.where(all_cls == cls)[0]
                    item_new['boxes'] = item['boxes'][target_idx]
                    item_new['gt_classes'] = item['gt_classes'][target_idx]
                    item_new['index'] = cnt
                    cnt += 1
                    new_roidb.append(item_new)
            print('original testing annotation number: ', len(new_roidb))  # new_roidb is after classified processing  14862
            roidb_img = []
            roidb_cls = []
            roidb_index = []
            for item in new_roidb:
                roidb_img.append(item['image'])
                roidb_cls.append(item['target_cls'])
                roidb_index.append(item['index'])
            data_dict = {'img_ls': roidb_img, 'cls_ls': roidb_cls, 'index': roidb_index}
            index_pd = pd.DataFrame.from_dict(data_dict)
            # above : new_roidb is after classified processing by lc

            # contrust query roidb: for each eposide, and for each cls, randomly pick shot_num images as query image.
            # for evaluation, each image and each cls in each episode is new, because we want to evaluate every episode seperately.
            # new id is used in ap evaluation, real_index and new cls is used in construct result array.
            # we need to change some code in the voc evaluation protocol according to these new parameters.
            all_cls = list(set(roidb_cls))
            final_roidb = []
            cls_cnt = 0
            img_cnt = 0
            txt_ls = ''
            self.classes = ['__background__']

            self.category_to_id_map = {}
            for ep in range(episode_num):
                cls_ls = np.random.choice(all_cls, way_num, replace=False)
                used_img_ls = []
                used_index_ls = []
                for query_cls in cls_ls:
                    cls_name = self.id_to_category_map[query_cls] + '_' + str(ep)
                    self.classes.append(cls_name)
                    cls_cnt += 1
                    self.category_to_id_map[cls_name] = cls_cnt
                    for shot in range(shot_num):
                        support_index = index_pd.loc[(index_pd['cls_ls'] == query_cls) & (~index_pd['img_ls'].isin(used_img_ls)) & (~index_pd['index'].isin(used_index_ls)), 'index'].sample(random_state=ep).tolist()[0]
                        support_cls = index_pd.loc[index_pd['index'] == support_index, 'cls_ls'].tolist()[0]
                        support_img = index_pd.loc[index_pd['index'] == support_index, 'img_ls'].tolist()[0]
                        used_img_ls.append(support_img)
                        used_index_ls.append(support_index)

                        target_roidb = new_roidb[support_index].copy()
                        original_id = new_roidb[support_index]['id']
                        assert target_roidb['target_cls'] == support_cls
                        assert target_roidb['index'] == support_index

                        target_roidb['id'] = target_roidb['id'] * 1000 + ep
                        #target_roidb['target_cls'] = # We use target_cls to indicate the original cls
                        target_roidb['max_classes'] = np.full_like(target_roidb['max_classes'], cls_cnt)
                        target_roidb['gt_classes'] = np.full_like(target_roidb['gt_classes'], cls_cnt)
                        target_roidb['real_index'] = img_cnt
                        txt_ls += str(original_id)  + '_' + str(query_cls) + '_' + str(ep) + '\n' #str(img_cnt) + '\n'
                        img_cnt += 1
                        final_roidb.append(target_roidb)

                    #cls_cnt += 1
            self.num_classes = len(self.classes)

            print(self.classes, self.num_classes)
            self.json_category_id_to_contiguous_id = {
               i : i
                for i in self.category_to_id_map.values()
            }
            self.contiguous_category_id_to_json_id = {
                v: k
                for k, v in self.json_category_id_to_contiguous_id.items()
            }

            with open('../data/fsod/new_val.txt', 'w') as f:
                f.write(txt_ls)

            return roidb, final_roidb

        return roidb

    def _prep_roidb_entry(self, entry):
        """Adds empty metadata fields to an roidb entry."""
        # Reference back to the parent dataset
        entry['dataset'] = self
        # Make file_name an abs path
        # im_path = '../data/fsod/images/' + entry['file_name']   # from ./ to ../
        im_path = self.image_directory +'/'+ entry['file_name']
        assert os.path.exists(im_path), 'Image \'{}\' not found'.format(im_path)
        entry['image'] = im_path
        entry['flipped'] = False   # False is ok
        entry['has_visible_keypoints'] = False
        # Empty placeholders
        entry['boxes'] = np.empty((0, 4), dtype=np.float32)
        entry['segms'] = []
        entry['gt_classes'] = np.empty((0), dtype=np.int32)
        entry['seg_areas'] = np.empty((0), dtype=np.float32)
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(
            np.empty((0, self.num_classes), dtype=np.float32)
        )
        entry['is_crowd'] = np.empty((0), dtype=np.bool)
        # 'box_to_gt_ind_map': Shape is (#rois). Maps from each roi to the index
        # in the list of rois that satisfy np.where(entry['gt_classes'] > 0)
        entry['box_to_gt_ind_map'] = np.empty((0), dtype=np.int32)

        # Remove unwanted fields that come from the json file (if they exist)
        for k in ['date_captured', 'url', 'license', 'file_name']:
            if k in entry:
                del entry[k]

    def _add_gt_annotations(self, entry):
        """Add ground truth annotation metadata to an roidb entry."""
        ann_ids = self.COCO.getAnnIds(imgIds=entry['id'], iscrowd=None)
        objs = self.COCO.loadAnns(ann_ids)
        # Sanitize bboxes -- some are invalid
        valid_objs = []
        valid_segms = []
        width = entry['width']
        height = entry['height']
        for obj in objs:
            # crowd regions are RLE encoded and stored as dicts
            if isinstance(obj['segmentation'], list):
                # Valid polygons have >= 3 points, so require >= 6 coordinates
                obj['segmentation'] = [
                    p for p in obj['segmentation'] if len(p) >= 6
                ]
            if obj['area'] < cfg.TRAIN.GT_MIN_AREA:
                continue
            if 'ignore' in obj and obj['ignore'] == 1:
                continue
            # Convert form (x1, y1, w, h) to (x1, y1, x2, y2)
            x1, y1, x2, y2 = xywh_to_xyxy(obj['bbox'])
            x1, y1, x2, y2 = clip_xyxy_to_image(
                x1, y1, x2, y2, height, width
            )
            # Require non-zero seg area and more than 1x1 box size
            #if obj['area'] > 0 and x2 > x1 and y2 > y1:
            if x2 > x1 and y2 > y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
                valid_segms.append(obj['segmentation'])
        num_valid_objs = len(valid_objs)

        boxes = np.zeros((num_valid_objs, 4), dtype=entry['boxes'].dtype)
        gt_classes = np.zeros((num_valid_objs), dtype=entry['gt_classes'].dtype)
        gt_overlaps = np.zeros(
            (num_valid_objs, self.num_classes),
            dtype=entry['gt_overlaps'].dtype
        )
        seg_areas = np.zeros((num_valid_objs), dtype=entry['seg_areas'].dtype)
        is_crowd = np.zeros((num_valid_objs), dtype=entry['is_crowd'].dtype)
        box_to_gt_ind_map = np.zeros(
            (num_valid_objs), dtype=entry['box_to_gt_ind_map'].dtype
        )

        im_has_visible_keypoints = False
        for ix, obj in enumerate(valid_objs):
            cls = self.json_category_id_to_contiguous_id[obj['category_id']]
            boxes[ix, :] = obj['clean_bbox']
            gt_classes[ix] = cls
            seg_areas[ix] = obj['area']
            is_crowd[ix] = obj['iscrowd']
            box_to_gt_ind_map[ix] = ix

            if obj['iscrowd']:
                # Set overlap to -1 for all classes for crowd objects
                # so they will be excluded during training
                gt_overlaps[ix, :] = -1.0
            else:
                gt_overlaps[ix, cls] = 1.0
        entry['boxes'] = np.append(entry['boxes'], boxes, axis=0)
        entry['segms'].extend(valid_segms)
        # To match the original implementation:
        # entry['boxes'] = np.append(
        #     entry['boxes'], boxes.astype(np.int).astype(np.float), axis=0)
        entry['gt_classes'] = np.append(entry['gt_classes'], gt_classes)
        entry['seg_areas'] = np.append(entry['seg_areas'], seg_areas)
        entry['gt_overlaps'] = np.append(
            entry['gt_overlaps'].toarray(), gt_overlaps, axis=0
        )
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(entry['gt_overlaps'])
        entry['is_crowd'] = np.append(entry['is_crowd'], is_crowd)
        entry['box_to_gt_ind_map'] = np.append(
            entry['box_to_gt_ind_map'], box_to_gt_ind_map
        )

    def _add_gt_from_cache(self, roidb, cache_filepath):
        """Add ground truth annotation metadata from cached file."""
        logger.info('Loading cached gt_roidb from %s', cache_filepath)
        with open(cache_filepath, 'rb') as fp:
            cached_roidb = pickle.load(fp)

        assert len(roidb) == len(cached_roidb)

        for entry, cached_entry in zip(roidb, cached_roidb):
            values = [cached_entry[key] for key in self.valid_cached_keys]
            boxes, segms, gt_classes, seg_areas, gt_overlaps, is_crowd, \
                box_to_gt_ind_map = values[:7]
            if self.keypoints is not None:
                gt_keypoints, has_visible_keypoints = values[7:]
            entry['boxes'] = np.append(entry['boxes'], boxes, axis=0)
            entry['segms'].extend(segms)
            # To match the original implementation:
            # entry['boxes'] = np.append(
            #     entry['boxes'], boxes.astype(np.int).astype(np.float), axis=0)
            entry['gt_classes'] = np.append(entry['gt_classes'], gt_classes)
            entry['seg_areas'] = np.append(entry['seg_areas'], seg_areas)
            entry['gt_overlaps'] = scipy.sparse.csr_matrix(gt_overlaps)
            entry['is_crowd'] = np.append(entry['is_crowd'], is_crowd)
            entry['box_to_gt_ind_map'] = np.append(
                entry['box_to_gt_ind_map'], box_to_gt_ind_map
            )


def _add_class_assignments(roidb):
    """Compute object category assignment for each box associated with each
    roidb entry.    # matbe is set for proposals. all the roidb is gt, this is none of business.  by lc
    """
    for entry in roidb:
        gt_overlaps = entry['gt_overlaps'].toarray()  # (n, 801)  lc
        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = gt_overlaps.argmax(axis=1)
        entry['max_classes'] = max_classes
        entry['max_overlaps'] = max_overlaps
        # sanity checks
        # if max overlap is 0, the class must be background (class 0)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # if max overlap > 0, the class must be a fg class (not class 0)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)

