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

"""Functions for common roidb manipulations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import six
import logging
import numpy as np

from .boxes import *
from .config import cfg
from .json_dataset import JsonDataset

logger = logging.getLogger(__name__)


def combined_roidb_for_training(root_dir, subset, dataset_names, proposal_files):
    """Load and concatenate roidbs for one or more datasets, along with optional
    object proposals. The roidb entries are then prepared for use in training,
    which involves caching certain types of metadata for each roidb entry.
    """
    def get_roidb(dataset_name, proposal_file):  # dataset_name: 'fsod_train'
        ds = JsonDataset(root_dir=root_dir, subset=subset)
        roidb = ds.get_roidb(
            gt=True,
            crowd_filter_thresh=cfg.TRAIN.CROWD_FILTER_THRESH   # 0.7
        )
        # flipping is not required here
        logger.info('Loaded dataset: {:s}'.format(ds.name))
        return roidb

    if isinstance(dataset_names, six.string_types):
        dataset_names = (dataset_names, )
    if isinstance(proposal_files, six.string_types):
        proposal_files = (proposal_files, )
    if len(proposal_files) == 0:
        proposal_files = (None, ) * len(dataset_names)
    assert len(dataset_names) == len(proposal_files)
    roidbs = [get_roidb(*args) for args in zip(dataset_names, proposal_files)]
    original_roidb = roidbs[0]

    # new dataset split according to class
    roidb = []
    for item in original_roidb:
        gt_classes = list(set(item['gt_classes']))
        all_cls = np.array(item['gt_classes'])

        for cls in gt_classes:
            item_new = item.copy()
            target_idx = np.where(all_cls == cls)[0]
            #item_new['id'] = item_new['id'] * 1000 + int(cls)
            item_new['target_cls'] = int(cls)
            item_new['boxes'] = item_new['boxes'][target_idx]
            item_new['max_classes'] = item_new['max_classes'][target_idx]
            item_new['gt_classes'] = item_new['gt_classes'][target_idx]
            item_new['is_crowd'] = item_new['is_crowd'][target_idx]
            item_new['segms'] = item_new['segms'][:target_idx.shape[0]]
            item_new['seg_areas'] = item_new['seg_areas'][target_idx]
            item_new['max_overlaps'] = item_new['max_overlaps'][target_idx]
            item_new['box_to_gt_ind_map'] = np.array(range(item_new['gt_classes'].shape[0]))
            item_new['gt_overlaps'] = item_new['gt_overlaps'][target_idx]
            roidb.append(item_new)

    for r in roidbs[1:]:
        roidb.extend(r)
    roidb = filter_for_training(roidb)

    if cfg.TRAIN.ASPECT_GROUPING or cfg.TRAIN.ASPECT_CROPPING:
        logger.info('Computing image aspect ratios and ordering the ratios...')
        ratio_list, ratio_index, cls_list, id_list = rank_for_training(roidb)
        logger.info('done')
    else:
        ratio_list, ratio_index, cls_list, id_list = None, None, None, None

    _compute_and_log_stats(roidb)

    print(len(roidb))
    return roidb, ratio_list, ratio_index, cls_list, id_list


def filter_for_training(roidb):
    """Remove roidb entries that have no usable RoIs based on config settings.
    """
    def is_valid(entry):
        # Valid images have:
        #   (1) At least one foreground RoI OR
        #   (2) At least one background RoI
        overlaps = entry['max_overlaps']
        # find boxes with sufficient overlap
        fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # image is only valid if such boxes exist
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    logger.info('Filtered {} roidb entries: {} -> {}'.
                format(num - num_after, num, num_after))
    return filtered_roidb


def rank_for_training(roidb):
    """Rank the roidb entries according to image aspect ration and mark for cropping
    for efficient batching if image is too long.

    Returns:
        ratio_list: ndarray, list of aspect ratios from small to large
        ratio_index: ndarray, list of roidb entry indices correspond to the ratios
    """
    RATIO_HI = cfg.TRAIN.ASPECT_HI  # largest ratio to preserve.
    RATIO_LO = cfg.TRAIN.ASPECT_LO  # smallest ratio to preserve.

    need_crop_cnt = 0

    ratio_list = []
    cls_list = []
    id_list = []
    for entry in roidb:
        width = entry['width']
        height = entry['height']
        ratio = width / float(height)
        target_cls = entry['target_cls']
        img_id = entry['id'] #int(str(entry['id'])[:-3])

        if cfg.TRAIN.ASPECT_CROPPING:
            if ratio > RATIO_HI:
                entry['need_crop'] = True
                ratio = RATIO_HI
                need_crop_cnt += 1
            elif ratio < RATIO_LO:
                entry['need_crop'] = True
                ratio = RATIO_LO
                need_crop_cnt += 1
            else:
                entry['need_crop'] = False
        else:
            entry['need_crop'] = False

        ratio_list.append(ratio)
        cls_list.append(target_cls)
        id_list.append(img_id)

    if cfg.TRAIN.ASPECT_CROPPING:
        logging.info('Number of entries that need to be cropped: %d. Ratio bound: [%.2f, %.2f]',
                     need_crop_cnt, RATIO_LO, RATIO_HI)
    ratio_list = np.array(ratio_list)
    ratio_index = np.argsort(ratio_list)
    cls_list = np.array(cls_list)
    id_list = np.array(id_list)
    return ratio_list[ratio_index], ratio_index, cls_list, id_list


def _compute_and_log_stats(roidb):
    classes = roidb[0]['dataset'].classes
    char_len = np.max([len(c) for c in classes])
    hist_bins = np.arange(len(classes) + 1)

    # Histogram of ground-truth objects
    gt_hist = np.zeros((len(classes)), dtype=np.int)
    for entry in roidb:
        gt_inds = np.where(
            (entry['gt_classes'] > 0) & (entry['is_crowd'] == 0))[0]
        gt_classes = entry['gt_classes'][gt_inds]
        gt_hist += np.histogram(gt_classes, bins=hist_bins)[0]
    logger.debug('Ground-truth class histogram:')
    for i, v in enumerate(gt_hist):
        logger.debug(
            '{:d}{:s}: {:d}'.format(
                i, classes[i].rjust(char_len), v))
    logger.debug('-' * char_len)
    logger.debug(
        '{:s}: {:d}'.format(
            'total'.rjust(char_len), np.sum(gt_hist)))
