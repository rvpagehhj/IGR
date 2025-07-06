# # Copyright (c) 2017-present, Facebook, Inc.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# ##############################################################################
# #
# # Based on:
# # --------------------------------------------------------
# # Fast/er R-CNN
# # Licensed under The MIT License [see LICENSE for details]
# # Written by Ross Girshick
# # --------------------------------------------------------
#
# """Box manipulation functions. The internal Detectron box format is
# [x1, y1, x2, y2] where (x1, y1) specify the top-left box corner and (x2, y2)
# specify the bottom-right box corner. Boxes from external sources, e.g.,
# datasets, may be in other formats (such as [x, y, w, h]) and require conversion.
#
# This module uses a convention that may seem strange at first: the width of a box
# is computed as x2 - x1 + 1 (likewise for height). The "+ 1" dates back to old
# object detection days when the coordinates were integer pixel indices, rather
# than floating point coordinates in a subpixel coordinate frame. A box with x2 =
# x1 and y2 = y1 was taken to include a single pixel, having a width of 1, and
# hence requiring the "+ 1". Now, most datasets will likely provide boxes with
# floating point coordinates and the width should be more reasonably computed as
# x2 - x1.
#
# In practice, as long as a model is trained and tested with a consistent
# convention either decision seems to be ok (at least in our experience on COCO).
# Since we have a long history of training models with the "+ 1" convention, we
# are reluctant to change it even if our modern tastes prefer not to use it.
# """
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
#
# import warnings
import numpy as np
#
# from .config import cfg
#
# import cython_bbox as cython_bbox
# from .cython_nms import cython_nms
#
# bbox_overlaps = cython_bbox.bbox_overlaps
#
#
# def boxes_area(boxes):
#     """Compute the area of an array of boxes."""
#     w = (boxes[:, 2] - boxes[:, 0] + 1)
#     h = (boxes[:, 3] - boxes[:, 1] + 1)
#     areas = w * h
#
#     neg_area_idx = np.where(areas < 0)[0]
#     if neg_area_idx.size:
#         warnings.warn("Negative areas founds: %d" % neg_area_idx.size, RuntimeWarning)
#     #TODO proper warm up and learning rate may reduce the prob of assertion fail
#     # assert np.all(areas >= 0), 'Negative areas founds'
#     return areas, neg_area_idx
#
#
# def unique_boxes(boxes, scale=1.0):
#     """Return indices of unique boxes."""
#     v = np.array([1, 1e3, 1e6, 1e9])
#     hashes = np.round(boxes * scale).dot(v)
#     _, index = np.unique(hashes, return_index=True)
#     return np.sort(index)
#
#
def xywh_to_xyxy(xywh):
    """Convert [x1 y1 w h] box format to [x1 y1 x2 y2] format."""
    if isinstance(xywh, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xywh) == 4
        x1, y1 = xywh[0], xywh[1]
        x2 = x1 + np.maximum(0., xywh[2] - 1.)
        y2 = y1 + np.maximum(0., xywh[3] - 1.)
        return (x1, y1, x2, y2)
    elif isinstance(xywh, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        return np.hstack(
            (xywh[:, 0:2], xywh[:, 0:2] + np.maximum(0, xywh[:, 2:4] - 1))
        )
    else:
        raise TypeError('Argument xywh must be a list, tuple, or numpy array.')

#
def xyxy_to_xywh(xyxy):
    """Convert [x1 y1 x2 y2] box format to [x1 y1 w h] format."""
    if isinstance(xyxy, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xyxy) == 4
        x1, y1 = xyxy[0], xyxy[1]
        w = xyxy[2] - x1 + 1
        h = xyxy[3] - y1 + 1
        return (x1, y1, w, h)
    elif isinstance(xyxy, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        return np.hstack((xyxy[:, 0:2], xyxy[:, 2:4] - xyxy[:, 0:2] + 1))
    else:
        raise TypeError('Argument xyxy must be a list, tuple, or numpy array.')
#
#
# def filter_small_boxes(boxes, min_size):
#     """Keep boxes with width and height both greater than min_size."""
#     w = boxes[:, 2] - boxes[:, 0] + 1
#     h = boxes[:, 3] - boxes[:, 1] + 1
#     keep = np.where((w > min_size) & (h > min_size))[0]
#     return keep
#
#
# def clip_boxes_to_image(boxes, height, width):
#     """Clip an array of boxes to an image with the given height and width."""
#     boxes[:, [0, 2]] = np.minimum(width - 1., np.maximum(0., boxes[:, [0, 2]]))
#     boxes[:, [1, 3]] = np.minimum(height - 1., np.maximum(0., boxes[:, [1, 3]]))
#     return boxes
#
#
def clip_xyxy_to_image(x1, y1, x2, y2, height, width):
    """Clip coordinates to an image with the given height and width."""
    x1 = np.minimum(width - 1., np.maximum(0., x1))
    y1 = np.minimum(height - 1., np.maximum(0., y1))
    x2 = np.minimum(width - 1., np.maximum(0., x2))
    y2 = np.minimum(height - 1., np.maximum(0., y2))
    return x1, y1, x2, y2
