import numpy as np
import time

import _submodules.neuron.neuron.ops as ops
from _submodules.neuron.neuron.models.model import Model


__all__ = ['Tracker', 'OxUvA_Tracker']


class Tracker(Model):

    def __init__(self, name, is_deterministic=True,
                 input_type='image', color_fmt='RGB'):
        assert input_type in ['image', 'file']
        assert color_fmt in ['RGB', 'BGR', 'GRAY']
        super(Tracker, self).__init__()
        self.name = name
        self.is_deterministic = is_deterministic
        self.input_type = input_type
        self.color_fmt = color_fmt
    
    def init(self, img, init_bbox):
        raise NotImplementedError
    
    def update(self, img, frame_num):
        raise NotImplementedError
    
    def forward_test(self, img_files, init_bbox,gt_list, visualize=False):
        # state variables
        frame_num = len(img_files)
        bboxes = np.zeros((frame_num, 4))
        bboxes[0] = init_bbox
        times = np.zeros(frame_num)
        scores = np.zeros(frame_num)
        recall = np.zeros((frame_num,6))

        for f, img_file in enumerate(img_files):
            if self.input_type == 'image':
                img = ops.read_image(img_file, self.color_fmt)
            elif self.input_type == 'file':
                img = img_file

            begin = time.time()
            if f == 0:
                self.init(img, init_bbox)
                attention = None
                # recall[0,:] = [1,1,1,1,1,1]
            else:
                # bboxes[f, :],scores[f] = self.update(img)
                # bboxes[f, :],recall[f, :] = self.update(img, gt_list[f])
                bboxes[f, :] = self.update(img)  # for normal
                # for one shot multiple
                # bboxes,scores = self.update(img, gt_list[f])  # wrong!!!!
            times[f] = time.time() - begin

            if visualize:
                ops.show_image(img, bboxes[f, :],scores[f])
                # # for one shot  # can't record correctly in this way !!!! but vis is ok!
                # ops.show_image_shot(img, bboxes, scores)  # 20,4   20,
                # ops.show_image(img, bboxes[f, :], scores[f], attention)

        # return bboxes, recall
        return bboxes, times


class OxUvA_Tracker(Tracker):

    def update(self, img):
        r'''One needs to return (bbox, score, present) in
            function `update`.
        '''
        raise NotImplementedError

    def forward_test(self, img_files, init_bbox, visualize=False):
        frame_num = len(img_files)
        times = np.zeros(frame_num)
        bboxes = np.zeros(frame_num,4)
        preds = [{
            'present': True,
            'score': 1.0,
            'xmin': init_bbox[0],
            'xmax': init_bbox[2],
            'ymin': init_bbox[1],
            'ymax': init_bbox[3]}]
        
        for f, img_file in enumerate(img_files):
            if self.input_type == 'image':
                img = ops.read_image(img_file, self.color_fmt)
            elif self.input_type == 'file':
                img = img_file

            begin = time.time()
            if f == 0:
                self.init(img, init_bbox)
            else:
                bbox, score, present = self.update(img)
                preds.append({
                    'present': present,
                    'score': score,
                    'xmin': bbox[0],
                    'xmax': bbox[2],
                    'ymin': bbox[1],
                    'ymax': bbox[3]})
                bboxes[f,:] = bbox
            times[f] = time.time() - begin

            if visualize:
                ops.show_image(img, bboxes[f, :])
        
        # update the preds as one-per-second
        frame_stride = 30
        preds = {f * frame_stride: pred for f, pred in enumerate(preds)}

        return preds, times
