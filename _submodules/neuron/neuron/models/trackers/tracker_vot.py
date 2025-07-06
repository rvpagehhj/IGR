import numpy as np
import time

import _submodules.neuron.neuron.ops as ops
from _submodules.neuron.neuron.models.model import Model

__all__ = ['TrackerVOT']


class TrackerVOT(Model):

    def __init__(self, name, is_deterministic=True,
                 input_type='image', color_fmt='RGB'):
        assert input_type in ['image', 'file']
        assert color_fmt in ['RGB', 'BGR', 'GRAY']
        super(TrackerVOT, self).__init__()
        self.name = name
        self.is_deterministic = is_deterministic
        self.input_type = input_type
        self.color_fmt = color_fmt

    def init(self, img, init_bbox):
        raise NotImplementedError

    def update(self, img):
        raise NotImplementedError

    def forward_test(self, img_files, init_bbox, gt_list, visualize=False):
        # state variables
        frame_num = len(img_files)
        bboxes = np.zeros((frame_num, 4))
        times_details = np.zeros((frame_num, 5))
        bboxes[0] = init_bbox
        # by lc
        scores = np.zeros(frame_num)
        # end
        times = np.zeros(frame_num)
        recall = np.zeros((frame_num,4))  # or np.zeros((frame_num,6))

        for f, img_file in enumerate(img_files):
            if self.input_type == 'image':
                img = ops.read_image(img_file, self.color_fmt)
            elif self.input_type == 'file':
                img = img_file

            begin = time.time()
            if f == 0:
                times_details[0, 0] = self.init(img, init_bbox)
                scores[0] = 0  # for vot18lt
                recall[0,:] = [1,1,1,1]  # or [1,1,1,1,1,1]
            else:

                # bboxes[f, :], scores[f] = self.update(img)
                # bboxes[f, :], recall[f,:], scores[f] = self.update(img, gt_list[f])
                bboxes[f, :], scores[f] = self.update(img, gt_list[f])

                #############    选项如下:
                #   原始版本：bboxes[f, :] = self.update(img)
                # 　改动后的原始版本，为了观察globaltrack的分段时间：bboxes[f, :], times_details[f, :] = self.update(img)
                # 　ssd改装版，ttf改装版，去掉了分段时间的记录:　bboxes[f, :] = self.update(img)
                # 　对于vot18lt的测试，需要置信度得分记录：bboxes[f, :],scores[f]  = self.update(img)
                #   为了观察globaltrack的分段时间，globaltrack额版本输出最多: bboxes[f, :], times_details[f, :], scores[f] = self.update(img)  # for vot18lt && globaltrack
            times[f] = time.time() - begin

            if visualize:
                ops.show_image(img, bboxes[f, :])

        # return bboxes, times , times_details    # 这是为了观察globaltrack的分段时间所需要的返回格式
        # return bboxes, times     #　这个是原始的版本也是通常使用的版本
        return bboxes, times, scores  # for vot2018lt, 两层返回不要搞混了
        # return bboxes, recall, scores  # for vot2018lt, 两层返回不要搞混了
