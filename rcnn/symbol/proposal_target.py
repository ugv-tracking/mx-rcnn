"""
Proposal Target Operator selects foreground and background roi and assigns label, bbox_transform to them.
"""

import mxnet as mx
import numpy as np
from rcnn.config import config

from rcnn.core.minibatch import sample_rois

DEBUG = False


class ProposalTargetOperator(mx.operator.CustomOp):
    def __init__(self, num_classes, batch_images, batch_rois, fg_fraction):
        super(ProposalTargetOperator, self).__init__()
        self._num_classes = num_classes
        self._batch_images = batch_images
        self._batch_rois = batch_rois
        self._fg_fraction = fg_fraction

        if DEBUG:
            self._count = 0
            self._fg_num = 0
            self._bg_num = 0


    def forward(self, is_train, req, in_data, out_data, aux):
        assert self._batch_rois % self._batch_images == 0, \
            'BATCHIMAGES {} must devide BATCH_ROIS {}'.format(self._batch_images, self._batch_rois)
        rois_per_image = self._batch_rois / self._batch_images
        fg_rois_per_image = np.round(self._fg_fraction * rois_per_image).astype(int)

        all_rois = in_data[0].asnumpy()
        gt_boxes = in_data[1].asnumpy()
        orientation_ry = in_data[2].asnumpy()
        orientation_alpha = in_data[3].asnumpy()
        im_info = in_data[4].asnumpy()

 

        # Include ground-truth boxes in the set of candidate rois
        zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        all_rois = np.vstack((all_rois, np.hstack((zeros, gt_boxes[:, :-1]))))
        # Sanity check: single batch only
        assert np.all(all_rois[:, 0] == 0), 'Only single item batches are supported'

        rois, labels, bbox_targets, bbox_inside_weights, orientation_ry_targets, orientation_alpha_targets, orientation_weight= \
            sample_rois(all_rois, fg_rois_per_image, rois_per_image, self._num_classes, gt_boxes=gt_boxes, orientation_ry = orientation_ry, orientation_alpha = orientation_alpha)
        bbox_outside_weight = np.array(bbox_inside_weights > 0).astype(np.float32)


        if DEBUG:
            print "labels=", labels
            print 'num fg: {}'.format((labels > 0).sum())
            print 'num bg: {}'.format((labels == 0).sum())
            self._count += 1
            self._fg_num += (labels > 0).sum()
            self._bg_num += (labels == 0).sum()
            print "self._count=", self._count
            print 'num fg avg: {}'.format(self._fg_num / self._count)
            print 'num bg avg: {}'.format(self._bg_num / self._count)
            print 'ratio: {:.3f}'.format(float(self._fg_num) / float(self._bg_num))

        output = [rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weight]
        if config.TRAIN.ORIENTATION:
            output = [rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weight, orientation_ry_targets, orientation_alpha_targets, orientation_weight]

        for ind, val in enumerate(output):
            self.assign(out_data[ind], req[ind], val)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)
        if config.TRAIN.ORIENTATION:
            self.assign(in_grad[2], req[2], 0)
            self.assign(in_grad[3], req[3], 0)

    @staticmethod
    def get_context_rois(rois, im_info):
        """
        generate context ROIs that are 1.5 times the size of input ROIs
        :param rois: sampled rois
        :param im_info: [height, width, scale]
        """
        rois_center_x = (rois[:, 3] + rois[:, 1]) / 2.0
        rois_center_y = (rois[:, 4] + rois[:, 2]) / 2.0
        x_max = im_info[0][1]  # width
        y_max = im_info[0][0]  # height

        # keep rois_ctx inside the image
        rois_ctx_x1 = np.maximum(1.5 * (rois[:, 1] - rois_center_x) + rois_center_x, 0)
        rois_ctx_x2 = np.minimum(1.5 * (rois[:, 3] - rois_center_x) + rois_center_x, x_max)
        rois_ctx_y1 = np.maximum(1.5 * (rois[:, 2] - rois_center_y) + rois_center_y, 0)
        rois_ctx_y2 = np.minimum(1.5 * (rois[:, 4] - rois_center_y) + rois_center_y, y_max)

        rois_ctx = np.vstack((rois[:, 0], rois_ctx_x1, rois_ctx_y1, rois_ctx_x2, rois_ctx_y2)).transpose()
        return rois_ctx

@mx.operator.register('proposal_target')
class ProposalTargetProp(mx.operator.CustomOpProp):
    def __init__(self, num_classes, batch_images, batch_rois, fg_fraction):
        super(ProposalTargetProp, self).__init__(need_top_grad=False)
        self._num_classes = int(num_classes)
        self._batch_images = int(batch_images)
        self._batch_rois = int(batch_rois)
        self._fg_fraction = float(fg_fraction)

    def list_arguments(self):
        return ['rois', 'gt_boxes', 'orientation_ry', 'orientation_alpha', 'im_info']

    def list_outputs(self):
        if config.TRAIN.ORIENTATION:
            return ['rois_output', 'label', 'bbox_target', 'bbox_inside_weight', 'bbox_outside_weight', 'orientation_ry_target', 'orientation_alpha_target', 'orientation_weight']
        else:
            return ['rois_output', 'label', 'bbox_target', 'bbox_inside_weight', 'bbox_outside_weight']

    def infer_shape(self, in_shape):
        rpn_rois_shape = in_shape[0]
        gt_boxes_shape = in_shape[1]
        if config.TRAIN.ORIENTATION:
            orientation_ry_reshape = in_shape[2]
            orientation_alpha_reshape = in_shape[3]
            im_info_shape = in_shape[4]

        output_rois_shape = (self._batch_rois, 5)
        label_shape = (self._batch_rois, )
        bbox_target_shape = (self._batch_rois, self._num_classes * 4)
        bbox_inside_weight_shape = (self._batch_rois, self._num_classes * 4)
        bbox_outside_weight_shape = (self._batch_rois, self._num_classes * 4)

        orientation_ry_target_shape = (self._batch_rois, 1)
        orientation_alpha_target_shape = (self._batch_rois, 1)
        orientation_weight_shape = (self._batch_rois, 1)


        if config.TRAIN.ORIENTATION:
            return [rpn_rois_shape, gt_boxes_shape, orientation_ry_reshape, orientation_alpha_reshape, im_info_shape], \
               [output_rois_shape, label_shape, bbox_target_shape, bbox_inside_weight_shape, bbox_outside_weight_shape, \
                orientation_ry_target_shape, orientation_alpha_target_shape, orientation_weight_shape]
        else:
            return [rpn_rois_shape, gt_boxes_shape], \
                   [output_rois_shape, label_shape, bbox_target_shape, bbox_inside_weight_shape, bbox_outside_weight_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return ProposalTargetOperator(self._num_classes, self._batch_images, self._batch_rois, self._fg_fraction)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
