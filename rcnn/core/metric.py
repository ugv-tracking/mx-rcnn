import mxnet as mx
import numpy as np

from rcnn.config import config


class RPNAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNAccMetric, self).__init__('RPNAcc')

    def update(self, labels, preds):
        pred = preds[0]
        label = labels[0]

        # pred (b, c, p) or (b, c, h, w)
        pred_label = mx.ndarray.argmax_channel(pred).asnumpy().astype('int32')
        pred_label = pred_label.reshape((pred_label.shape[0], -1))
        # label (b, p)
        label = label.asnumpy().astype('int32')
        non_ignore_inds = np.where(label != -1)
        pred_label = pred_label[non_ignore_inds]
        label = label[non_ignore_inds]

        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)


class RCNNAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNAccMetric, self).__init__('RCNNAcc')

    def update(self, labels, preds):
        if config.TRAIN.END2END:
            pred = preds[2]
            label = preds[4]
        else:
            pred = preds[0]
            label = labels[0]

        last_dim = pred.shape[-1]
        pred_label = pred.asnumpy().reshape(-1, last_dim).argmax(axis=1).astype('int32')
        label = label.asnumpy().reshape(-1,).astype('int32')

        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)


class RPNLogLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNLogLossMetric, self).__init__('RPNLogLoss')

    def update(self, labels, preds):
        pred = preds[0]
        label = labels[0]

        # label (b, p)
        label = label.asnumpy().astype('int32').reshape((-1))
        # pred (b, c, p) or (b, c, h, w) --> (b, p, c) --> (b*p, c)
        pred = pred.asnumpy().reshape((pred.shape[0], pred.shape[1], -1)).transpose((0, 2, 1))
        pred = pred.reshape((label.shape[0], -1))

        non_ignore_inds = np.where(label != -1)[0]
        label = label[non_ignore_inds]
        cls = pred[non_ignore_inds, label]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]


class RCNNLogLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNLogLossMetric, self).__init__('RCNNLogLoss')

    def update(self, labels, preds):
        if config.TRAIN.END2END:
            pred = preds[2]
            label = preds[4]
        else:
            pred = preds[0]
            label = labels[0]

        last_dim = pred.shape[-1]
        pred = pred.asnumpy().reshape(-1, last_dim)
        label = label.asnumpy().reshape(-1,).astype('int32')
        cls = pred[np.arange(label.shape[0]), label]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]

class RPNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNL1LossMetric, self).__init__('RPNL1Loss')

    def update(self, labels, preds):
        pred = preds[1]

        bbox_loss = pred.asnumpy()
        bbox_loss = bbox_loss.reshape((bbox_loss.shape[0], -1))

        self.sum_metric += np.sum(bbox_loss)
        self.num_inst += bbox_loss.shape[0]


class RCNNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNL1LossMetric, self).__init__('RCNNL1Loss')
        self.has_rpn = config.TRAIN.HAS_RPN

    def update(self, labels, preds):
        if config.TRAIN.END2END:
            pred = preds[3]
        else:
            pred = preds[1]

        bbox_loss = pred.asnumpy()
        first_dim = bbox_loss.shape[0] * bbox_loss.shape[1]
        bbox_loss = bbox_loss.reshape(first_dim, -1)

        self.sum_metric += np.sum(bbox_loss)
        self.num_inst += bbox_loss.shape[0]


#######################################################################################


class RCNNOrientationRyLogLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNOrientationRyLogLossMetric, self).__init__('RCNNOrientationRyLogLoss')

    def update(self, labels, preds):
        assert config.TRAIN.ORIENTATION

        obj_label = preds[4]
        orientatio_pred = preds[5]
        orientatio_label = preds[6]

        last_dim = orientatio_pred.shape[-1]
        obj_label        = obj_label.asnumpy().reshape(-1,).astype('int32')
        orientatio_pred  = orientatio_pred.asnumpy().reshape(-1, last_dim)
        orientatio_label = orientatio_label.asnumpy().reshape(-1,).astype('int32')

        non_ignore_inds = np.where(obj_label > 0)[0]
        # only BP for the non-negative samples
        if len(non_ignore_inds) == 0:
            self.sum_metric += 0
            self.num_inst += 0
            print 'no pos samples'
            return 

        orientatio_label = orientatio_label[non_ignore_inds]
        cls = orientatio_pred[non_ignore_inds, orientatio_label]
        #print 'non_ignore_inds: ', non_ignore_inds
        #print 'orientatio_label: ', orientatio_label
        #print 'cls: ', cls

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += orientatio_label.shape[0]



class RCNNOrientationRyAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNOrientationRyAccMetric, self).__init__('RCNNOrientationRyAcc')

    def update(self, labels, preds):
        assert config.TRAIN.ORIENTATION

        obj_label = preds[4]

        pred = preds[5]
        label = preds[6]
      

        last_dim = pred.shape[-1]
        pred_label = pred.asnumpy().reshape(-1, last_dim).argmax(axis=1).astype('int32')
        label      = label.asnumpy().reshape(-1,).astype('int32')
        obj_label  = obj_label.asnumpy().reshape(-1,).astype('int32')


        non_ignore_inds = np.where(obj_label != 0)[0]
        pred_label = pred_label[non_ignore_inds]
        label = label[non_ignore_inds]


        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)


##############################################
class RCNNOrientationAlphaLogLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNOrientationAlphaLogLossMetric, self).__init__('RCNNOrientationAlphaLogLoss')

    def update(self, labels, preds):
        assert config.TRAIN.ORIENTATION

        obj_label = preds[4]
        orientatio_pred = preds[7]
        orientatio_label = preds[8]

        last_dim = orientatio_pred.shape[-1]
        obj_label        = obj_label.asnumpy().reshape(-1,).astype('int32')
        orientatio_pred  = orientatio_pred.asnumpy().reshape(-1, last_dim)
        orientatio_label = orientatio_label.asnumpy().reshape(-1,).astype('int32')

        non_ignore_inds = np.where(obj_label > 0)[0]
        # only BP for the non-negative samples
        if len(non_ignore_inds) == 0:
            self.sum_metric += 0
            self.num_inst += 0
            print 'no pos samples'
            return 

        orientatio_label = orientatio_label[non_ignore_inds]
        cls = orientatio_pred[non_ignore_inds, orientatio_label]
        #print 'non_ignore_inds: ', non_ignore_inds
        #print 'orientatio_label: ', orientatio_label
        #print 'cls: ', cls

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += orientatio_label.shape[0]



class RCNNOrientationAlphaAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNOrientationAlphaAccMetric, self).__init__('RCNNOrientationAlphaAcc')

    def update(self, labels, preds):
        assert config.TRAIN.ORIENTATION

        obj_label = preds[4]

        pred = preds[7]
        label = preds[8]
      

        last_dim = pred.shape[-1]
        pred_label = pred.asnumpy().reshape(-1, last_dim).argmax(axis=1).astype('int32')
        label      = label.asnumpy().reshape(-1,).astype('int32')
        obj_label  = obj_label.asnumpy().reshape(-1,).astype('int32')


        non_ignore_inds = np.where(obj_label != 0)[0]
        pred_label = pred_label[non_ignore_inds]
        label = label[non_ignore_inds]


        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)