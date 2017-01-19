import argparse
import logging
import os
import pprint
import mxnet as mx

from ..config import config
from ..symbol import *
from ..dataset import *
from ..core import callback, metric
from ..core.loader import ROIIter
from ..core.module import MutableModule
from ..processing.bbox_regression import add_bbox_regression_targets
from ..utils.load_model import load_param

# config.TRAIN.BG_THRESH_LO = 0.0 [not used for Fast R-CNN]


def train_rcnn(args, ctx, pretrained, epoch, prefix, begin_epoch, end_epoch,
               finetune=False, lr=0.001, lr_step=30000, proposal='rpn'):
    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # set up config
    config.TRAIN.BATCH_IMAGES = 2
    config.TRAIN.BATCH_ROIS = 128

    # load symbol
    sym = eval('get_' + args.network + '_rcnn')()

    # setup multi-gpu
    batch_size = len(ctx)
    input_batch_size = config.TRAIN.BATCH_IMAGES * batch_size

    # print config
    pprint.pprint(args)
    pprint.pprint(config)

    # load dataset and prepare imdb for training
    imdb = eval(args.dataset)(args.image_set, args.root_path, args.dataset_path)
    gt_roidb = imdb.gt_roidb()
    roidb = eval('imdb.' + proposal + '_roidb')(gt_roidb)
    if args.flip:
        roidb = imdb.append_flipped_images(roidb)
    means, stds = add_bbox_regression_targets(roidb)

    # load training data
    train_data = ROIIter(roidb, batch_size=input_batch_size, shuffle=True,
                         ctx=ctx, work_load_list=args.work_load_list)

    # infer max shape
    max_data_shape = [('data', (input_batch_size, 3, 1000, 1000))]

    # load pretrained
    arg_params, aux_params = load_param(pretrained, epoch, convert=True)

    # infer shape
    data_shape_dict = dict(train_data.provide_data + train_data.provide_label)
    arg_shape, out_shape, aux_shape = sym.infer_shape(**data_shape_dict)
    arg_shape_dict = dict(zip(sym.list_arguments(), arg_shape))
    out_shape_dict = dict(zip(sym.list_outputs(), out_shape))
    aux_shape_dict = dict(zip(sym.list_auxiliary_states(), aux_shape))
    print 'output shape'
    pprint.pprint(out_shape_dict)

    # initialize params
    if not args.resume:
        arg_params['cls_score_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['cls_score_weight'])
        arg_params['cls_score_bias'] = mx.nd.zeros(shape=arg_shape_dict['cls_score_bias'])
        arg_params['bbox_pred_weight'] = mx.random.normal(0, 0.001, shape=arg_shape_dict['bbox_pred_weight'])
        arg_params['bbox_pred_bias'] = mx.nd.zeros(shape=arg_shape_dict['bbox_pred_bias'])

    # check parameter shapes
    for k in sym.list_arguments():
        if k in data_shape_dict:
            continue
        assert k in arg_params, k + ' not initialized'
        assert arg_params[k].shape == arg_shape_dict[k], \
            'shape inconsistent for ' + k + ' inferred ' + str(arg_shape_dict[k]) + ' provided ' + str(arg_params[k].shape)
    for k in sym.list_auxiliary_states():
        assert k in aux_params, k + ' not initialized'
        assert aux_params[k].shape == aux_shape_dict[k], \
            'shape inconsistent for ' + k + ' inferred ' + str(aux_shape_dict[k]) + ' provided ' + str(aux_params[k].shape)

    # prepare training
    # create solver
    data_names = [k[0] for k in train_data.provide_data]
    label_names = [k[0] for k in train_data.provide_label]
    if finetune:
        fixed_param_prefix = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
    else:
        fixed_param_prefix = ['conv1', 'conv2']
    mod = MutableModule(sym, data_names=data_names, label_names=label_names,
                        logger=logger, context=ctx, work_load_list=args.work_load_list,
                        max_data_shapes=max_data_shape, fixed_param_prefix=fixed_param_prefix)

    # decide training params
    # metric
    eval_metric = metric.RCNNAccMetric()
    cls_metric = metric.RCNNLogLossMetric()
    bbox_metric = metric.RCNNL1LossMetric()
    eval_metrics = mx.metric.CompositeEvalMetric()
    for child_metric in [eval_metric, cls_metric, bbox_metric]:
        eval_metrics.add(child_metric)
    # callback
    batch_end_callback = callback.Speedometer(train_data.batch_size, frequent=args.frequent)
    epoch_end_callback = callback.do_checkpoint(prefix, means, stds)
    # optimizer
    optimizer_params = {'momentum': 0.9,
                        'wd': 0.0005,
                        'learning_rate': lr,
                        'lr_scheduler': mx.lr_scheduler.FactorScheduler(lr_step, 0.1),
                        'rescale_grad': (1.0 / batch_size)}

    # train
    mod.fit(train_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callback,
            batch_end_callback=batch_end_callback, kvstore=args.kvstore,
            optimizer='sgd', optimizer_params=optimizer_params,
            arg_params=arg_params, aux_params=aux_params, begin_epoch=begin_epoch, num_epoch=end_epoch)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN Network')
    # general
    parser.add_argument('--network', help='network name',
                        default='vgg', type=str)
    parser.add_argument('--dataset', help='dataset name',
                        default='PascalVOC', type=str)
    parser.add_argument('--image_set', help='image_set name',
                        default='2007_trainval', type=str)
    parser.add_argument('--root_path', help='output data folder',
                        default='data', type=str)
    parser.add_argument('--dataset_path', help='dataset path',
                        default=os.path.join('data', 'VOCdevkit'), type=str)
    # training
    parser.add_argument('--frequent', help='frequency of logging',
                        default=20, type=int)
    parser.add_argument('--kvstore', help='the kv-store type',
                        default='device', type=str)
    parser.add_argument('--work_load_list', help='work load for different devices',
                        default=None, type=list)
    parser.add_argument('--flip', help='flip images', action='store_true', default=True)
    parser.add_argument('--resume', help='continue training', action='store_true')
    # rcnn
    parser.add_argument('--gpus', help='GPU device to train with',
                        default='0', type=str)
    parser.add_argument('--pretrained', help='pretrained model prefix',
                        default=os.path.join('model', 'vgg16'), type=str)
    parser.add_argument('--epoch', help='epoch of pretrained model',
                        default=1, type=int)
    parser.add_argument('--prefix', help='new model prefix',
                        default=os.path.join('model', 'rcnn'), type=str)
    parser.add_argument('--begin_epoch', help='begin epoch of training',
                        default=0, type=int)
    parser.add_argument('--end_epoch', help='end epoch of training',
                        default=8, type=int)
    parser.add_argument('--finetune', help='second round finetune', action='store_true')
    parser.add_argument('--lr', help='base learning rate', default=0.001, type=float)
    parser.add_argument('--lr_step', help='learning rate step', default=30000, type=int)
    parser.add_argument('--proposal', help='can be ss for selective search or rpn',
                        default='rpn', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    train_rcnn(args, ctx, args.pretrained, args.epoch, args.prefix, args.begin_epoch, args.end_epoch,
               finetune=args.finetune, lr=args.lr, lr_step=args.lr_step, proposal=args.proposal)
