import argparse
import logging
import os
import pprint
import mxnet as mx

from ..config import config
from ..symbol import *
from ..dataset import *
from ..core import callback, metric
from ..core.loader import AnchorLoader
from ..core.module import MutableModule
from ..utils.load_model import load_param


def train_rpn(args, ctx, pretrained, epoch, prefix, begin_epoch, end_epoch,
              finetune=False, lr=0.001, lr_step=60000):
    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # setup config
    config.TRAIN.HAS_RPN = True
    config.TRAIN.BATCH_SIZE = 1

    # load symbol
    sym = eval('get_' + args.network + '_rpn')()
    feat_sym = get_vgg_rpn().get_internals()['rpn_cls_score_output']

    # setup multi-gpu
    config.TRAIN.BATCH_IMAGES *= len(ctx)
    config.TRAIN.BATCH_SIZE *= len(ctx)

    # print config
    pprint.pprint(config)

    # load dataset and prepare imdb for training
    imdb = eval(args.dataset)(args.image_set, args.root_path, args.dataset_path)
    roidb = imdb.gt_roidb()
    if args.flip:
        roidb = imdb.append_flipped_images(roidb)

    # load training data
    train_data = AnchorLoader(feat_sym, roidb, batch_size=config.TRAIN.BATCH_SIZE, shuffle=True,
                              ctx=ctx, work_load_list=args.work_load_list,
                              feat_stride=config.RPN_FEAT_STRIDE, anchor_scales=config.ANCHOR_SCALES,
                              anchor_ratios=config.ANCHOR_RATIOS)

    # infer max shape
    max_data_shape = [('data', (config.TRAIN.BATCH_IMAGES, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]
    max_data_shape, max_label_shape = train_data.infer_shape(max_data_shape)
    print 'providing maximum shape', max_data_shape, max_label_shape

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
        arg_params['rpn_conv_3x3_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['rpn_conv_3x3_weight'])
        arg_params['rpn_conv_3x3_bias'] = mx.nd.zeros(shape=arg_shape_dict['rpn_conv_3x3_bias'])
        arg_params['rpn_cls_score_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['rpn_cls_score_weight'])
        arg_params['rpn_cls_score_bias'] = mx.nd.zeros(shape=arg_shape_dict['rpn_cls_score_bias'])
        arg_params['rpn_bbox_pred_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['rpn_bbox_pred_weight'])
        arg_params['rpn_bbox_pred_bias'] = mx.nd.zeros(shape=arg_shape_dict['rpn_bbox_pred_bias'])

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

    # create solver
    data_names = [k[0] for k in train_data.provide_data]
    label_names = [k[0] for k in train_data.provide_label]
    if finetune:
        fixed_param_prefix = config.FIXED_PARAMS_FINETUNE
    else:
        fixed_param_prefix = config.FIXED_PARAMS
    mod = MutableModule(sym, data_names=data_names, label_names=label_names,
                        logger=logger, context=ctx, work_load_list=args.work_load_list,
                        max_data_shapes=max_data_shape, max_label_shapes=max_label_shape,
                        fixed_param_prefix=fixed_param_prefix)

    # decide training params
    # metric
    eval_metric = metric.RPNAccMetric()
    cls_metric = metric.RPNLogLossMetric()
    bbox_metric = metric.RPNL1LossMetric()
    eval_metrics = mx.metric.CompositeEvalMetric()
    for child_metric in [eval_metric, cls_metric, bbox_metric]:
        eval_metrics.add(child_metric)
    # callback
    batch_end_callback = callback.Speedometer(train_data.batch_size, frequent=args.frequent)
    epoch_end_callback = mx.callback.do_checkpoint(prefix)
    # optimizer
    optimizer_params = {'momentum': 0.9,
                        'wd': 0.0005,
                        'learning_rate': lr,
                        'lr_scheduler': mx.lr_scheduler.FactorScheduler(lr_step, 0.1),
                        'rescale_grad': (1.0 / config.TRAIN.BATCH_SIZE)}

    # train
    mod.fit(train_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callback,
            batch_end_callback=batch_end_callback, kvstore=args.kvstore,
            optimizer='sgd', optimizer_params=optimizer_params,
            arg_params=arg_params, aux_params=aux_params, begin_epoch=begin_epoch, num_epoch=end_epoch)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Region Proposal Network')
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
    # rpn
    parser.add_argument('--gpus', help='GPU device to train with',
                        default='0', type=str)
    parser.add_argument('--pretrained', help='pretrained model prefix',
                        default=os.path.join('model', 'vgg16'), type=str)
    parser.add_argument('--epoch', help='epoch of pretrained model',
                        default=1, type=int)
    parser.add_argument('--prefix', help='new model prefix',
                        default=os.path.join('model', 'rpn'), type=str)
    parser.add_argument('--begin_epoch', help='begin epoch of training',
                        default=0, type=int)
    parser.add_argument('--end_epoch', help='end epoch of training',
                        default=8, type=int)
    parser.add_argument('--finetune', help='second round finetune', action='store_true')
    parser.add_argument('--lr', help='base learning rate', default=0.001, type=float)
    parser.add_argument('--lr_step', help='learning rate step', default=60000, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    train_rpn(args, ctx, args.pretrained, args.epoch, args.prefix, args.begin_epoch, args.end_epoch,
              finetune=args.finetune, lr=args.lr, lr_step=args.lr_step)
