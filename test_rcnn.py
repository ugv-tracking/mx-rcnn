import argparse
import os
import mxnet as mx

from rcnn.config import config
from rcnn.symbol import *
from rcnn.dataset import *
from rcnn.core.loader import TestLoader
from rcnn.core.tester import Predictor, pred_eval
from rcnn.utils.load_model import load_param


def test_rcnn(args, ctx, prefix, epoch,
              vis=False, shuffle=False, has_rpn=True, proposal='rpn'):
    # load symbol and testing data
    if has_rpn:
        config.TEST.HAS_RPN = True
        config.TEST.RPN_PRE_NMS_TOP_N = 6000
        config.TEST.RPN_POST_NMS_TOP_N = 300
        #sym = eval('get_' + args.network + '_test_ry_alpha_cls')()
        sym = eval('get_' + args.network + '_test_ry_alpha_regression')()
        imdb = eval(args.dataset)(args.image_set, args.root_path, args.dataset_path)
        roidb = imdb.gt_roidb()

    else:
        sym = eval('get_' + args.network + '_rcnn_test')()
        imdb = eval(args.dataset)(args.image_set, args.root_path, args.dataset_path)
        gt_roidb = imdb.gt_roidb()
        roidb = eval('imdb.' + proposal + '_roidb')(gt_roidb)

    # get test data iter
    test_data = TestLoader(roidb, batch_size=1, shuffle=shuffle, has_rpn=has_rpn)

    # load model
    arg_params, aux_params = load_param(prefix, epoch, convert=True, ctx=ctx, process=True)

    # check parameters
    param_names = [k for k in sym.list_arguments() + sym.list_auxiliary_states()
                   if k not in dict(test_data.provide_data) and 'label' not in k]
    missing_names = [k for k in param_names if k not in arg_params and k not in aux_params]
    if len(missing_names):
        print 'detected missing params', missing_names

    # decide maximum shape
    data_names = [k[0] for k in test_data.provide_data]
    label_names = ['cls_prob_label', 'orientation_ry_prob_label', 'orientation_alpha_prob_label']
    max_data_shape = [('data', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]
    print 'data_names: ', data_names
    print 'label_names: ', label_names
    print 'max_data_shape: ', max_data_shape

    predictor = Predictor(sym, data_names, label_names,
                          context=ctx, max_data_shapes=max_data_shape,
                          test_data=test_data, arg_params=arg_params, aux_params=aux_params)

    # start detection
    pred_eval(predictor, test_data, imdb, vis=vis)


def parse_args():
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    # general
    parser.add_argument('--network', help='network name',
                        default='vgg', type=str)
    parser.add_argument('--dataset', help='dataset name',
                        default='PascalVOC', type=str)
    parser.add_argument('--image_set', help='image_set name',
                        default='2007_test', type=str)
    parser.add_argument('--root_path', help='output data folder',
                        default='data', type=str)
    parser.add_argument('--dataset_path', help='dataset path',
                        default=os.path.join('data', 'VOCdevkit'), type=str)
    # testing
    parser.add_argument('--prefix', help='model to test with', type=str)
    parser.add_argument('--epoch', help='model to test with', type=int)
    parser.add_argument('--gpu', help='GPU device to test with', type=int)
    # rcnn
    parser.add_argument('--vis', dest='vis', help='turn on visualization', action='store_true')
    parser.add_argument('--shuffle', help='shuffle data on visualization', action='store_true')
    parser.add_argument('--has_rpn', help='generate proposals on the fly',
                        action='store_true')
    parser.add_argument('--proposal', dest='proposal', help='can be ss for selective search or rpn',
                        default='rpn', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    ctx = mx.gpu(args.gpu)
    test_rcnn(args, ctx, args.prefix, args.epoch,
              vis=args.vis, shuffle=args.shuffle, has_rpn=args.has_rpn, proposal=args.proposal)
