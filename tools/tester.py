import cPickle
import os
import time
import mxnet as mx
import numpy as np

from rcnn.config import config
from rcnn.processing import image_processing
from rcnn.processing.bbox_transform import bbox_pred, clip_boxes
from rcnn.processing.nms import nms
from module import MutableModule


class Predictor(object):
    def __init__(self, symbol, data_names, label_names,
                 context=mx.cpu(), max_data_shapes=None,
                 test_data=None, arg_params=None, aux_params=None):
        self._mod = MutableModule(symbol, data_names, label_names,
                                  context=context, max_data_shapes=max_data_shapes)

        print '=========================='
        print 'max_data_shapes: ', max_data_shapes
        print 'test_data.provide_data: ', test_data.provide_data
        print 'test_data.provide_label: ', test_data.provide_label

        self._mod.bind(test_data.provide_data, test_data.provide_label, for_training=False)
        self._mod.init_params(arg_params=arg_params, aux_params=aux_params)

    def predict(self, data_batch):
        self._mod.forward(data_batch)
        return dict(zip(self._mod.output_names, self._mod.get_outputs()))


def generate_proposals(predictor, test_data, imdb, vis=False, thresh=0):
    """
    Generate detections results using RPN.
    :param predictor: Predictor
    :param test_data: data iterator, must be non-shuffled
    :param imdb: image database
    :param vis: controls visualization
    :param thresh: thresh for valid detections
    :return: list of detected boxes
    """
    assert not test_data.shuffle
    data_names = [k[0] for k in test_data.provide_data]

    i = 0
    imdb_boxes = list()
    for im_info, data_batch in test_data:
        if i % 10 == 0:
            print 'generating detections {}/{}'.format(i, imdb.num_images)

        output = predictor.predict(data_batch)
        # drop the batch index
        boxes = output['rois_output'].asnumpy()[:, 1:]
        scores = output['rois_score'].asnumpy()

        data_dict = dict(zip(data_names, data_batch.data))
        # transform to original scale
        scale = im_info[0, 2]
        boxes = boxes / scale
        keep = np.where(scores > thresh)[0]
        imdb_boxes.append(boxes[keep, :])

        if vis:
            dets = [np.hstack((boxes[keep, :] * scale, scores[keep, :]))]
            vis_all_detection(data_dict['data'].asnumpy(), dets, ['obj'], thresh)
        i += 1

    assert len(imdb_boxes) == imdb.num_images, 'calculations not complete'
    rpn_folder = os.path.join(imdb.root_path, 'rpn_data')
    if not os.path.exists(rpn_folder):
        os.mkdir(rpn_folder)
    rpn_file = os.path.join(rpn_folder, imdb.name + '_rpn.pkl')
    with open(rpn_file, 'wb') as f:
        cPickle.dump(imdb_boxes, f, cPickle.HIGHEST_PROTOCOL)
    print 'wrote rpn proposals to {}'.format(rpn_file)
    return imdb_boxes


def im_detect(predictor, data_batch, data_names):
    output = predictor.predict(data_batch)

    data_dict = dict(zip(data_names, data_batch.data))
    if config.TEST.HAS_RPN:
        rois = output['rois_output'].asnumpy()[:, 1:]
    else:
        rois = data_dict['rois'].asnumpy()[:, 1:]
    im_shape = data_dict['data'].shape

    # save output
    scores = output['cls_prob_reshape_output'].asnumpy()[0]
    bbox_deltas = output['bbox_pred_reshape_output'].asnumpy()[0]

    orientation_ry = np.zeros((scores.shape[0], scores.shape[1]), dtype=np.float32)
    orientation_alpha = np.zeros((scores.shape[0], scores.shape[1]), dtype=np.float32)

    if config.TRAIN.ORIENTATION:
        #orientation_ry = output['orientation_ry_prob_reshape_output'].asnumpy()[0]
        #orientation_alpha = output['orientation_alpha_prob_reshape_output'].asnumpy()[0]

        orientation_ry = output['orientation_ry_pred_reshape_output'].asnumpy()[0]
        orientation_alpha = output['orientation_alpha_pred_reshape_output'].asnumpy()[0]
    # post processing
    pred_boxes = bbox_pred(rois, bbox_deltas)
    pred_boxes = clip_boxes(pred_boxes, im_shape[-2:])

    return scores, pred_boxes, data_dict, orientation_ry, orientation_alpha


def pred_eval(predictor, test_data, imdb, vis=False):
    """
    wrapper for calculating offline validation for faster data analysis
    in this example, all threshold are set by hand
    :param predictor: Predictor
    :param test_data: data iterator, must be non-shuffle
    :param imdb: image database
    :param vis: controls visualization
    :return:
    """
    #vis = True
    assert not test_data.shuffle
    data_names = [k[0] for k in test_data.provide_data]

    thresh = 0.05
    # limit detections to max_per_image over all classes
    max_per_image = 100

    num_images = imdb.num_images
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]
    all_orientation_ry = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]
    all_orientation_alpha = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    margin_ry = 2 * config.PI / config.RY_CLASSES 
    ori_bin = np.zeros(config.RY_CLASSES)
    for bin_id in range(0, config.RY_CLASSES):
        ori_bin[bin_id] = bin_id * margin_ry + margin_ry * 0.5 - config.PI

    i = 0
    t = time.time()
    for im_info, data_batch in test_data:
        t1 = time.time() - t
        t = time.time()
        

        scores, boxes, data_dict, orientation_ry, orientation_alpha = im_detect(predictor, data_batch, data_names)
        # we used scaled image & roi to train, so it is necessary to transform them back
        # however, visualization will be scaled
        scale = im_info[0, 2]

        t2 = time.time() - t
        t = time.time()

        for j in range(0, imdb.num_classes):
            indexes = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[indexes, j]
            cls_boxes = boxes[indexes, j * 4:(j + 1) * 4] / scale
            cls_orientation_ry    = orientation_ry[indexes, :] 
            cls_orientation_alpha = orientation_alpha[indexes, :] 


            is_orientation_cls = False

            if is_orientation_cls:
                assert 1 == 0
                cls_orientation_ry_pred_label    = cls_orientation_ry.argmax(axis=1).astype('int32')
                cls_orientation_alpha_pred_label = cls_orientation_alpha.argmax(axis=1).astype('int32')

                is_aver = True
                if is_aver: 
                    cls_orientation_ry_aver    = (cls_orientation_ry_pred_label - 1) * margin_ry + margin_ry * 0.5 - config.PI
                    cls_orientation_alpha_aver = (cls_orientation_alpha_pred_label - 1) * margin_ry + margin_ry * 0.5 - config.PI

                    for obj_id in range(0, len(cls_orientation_ry_pred_label)):
                        cls_orientation_ry_aver[obj_id] = 0
                        cls_orientation_alpha_aver[obj_id] = 0

                        for cls_id in range(len(ori_bin)):
                            cls_orientation_ry_aver[obj_id]    = cls_orientation_ry_aver[obj_id]    + ori_bin[cls_id] * cls_orientation_ry[obj_id, cls_id]
                            cls_orientation_alpha_aver[obj_id] = cls_orientation_alpha_aver[obj_id] + ori_bin[cls_id] * cls_orientation_alpha[obj_id, cls_id]

                    cls_orientation_ry    = cls_orientation_ry_aver
                    cls_orientation_alpha = cls_orientation_alpha_aver
                else:
                    cls_orientation_ry    = (cls_orientation_ry_pred_label - 1) * margin_ry + margin_ry * 0.5 - config.PI
                    cls_orientation_alpha = (cls_orientation_alpha_pred_label - 1) * margin_ry + margin_ry * 0.5 - config.PI
            else:
                cls_orientation_ry    = cls_orientation_ry[:, 0] 
                cls_orientation_alpha = cls_orientation_alpha[:, 0] 


            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis]))
            keep = nms(cls_dets, config.TEST.NMS)
            all_boxes[j][i] = cls_dets[keep, :]
            all_orientation_ry[j][i] = cls_orientation_ry[keep]
            all_orientation_alpha[j][i] = cls_orientation_alpha[keep]

        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in range(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
                    all_orientation_ry[j][i]    = all_orientation_ry[j][i][keep]
                    all_orientation_alpha[j][i] = all_orientation_alpha[j][i][keep]

        boxes_this_image = [[]] + [all_boxes[j][i] for j in range(1, imdb.num_classes)]
        orientation_ry_this_image    = [[]] + [all_orientation_ry[j][i] for j in range(1, imdb.num_classes)]
        orientation_alpha_this_image = [[]] + [all_orientation_alpha[j][i] for j in range(1, imdb.num_classes)]

        if vis:
            # visualize the testing scale
            for box in boxes_this_image:
                if isinstance(box, np.ndarray):
                    box[:, :4] *= scale
            vis_all_detection(data_dict['data'].asnumpy(), boxes_this_image, orientation_ry_this_image, orientation_alpha_this_image,
                              class_names=imdb.classes)

        t3 = time.time() - t
        t = time.time()
        print 'testing {}/{} data {:.4f}s net {:.4f}s post {:.4f}s'.format(i, imdb.num_images, t1, t2, t3)
        i += 1

    cache_folder = os.path.join(imdb.cache_path, imdb.name)
    if not os.path.exists(cache_folder):
        os.mkdir(cache_folder)

    det_file = os.path.join(cache_folder, 'detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f)

    orientation_ry_file = os.path.join(cache_folder, 'orientation_ry.pkl')
    with open(orientation_ry_file, 'wb') as f:
        cPickle.dump(all_orientation_ry, f)

    orientation_alpha_file = os.path.join(cache_folder, 'orientation_alpha.pkl')
    with open(orientation_alpha_file, 'wb') as f:
        cPickle.dump(all_orientation_alpha, f)

    imdb.evaluate_detections(all_boxes, all_orientation_ry, all_orientation_alpha)


def vis_all_detection(im_array, detections, orientation_ry, orientation_alpha , class_names=None, thresh=0.7):
    """
    visualize all detections in one image
    :param im_array: [b=1 c h w] in rgb
    :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
    :param class_names: list of names in imdb
    :param thresh: threshold for valid detections
    :return:
    """
    import matplotlib.pyplot as plt
    import random
    im = image_processing.transform_inverse(im_array, config.PIXEL_MEANS)
    plt.imshow(im)
    for j, name in enumerate(class_names):
        if name == '__background__':
            continue
        color = (random.random(), random.random(), random.random())  # generate a random color
        dets = detections[j]
        oris_ry    = orientation_ry[j]
        oris_alpha = orientation_alpha[j]

        for i in range(dets.shape[0]):
            bbox = dets[i, :4]
            score = dets[i, -1]
            if score > thresh:
                rect = plt.Rectangle((bbox[0], bbox[1]),
                                     bbox[2] - bbox[0],
                                     bbox[3] - bbox[1], fill=False,
                                     edgecolor=color, linewidth=3.5)
                plt.gca().add_patch(rect)
                plt.gca().text(bbox[0], bbox[1] - 2,
                               '{:s} {:.3f}'.format(name, score),
                               bbox=dict(facecolor=color, alpha=0.5), fontsize=12, color='white')

                plt.gca().text(bbox[0], bbox[3] - 2,
                               '{:s} {:.3f}'.format('Ori', oris_ry[i]),
                               bbox=dict(facecolor=color, alpha=0.5), fontsize=12, color='white')
    plt.show()
