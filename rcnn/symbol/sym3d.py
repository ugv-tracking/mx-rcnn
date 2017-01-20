from __future__ import division
import os
# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
os.environ["MXNET_CPU_WORKER_NTHREADS"] = "4"
import mxnet as mx
import numpy as np
import math

class AngleOutput(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        
        self.assign(out_data[0], req[0], in_data[0])
        
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        num_bin = 4
        overlap = math.pi/2
        #logging.info('backward')
        
        lbl = in_data[1].asnumpy()
        y = np.zeros_like(in_data[0].asnumpy())
        #for i in xrange(num_bin):
            #y[:,i] = np.arctan2(y[:,i*2+1],y[:,i*2])
        
        #y = y[:,:num_bin] + math.pi
        bins_angle = np.linspace(0, 2*math.pi, num = num_bin, endpoint = False)
        bins_angle = np.vstack((bins_angle,bins_angle)).reshape(-1, order='F')
        #y = y.reshape(-1)[:, np.newaxis] - bins_angle

        dist = np.abs(lbl[:, np.newaxis] - bins_angle)

        cover_bins = np.zeros_like(dist)
        cover_bins[dist < overlap] = 1
        
        for i in xrange(2*num_bin):
            if i%2 == 0:
                y[:,i] = -np.cos(lbl - bins_angle[i])
            else:
                y[:,i] = -np.sin(lbl - bins_angle[i])

        y *= cover_bins/cover_bins.sum(axis=1)[:, np.newaxis]
        
        self.assign(in_grad[0], req[0], mx.nd.array(y))

        
@mx.operator.register("angle")
class AngleProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(AngleProp, self).__init__(need_top_grad=False)
    
    def list_arguments(self):
        return ['data', 'label'] 

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = (in_shape[0][0],)
        output_shape = in_shape[0]
        return [data_shape, label_shape], [output_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return AngleOutput()

def get_symbol(num_classes, **kwargs):
    
    dims_label = mx.symbol.Variable('dims_label')
    angle_label = mx.symbol.Variable('angle_label')
    softmax_label = mx.symbol.Variable('softmax_label')
    data = mx.symbol.Variable('data')
    # group 1
    conv1_1 = mx.symbol.Convolution(data=data, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_1")
    relu1_1 = mx.symbol.Activation(data=conv1_1, act_type="relu", name="relu1_1")
    pool1 = mx.symbol.Pooling(
        data=relu1_1, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool1")
    # group 2
    conv2_1 = mx.symbol.Convolution(
        data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_1")
    relu2_1 = mx.symbol.Activation(data=conv2_1, act_type="relu", name="relu2_1")
    pool2 = mx.symbol.Pooling(
        data=relu2_1, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool2")
    # group 3
    conv3_1 = mx.symbol.Convolution(
        data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_1")
    relu3_1 = mx.symbol.Activation(data=conv3_1, act_type="relu", name="relu3_1")
    conv3_2 = mx.symbol.Convolution(
        data=relu3_1, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_2")
    relu3_2 = mx.symbol.Activation(data=conv3_2, act_type="relu", name="relu3_2")
    pool3 = mx.symbol.Pooling(
        data=relu3_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool3")
    # group 4
    conv4_1 = mx.symbol.Convolution(
        data=pool3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_1")
    relu4_1 = mx.symbol.Activation(data=conv4_1, act_type="relu", name="relu4_1")
    conv4_2 = mx.symbol.Convolution(
        data=relu4_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_2")
    relu4_2 = mx.symbol.Activation(data=conv4_2, act_type="relu", name="relu4_2")
    pool4 = mx.symbol.Pooling(
        data=relu4_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool4")
    # group 5
    conv5_1 = mx.symbol.Convolution(
        data=pool4, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_1")
    relu5_1 = mx.symbol.Activation(data=conv5_1, act_type="relu", name="relu5_1")
    conv5_2 = mx.symbol.Convolution(
        data=relu5_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_2")
    relu5_2 = mx.symbol.Activation(data=conv5_2, act_type="relu", name="conv1_2")
    pool5 = mx.symbol.Pooling(
        data=relu5_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool5")
    
    #3d bbox module

    flatten = mx.symbol.Flatten(data=pool5, name="flatten")

    #dim branch
    fc6_dim = mx.symbol.FullyConnected(data=flatten, num_hidden=512, name="fc6_dim")
    relu6_dim = mx.symbol.Activation(data=fc6_dim, act_type="relu", name="relu6_dim")
    drop6_dim = mx.symbol.Dropout(data=relu6_dim, p=0.5, name="drop6_dim")
    
    fc7_dim = mx.symbol.FullyConnected(data=drop6_dim, num_hidden=512, name="fc7_dim")
    relu7_dim = mx.symbol.Activation(data=fc7_dim, act_type="relu", name="relu7_dim")
    drop7_dim = mx.symbol.Dropout(data=relu7_dim, p=0.5, name="drop7_dim")
    
    fc8_dim = mx.symbol.FullyConnected(data=drop7_dim, num_hidden=3, name="fc8_dim")
    loss_dim = mx.symbol.LinearRegressionOutput(data = fc8_dim, label = dims_label, name='dims')

    #angle branch

    num_bin = 4

    fc6_angle = mx.symbol.FullyConnected(data=flatten, num_hidden=256, name="fc6_angle")
    relu6_angle = mx.symbol.Activation(data=fc6_angle, act_type="relu", name="relu6_angle")
    drop6_angle = mx.symbol.Dropout(data=relu6_angle, p=0.5, name="drop6_angle")

    fc7_angle = mx.symbol.FullyConnected(data=drop6_angle, num_hidden=256, name="fc7_angle")
    relu7_angle = mx.symbol.Activation(data=fc7_angle, act_type="relu", name="relu7_angle")
    drop7_angle = mx.symbol.Dropout(data=relu7_angle, p=0.5, name="drop7_angle")

    fc8_angle = mx.symbol.FullyConnected(data=drop7_angle, num_hidden=num_bin*2, name="fc8_angle")
    
    fc8_angle_reshape = mx.symbol.Reshape(data=fc8_angle, shape=(-1, num_bin, 2), name='fc8_angle_reshape')
    L2_norm = mxnet.symbol.L2Normalization(data=fc8_angle_reshape, mode='spatial', name='L2_norm')
    angle_flatten = mx.symbol.Reshape(data=L2_norm, shape=(-1, num_bin*2), name='angle_flatten')

    loss_angle = mx.symbol.Custom(data=angle_flatten, label=angle_label, name='angle', op_type='angle')
    

    #confidence branch

    fc6_conf = mx.symbol.FullyConnected(data=flatten, num_hidden=256, name="fc6_conf")
    relu6_conf = mx.symbol.Activation(data=fc6_conf, act_type="relu", name="relu6_conf")
    drop6_conf = mx.symbol.Dropout(data=relu6_conf, p=0.5, name="drop6_conf")

    fc7_conf = mx.symbol.FullyConnected(data=drop6_conf, num_hidden=256, name="fc7_conf")
    relu7_conf = mx.symbol.Activation(data=fc7_conf, act_type="relu", name="relu7_conf")
    drop7_conf = mx.symbol.Dropout(data=relu7_conf, p=0.5, name="drop7_conf")

    fc8_conf = mx.symbol.FullyConnected(data=drop7_conf, num_hidden=num_bin, name="fc8_conf")
    loss_softmax = mx.symbol.SoftmaxOutput(data=fc8_conf, label=softmax_label , name='softmax')
    

    net = mx.symbol.Group([loss_dim, loss_angle, loss_softmax])
    return net
