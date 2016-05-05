#!/usr/bin/env python

from __future__ import division

import argparse
import numpy as np
import os
import tempfile
import time
import json
import random

parser = argparse.ArgumentParser(
    description='Train and evaluate a net on the product images dataset.')
parser.add_argument('--image_root', default='./images/',
    help='Directory where images are stored')
parser.add_argument('--crop', type=int, default=36,
    help=('The edge length of the random image crops'
          '(defaults to 96 for 96x96 crops)'))
parser.add_argument('--disp', type=int, default=10,
    help='Print loss/accuracy every --disp training iterations')
parser.add_argument('--snapshot_dir', default='./snapshot',
    help='Path to directory where snapshots are saved')
parser.add_argument('--snapshot_prefix', default='place_net',
    help='Snapshot filename prefix')
parser.add_argument('--iters', type=int, default= 10000,
    help='Total number of iterations to train the network')
parser.add_argument('--batch', type=int, default=100,
    help='The batch size to use for training')
parser.add_argument('--iter_size', type=int, default=2,
    help=('The number of iterations (batches) over which to average the '
          'gradient computation. Effectively increases the batch size '
          '(--batch) by this factor, but without increasing memory use '))
parser.add_argument('--lr', type=float, default=0.000001,
    help='The initial learning rate')
parser.add_argument('--gamma', type=float, default=0.01,
    help='Factor by which to drop the learning rate')
parser.add_argument('--stepsize', type=int, default=500,
    help='Drop the learning rate every N iters -- this specifies N')
parser.add_argument('--momentum', type=float, default=0.7,
    help='The momentum hyperparameter to use for momentum SGD')
parser.add_argument('--decay', type=float, default=5e-4,
    help='The L2 weight decay coefficient')
parser.add_argument('--seed', type=int, default=1,
    help='Seed for the random number generator')
parser.add_argument('--cudnn', action='store_true',
    help='Use CuDNN at training time -- usually faster, but non-deterministic')
parser.add_argument('--gpu', type=int, default=0,
    help='GPU ID to use for training and inference (-1 for CPU)')
args = parser.parse_args()

# disable most Caffe logging (unless env var $GLOG_minloglevel is already set)
key = 'GLOG_minloglevel'
if not os.environ.get(key, ''):
    os.environ[key] = '3'

import caffe
from caffe.proto import caffe_pb2
from caffe import layers as L
from caffe import params as P

if args.gpu >= 0:
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu)
else:
    caffe.set_mode_cpu()

def to_tempfile(file_content):
    """Serialize a Python protobuf object str(proto), dump to a temporary file,
       and return its filename."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(file_content)
        return f.name

weight_param = dict(lr_mult=1, decay_mult=1)
bias_param   = dict(lr_mult=2, decay_mult=0)
learned_param = [weight_param, bias_param]
frozen_param = [dict(lr_mult=0)] * 2

zero_filler     = dict(type='constant', value=0)
msra_filler     = dict(type='msra')
uniform_filler  = dict(type='uniform', min=-0.1, max=0.1)
fc_filler       = dict(type='gaussian', std=0.005)
# Original AlexNet used the following commented out Gaussian initialization;
# we'll use the "MSRA" one instead, which scales the Gaussian initialization
# of a convolutional filter based on its receptive field size.
# conv_filler     = dict(type='gaussian', std=0.01)
conv_filler     = dict(type='msra')

def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1,
              param=learned_param,
              weight_filler=conv_filler, bias_filler=zero_filler,
              train=False):
    # set CAFFE engine to avoid CuDNN convolution -- non-deterministic results
    engine = {}
    if train and not args.cudnn:
        engine.update(engine=P.Pooling.CAFFE)
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad, group=group, param=param,
                         weight_filler=weight_filler, bias_filler=bias_filler,
                         **engine)
    return conv, L.ReLU(conv, in_place=True)

def batch_norm(bottom):
    bn = L.LRN(bottom, local_size=3, alpha=5e-5, beta=.75, norm_region=1)
    return bn

def fc_relu(bottom, nout, param=learned_param,
            weight_filler=fc_filler, bias_filler=zero_filler):
    fc = L.InnerProduct(bottom, num_output=nout, param=param,
                        weight_filler=weight_filler, bias_filler=bias_filler)
    return fc, L.ReLU(fc, in_place=True)

def max_pool(bottom, ks, stride=1, train=False):
    # set CAFFE engine to avoid CuDNN pooling -- non-deterministic results
    engine = {}
    if train and not args.cudnn:
        engine.update(engine=P.Pooling.CAFFE)
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride,
                     **engine)

def minialexnet(data, labels=None, train=False, param=learned_param,
                num_classes=1, with_labels=True):
    """
    Returns a protobuf text file specifying a variant of AlexNet, following the
    original specification (<caffe>/models/bvlc_alexnet/train_val.prototxt).
    The changes with respect to the original AlexNet are:
        - LRN (local response normalization) layers are not included
        - The Fully Connected (FC) layers (fc6 and fc7) have smaller dimensions
          due to the lower resolution of mini-places images (128x128) compared
          with ImageNet images (usually resized to 256x256)
    """
    n = caffe.NetSpec()
    n.data = data
    conv_kwargs = dict(param=param, train=train)
    n.conv1, n.relu1 = conv_relu(n.data, 12, 96, pad=4, **conv_kwargs)
    n.conv2, n.relu2 = conv_relu(n.relu1, 9, 96, pad=3, **conv_kwargs)
    n.pool2 = max_pool(n.relu2, 2, stride=2, train=train)
    n.bn2 = batch_norm(n.pool2)
    n.conv3, n.relu3 = conv_relu(n.bn2, 6, 192, pad=2, group = 2, **conv_kwargs)
    n.pool4 = max_pool(n.relu3, 2, stride=2, train=train)
    n.bn4 = batch_norm(n.pool4)
    n.conv4, n.relu4 = conv_relu(n.bn4, 3, 192, pad=1, group = 2, **conv_kwargs)
    n.pool5 = max_pool(n.relu4, 3, stride=2, train=train)
    n.pool6 = max_pool(n.pool5, 3, stride=1, train=train)
    n.bn6 = batch_norm(n.pool6)
    n.fc6, n.relu6 = fc_relu(n.bn6, 1024, param=param)
    n.drop6 = L.Dropout(n.relu6, in_place=True)
    n.fc7, n.relu7 = fc_relu(n.drop6, 1024, param=param)
    n.drop7 = L.Dropout(n.relu7, in_place=True)
    n.pred = L.InnerProduct(n.drop7, num_output=1, param=param)

    if train:
        n.label = labels
        n.loss = L.EuclideanLoss(n.pred, n.label)

    return to_tempfile(str(n.to_proto()))

def get_split(split):
    filename = './%s.txt' % split
    if not os.path.exists(filename):
        raise IOError('Split data file not found: %s' % split)
    return filename

def miniplaces_net(source, train=False, with_labels=True):
    mean = [104, 117, 123]  # per-channel mean of the BGR image pixels
    #mean = [int(255*random.random()),int(255*random.random()),int(255*random.random())]
    transform_param = dict(mirror=train, crop_size=args.crop, mean_value=mean)
    batch_size = args.batch if train else 100
    places_data, places_labels = L.ImageData(transform_param=transform_param,
        source=source, root_folder=args.image_root, shuffle=train,
        batch_size=batch_size, ntop=2)
    return minialexnet(data=places_data, labels=places_labels, train=train,
                       with_labels=with_labels)

def snapshot_prefix():
    return os.path.join(args.snapshot_dir, args.snapshot_prefix)

def snapshot_at_iteration(iteration):
    return '%s_iter_%d.caffemodel' % (snapshot_prefix(), iteration)

def miniplaces_solver(train_net_path, test_net_path=None):
    s = caffe_pb2.SolverParameter()

    # Specify locations of the train and (maybe) test networks.
    s.train_net = train_net_path
    if test_net_path is not None:
        s.test_net.append(test_net_path)
        # Test after every 1000 training iterations.
        s.test_interval = 1000
        # Set `test_iter` to test on 100 batches each time we test.
        # With test batch size 100, this covers the entire validation set of
        # 10K images (100 * 100 = 10K).
        s.test_iter.append(100)
    else:
        s.test_interval = args.iters + 1  # don't test during training

    # The number of batches over which to average the gradient.
    # Effectively boosts the training batch size by the given factor, without
    # affecting memory utilization.
    s.iter_size = args.iter_size

    # Solve using the stochastic gradient descent (SGD) algorithm.
    # Other choices include 'Adam' and 'RMSProp'.
    s.type = 'SGD'

    # The following settings (base_lr, lr_policy, gamma, stepsize, and max_iter),
    # define the following learning rate schedule:
    #   Iterations [  0, 20K) -> learning rate 0.01   = base_lr
    #   Iterations [20K, 40K) -> learning rate 0.001  = base_lr * gamma
    #   Iterations [40K, 50K) -> learning rate 0.0001 = base_lr * gamma^2

    # Set the initial learning rate for SGD.
    s.base_lr = args.lr

    # Set `lr_policy` to define how the learning rate changes during training.
    # Here, we 'step' the learning rate by multiplying it by a factor `gamma`
    # every `stepsize` iterations.
    s.lr_policy = 'step'
    s.gamma = args.gamma
    s.stepsize = args.stepsize

    # `max_iter` is the number of times to update the net (training iterations).
    s.max_iter = args.iters

    # Set other SGD hyperparameters. Setting a non-zero `momentum` takes a
    # weighted average of the current gradient and previous gradients to make
    # learning more stable. L2 weight decay regularizes learning, to help
    # prevent the model from overfitting.
    s.momentum = args.momentum
    s.weight_decay = args.decay

    # Display the current training loss and accuracy every `display` iterations.
    # This doesn't have an effect for Python training here as logging is
    # disabled by this script (see the GLOG_minloglevel setting).
    s.display = args.disp

    # Number of training iterations over which to smooth the displayed loss.
    # The summed loss value (Iteration N, loss = X) will be averaged,
    # but individual loss values (Train net output #K: my_loss = X) won't be.
    s.average_loss = 10

    # Seed the RNG for deterministic results.
    # (May not be so deterministic if using CuDNN.)
    s.random_seed = args.seed

    # Snapshots are files used to store networks we've trained.  Here, we'll
    # snapshot twice per learning rate step to the location specified by the
    # --snapshot_dir and --snapshot_prefix args.
    s.snapshot = args.stepsize // 2
    s.snapshot_prefix = snapshot_prefix()

    # Create snapshot dir if it doesn't already exist.
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    return to_tempfile(str(s))

def train_net(with_val_net=False):
    train_net_file = miniplaces_net(get_split('train'), train=True)
    # Set with_val_net=True to test during training.
    # Environment variable GLOG_minloglevel should be set to 0 to display
    # Caffe output in this case; otherwise, the test result will not be
    # displayed.
    if with_val_net:
        val_net_file = miniplaces_net(get_split('val'), train=False)
    else:
        val_net_file = None
    solver_file = miniplaces_solver(train_net_file, val_net_file)
    solver = caffe.get_solver(solver_file)
    outputs = sorted(solver.net.outputs)
    def str_output(output):
        value = solver.net.blobs[output].data
        if output.startswith('accuracy'):
            valstr = '%5.2f%%' % (100 * value, )
        else:
            valstr = '%6f' % value
        return '%s = %s' % (output, valstr)
    def disp_outputs(iteration, iter_pad_len=len(str(args.iters))):
        metrics = '; '.join(str_output(o) for o in outputs)
        return 'Iteration %*d: %s' % (iter_pad_len, iteration, metrics)
    # We could just call `solver.solve()` rather than `step()`ing in a loop.
    # (If we hadn't set GLOG_minloglevel = 3 at the top of this file, Caffe
    # would display loss/accuracy information during training.)
    previous_time = None
    for iteration in xrange(args.iters):
        solver.step(1)
        if (args.disp > 0) and (iteration % args.disp == 0):
            current_time = time.clock()
            if previous_time is None:
                benchmark = ''
            else:
                time_per_iter = (current_time - previous_time) / args.disp
                benchmark = ' (%5f s/it)' % time_per_iter
            previous_time = current_time
            print disp_outputs(iteration), benchmark
    # Print accuracy for last iteration.
    solver.net.forward()
    disp_outputs(args.iters)
    solver.net.save(snapshot_at_iteration(args.iters))

def eval_net(split):
    print 'Running evaluation for split:', split
    filenames = []
    labels = []
    split_file = get_split(split)
    with open(split_file, 'r') as f:
        for line in f.readlines():
            parts = line.split()
            assert 1 <= len(parts) <= 2, 'malformed line'
            filenames.append(parts[0])
            if len(parts) > 1:
                labels.append(float(parts[1]))
    known_labels = (len(labels) > 0)
    if known_labels:
        assert len(labels) == len(filenames)
    else:
        # create file with 'dummy' labels (all 0s)
        split_file = to_tempfile(''.join('%s 0\n' % name for name in filenames))
    test_net_file = miniplaces_net(split_file, train=False, with_labels=True)
    weights_file = snapshot_at_iteration(args.iters)
    net = caffe.Net(test_net_file, weights_file, caffe.TEST)
    price_predictions = np.zeros(len(filenames), dtype = np.float32)

    offset = 0
    while offset < len(filenames):
        batch_pred = net.forward()['pred']
        for pred in batch_pred:
            price_predictions[offset] = pred
            offset += 1
            if offset >= len(filenames):
                break

    price_diff = price_predictions - labels
    mean_price_diff = np.mean(price_diff)
    rms_price_diff = np.sqrt(np.mean( (price_diff)**2 ))

    indexed_price_diff = [ (i, labels[i], price_diff[i]) for i in range(len(filenames)) ]
    sorted_price_diff = sorted(indexed_price_diff, key = lambda x : x[2])
    


    print "Mean price difference: ",  mean_price_diff
    print "RMS price difference: ", rms_price_diff
    print "Top 10 underestimations: "
    for i, p, pdiff in sorted_price_diff[:10]:
        print "Image{:6d}".format(i), ": Original price = {0:6.2f}, Predicted price = {1:6.2f}, Error = {2:6.2f} ".format(p, p + pdiff, pdiff)
    sorted_price_diff[:10]
    print "\n"
    print "Top 10 overestimation: "
    for i, p, pdiff in sorted_price_diff[-10:]:
        print "Image{:6d}".format(i), ": Original price = {0:6.2f}, Predicted price = {1:6.2f}, Error = {2:6.2f} ".format(p, p + pdiff, pdiff)
    sorted_price_diff[:10]

    with open('./indexed_price_diff.json', 'w') as f:
        json.dump(indexed_price_diff,f)
        #f.write(str(indexed_price_diff))

if __name__ == '__main__':
    print 'Training net...\n'
    train_net()

    print '\nTraining complete. Evaluating...\n'
    #for split in ('train', 'val', 'test'):
    for split in ('train', ):
        eval_net(split)
        print
    print 'Evaluation complete.'
