#!/usr/bin/env python
import argparse
import models
import dataset
import sys
import numpy as np
import tensorflow as tf
import os.path as osp
from glob import glob
sys.path.append('../')
sys.path.append('../../../')
from ops.utils import print_prob
from exp_ops.helper_functions import *
from exp_ops.resnet_utils import *

absolute_home = '/home/drew/Documents/tensorflow-vgg' #need to figure out a better system
syn_file = absolute_home + '/data/ilsvrc_2012/synset_names.txt'
full_syn = absolute_home + '/data/ilsvrc_2012/synset.txt'

im_ext = '.JPEG'
im_size = [224,224]
batch_size = 25
resnet_type = 152

model_data_path = '/home/drew/Documents/caffe-tensorflow/resnet_conversions/resnet_' + str(resnet_type) + '_data.npy'
test_im_dir = '/home/drew/Downloads/p2p_MIRCs/imgs/all_validation'

#Prepare network
net, spec = interpret_resnet(resnet_type)

#Images
_,_,test_names = prepare_testing_images(test_im_dir,im_size,im_ext,grayscale=False,apply_preprocess=True)
syn, skeys = get_synkeys()
gt,gt_ids = get_labels(test_names,syn,skeys,syn_file)
image_paths = sorted(glob(test_im_dir + '/*' + im_ext)) 

# Create a placeholder for the input image
input_node = tf.placeholder(tf.float32,
                                shape=(None, spec.crop_size, spec.crop_size, spec.channels))

# Construct the network
net = net({'data': input_node})

# Create an image producer (loads and processes images in parallel)
image_producer = dataset.ImageProducer(image_paths=image_paths, data_spec=spec, batch_size=len(image_paths))

#Work through all batches
with tf.Session() as sesh:
    # Start the image processing workers
    coordinator = tf.train.Coordinator()
    threads = image_producer.start(session=sesh, coordinator=coordinator)

    # Load the converted parameters
    print('Loading the model')
    net.load(model_data_path, sesh)

    # Load the input image
    print('Loading the images')
    indices, input_images = image_producer.get(sesh)

    # Perform a forward pass through the network to get the class probabilities
    print('Classifying')
    prob = sesh.run(net.get_output(), feed_dict={input_node: input_images})

    # Stop the worker threads
    coordinator.request_stop()
    coordinator.join(threads, stop_grace_period_secs=2)

sorted_indices = np.argsort(indices)
prob = prob[sorted_indices,:]
class_accuracy, t1_preds, t5_preds, t1_true_acc, t5_true_acc = evaluate_model(gt,gt_ids,prob,test_names,im_ext,full_syn)

