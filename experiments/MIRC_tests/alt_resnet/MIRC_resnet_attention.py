#!/usr/bin/env python
import argparse
import numpy as np
import tensorflow as tf
import os.path as osp
import models
import dataset
import sys
sys.path.append('../../../')
sys.path.append('../')
from exp_ops.helper_functions import *
from ops.utils import print_prob
from glob import glob

absolute_home = '/home/drew/Documents/tensorflow-vgg' #need to figure out a better system
syn_file = absolute_home + '/data/ilsvrc_2012/synset_names.txt'
full_syn = absolute_home + '/data/ilsvrc_2012/synset.txt'
#model_data_path = '/home/drew/Documents/caffe-tensorflow/resnet_conversions/resnet_50_data.npy'
model_data_path = '/home/drew/Documents/caffe-tensorflow/resnet_conversions/resnet_102_data.npy'
#model_data_path = '/home/drew/Documents/caffe-tensorflow/resnet_conversions/resnet_152_data.npy'
test_im_dir = '/home/drew/Downloads/p2p_MIRCs/imgs/all_validation'
attention_path = ['/home/drew/Documents/MIRC_behavior/heat_map_output/pooled_p2p_alt/uniform_weight_overlap_human/heatmaps.npz',\
'/home/drew/Documents/MIRC_behavior/click_comparisons/output/labelme.npz']
im_ext = '.JPEG'
im_size = [224,224]
batch_size = 25

# Get the data specifications for the GoogleNet model
spec = models.get_data_spec(model_class=models.ResNet50)

#Images
_,_,test_names = prepare_testing_images(test_im_dir,im_size,im_ext,grayscale=False,apply_preprocess=True)
syn, skeys = get_synkeys()
gt,gt_ids = get_labels(test_names,syn,skeys,syn_file)
#gt_ids = np.asarray(gt_ids)
image_paths = sorted(glob(test_im_dir + '/*' + im_ext)) 
#image_paths = np.asarray(image_paths)
#image_paths = image_paths[gt_ids!=-1].tolist()
attention_batch = get_attention_maps(attention_path,[spec.crop_size,spec.crop_size])

# Create a placeholder for the input image
input_node = tf.placeholder(tf.float32,
                                shape=(None, spec.crop_size, spec.crop_size, spec.channels))
attention = tf.placeholder(tf.float32,shape=(None, spec.crop_size, spec.crop_size, 1))

# Construct the network
#net = models.attResNet50({'data': input_node, 'attention' : attention})
net = models.attResNet101({'data': input_node, 'attention' : attention})
#net = models.attResNet152({'data': input_node, 'attention' : attention})

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
    prob = sesh.run(net.get_output(), feed_dict={input_node: input_images, attention:attention_batch})

    # Stop the worker threads
    coordinator.request_stop()
    coordinator.join(threads, stop_grace_period_secs=2)

class_accuracy, t1_preds, t5_preds, t1_true_acc, t5_true_acc = evaluate_model(gt,gt_ids,prob,test_names,im_ext,full_syn)

