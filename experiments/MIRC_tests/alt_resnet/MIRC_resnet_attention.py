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
from copy import deepcopy

absolute_home = '/home/drew/Documents/tensorflow-vgg' #need to figure out a better system
syn_file = absolute_home + '/data/ilsvrc_2012/synset_names.txt'
full_syn = absolute_home + '/data/ilsvrc_2012/synset.txt'

im_ext = '.JPEG'
im_size = [224,224]
batch_size = 25
resnet_type = 101
ptest = True
num_perms = 1000

model_data_path = '/home/drew/Documents/caffe-tensorflow/resnet_conversions/resnet_' + str(resnet_type) + '_data.npy'
test_im_dir = '/home/drew/Downloads/p2p_MIRCs/imgs/all_validation'
attention_path = ['/home/drew/Documents/MIRC_behavior/heat_map_output/pooled_p2p_alt/uniform_weight_overlap_human/heatmaps.npz',\
'/home/drew/Documents/MIRC_behavior/click_comparisons/output/labelme.npz']

#Prepare network
net, spec = interpret_resnet(resnet_type)

#Images
_,_,test_names = prepare_testing_images(test_im_dir,im_size,im_ext,grayscale=False,apply_preprocess=True)
syn, skeys = get_synkeys()
gt,gt_ids = get_labels(test_names,syn,skeys,syn_file)
image_paths = sorted(glob(test_im_dir + '/*' + im_ext)) 
attention_batch = get_attention_maps(attention_path,[spec.crop_size,spec.crop_size])

# Create a placeholder for the input image and attention
input_node = tf.placeholder(tf.float32,
                                shape=(None, spec.crop_size, spec.crop_size, spec.channels))
attention = tf.placeholder(tf.float32,shape=(None, spec.crop_size, spec.crop_size, 1),name='attention_maps')

# Construct the network
net = net({'data': input_node, 'attention' : attention})

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


    if ptest:
        _, _, _, t1_true_acc, t5_true_acc = evaluate_model(gt,gt_ids,prob,test_names,im_ext,full_syn,print_results=False)
        for idx in tqdm(range(num_perms)):
            shuff_att = shuffle_attention(deepcopy(attention_batch),shuffle_or_warp)
            feed_dict = {images: test_X, attention_maps: shuff_att}
            prob = sess.run(vgg.prob, feed_dict=feed_dict)
            _, _, _, t1_perm_accs[idx], t5_perm_accs[idx] = evaluate_model(gt,gt_ids,prob,test_names,im_ext,full_syn,print_results=False)
            if t1_perm_accs[idx] > max_perm_acc:
                max_perm_acc = t1_perm_accs[idx]
                opt_perm_maps = shuff_att
        t1_p = (np.sum(t1_true_acc < t1_perm_accs) + 1).astype(np.float32) / (num_perms + 1)
        t5_p = (np.sum(t5_true_acc < t5_perm_accs) + 1).astype(np.float32) / (num_perms + 1)



    # Stop the worker threads
    coordinator.request_stop()
    coordinator.join(threads, stop_grace_period_secs=2)

sorted_indices = np.argsort(indices)
prob = prob[sorted_indices,:]
class_accuracy, t1_preds, t5_preds, t1_true_acc, t5_true_acc = evaluate_model(gt,gt_ids,prob,test_names,im_ext,full_syn)

