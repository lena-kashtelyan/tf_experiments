#!/usr/bin/env python
import os, sys
sys.path.append('../../../')
import tensorflow as tf
from copy import deepcopy
from exp_ops.helper_functions import *
from tf_experiments.experiments.config import * # Path configurations
from tf_experiments.model_depo import att_vgg16 as vgg16

def svm_attention_vgg16():

    im_ext = '.JPEG'
    im_size = [224,224]
    grayscale=False
    attention_conv = '1_1'
    num_batches = 2

    #Load model and relevant info
    syn, skeys = get_synkeys()

    #Load seperate images/labels in a loop
    test_X = []
    test_y = []
    test_names = []
    max_test_y = 0
    for idx in test_im_dir:
        it_test_X,it_test_y,it_test_names = prepare_testing_images(idx,im_size,im_ext,grayscale=grayscale)
        test_X.append(it_test_X)
        test_y.append(it_test_y + max_test_y)
        test_names.append(it_test_names)
        max_test_y += np.max(it_test_y) + 1
    test_X = np.vstack((test_X[:]))
    test_y = np.hstack((test_y[:]))
    test_names = np.hstack((test_names[:]))
    gt,gt_ids = get_labels(test_names,syn,skeys,syn_file)

    #Now do the same for attention (should factor this code at some point...)
    attention_batch = []
    for idx in attention_path:
        it_attention_batch = get_attention_maps([idx],im_size,test_names)
        attention_batch.append(it_attention_batch)
    attention_batch = np.vstack((attention_batch[:]))

    #Create a batch index
    bi = np.asarray(np.arange(attention_batch.shape[0]) >= attention_batch.shape[0] // 2).astype(np.int)
    test_X_shape = np.asarray(test_X.shape)
    test_X_shape[0] = test_X_shape[0] // 2
    test_X_shape = tuple(test_X_shape)
    attention_batch_shape = np.asarray(attention_batch.shape)
    attention_batch_shape[0] = attention_batch_shape[0] // 2
    attention_batch_shape = tuple(attention_batch_shape)

    #Get features
    fc_batches = []
    with tf.device('/gpu:0'):
        with tf.Session(
                config=tf.ConfigProto(allow_soft_placement = True, gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=.8)))) as sess:
            vgg = vgg16.Vgg16(vgg16_npy_path=vgg16_weight_path)
            images = tf.placeholder("float", test_X_shape)
            attention_maps = tf.placeholder("float", attention_batch_shape)
            with tf.name_scope("content_vgg"):
                vgg.build(images,attention_maps,attention_conv)

            #prob = sess.run(vgg.prob, feed_dict=feed_dict)
            for idx in range(num_batches):
                feed_dict = {images: test_X[bi==idx,:,:,:], attention_maps: attention_batch[bi==idx,:,:,:]}
                fc_batches.append(sess.run(vgg.fc8, feed_dict=feed_dict))

    fc_batches = np.vstack((fc_batches[:]))
    np.savez('svm_data/attention_svm_data_vgg16',fc_batches=fc_batches,gt=gt,gt_ids=gt_ids,test_names=test_names,full_syn=full_syn)


if __name__ == '__main__':
    svm_attention_vgg16()
