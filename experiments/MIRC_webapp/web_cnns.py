#!/usr/bin/env python
import os
import tensorflow as tf
from copy import deepcopy
from exp_ops.helper_functions import *
from model_depo import att_vgg16, vgg16, att_vgg19, vgg19

def global_settings(test_im_dir,syn_file):
    im_ext = '.JPEG'
    im_size = [224,224]
    grayscale=False
    attention_conv = '1_1'
    batch_size = 10

    #Load model and relevant info
    syn, skeys = get_synkeys()

    #Prepare training data and train an svm
    test_X,test_y,test_names = prepare_testing_images(test_im_dir,im_size,im_ext,grayscale=grayscale)
    gt,gt_ids = get_labels(test_names,syn,skeys,syn_file)
    return batch_size, im_ext, im_size, grayscale, attention_conv, syn, skeys, test_X, test_y, test_names, gt, gt_ids

def run_batches(gt,gt_ids,test_names,im_ext,full_syn,bs,sess,images,vgg,test_X,attention_maps=None,attention_batch=None):
    uni_batches = np.arange(test_X.shape[0]//bs)
    cv_ind = np.repeat(uni_batches,bs,axis=0)
    class_accuracy = []
    t1_true_acc = np.zeros((len(cv_ind)))
    t5_true_acc = np.zeros((len(cv_ind)))
    t1_preds = []
    t5_preds = []
    test_names = np.asarray(test_names)
    gt = np.asarray(gt)
    gt_ids = np.asarray(gt_ids)
    for idx in uni_batches:
        if attention_maps != None:
            feed_dict = {images: test_X[cv_ind==idx,:,:,:], attention_maps: attention_batch[cv_ind==idx,:,:,:]}
        else:
            feed_dict = {images: test_X[cv_ind==idx,:,:,:]}
        prob = sess.run(vgg.prob, feed_dict=feed_dict)
        it_class_accuracy, it_t1_preds, it_t5_preds, t1_true_acc[cv_ind==idx], t5_true_acc[cv_ind==idx] = \
            evaluate_model(gt[cv_ind==idx],gt_ids[cv_ind==idx],prob,test_names[cv_ind==idx],im_ext,full_syn)
        class_accuracy.append(it_class_accuracy)
        t1_preds.append(it_t1_preds)
        t5_preds.append(it_t5_preds)
    return class_accuracy, t1_preds, t5_preds, t1_true_acc, t5_true_acc

def baseline_vgg16(test_im_dir,syn_file,full_syn,model_weight_path):
    bs, im_ext, im_size, grayscale, _, syn, skeys, test_X, test_y, test_names, gt, gt_ids = \
        global_settings(test_im_dir,syn_file)
    with tf.device('/gpu:0'):
        with tf.Session(
                config=tf.ConfigProto(allow_soft_placement = True, gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=.8)))) as sess:
            vgg = vgg16.Vgg16(vgg16_npy_path=model_weight_path)
            images = tf.placeholder("float", [bs,test_X.shape[1],test_X.shape[2],test_X.shape[3]])
            with tf.name_scope("content_vgg"):
                vgg.build(images)
            class_accuracy, t1_preds, t5_preds, t1_true_acc, t5_true_acc = run_batches(gt,gt_ids,test_names,im_ext,full_syn,bs,sess,images,vgg,test_X)
    return class_accuracy, t1_true_acc, t5_true_acc, t1_preds, t5_preds

def baseline_vgg19(test_im_dir,syn_file,full_syn,model_weight_path):
    bs, im_ext, im_size, grayscale, _, syn, skeys, test_X, test_y, test_names, gt, gt_ids = \
        global_settings(test_im_dir,syn_file)
    with tf.device('/gpu:0'):
        with tf.Session(
                config=tf.ConfigProto(allow_soft_placement = True, gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=.8)))) as sess:
            vgg = vgg19.Vgg19(vgg19_npy_path=model_weight_path)
            images = tf.placeholder("float", [bs,test_X.shape[1],test_X.shape[2],test_X.shape[3]])
            with tf.name_scope("content_vgg"):
                vgg.build(images)
            class_accuracy, t1_preds, t5_preds, t1_true_acc, t5_true_acc = run_batches(gt,gt_ids,test_names,im_ext,full_syn,bs,sess,images,vgg,test_X)
    return class_accuracy, t1_true_acc, t5_true_acc, t1_preds, t5_preds

def attention_vgg16(test_im_dir,syn_file,full_syn,model_weight_path,attention_path):
    bs, im_ext, im_size, grayscale, attention_conv, syn, skeys, test_X, test_y, test_names, gt, gt_ids = \
        global_settings(test_im_dir,syn_file)
    attention_batch = get_attention_maps(attention_path,im_size,test_names)
    with tf.device('/gpu:0'):
        with tf.Session(
                config=tf.ConfigProto(allow_soft_placement = True, gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=.8)))) as sess:
            vgg = att_vgg16.Vgg16(vgg16_npy_path=model_weight_path)
            images = tf.placeholder("float", [bs,test_X.shape[1],test_X.shape[2],test_X.shape[3]])
            attention_maps = tf.placeholder("float", [bs,attention_batch.shape[1],attention_batch.shape[2],attention_batch.shape[3]])
            with tf.name_scope("content_vgg"):
                vgg.build(images,attention_maps,attention_conv)
            class_accuracy, t1_preds, t5_preds, t1_true_acc, t5_true_acc = run_batches(gt,gt_ids,test_names,im_ext,full_syn,bs,sess,images,vgg,test_X,attention_maps,attention_batch)
    return class_accuracy, t1_true_acc, t5_true_acc, t1_preds, t5_preds

def attention_vgg19(test_im_dir,syn_file,full_syn,model_weight_path,attention_path):
    bs, im_ext, im_size, grayscale, attention_conv, syn, skeys, test_X, test_y, test_names, gt, gt_ids = \
        global_settings(test_im_dir,syn_file)
    attention_batch = get_attention_maps(attention_path,im_size,test_names)
    with tf.device('/gpu:0'):
        with tf.Session(
                config=tf.ConfigProto(allow_soft_placement = True, gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=.8)))) as sess:
            vgg = att_vgg19.Vgg19(vgg19_npy_path=model_weight_path)
            images = tf.placeholder("float", [bs,test_X.shape[1],test_X.shape[2],test_X.shape[3]])
            attention_maps = tf.placeholder("float", [bs,attention_batch.shape[1],attention_batch.shape[2],attention_batch.shape[3]])
            with tf.name_scope("content_vgg"):
                vgg.build(images,attention_maps,attention_conv)
            class_accuracy, t1_preds, t5_preds, t1_true_acc, t5_true_acc = run_batches(gt,gt_ids,test_names,im_ext,full_syn,bs,sess,images,vgg,test_X,attention_maps,attention_batch)
    return class_accuracy, t1_true_acc, t5_true_acc, t1_preds, t5_preds

def run_model(model_type,test_im_dir,syn_file,full_syn,model_weight_path,attention_path=None):
    if model_type == 'baseline_vgg16':
        model_weight_path = model_weight_path + 'vgg16.npy'
        class_accuracy, t1_true_acc, t5_true_acc, t1_preds, t5_preds = \
            baseline_vgg16(test_im_dir,syn_file,full_syn,model_weight_path)
    elif model_type == 'baseline_vgg19':
        model_weight_path = model_weight_path + 'vgg19.npy'
        class_accuracy, t1_true_acc, t5_true_acc, t1_preds, t5_preds = \
            baseline_vgg19(test_im_dir,syn_file,full_syn,model_weight_path)
    elif model_type == 'attention_vgg16':
        model_weight_path = model_weight_path + 'vgg16.npy'
        class_accuracy, t1_true_acc, t5_true_acc, t1_preds, t5_preds = \
            attention_vgg16(test_im_dir,syn_file,full_syn,model_weight_path,attention_path)
    elif model_type == 'attention_vgg19':
        model_weight_path = model_weight_path + 'vgg19.npy'
        class_accuracy, t1_true_acc, t5_true_acc, t1_preds, t5_preds = \
            attention_vgg19(test_im_dir,syn_file,full_syn,model_weight_path,attention_path)
    return class_accuracy, t1_true_acc, t5_true_acc, t1_preds, t5_preds
