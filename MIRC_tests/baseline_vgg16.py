import numpy as np
import tensorflow as tf
from helper_functions import *
import sys
sys.path.append('/home/drew/Documents/tensorflow-vgg/')
import vgg16
from utils import print_prob

#settings
#test_im_dir = '/home/drew/Documents/MIRC_behavior/all_images'
test_im_dir = '/home/drew/Downloads/p2p_MIRCs/imgs/all_validation'
train_im_dir = '/home/drew/Downloads/p2p_MIRCs/imgs/train'
syn_file = '/home/drew/caffe/data/ilsvrc12/synsets.txt'
full_syn = '../synset.txt'
weight_path = '../pretrained_weights/vgg16.npy'
im_ext = '.JPEG'
im_size = [224,224]
grayscale=False

#Load model and relevant info
syn, skeys = get_synkeys()

#Prepare training data and train an svm
test_X,test_y,test_names = prepare_testing_images(test_im_dir,im_size,im_ext,grayscale=grayscale)
gt,gt_ids = get_labels(test_names,syn,skeys,syn_file)

with tf.device('/gpu:0'):
    with tf.Session(
            config=tf.ConfigProto(allow_soft_placement = True, gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=.8)))) as sess:
	    vgg = vgg16.Vgg16(vgg16_npy_path=weight_path)

	    images = tf.placeholder("float", test_X.shape)    
	    with tf.name_scope("content_vgg"):
	        vgg.build(images)

	    feed_dict = {images: test_X}

	    prob = sess.run(vgg.prob, feed_dict=feed_dict)

class_accuracy, t1_preds, t5_preds, t1_true_acc, t5_true_acc = evaluate_model(gt,gt_ids,prob,test_names,im_ext,full_syn)
