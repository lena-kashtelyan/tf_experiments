import numpy as np
import tensorflow as tf
from helper_functions import *
import sys
sys.path.append('/home/drew/Documents/tensorflow-vgg/')
import att_vgg16 as vgg16
from utils import print_prob

#settings
absolute_home = '/home/drew/Documents/tensorflow-vgg' #need to figure out a better system
test_im_dir = '/home/drew/Downloads/p2p_MIRCs/imgs/all_validation'
train_im_dir = '/home/drew/Downloads/p2p_MIRCs/imgs/train'
syn_file = absolute_home + '/' + ilsvrc_2012 + '/synset_names.txt'
full_syn = absolute_home + '/' + ilsvrc_2012 + '/synset.txt'
attention_path = '/home/drew/Documents/MIRC_behavior/heat_map_output/pooled_p2p_alt/uniform_weight_overlap_human/heatmaps.npz'
weight_path = absolute_home + 'pretrained_weights/vgg16.npy'
im_ext = '.JPEG'
im_size = [224,224]
grayscale=False
attention_conv = '2_2'

#Load model and relevant info
syn, skeys = get_synkeys()

#Prepare training data and train an svm
test_X,test_y,test_names = prepare_testing_images(test_im_dir,im_size,im_ext,grayscale=grayscale)
gt,gt_ids = get_labels(test_names,syn,skeys,syn_file)
attention_batch = get_attention_maps(attention_path,im_size)

with tf.device('/gpu:0'):
    with tf.Session(
            config=tf.ConfigProto(allow_soft_placement = True, gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=.8)))) as sess:
        vgg = vgg16.Vgg16(vgg16_npy_path=weight_path)
        images = tf.placeholder("float", test_X.shape)    
        attention_maps = tf.placeholder("float", attention_batch.shape)   
        with tf.name_scope("content_vgg"):
            vgg.build(images,attention_maps,attention_conv)

        feed_dict = {images: test_X, attention_maps: attention_batch}

        prob = sess.run(vgg.prob, feed_dict=feed_dict)
        #l1 = sess.run(vgg.conv1_1, feed_dict=feed_dict)
        #l2 = sess.run(vgg.conv2_1, feed_dict=feed_dict)

class_accuracy, t1_preds, t5_preds, t1_true_acc, t5_true_acc = evaluate_model(gt,gt_ids,prob,test_names,im_ext,full_syn)

#im_mosaic(np.squeeze(l1[0,:,:,:]))  
#im_mosaic(np.squeeze(att[0,:,:,:]))  
#im_mosaic(np.squeeze(l2[0,:,:,:]))  