#!/usr/bin/env python
# Per-machine/user configuration (mostly paths)

import os,sys
from os.path import join as pjoin
node_name = os.uname()[1]


# Path settings
if node_name == 'x9':
    # Drew
    # Home of this source code: Three folders up form the config file
    src_dir = '/home/drew/Documents/tensorflow-vgg'
    sys.path.append(src_dir)
    test_im_dir = '/home/drew/Downloads/p2p_MIRCs/imgs/all_validation'
    train_im_dir = '/home/drew/Downloads/p2p_MIRCs/imgs/train'
    syn_file = pjoin(src_dir, 'data', 'ilsvrc_2012', 'synset_names.txt')
    full_syn = pjoin(src_dir, 'data', 'ilsvrc_2012', 'synset.txt')
    vgg16_weight_path = pjoin(src_dir, 'pretrained_weights', 'vgg16.npy')
    resnet_weight_path = '/home/drew/Documents/caffe-tensorflow/resnet_conversions/'

    #attention_path = ['/home/drew/Documents/MIRC_behavior/heat_map_output/pooled_p2p_alt/uniform_weight_overlap_human/heatmaps.npz']
    attention_path = ['/home/drew/Documents/MIRC_behavior/click_comparisons/output/labelme.npz']
    #attention_path = ['/home/drew/Documents/MIRC_behavior/heat_map_output/pooled_p2p_alt/uniform_weight_overlap_human/heatmaps.npz',
    #    '/home/drew/Documents/MIRC_behavior/click_comparisons/output/labelme.npz']
else:
    # Sven2
    data_dir = '/media/data_gluster/attention'
    test_im_dir = pjoin(data_dir, 'p2p_MIRCs', 'imgs', 'all_validation')
    train_im_dir = pjoin(data_dir, 'p2p_MIRCs', 'imgs', 'train')
    syn_file = pjoin(data_dir, 'data', 'ilsvrc_2012', 'synset_names.txt')
    full_syn = pjoin(data_dir, 'data', 'ilsvrc_2012', 'synset.txt')
    vgg16_weight_path = pjoin(data_dir, 'pretrained_weights', 'vgg16.npy')
    attention_path = [
        pjoin(data_dir, 'MIRC_behavior', 'heat_map_output', 'pooled_p2p_alt', 'uniform_weight_overlap_human', 'heatmaps.npz'),
        pjoin(data_dir, 'MIRC_behavior', 'click_comparisons', 'output', 'labelme.npz')]