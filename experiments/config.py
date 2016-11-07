#!/usr/bin/env python
# Per-machine/user configuration (mostly paths)

import os,sys
from os.path import join as pjoin
node_name = os.uname()[1]


# Path settings
if node_name == 'x9':
    # Drew
    # Home of this source code: Three folders up form the config file
    src_dir = '/home/drew/Documents/tf_experiments/'
    sys.path.append(src_dir)
    image_set = 2 #1 or 2
    train_im_dir = '/home/drew/Downloads/p2p_MIRCs/imgs/train' #Pretty sure this is vestigial
    syn_file = pjoin(src_dir, 'data', 'ilsvrc_2012', 'synset_names.txt')
    full_syn = pjoin(src_dir, 'data', 'ilsvrc_2012', 'synset.txt')
    vgg16_weight_path = pjoin(src_dir, 'pretrained_weights', 'vgg16.npy')
    vgg19_weight_path = pjoin(src_dir, 'pretrained_weights', 'vgg19.npy')
    resnet_weight_path = '/home/drew/Documents/caffe-tensorflow/resnet_conversions/'
    #Set up experiment variables
    if image_set == 1:
        test_im_dir = '/home/drew/Downloads/p2p_MIRCs/imgs/all_validation'
        #attention_path = ['/home/drew/Documents/MIRC_behavior/heat_map_output/pooled_p2p_alt/uniform_weight_overlap_human/heatmaps.npz'] #Human clicks
        #attention_path = ['/home/drew/Documents/MIRC_behavior/heat_map_output/pooled_p2p_alt/human_none/heatmaps.npz'] #Human clicks
        #attention_path = ['/home/drew/Documents/MIRC_behavior/heat_map_output/pooled_vgg/uniform_weight_overlap_machine/heatmaps.npz'] #Machine clicks
        attention_path = ['/home/drew/Documents/MIRC_behavior/heat_map_output/bu_saliency/images_1_saliency.npz'] #bottom up saliency
        #--------------------------------------------------
        #attention_path = ['/home/drew/Documents/MIRC_behavior/click_comparisons/heatmaps_for_paper/vgg_bubbles_images_1/maps.npz'] #Human clicks
        #attention_path = ['/home/drew/Documents/MIRC_behavior/click_comparisons/heatmaps_for_paper/vgg_maps_images_1/maps.npz'] #Human clicks
        #--------------------------------------------------
        #attention_path = ['/home/drew/Documents/MIRC_behavior/click_comparisons/output/labelme.npz'] #Labelme
        #attention_path = ['/home/drew/Documents/MIRC_behavior/heat_map_output/pooled_p2p_alt/uniform_weight_overlap_human/heatmaps.npz',
        #    '/home/drew/Documents/MIRC_behavior/click_comparisons/raw_alternative_attention_maps/output/labelme.npz'] #Combined
    elif image_set == 2:
        test_im_dir = '/home/drew/Downloads/p2p_MIRCs/imgs/all_validation_replication'
        #attention_path = ['/home/drew/Documents/MIRC_behavior/replication_heat_map_output/pooled_p2p/uniform_weight_overlap/heatmaps.npz'] #Human clicks
        #attention_path = ['/home/drew/Documents/MIRC_behavior/replication_heat_map_output/pooled_p2p/human_none/heatmaps.npz'] #Human clicks
        #attention_path = ['/home/drew/Documents/MIRC_behavior/replication_heat_map_output/pooled_vgg/uniform_weight_overlap/heatmaps.npz'] #Machine clicks
        attention_path = ['/home/drew/Documents/MIRC_behavior/heat_map_output/bu_saliency/images_2_saliency.npz'] #bottom up saliency
        #--------------------------------------------------        
	#attention_path = ['/home/drew/Documents/MIRC_behavior/click_comparisons/heatmaps_for_paper/vgg_bubbles_images_2/maps.npz'] #Human clicks        
	#attention_path = ['/home/drew/Documents/MIRC_behavior/click_comparisons/heatmaps_for_paper/vgg_maps_images_2/maps.npz'] #Human clicks

    else:
        print('Cannot understand what images you want to use. Exiting.')
        sys.exit()
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
        pjoin(data_dir, 'MIRC_behavior', 'click_comparisons', 'raw_alternative_attention_maps', 'output', 'labelme.npz')]
    pretrained_weights_path = pjoin(data_dir, 'pretrained_weights')
    heatmap_path = pjoin(data_dir, 'heatmaps')
