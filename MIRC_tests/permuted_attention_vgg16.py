import numpy as np
import tensorflow as tf
from helper_functions import *
import sys
sys.path.append('/home/drew/Documents/tensorflow-vgg/')
import vgg16
from utils import print_prob
from tqdm import tqdm
from skimage import transform as tf

#settings
#test_im_dir = '/home/drew/Documents/MIRC_behavior/all_images'
test_im_dir = '/home/drew/Downloads/p2p_MIRCs/imgs/all_validation'
train_im_dir = '/home/drew/Downloads/p2p_MIRCs/imgs/train'
syn_file = '/home/drew/caffe/data/ilsvrc12/synsets.txt'
full_syn = '../synset.txt'
attention_path = '/home/drew/Documents/MIRC_behavior/heat_map_output/pooled_p2p_alt/uniform_weight_overlap_human/heatmaps.npz'
#attention_path = '/home/drew/Documents/MIRC_behavior/heat_map_output/pooled_p2p_alt/linear_accumulation_human/heatmaps.npz'
weight_path = '../pretrained_weights/vgg16.npy'
im_ext = '.JPEG'
im_size = [224,224]
grayscale=False
divisive_normalization = False
num_perms = 10000
shuffle_or_warp = 'shuffle'

#Load model and relevant info
syn, skeys = get_synkeys()

#Prepare training data and train an svm
test_X,test_y,test_names = prepare_testing_images(test_im_dir,im_size,im_ext,grayscale=grayscale)
gt,gt_ids = get_labels(test_names,syn,skeys,syn_file)
attention_batch = get_attention_maps(attention_path,im_size)

t1_perm_accs = np.zeros((num_perms))
t5_perm_accs = np.zeros((num_perms))
max_perm_acc = 0
with tf.Session(
        config=tf.ConfigProto(allow_soft_placement = True, log_device_placement=True,\
            gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=.8)))) as sess:
    with tf.device('/gpu:0'):
        vgg = vgg16.Vgg16(vgg16_npy_path=weight_path)
        images = tf.placeholder("float", test_X.shape)    
        attention_maps = tf.placeholder("float", attention_batch.shape)   
        with tf.name_scope("content_vgg"):
            vgg.build(images,attention_maps,divisive_normalization)
        feed_dict = {images: test_X, attention_maps: attention_batch}
        prob = sess.run(vgg.prob, feed_dict=feed_dict)
        _, _, _, t1_true_acc, t5_true_acc = evaluate_model(gt,gt_ids,prob,test_names,im_ext,full_syn,print_results=False)
        for idx in tqdm(range(num_perms)):
            if shuffle_or_warp == 'shuffle':
                shuff_att = shuffle_attention(attention_batch)
            else:
                tform = tf.SimilarityTransform(scale=np.random.rand() + 0.5, rotation=np.random.vonmises(),
                               translation=(np.random.randint(0,attention_batch.shape[1]*.5)))
                shuff_att = tf.warp(attention_batch,tform)
            feed_dict = {images: test_X, attention_maps: shuff_att}
            prob = sess.run(vgg.prob, feed_dict=feed_dict)
            _, _, _, t1_perm_accs[idx], t5_perm_accs[idx] = evaluate_model(gt,gt_ids,prob,test_names,im_ext,full_syn,print_results=False)
            if t1_perm_accs[idx] > max_perm_acc:
                max_perm_acc = t1_perm_accs[idx]
                opt_perm_maps = shuff_att
t1_p = (np.sum(t1_true_acc < t1_perm_accs) + 1).astype(np.float32) / (num_perms + 1)
t5_p = (np.sum(t5_true_acc < t5_perm_accs) + 1).astype(np.float32) / (num_perms + 1)

print('t1 p = ',t1_p)
print('t5 p = ',t5_p)
