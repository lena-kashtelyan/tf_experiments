import numpy as np
import tensorflow as tf
from exp_ops.helper_functions import *
from exp_ops.convert import print_prob, load_image, checkpoint_fn, meta_fn, create_att_net
from ops.utils import print_prob
from model_depo import resnet 


#settings
absolute_home = '/home/drew/Documents/tensorflow-vgg' #need to figure out a better system
test_im_dir = '/home/drew/Downloads/p2p_MIRCs/imgs/all_validation'
train_im_dir = '/home/drew/Downloads/p2p_MIRCs/imgs/train'
syn_file = absolute_home + '/data/ilsvrc_2012/synset_names.txt'
full_syn = absolute_home + '/data/ilsvrc_2012/synset.txt'
weight_path = absolute_home + '/pretrained_weights/vgg16.npy'
attention_path = ['/home/drew/Documents/MIRC_behavior/heat_map_output/pooled_p2p_alt/uniform_weight_overlap_human/heatmaps.npz',\
'/home/drew/Documents/MIRC_behavior/click_comparisons/output/labelme.npz']
weight_dir = absolute_home + '/pretrained_weights/'
im_ext = '.JPEG'
im_size = [224,224]
grayscale=False
layers = 50

#Load model and relevant info
syn, skeys = get_synkeys()

#Prepare training data and train an svm
test_X,test_y,test_names = prepare_testing_images(test_im_dir,im_size,im_ext,grayscale=grayscale,apply_preprocess=True)
test_X = test_X.transpose(0,2,3,1)
gt,gt_ids = get_labels(test_names,syn,skeys,syn_file)
attention_batch = get_attention_maps(attention_path,im_size)

with tf.device('/gpu:0'):
    with tf.Session(
            config=tf.ConfigProto(allow_soft_placement = True, gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=.95)))) as sess:

        attention_maps = tf.placeholder("float32", attention_batch.shape)
        logits, prob_tensor, images = create_att_net(weight_dir + checkpoint_fn(layers), layers)#, attention_maps)
        #sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver(tf.all_variables())#tf.trainable_variables())
        saver.restore(sess,weight_dir + checkpoint_fn(layers))

        feed_dict = {images: test_X}#, attention_maps: attention_batch}
        prob = sess.run(prob_tensor, feed_dict=feed_dict)


class_accuracy, t1_preds, t5_preds, t1_true_acc, t5_true_acc = evaluate_model(gt,gt_ids,prob,test_names,im_ext,full_syn)
