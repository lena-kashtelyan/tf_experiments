import numpy as np
import tensorflow as tf
from exp_ops.helper_functions import *
from exp_ops.convert import print_prob, load_image, checkpoint_fn, meta_fn
from ops.utils import print_prob



#settings
absolute_home = '/home/drew/Documents/tensorflow-vgg' #need to figure out a better system
test_im_dir = '/home/drew/Downloads/p2p_MIRCs/imgs/all_validation'
train_im_dir = '/home/drew/Downloads/p2p_MIRCs/imgs/train'
syn_file = absolute_home + '/data/ilsvrc_2012/synset_names.txt'
full_syn = absolute_home + '/data/ilsvrc_2012/synset.txt'
weight_path = absolute_home + '/pretrained_weights/vgg16.npy'
weight_dir = absolute_home + '/pretrained_weights/'
attention_conv = '1_1'
im_ext = '.JPEG'
im_size = [224,224]
grayscale=False
layers = 50

test_im = '/home/drew/Downloads/p2p_MIRCs/imgs/all_validation/sorrel0.JPEG'
img = load_image(test_im)

#Load model and relevant info
syn, skeys = get_synkeys()

#Prepare training data and train an svm
test_X,test_y,test_names = prepare_testing_images(test_im_dir,im_size,im_ext,grayscale=grayscale)
gt,gt_ids = get_labels(test_names,syn,skeys,syn_file)

with tf.device('/gpu:0'):
    with tf.Session(
            config=tf.ConfigProto(allow_soft_placement = True, gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=.95)))) as sess:
        new_saver = tf.train.import_meta_graph(weight_dir + meta_fn(layers))
        new_saver.restore(sess, weight_dir + checkpoint_fn(layers))

        graph = tf.get_default_graph()
        prob_tensor = graph.get_tensor_by_name("prob:0")
        images = graph.get_tensor_by_name("images:0")
        feed_dict = {images: test_X}
        prob = sess.run(prob_tensor, feed_dict=feed_dict)

class_accuracy, t1_preds, t5_preds, t1_true_acc, t5_true_acc = evaluate_model(gt,gt_ids,prob,test_names,im_ext,full_syn)

