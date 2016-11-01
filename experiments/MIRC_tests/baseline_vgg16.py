import os, sys
sys.path.append('/home/drew/Documents/tensorflow-vgg/')

import tensorflow as tf
from exp_ops.helper_functions import *
from experiments.config import * # Path configurations
from model_depo import vgg16

def baseline_vgg16():
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
            vgg = vgg16.Vgg16(vgg16_npy_path=vgg16_weight_path)

            images = tf.placeholder("float", test_X.shape)    
            with tf.name_scope("content_vgg"):
                vgg.build(images)

            feed_dict = {images: test_X}

            prob = sess.run(vgg.prob, feed_dict=feed_dict)

    class_accuracy, t1_preds, t5_preds, t1_true_acc, t5_true_acc = evaluate_model(gt,gt_ids,prob,test_names,im_ext,full_syn)

    return class_accuracy, t1_true_acc, t5_true_acc, t1_preds, t5_preds, 100, 100


if __name__ == '__main__':
    baseline_vgg16()