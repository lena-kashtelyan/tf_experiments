#!/usr/bin/env python
import os, sys
sys.path.append('../../../')
import tensorflow as tf
from copy import deepcopy
from exp_ops.helper_functions import *
from tf_experiments.experiments.config import * # Path configurations
from model_depo import att_vgg19 as vgg19

def attention_vgg19(ptest=False,num_perms=1000,shuffle_or_warp='shuffle',insert_into_database=True):

    im_ext = '.JPEG'
    im_size = [224,224]
    grayscale=False
    attention_conv = '1_1'

    #Load model and relevant info
    syn, skeys = get_synkeys()

    #Prepare training data and train an svm
    test_X,test_y,test_names = prepare_testing_images(test_im_dir,im_size,im_ext,grayscale=grayscale)
    gt,gt_ids = get_labels(test_names,syn,skeys,syn_file)
    attention_batch = get_attention_maps(attention_path,im_size,test_names)
    with tf.device('/gpu:0'):
        with tf.Session(
                config=tf.ConfigProto(allow_soft_placement = True, gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=.8)))) as sess:
            vgg = vgg19.Vgg19(vgg19_npy_path=vgg19_weight_path)
            images = tf.placeholder("float", test_X.shape)
            attention_maps = tf.placeholder("float", attention_batch.shape)
            with tf.name_scope("content_vgg"):
                vgg.build(images,attention_maps,attention_conv)

            feed_dict = {images: test_X, attention_maps: attention_batch}

            prob = sess.run(vgg.prob, feed_dict=feed_dict)
            #l1 = sess.run(vgg.conv1_1, feed_dict=feed_dict)
            #l2 = sess.run(vgg.conv2_1, feed_dict=feed_dict)
            if ptest:
                t1_perm_accs = np.zeros((num_perms))
                t5_perm_accs = np.zeros((num_perms))
                max_perm_acc = 0
                _, _, _, t1_true_acc, t5_true_acc = evaluate_model(gt,gt_ids,prob,test_names,im_ext,full_syn,print_results=False)
                for idx in tqdm(range(num_perms)):
                    shuff_att = shuffle_attention(deepcopy(attention_batch),shuffle_or_warp)
                    feed_dict = {images: test_X, attention_maps: shuff_att}
                    prob = sess.run(vgg.prob, feed_dict=feed_dict)
                    _, _, _, t1_perm_accs[idx], t5_perm_accs[idx] = evaluate_model(gt,gt_ids,prob,test_names,im_ext,full_syn,print_results=False)
                    if t1_perm_accs[idx] > max_perm_acc:
                        max_perm_acc = t1_perm_accs[idx]
                        opt_perm_maps = shuff_att
                t1_p = (np.sum(t1_true_acc < t1_perm_accs) + 1).astype(np.float32) / (num_perms + 1)
                t5_p = (np.sum(t5_true_acc < t5_perm_accs) + 1).astype(np.float32) / (num_perms + 1)

        class_accuracy, t1_preds, t5_preds, t1_true_acc, t5_true_acc = evaluate_model(gt,gt_ids,prob,test_names,im_ext,full_syn)
        if insert_into_database:
            add_to_mirc_database(class_accuracy,experiment=image_set,model_name=sys.argv[0],attention=attention_path)

    if ptest:
        print('top-1 p value:',t1_p)
        print('top-5 p value:',t5_p)
    else:
        t1_p = 100
        t5_p = 100
    #im_mosaic(np.squeeze(l1[0,:,:,:]))
    #im_mosaic(np.squeeze(l2[0,:,:,:]))
    return class_accuracy, t1_true_acc, t5_true_acc, t1_preds, t5_preds, t1_p, t5_p

if __name__ == '__main__':
    if len(sys.argv) > 1:
        attention_vgg19(ptest=(sys.argv[1]=='True'))
    else:
        attention_vgg19()
