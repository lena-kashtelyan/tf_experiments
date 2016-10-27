import numpy as np
import tensorflow as tf

import vgg16
import utils



weight_path = 'pretrained_weights/vgg16.npy'
l1_filters = 64

img1 = utils.load_image("./test_data/tiger.jpeg")
img2 = utils.load_image("./test_data/puzzle.jpeg")

batch1 = img1.reshape((1, 224, 224, 3))
batch2 = img2.reshape((1, 224, 224, 3))
batch = np.concatenate((batch1, batch2), 0)

attention1 = np.zeros((1,224,224))
attention2 = np.ones((1,224,224))
attention_batch = np.concatenate((attention1, attention2), 0)[:,:,:,None]

with tf.Session(
        config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=1)))) as sess:
    vgg = vgg16.Vgg16(vgg16_npy_path=weight_path)

    images = tf.placeholder("float", batch.shape)    
    attention_maps = tf.placeholder("float", attention_batch.shape)   
    with tf.name_scope("content_vgg"):
        vgg.build(images,attention_maps)

    feed_dict = {images: batch, attention_maps: attention_batch}

    prob = sess.run(vgg.prob, feed_dict=feed_dict)
    print(prob)
    utils.print_prob(prob[0], './synset.txt')
    utils.print_prob(prob[1], './synset.txt')
