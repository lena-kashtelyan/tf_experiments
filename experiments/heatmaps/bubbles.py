#!/usr/bin/env python
# Create a bubbles heatmap

import os, sys
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Run only on GPU 0 to speed up init time
import tensorflow as tf

from tf_experiments.experiments.config import pretrained_weights_path, heatmap_path
from tf_experiments.model_depo import vgg16
from tf_experiments.ops import utils
from scipy.ndimage.interpolation import zoom

def init_session():
    return tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.95))))

def load_model_vgg16(batch_size):
    # Load a default vgg16
    image_shape = (224, 224, 3)
    weight_path = os.path.join(pretrained_weights_path, 'vgg16.npy')
    vgg = vgg16.Vgg16(vgg16_npy_path=weight_path)
    input = tf.placeholder("float", (batch_size,) + image_shape)
    with tf.name_scope("content_vgg"):
        vgg.build(input)
    # Return model and input tensor
    return vgg

def get_bubbles_heatmap(sess, model, input_image, class_index=None, block_size=10, block_stride=12, variant='neg'):
    # Compute bubbles heatmap of class_index for given image on model
    assert variant in ['pos', 'neg']
    # Get shape information from model
    input_shape = model.input.get_shape().as_list()
    batch_size = input_shape[0]
    assert list(input_image.shape) == input_shape[1:]

    # Prepare batch information
    coords = [None] * batch_size
    batch = np.zeros(input_shape)
    feed_dict = {model.input: batch}

    # Get class index
    if class_index is None:
        batch[0, ...] = input_image
        prob = sess.run(model.prob, feed_dict=feed_dict)[0].squeeze()
        class_index = np.argmax(prob)
        print 'Using class index %d (prob %.3f)' % (class_index, prob[class_index])

    # Prepare output (zoomed down for block_stride>1)
    output_size = [c / block_stride for c in input_shape[1:3]]
    heatmap = np.zeros(output_size)

    # Get output from one batch and put it into the heatmap
    def process_batch(n=batch_size):
        prob = sess.run(model.prob, feed_dict=feed_dict).squeeze()
        for i, c in enumerate(coords[:n]):
            heatmap[c[0], c[1]] = prob[i, class_index]

    # Accumulate image regions into batch and process them
    i_batch = 0
    print '%d lines...' % input_shape[1]
    for iy in xrange(output_size[0]):
        print str(iy),
        y = iy * block_stride
        for ix in xrange(output_size[1]):
            x = ix * block_stride
            y0 = max(0, y - block_size / 2)
            y1 = min(input_shape[1], y + (block_size + 1) / 2)
            x0 = max(0, x - block_size / 2)
            x1 = min(input_shape[2], x + (block_size + 1) / 2)
            if variant == 'pos':
                image_masked = np.zeros(input_shape[1:])
                image_masked[y0:y1, x0:x1, :] = input_image[y0:y1, x0:x1, :]
            else:
                image_masked = input_image.copy()
                image_masked[y0:y1, x0:x1, :] = 0
            batch[i_batch, ...] = image_masked
            coords[i_batch] = [iy, ix]
            i_batch += 1
            if i_batch == batch_size:
                print ".",
                process_batch()
                i_batch = 0
        print "done"
    # Process remainder
    if i_batch:
        process_batch(i_batch)

    # Undo zoom
    heatmap = zoom(heatmap, block_stride)

    # Reverse signal of blanked-out
    if variant == 'neg':
        heatmap = 1.0 - heatmap

    return heatmap

def get_heatmap_filename(model_name, method_name, variant_name, class_index, image_filename):
    # Derive filename to save heatmap in from model + image
    path = os.path.join(heatmap_path, method_name)
    if not os.path.isdir(path):
        os.makedirs(path)
    return os.path.join(path, '%s_%s_%s_%s.npy' % (model_name, variant_name, str(class_index), os.path.basename(image_filename)))

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    image_filename = "../../forward_pass/test_data/tiger.jpeg"
    variant = 'neg'
    class_index = None
    heatmap_fn = get_heatmap_filename('vgg16', 'bubbles', variant, class_index, image_filename)
    if os.path.isfile(heatmap_fn):
        img = utils.load_image(image_filename)
        heatmap = np.load(heatmap_fn)
    else:
        with init_session() as sess:
            img = utils.load_image(image_filename)
            vgg = load_model_vgg16(batch_size=16)
            heatmap = get_bubbles_heatmap(sess, vgg, img, class_index, variant=variant)
            np.save(heatmap_fn, heatmap)

    f, axarr = plt.subplots(2, 1)
    axarr[0].imshow(img)
    m = axarr[1].matshow(heatmap)
    f.colorbar(m)
    plt.show()