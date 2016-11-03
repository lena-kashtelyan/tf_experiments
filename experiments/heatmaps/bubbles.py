#!/usr/bin/env python
# Create a bubbles heatmap

import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Run only on GPU 0 to speed up init time
import tensorflow as tf
import glob
from scipy.misc import imresize

from tf_experiments.experiments.config import pretrained_weights_path, heatmap_path, data_dir
from tf_experiments.model_depo import vgg16
from tf_experiments.ops import utils
from scipy.ndimage.interpolation import zoom
from tf_experiments.experiments.MIRC_tests.exp_ops.helper_functions import get_synkeys, get_class_index_for_filename

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
    if len(input_image.shape) == 2:
        # Grayscale to RGB
        input_image = np.dstack((input_image,)*3)
    if list(input_image.shape) != input_shape[1:]:
        # Resize to fit model
        print '  Reshaping image from %s to %s.' % (str(input_image.shape), str(input_shape[1:]))
        input_image = imresize(input_image, input_shape[1:])

    # Prepare batch information
    coords = [None] * batch_size
    batch = np.zeros(input_shape)
    feed_dict = {model.input: batch}

    # Get class index
    if class_index is None:
        batch[0, ...] = input_image
        prob = sess.run(model.prob, feed_dict=feed_dict)[0].squeeze()
        class_index = np.argmax(prob)
        print '  Using class index %d (prob %.3f)' % (class_index, prob[class_index])
    else:
        print '  Using class index %d (true label)' % (class_index)

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
    print ('  Processing %s...\n  ' % str(output_size)),
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
        if not (iy % 10): print '\n  ',
    # Process remainder
    if i_batch:
        process_batch(i_batch)
    print '  Heatmap done.',

    # Undo zoom
    heatmap = zoom(heatmap, block_stride)

    # Reverse signal of blanked-out
    if variant == 'neg':
        heatmap = 1.0 - heatmap

    return heatmap

def get_heatmap_filename(model_name, method_name, variant_name, class_index, image_filename):
    # Derive filename to save heatmap in from model + image
    path = os.path.join(heatmap_path, method_name, variant_name)
    if not os.path.isdir(path):
        os.makedirs(path)
    return os.path.join(path, '%s_%s_%s.npy' % (model_name, str(class_index), os.path.basename(image_filename)))

def generate_heatmaps_for_images(image_filenames, model_name, method_name, variant, block_size=10, block_stride=1):
    # Generate all heatmaps for images in list
    # Load synkeys to class index
    syn, _skeys = get_synkeys()
    synset_names = open(os.path.join(data_dir, 'data', 'ilsvrc_2012', 'synset_names.txt'), 'rt').read().splitlines()
    name_to_class_index = {}
    for k, wordnet_id in syn.iteritems():
        name_to_class_index[k] = synset_names.index(wordnet_id)
    # Get class indices for all files
    class_indices = [get_class_index_for_filename(fn, name_to_class_index) for fn in image_filenames]
    # Process all files
    variant_name = '%s_%d_%d' % (variant, block_size, block_stride)
    with init_session() as sess:
        vgg = load_model_vgg16(batch_size=16)
        for class_index, image_filename in zip(class_indices, image_filenames):
            heatmap_filename = get_heatmap_filename(model_name=model_name, method_name=method_name, variant_name=variant_name, class_index=class_index, image_filename=image_filename)
            print 'Heatmap for %s...' % os.path.basename(heatmap_filename)
            if os.path.isfile(heatmap_filename):
                print ' Skipping existing heatmap at %s' % heatmap_filename
            else:
                img = utils.load_image(image_filename)
                heatmap = get_bubbles_heatmap(sess, vgg, img, class_index, variant=variant, block_size=block_size, block_stride=block_stride)
                print ' Saving heatmap to %s...' % heatmap_filename
                np.save(heatmap_filename, heatmap)


def test_heatmap():
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

if __name__ == "__main__":
    image_filenames = glob.glob(os.path.join(data_dir, 'MIRC_images_for_sven', 'bw_validation', 'all_images', '*.JPEG'))
    image_filenames = [fn for fn in image_filenames if not os.path.basename(fn).startswith('mircs')]
    generate_heatmaps_for_images(image_filenames, 'vgg16', 'bubbles', 'neg', block_size=10, block_stride=1)