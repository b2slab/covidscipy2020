import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import pylab
import random
import os

basepath = 'C:/Users/Guillem/Desktop/HACKATHON 2020/Labeled audio/spectrograms/'
sp = os.path.join(basepath, 'spectrograms_pos/')
sn = os.path.join(basepath, 'spectrograms_neg/')

for spectr in os.listdir(sp):

    spectr_path = os.path.join(sp, spectr)
    image = tf.io.read_file(spectr_path)
    image = tf.image.decode_jpeg(image)

    times_augmented = 1

    for i in range(times_augmented):
        aug_image, aug_name = image_augmentation(image)
        output_path = spectr_path.strip('.jpg') + '_{}_augm_{}.jpg'.format(aug_name, i)
        plot_augmented_spectrogram(aug_image, output_path)

# Apply if we want to augment Negative dataset
for spectr in os.listdir(sn):

    spectr_path = os.path.join(sn, spectr)
    image = tf.io.read_file(spectr_path)
    image = tf.image.decode_jpeg(image)

    times_augmented = 1

    for i in range(times_augmented):
        aug_image, aug_name = image_augmentation(image)
        output_path = spectr_path.strip('.jpg') + '_{}_augm_{}.jpg'.format(aug_name, i)
        plot_augmented_spectrogram(aug_image, output_path)


def image_augmentation(image):

    augmentation = ['rb','rc','rh','rcrop', 'rn', 'rs']
    _sample = random.sample(augmentation, k = 1)

    if _sample == 'rb':
        rb = tf.image.random_brightness(image, 0.2)
        return rb,_sample

    elif _sample == 'rc':
        rc = tf.image.random_contrast(image, 0.2, 0.5)
        return rc,_sample

    elif _sample == 'rh':
        rh = tf.image.random_hue(image, max_delta = 0.4)
        return rh,_sample

    elif _sample == 'rcrop':
        rcrop = tf.image.random_crop(image, size = [170, 170, 3])
        return rcrop,_sample

    elif _sample == 'rn':
        rn = tf.image.random_jpeg_quality(image, 10, 30)
        return rn,_sample

    else:
        rs = tf.image.random_saturation(image, 1, 2)
        return rs,_sample


def plot_augmented_spectrogram(aug_image, output_path, height = 448, width = 448):
    dpi = pylab.gcf().get_dpi()
    fig, ax = plt.subplots(figsize=(height/dpi, width/dpi))
    plt.imshow(aug_image.numpy().astype("uint8"))
    plt.axis('off')
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(output_path, bbox_inches = 'tight',pad_inches = 0)
    plt.close(fig)
