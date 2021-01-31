import numpy as np
import pandas as pd
import tensorflow as tf
import cv2

# Helper functions

def _int64_feature(value):
  return tf.train.Feature(
            int64_list=tf.train.Int64List(value=value)
         )
def _floats_feature(value):
    return tf.train.Feature(
               float_list=tf.train.FloatList(value=value)
           )
def _bytes_feature(value):
  return tf.train.Feature(
              bytes_list=tf.train.BytesList(value=value)
         )

# Read images

def load_image(path):
    img = cv2.imread(path)
    # cv2 load images as BGR, convert it to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return img


# Convert the whole image as array

def get_feature(image, label):
    return { # split to channels just for clearer representation, you may also encode it as image.reshape(-1)
        'r': _floats_feature(image[:, :, 0].reshape(-1)),
        'g': _floats_feature(image[:, :, 1].reshape(-1)),
        'b': _floats_feature(image[:, :, 2].reshape(-1)),
        'label': _int64_feature([label])
    }


# Create TFRecords
import os

basepath = 'C:/Users/Guillem/Desktop/HACKATHON 2020/CNN for audio classification/audio_preprocessing/spectrograms/'
pos_path = os.path.join(basepath, 'pos/')
neg_path = os.path.join(basepath, 'neg/')

paths = []
for path in os.listdir(pos_path):
    paths.append(os.path.join(pos_path,path))

for path in os.listdir(neg_path):
    paths.append(os.path.join(neg_path,path))

# paths = ['path/to/image1', 'path/to/image2', ...]

label = np.full(len(os.listdir(pos_path)), 1)
label = np.append(label, np.full(len(os.listdir(neg_path)), 0)).tolist()

# label = 1 ---> POSITIVE
# label = 0 ---> NEGATIVE

output_file_path = 'C:/Users/Guillem/Desktop/HACKATHON 2020/CNN for audio classification/audio_preprocessing/spectrograms/spectrograms.tfrecords'
with tf.io.TFRecordWriter(output_file_path) as writer:
    for idx, path in enumerate(paths):
        img = load_image(paths[idx])
        example = tf.train.Example(features=tf.train.Features(feature = get_feature(img,label[idx])))
        writer.write(example.SerializeToString())
        print('\r{:.1%}'.format((idx+1)/len(paths)), end='')



# Read TFRecords -- From array encoded values

img_height = 223
img_width = 223

def get_tfrecords_features():
    return {
        'r': tf.io.FixedLenFeature([img_height*img_width], tf.float32),
        'g': tf.io.FixedLenFeature([img_height*img_width], tf.float32),
        'b': tf.io.FixedLenFeature([img_height*img_width], tf.float32),
        'label': tf.io.FixedLenFeature([1], tf.int64)
        }


def feature_retrieval(serialized_example, width = 223, height = 223):

    features = tf.io.parse_single_example(
                  serialized_example,
                  features=get_tfrecords_features()
               )

    _r = tf.cast(features['r'], tf.float32)
    _g = tf.cast(features['g'], tf.float32)
    _b = tf.cast(features['b'], tf.float32)
    _label = tf.cast(features['label'], tf.int64)

    data = tf.transpose(
            tf.stack([
                tf.reshape(_r, (height, width)),
                tf.reshape(_g, (height, width)),
                tf.reshape(_b, (height, width))
            ])
        )
    label = tf.transpose(_label)
    return data, label

# Output the values

def load_tfrecords(tfrecords_filepath):
    items = []
    labels = []
    print("Loading %s" % tfrecords_filepath)
    for serialized_example in tf.data.TFRecordDataset(tfrecords_filepath):
        data, label = feature_retrieval(serialized_example)
        items.append(data)
        labels.append(label)
    print("Finished Loading %s" % tfrecords_filepath)
    return (tf.stack(items), tf.stack(labels))

tfrecords_filepath = 'C:/Users/Guillem/Desktop/HACKATHON 2020/CNN for audio classification/audio_preprocessing/spectrograms/spectrograms.tfrecords'
items, labels = load_tfrecords(tfrecords_filepath)
