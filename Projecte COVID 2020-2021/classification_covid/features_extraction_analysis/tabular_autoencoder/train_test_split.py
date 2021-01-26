import os
import random
import shutil
import pandas as pd
import numpy as np

basepath = 'C:/Users/Guillem/Desktop/Anomaly detection (autoencoders)/spectrograms/'
pos_path = basepath + 'pos/'
neg_path = basepath + 'neg/'

train_path = basepath + 'train/'
if not os.path.exists(train_path):
    os.makedirs(train_path)
    os.makedirs(train_path + 'neg/')

test_path = basepath + 'test/'
if not os.path.exists(test_path):
    os.makedirs(test_path)
    os.makedirs(test_path + 'neg/')
    os.makedirs(test_path + 'pos/')


for pos_spec in os.listdir(pos_path):
    old = os.path.join(pos_path, pos_spec)
    new = os.path.join(test_path+'pos/', pos_spec)
    shutil.copy(old, new)


def train_test_split(directory, training_size):
    random.seed(12345)
    list = os.listdir(directory)
    list = pd.DataFrame(list)
    lenght = len(list)
    num_sample = int(training_size * lenght)
    _sample = random.sample(list[0].index.to_list(), num_sample)

    training = list.iloc[_sample]
    testing = list.iloc[~list.index.isin(_sample)]

    for i in training[0]:
        old = os.path.join(neg_path, i)
        new = os.path.join(train_path + 'neg/', i)
        shutil.copy(old, new)

    for i in testing[0]:
        old = os.path.join(neg_path, i)
        new = os.path.join(test_path + 'neg/', i)
        shutil.copy(old, new)

train_test_split(neg_path, 0.935)





import tensorflow as tf

num_images = len(os.listdir(train_path + 'neg/'))
img_height = 224
img_width = 224

df = []
for idx, filename in enumerate(os.listdir(train_path + 'neg/')):

    image_path = os.path.join(train_path + 'neg/', filename)
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize(image, size = [img_height,img_width])
    image = image.numpy()/255.
    # image = image[:,:, 0]

    df.append(image)

x_train = np.array(df).astype('float32')
x_test = x_train[int(num_images*0.9):,:,:,:]
x_train = x_train[0:int(num_images*0.9),:,:,:]

x_train.shape
x_test.shape


from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

class Denoise(Model):
  def __init__(self):
    super(Denoise, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(img_height, img_width, 1)),
      layers.Conv2D(112, (3,3), activation='relu', padding='same', strides=2),
      layers.Conv2D(56, (3,3), activation='relu', padding='same', strides=2),
      layers.Conv2D(28, (3,3), activation='relu', padding='same', strides=2),
      layers.Conv2D(14, (3,3), activation='relu', padding='same', strides=2)])

    self.decoder = tf.keras.Sequential([
      layers.Conv2DTranspose(14, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2DTranspose(28, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2DTranspose(56, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2DTranspose(112, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2D(1, kernel_size=(3,3), activation='sigmoid', padding='same')])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = Denoise()

autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

autoencoder.fit(x_train, x_train,
                epochs=10,
                shuffle=True,
                validation_data=(x_test, x_test))
                
