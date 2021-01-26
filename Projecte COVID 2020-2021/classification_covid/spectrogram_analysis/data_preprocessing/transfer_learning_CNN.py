'''
Creation of TRAIN / TEST directories
'''

import os
import shutil
import numpy as np
import pandas as pd
import random

# Creation manually
basepath = 'C:/Users/Guillem/Desktop/HACKATHON 2020/Labeled audio/spectrograms/'
sp = os.path.join(basepath + 'spectrograms_pos/')
sn = os.path.join(basepath + 'spectrograms_neg/')

train_dir = basepath + 'train/'
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
    os.makedirs(train_dir + 'neg/')
    os.makedirs(train_dir + 'pos/')

test_dir = basepath + 'test/'
if not os.path.exists(test_dir):
    os.makedirs(test_dir)
    os.makedirs(test_dir + 'neg/')
    os.makedirs(test_dir + 'pos/')


def train_test_split(directory, training_size, pos = True):
    random.seed(12345)
    list = os.listdir(directory)
    list = pd.DataFrame(list)
    lenght = len(list)
    num_sample = int(training_size * lenght)
    _sample = random.sample(list[0].index.to_list(), num_sample)

    training = list.iloc[_sample]
    testing = list.iloc[~list.index.isin(_sample)]

    if pos:
        pos = '/pos/'
    else:
        pos = '/neg/'

    for i in training[0]:
        dir = directory.split('/')[-2]
        original_path = basepath + '/' + dir + '/' + '{}'.format(i)
        new_path = train_dir + pos + '{}'.format(i)
        shutil.move(original_path, new_path)

    for i in testing[0]:
        dir = directory.split('/')[-2]
        original_path = basepath + '/' + dir + '/' +'{}'.format(i)
        new_path = test_dir + pos + '{}'.format(i)
        shutil.move(original_path, new_path)


train_test_split(sp, training_size = 0.7, pos = True)
train_test_split(sn, training_size = 0.7, pos = False)

if len(os.listdir(sp)) == 0:
    shutil.rmtree(sp)

if len(os.listdir(sn)) == 0:
    shutil.rmtree(sn)



### Zip the folder and upload to Google Drive ---> Mount in Google Colab in order to compute everything with GPU

'''
Load dataset
'''

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image_dataset_from_directory

img_height, img_width, channels = 223, 223, 3
BATCH_SIZE = 32
IMG_SIZE = (img_height, img_width)

train_dir = 'C:/Users/Guillem/Desktop/HACKATHON 2020/Labeled audio/spectrograms/train/'
validation_dir = 'C:/Users/Guillem/Desktop/HACKATHON 2020/Labeled audio/spectrograms/test/'

train_dataset = image_dataset_from_directory(train_dir,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE)

validation_dataset = image_dataset_from_directory(validation_dir,
                                            shuffle=True,
                                            batch_size=BATCH_SIZE,
                                            image_size=IMG_SIZE)


class_names = train_dataset.class_names
len(class_names)

plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")


val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)

print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))


AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)


preprocess_input = tf.keras.applications.vgg16.preprocess_input

# Create the base model from the pre-trained model MobileNet V2
IMG_SHAPE = IMG_SIZE + (3,)


base_model = tf.keras.applications.VGG16(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')

image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print(feature_batch.shape)

base_model.trainable = False
base_model.summary()

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

prediction_layer = tf.keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

inputs = tf.keras.Input(shape=(img_height, img_width, channels))
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)


base_learning_rate = 0.01
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])


model.summary()


model.save('VGG16_model_covid.h5')
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('VGG16_model_covid.h5',save_best_only=True)

initial_epochs = 5
'''
loss0, accuracy0 = model.evaluate(validation_dataset)

print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))


history = model.fit(train_dataset,
                    epochs=150,
                    validation_data=validation_dataset,
                    callbacks=[checkpoint_cb, early_stopping_cb])
'''
history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=validation_dataset)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


base_model.trainable = True

# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 15

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
              metrics=['accuracy'])

model.summary()

initial_epochs = 3
fine_tune_epochs = 10
total_epochs =  initial_epochs + fine_tune_epochs

history_fine = model.fit(train_dataset,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         validation_data=validation_dataset)
