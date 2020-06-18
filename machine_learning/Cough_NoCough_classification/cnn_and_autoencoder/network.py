from datetime import datetime

import numpy as np
from keras import Model, Input

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, UpSampling2D
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop


class NN:
    def __init__(self, num_labels, num_rows=40, num_columns=174, num_channels=1, filter_size=2):
        self.train_set = []
        self.test_set = []

        self.num_rows = num_rows
        self.num_columns = num_columns
        self.num_channels = num_channels
        self.filter_size = filter_size
        self.num_labels = num_labels

    def add_data(self, x_train, x_test, y_train, y_test):
        # import code
        # code.interact(local=locals())
        x_train = x_train.reshape(x_train.shape[0], self.num_rows, self.num_columns, self.num_channels)
        x_test = x_test.reshape(x_test.shape[0], self.num_rows, self.num_columns, self.num_channels)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def build_nn(self):
        # Construct model
        self.model = Sequential()
        self.model.add(
            Conv2D(filters=16, kernel_size=2, input_shape=(self.num_rows, self.num_columns, self.num_channels), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=2))
        self.model.add(Dropout(0.2))

        self.model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
        self.model.add(MaxPooling2D(pool_size=2))
        self.model.add(Dropout(0.2))

        self.model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
        self.model.add(MaxPooling2D(pool_size=2))
        self.model.add(Dropout(0.2))

        self.model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
        self.model.add(MaxPooling2D(pool_size=2))
        self.model.add(Dropout(0.2))
        self.model.add(GlobalAveragePooling2D())

        self.model.add(Dense(self.num_labels, activation='softmax'))

    def encode(self):
        # encoder
        # input = 28 x 28 x 1 (wide and thin)
        input_shape = Input(self.num_rows, self.num_columns, self.num_channels)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_shape)  # 28 x 28 x 32
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 14 x 14 x 32
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)  # 14 x 14 x 64
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 7 x 7 x 64
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)  # 7 x 7 x 128 (small and thick)
        conv3 = BatchNormalization()(conv3)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        conv3 = BatchNormalization()(conv3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)  # 7 x 7 x 256 (small and thick)
        conv4 = BatchNormalization()(conv4)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        self.encoder = BatchNormalization()(conv4)
        return self.encoder

    def decode(self):
        # decoder
        conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(self.encoder)  # 7 x 7 x 128
        conv5 = BatchNormalization()(conv5)
        conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
        conv5 = BatchNormalization()(conv5)
        conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)  # 7 x 7 x 64
        conv6 = BatchNormalization()(conv6)
        conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)
        conv6 = BatchNormalization()(conv6)
        up1 = UpSampling2D((2, 2))(conv6)  # 14 x 14 x 64
        conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1)  # 14 x 14 x 32
        conv7 = BatchNormalization()(conv7)
        conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)
        conv7 = BatchNormalization()(conv7)
        up2 = UpSampling2D((2, 2))(conv7)  # 28 x 28 x 32
        self.decoder = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2)  # 28 x 28 x 1
        return self.decoder

    def compile_autoencoder(self):
        input_shape = Input(self.num_rows, self.num_columns, self.num_channels)
        self.autoencoder = Model(input_shape, self.decode(self.encode()))

        # Compile the model
        self.autoencoder.compile(loss='mean_squared_error', optimizer=RMSprop())

        # Display self.model architecture summary
        self.autoencoder.summary()

        # Calculate pre-training accuracy
        score = self.autoencoder.evaluate(self.x_test, self.y_test, verbose=1)
        accuracy = 100 * score[1]

        print("Pre-training accuracy: %.4f%%" % accuracy)

    def compile_nn(self):
        # Compile the model
        self.model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

        # Display self.model architecture summary 
        self.model.summary()

        # Calculate pre-training accuracy 
        score = self.model.evaluate(self.x_test, self.y_test, verbose=1)
        accuracy = 100 * score[1]

        print("Pre-training accuracy: %.4f%%" % accuracy) 

    def train_autoencoder(self, epochs=200, batch_size=64):
        checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.autoencoder_cough.hdf5',
                                       verbose=1, save_best_only=True)

        start = datetime.now()

        self.autoencoder.fit(self.x_train, self.y_test, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(self.x_test, self.y_test),
                             callbacks=[checkpointer])

        duration = datetime.now() - start
        print("Training completed in time: ", duration)

    def train_nn(self, epochs=72, batch_size=256):

        checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.cnn_cough.hdf5',
                                       verbose=1, save_best_only=True)
        start = datetime.now()

        self.model.fit(self.x_train, self.y_train, batch_size=batch_size, epochs=epochs, validation_data=(self.x_test, self.y_test),
                       callbacks=[checkpointer], verbose=1)

        duration = datetime.now() - start
        print("Training completed in time: ", duration)

    def evaluate_nn(self):
        # Evaluating the model on the training and testing set
        score = self.model.evaluate(self.x_train, self.y_train, verbose=0)
        print("Training Accuracy: ", score[1])

        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print("Testing Accuracy: ", score[1])

    def predict_label(self, label_encoder, features):
        features = features.reshape(1, self.num_rows, self.num_columns, self.num_channels)

        pred_vector = self.model.predict_classes(features)
        pred_class = label_encoder.inverse_transform(pred_vector)
        print(f"The predicted class is {pred_class}")

        pred_prob_vector = self.model.predict_proba(features)
        pred_prob = pred_prob_vector[0]
        for i in range(len(pred_prob)):
            category = label_encoder.inverse_transform(np.array([i]))
            print(f"{category[0]} \t {format([pred_prob[i], '.32f'])} ")

    def load_model(self, file):
        self.build_nn()
        self.model.load_weights(file)
