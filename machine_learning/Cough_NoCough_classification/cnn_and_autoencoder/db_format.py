# Load various imports
import pandas as pd
import os
import librosa
import librosa.display
import struct
import numpy as np

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split


class WavFileHelper:

    def __init__(self, db_dir):
        self.db_dir = db_dir
        metadata_path = os.path.join(db_dir, "metadata", "just_cough.csv")
        self.metadata = pd.read_csv(metadata_path)
        self.audio_files = self.get_files()
        self.le = None

    def get_files(self):
        files = []
        for index, row in self.metadata.iterrows():
            file_name = os.path.join(self.db_dir, 'audio', f'fold{row["fold"]}',
                                     str(row["slice_file_name"]))
            files.append(file_name)
        return files

    def data_properties_df(self):
        audiodata = []
        for file in self.audio_files:
            data = self.read_file_properties(file)
            audiodata.append(data)

        # Convert into a Panda dataframe
        return pd.DataFrame(audiodata,
                            columns=['num_channels', 'sample_rate', 'bit_depth'])

    def features_df(self):
        features = []

        # Iterate through each sound file and extract the features
        for index, row in self.metadata.iterrows():
            file_name = os.path.join(self.db_dir, 'audio', f'fold{row["fold"]}',
                                     str(row["slice_file_name"]))
            class_label = row["class"]
            data = self.extract_features(file_name)
            features.append([data, class_label])

        # Convert into a Panda dataframe
        return pd.DataFrame(features, columns=['feature', 'class_label'])

    def get_train_data(self, features_df):
        # features_df = pd.read_csv(features_df)

        X = np.array(features_df.feature.tolist())
        y = np.array(features_df.class_label.tolist())

        self.le = LabelEncoder()
        yy = to_categorical(self.le.fit_transform(y))
        self.num_labels = yy.shape[1]

        x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state=42)
        return yy, x_train, x_test, y_train, y_test


    def get_cough_train_data(self, features_df):
        # features_df = pd.read_csv(features_df)

        X = np.array(features_df.feature.tolist())
        y = np.array(features_df.class_label.tolist())

        self.le = LabelEncoder()
        yy = to_categorical(self.le.fit_transform(y))
        self.num_labels = yy.shape[1]

        x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state=42)
        return yy, x_train, x_test, y_train, y_test

    @staticmethod
    def read_file_properties(filename):
        wave_file = open(filename, "rb")

        riff = wave_file.read(12)
        fmt = wave_file.read(36)

        num_channels_string = fmt[10:12]
        num_channels = struct.unpack('<H', num_channels_string)[0]

        sample_rate_string = fmt[12:16]
        sample_rate = struct.unpack("<I", sample_rate_string)[0]

        bit_depth_string = fmt[22:24]
        bit_depth = struct.unpack("<H", bit_depth_string)[0]

        return num_channels, sample_rate, bit_depth

    @staticmethod
    def extract_features(file_name, max_pad_len=174):
        try:
            audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
            # mfccsscaled = np.mean(mfccs.T, axis=0)

        except Exception as e:
            # import code
            # code.interact(local=locals())
            print("Error encountered while parsing file: ", file_name)
            return None

        return mfccs

