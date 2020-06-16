import os

import soundfile as sf
import tensorflow as tf
import numpy as np

import machine_learning.Cough_NoCough_classification.yamnet.yamnet as yamnet_model
from machine_learning.Cough_NoCough_classification.yamnet import params

YAMNET_DIR = os.path.join("machine_learning", "Cough_NoCough_classification", "yamnet")


def classify(file):
    params.PATCH_HOP_SECONDS = 0.1  # 10 Hz scores frame rate.
    wav_data, sr = sf.read(file, dtype=np.int16)
    params.SAMPLE_RATE = sr
    waveform = wav_data / 32768.0
    class_names = yamnet_model.class_names(os.path.join(YAMNET_DIR, 'yamnet_class_map.csv'))
    params.PATCH_HOP_SECONDS = 0.1  # 10 Hz scores frame rate.
    graph = tf.Graph()
    with graph.as_default():
        yamnet = yamnet_model.yamnet_frames_model(params)
        yamnet.load_weights(os.path.join(YAMNET_DIR, 'yamnet.h5'))
    with graph.as_default():
        scores, spectrogram = yamnet.predict(np.reshape(waveform, [1, -1]), steps=1)
    mean_scores = np.mean(scores, axis=0)
    top_N = 10
    top_class_indices = np.argsort(mean_scores)[::-1][:top_N]
    return class_names[top_class_indices]
