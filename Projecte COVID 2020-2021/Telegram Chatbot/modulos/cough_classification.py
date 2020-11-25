from modulos.yamnet_importation import *
from pydub import AudioSegment
from pyAudioAnalysis import audioTrainTest as aT

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio
from scipy.io import wavfile
from scipy import signal
import os
import json
import pandas as pd
import joblib

# Import the stacking classifier of Yamnet and SVM
stacking_classifier = joblib.load("C:/Users/Guillem/Desktop/Bot_Telegram/modulos/stacking_classifier.pkl")

def is_cough(file_path):
    wav_file_path = convert_to_wav(file_path)
    yamnet_veredict = yamnet_classifier(wav_file_path)
    svm_veredict = aT.file_classification(wav_file_path, "cough_classifier/svm_cough", "svm")
    svm_predict = svm_veredict[1][0]

    #accepted = [yamnet_veredict, svm_predict]
    #return accepted

    X_new = pd.DataFrame({'Yamnet':[yamnet_veredict], 'SVM': [svm_predict]})
    stacking_prediction = stacking_classifier.predict_proba(X_new)[:,1]
    optimal_threshold = 0.628

    if (stacking_prediction >= optimal_threshold):
        return True
    else:
        return False

def convert_to_wav(input_file):
    file_dir, filename = os.path.split(os.path.abspath(input_file))
    input_file_path = os.path.abspath(input_file)
    basename = filename.split('.')[0]
    output_file = os.path.join(file_dir, '{}.wav'.format(basename))

    ffmpeg_instruction = 'ffmpeg -y -i {} {}'.format(input_file_path,output_file)
    os.system(ffmpeg_instruction)
    return output_file


def yamnet_classifier(wav_file_path, visualization = False):
    sample_rate, wav_data = wavfile.read(wav_file_path)
    sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)

    waveform = wav_data / tf.int16.max
    waveform = tf.cast(waveform, tf.float32)

    try:
        if (np.shape(waveform)[1] == 2):
            waveform = np.mean(waveform, axis = 1)   # If the audio is stereo and not mono
    except Exception:
        pass

    # Run the model, check the output.
    scores, embeddings, spectrogram = model(waveform)

    scores_np = scores.numpy()
    spectrogram_np = spectrogram.numpy()
    infered_class = class_names[scores_np.mean(axis=0).argmax()]

    if (visualization):

        plt.figure(figsize=(10, 6))

        # Plot the waveform.
        plt.subplot(2, 1, 1)
        plt.plot(waveform)
        plt.xlim([0, len(waveform)])

        # Plot the log-mel spectrogram (returned by the model).
        plt.subplot(2, 1, 2)
        plt.imshow(spectrogram_np.T, aspect='auto', interpolation='nearest', origin='lower')


    if infered_class == 'Cough':
        # return True
        return 1
    else:
        # return False
        return 0
