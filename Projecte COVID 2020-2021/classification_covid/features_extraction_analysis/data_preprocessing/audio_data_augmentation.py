import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import librosa
import librosa.display
import pylab
import os
import numpy as np
import tensorflow as tf
import random

def wav_to_spectrogram(wav_file_path, spectrograms_path, augment_data = False, times_augmented = 1, height = 224, width = 224):
    dpi = pylab.gcf().get_dpi()
    name_file = os.path.basename(wav_file_path).split('.')[0] + '.jpg'
    output_path = os.path.join(spectrograms_path, name_file)

    with open(wav_file_path, 'rb') as f:
        waveform, sample_rate = librosa.load(f, sr = None)
        duration = len(waveform)/sample_rate

        if (duration < 1):
            f.close()
            os.remove(wav_file_path)
            return

        fig, ax = plt.subplots(figsize=(height/dpi, width/dpi))
        M = librosa.feature.melspectrogram(y=waveform, sr=sample_rate)
        M_db = librosa.power_to_db(M, ref=np.max)
        # img = librosa.display.specshow(M_db, y_axis='mel', x_axis='time', ax=ax, cmap='gray_r')
        img = librosa.display.specshow(M_db, y_axis='mel', x_axis='time', ax=ax)
        ax.set_ylim([0,5000])
        plt.axis('off')

        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        plt.savefig(output_path, bbox_inches = 'tight',pad_inches = 0)
        plt.close(fig)

        if (augment_data == True):
            shape = M.shape
            M = np.reshape(M, (-1, shape[0], shape[1], 1))
            for i in range(times_augmented):

                name_file_augmented = os.path.basename(wav_file_path).split('.')[0] + '_augmented_{}.jpg'.format(i)
                output_path_augmented = os.path.join(spectrograms_path, name_file_augmented)

                mask_num = random.randint(1,5)
                freq_mask = random.randint(5,15)
                fm = frequency_masking(M, v = M.shape[0],frequency_masking_para=freq_mask, frequency_mask_num=mask_num)

                mask_num = random.randint(1,5)
                time_mask = random.randint(5,15)
                tm = time_masking(fm, tau=fm.shape[1], time_masking_para=time_mask, time_mask_num=mask_num)

                fig, ax = plt.subplots(figsize=(height/dpi, width/dpi))
                # img = librosa.display.specshow(librosa.power_to_db(tm[0, :, :, 0], ref=np.max), y_axis='mel', x_axis='time', cmap='gray_r')
                img = librosa.display.specshow(librosa.power_to_db(tm[0, :, :, 0], ref=np.max), y_axis='mel', x_axis='time')
                ax.set_ylim([0,5000])
                plt.axis('off')

                plt.gca().set_axis_off()
                plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
                plt.margins(0,0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())

                plt.savefig(output_path_augmented, bbox_inches = 'tight',pad_inches = 0)
                plt.close(fig)


                '''
                mask_num = random.randint(1,3)

                name_file_augmented_freq = os.path.basename(wav_file_path).split('.')[0] + '_augmented_freq_{}.jpg'.format(i)
                output_path_augmented_freq = os.path.join(spectrograms_path, name_file_augmented_freq)

                fm = frequency_masking(M, v = M.shape[0],frequency_masking_para=10, frequency_mask_num=mask_num)
                fig, ax = plt.subplots(figsize=(height/dpi, width/dpi))
                # img = librosa.display.specshow(librosa.power_to_db(fm[0, :, :, 0], ref=np.max), y_axis='mel', x_axis='time', cmap='gray_r')
                img = librosa.display.specshow(librosa.power_to_db(fm[0, :, :, 0], ref=np.max), y_axis='mel', x_axis='time')
                ax.set_ylim([0,5000])
                plt.axis('off')

                plt.gca().set_axis_off()
                plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
                plt.margins(0,0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())

                plt.savefig(output_path_augmented_freq, bbox_inches = 'tight',pad_inches = 0)
                plt.close(fig)

                name_file_augmented_time = os.path.basename(wav_file_path).split('.')[0] + '_augmented_time_{}.jpg'.format(i)
                output_path_augmented_time = os.path.join(spectrograms_path, name_file_augmented_time)

                tm = time_masking(M, tau=M.shape[1], time_masking_para=10, time_mask_num=mask_num)
                fig, ax = plt.subplots(figsize=(height/dpi, width/dpi))
                # img = librosa.display.specshow(librosa.power_to_db(tm[0, :, :, 0], ref=np.max), y_axis='mel', x_axis='time', cmap='gray_r')
                img = librosa.display.specshow(librosa.power_to_db(tm[0, :, :, 0], ref=np.max), y_axis='mel', x_axis='time')
                ax.set_ylim([0,5000])
                plt.axis('off')

                plt.gca().set_axis_off()
                plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
                plt.margins(0,0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())

                plt.savefig(output_path_augmented_time, bbox_inches = 'tight',pad_inches = 0)
                plt.close(fig)

                '''


def frequency_masking(mel_spectrogram, v, frequency_masking_para=27, frequency_mask_num=2):
    """Spec augmentation Calculation Function.
    'SpecAugment' have 3 steps for audio data augmentation.
    first step is time warping using Tensorflow's image_sparse_warp function.
    Second step is frequency masking, last step is time masking.
    # Arguments:
      mel_spectrogram(numpy array): audio file path of you want to warping and masking.
      frequency_masking_para(float): Augmentation parameter, "frequency mask parameter F"
        If none, default = 100 for LibriSpeech.
      frequency_mask_num(float): number of frequency masking lines, "m_F".
        If none, default = 1 for LibriSpeech.
    # Returns
      mel_spectrogram(numpy array): warped and masked mel spectrogram.
    """
    # Step 2 : Frequency masking
    fbank_size = tf.shape(mel_spectrogram)
    n, v = fbank_size[1], fbank_size[2]

    for i in range(frequency_mask_num):
        f = tf.random.uniform([], minval=0, maxval=frequency_masking_para, dtype=tf.int32)
        v = tf.cast(v, dtype=tf.int32)
        f0 = tf.random.uniform([], minval=0, maxval=v-f, dtype=tf.int32)

        # warped_mel_spectrogram[f0:f0 + f, :] = 0
        mask = tf.concat((tf.ones(shape=(1, n, v - f0 - f, 1)),
                          tf.zeros(shape=(1, n, f, 1)),
                          tf.ones(shape=(1, n, f0, 1)),
                          ), 2)
        mel_spectrogram = mel_spectrogram * mask
    return tf.cast(mel_spectrogram, dtype=tf.float32)


def time_masking(mel_spectrogram, tau, time_masking_para=100, time_mask_num=2):
    """Spec augmentation Calculation Function.
    'SpecAugment' have 3 steps for audio data augmentation.
    first step is time warping using Tensorflow's image_sparse_warp function.
    Second step is frequency masking, last step is time masking.
    # Arguments:
      mel_spectrogram(numpy array): audio file path of you want to warping and masking.
      time_masking_para(float): Augmentation parameter, "time mask parameter T"
        If none, default = 27 for LibriSpeech.
      time_mask_num(float): number of time masking lines, "m_T".
        If none, default = 1 for LibriSpeech.
    # Returns
      mel_spectrogram(numpy array): warped and masked mel spectrogram.
    """
    fbank_size = tf.shape(mel_spectrogram)
    n, v = fbank_size[1], fbank_size[2]

    # Step 3 : Time masking
    for i in range(time_mask_num):
        t = tf.random.uniform([], minval=0, maxval=time_masking_para, dtype=tf.int32)
        t0 = tf.random.uniform([], minval=0, maxval=tau-t, dtype=tf.int32)

        # mel_spectrogram[:, t0:t0 + t] = 0
        mask = tf.concat((tf.ones(shape=(1, n-t0-t, v, 1)),
                          tf.zeros(shape=(1, t, v, 1)),
                          tf.ones(shape=(1, t0, v, 1)),
                          ), 1)
        mel_spectrogram = mel_spectrogram * mask

    return tf.cast(mel_spectrogram, dtype=tf.float32)
