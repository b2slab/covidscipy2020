import numpy as np
import pandas as pd
import shutil
import os

'''
LABELED DATA
'''

basepath = 'C:/Users/Guillem/Desktop/HACKATHON 2020/'

labeled_path = basepath + 'Labeled audio/'
pos_path = labeled_path + 'Pos/'
neg_path = labeled_path + 'Neg/'
pos_asymp_path = labeled_path + 'Pos_asymp/'

'''
We can extract both cough audios recordings from each participant in a single directory.
'''

positives_path = labeled_path + 'Positives_audios/'
if not os.path.exists(positives_path):
    os.makedirs(positives_path)

for i in os.listdir(pos_path):

    participant_path = pos_path + i

    if ('cough-heavy.wav' in os.listdir(participant_path)):
        old_path = participant_path + '/cough-heavy.wav'
        new_path = positives_path + i + '_cough-heavy.wav'
        shutil.copy(old_path, new_path)

    if ('cough-shallow.wav' in os.listdir(participant_path)):
        old_path = participant_path + '/cough-shallow.wav'
        new_path = positives_path + i + '_cough-shallow.wav'
        shutil.copy(old_path, new_path)


negatives_path = labeled_path + 'Negatives_audios/'
if not os.path.exists(negatives_path):
    os.makedirs(negatives_path)

for i in os.listdir(neg_path):

    participant_path = neg_path + i

    if ('cough-heavy.wav' in os.listdir(participant_path)):
        old_path = participant_path + '/cough-heavy.wav'
        new_path = negatives_path + i + '_cough-heavy.wav'
        shutil.copy(old_path, new_path)

    if ('cough-shallow.wav' in os.listdir(participant_path)):
        old_path = participant_path + '/cough-shallow.wav'
        new_path = negatives_path + i + '_cough-shallow.wav'
        shutil.copy(old_path, new_path)


'''
Transform audios to spectrograms (images)
'''

from audio_data_augmentation import *

spectrograms_positives_path = positives_path + 'spectrograms/'
if not os.path.exists(spectrograms_positives_path):
    os.makedirs(spectrograms_positives_path)

onlyfiles_pos = [os.path.join(positives_path, f) for f in os.listdir(positives_path) if os.path.isfile(os.path.join(positives_path, f))]

for audio_path in onlyfiles_pos:
    wav_to_spectrogram(audio_path, spectrograms_positives_path, augment_data=True, times_augmented=2, height=448, width=448)


spectrograms_negatives_path = negatives_path + 'spectrograms/'
if not os.path.exists(spectrograms_negatives_path):
    os.makedirs(spectrograms_negatives_path)


onlyfiles_neg = [os.path.join(negatives_path, f) for f in os.listdir(negatives_path) if os.path.isfile(os.path.join(negatives_path, f))]

for audio_path in onlyfiles_neg:
    wav_to_spectrogram(audio_path, spectrograms_negatives_path, augment_data=False,height=448, width=448)





'''
Let's sample few negatives spectrograms in order to balance the dataset

We could do this or apply class-weight in order to balance the loss function
'''

import random

num_pos = len(os.listdir(spectrograms_positives_path))
num_neg = len(os.listdir(spectrograms_negatives_path))

selected_spectrograms = random.sample(population = os.listdir(spectrograms_negatives_path), k = num_pos)

selected_neg_path = spectrograms_negatives_path + 'selected/'
if not os.path.exists(selected_neg_path):
    os.makedirs(selected_neg_path)

for path in os.listdir(spectrograms_negatives_path):

    if (path in selected_spectrograms):
        old_path = os.path.join(spectrograms_negatives_path, path)
        new_path = os.path.join(selected_neg_path, path)
        shutil.copy(old_path, new_path)


##### Move manually to spectrograms folder

'''

wav_file_path = os.path.join(positives_path,os.listdir(positives_path)[1])


with open(wav_file_path, 'rb') as f:
    dpi = pylab.gcf().get_dpi()
    height = 224
    width = 224
    waveform, sample_rate = librosa.load(f, sr = None)

    fig, ax = plt.subplots(figsize=(height/dpi, width/dpi))
    M = librosa.feature.melspectrogram(y=waveform, sr=sample_rate)
    M_db = librosa.power_to_db(M, ref=np.max)
    # img = librosa.display.specshow(M_db, y_axis='mel', x_axis='time', ax=ax, cmap='gray_r')
    img = librosa.display.specshow(M_db, y_axis='mel', x_axis='time', ax=ax)
    ax.set_ylim([0,5000])
    plt.axis('off')
    #plt.savefig('C:/Users/Guillem/Desktop/prueba.jpg', bbox_inches='tight', dpi = dpi)

    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig('C:/Users/Guillem/Desktop/prueba.jpg', bbox_inches = 'tight',pad_inches = 0)

    plt.close()


'''




'''

import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import librosa
import librosa.display

spectrograms_positives_path = positives_path + 'spectrograms/'
if not os.path.exists(spectrograms_positives_path):
    os.makedirs(spectrograms_positives_path)

def wav_to_spectrogram(wav_file_path, spectrograms_path, height = 224, width = 224):
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
        img = librosa.display.specshow(M_db, y_axis='mel', x_axis='time', ax=ax, cmap='gray_r')
        ax.set_ylim([0,5000])
        plt.axis('off')
        plt.savefig(output_path)
        plt.close(fig)

onlyfiles_pos = [os.path.join(positives_path, f) for f in os.listdir(positives_path) if os.path.isfile(os.path.join(positives_path, f))]

for audio_path in onlyfiles_pos:
    wav_to_spectrogram(audio_path, spectrograms_positives_path)


spectrograms_negatives_path = negatives_path + 'spectrograms/'
if not os.path.exists(spectrograms_negatives_path):
    os.makedirs(spectrograms_negatives_path)


onlyfiles_neg = [os.path.join(negatives_path, f) for f in os.listdir(negatives_path) if os.path.isfile(os.path.join(negatives_path, f))]

for audio_path in onlyfiles_neg:
    wav_to_spectrogram(audio_path, spectrograms_negatives_path)






#AUDIO DATA AUGMENTATION
import librosa
import spec_augment_tensorflow


audio = os.listdir(positives_path)[1]
audio_path = os.path.join(positives_path,audio)

audio, sampling_rate = librosa.load(audio_path)
mel_spectrogram = librosa.feature.melspectrogram(y=audio,sr=sampling_rate)

shape = mel_spectrogram.shape
mel_spectrogram = np.reshape(mel_spectrogram, (-1, shape[0], shape[1], 1))

# Show Raw mel-spectrogram
spec_augment_tensorflow.visualization_spectrogram(mel_spectrogram=mel_spectrogram,
                                                      title="Raw Mel Spectrogram")


tt = spec_augment_tensorflow.frequency_masking(mel_spectrogram, v = mel_spectrogram.shape[0],frequency_masking_para=10, frequency_mask_num=1)
tf = spec_augment_tensorflow.time_masking(mel_spectrogram, tau=mel_spectrogram.shape[1], time_masking_para=10, time_mask_num=1)


fig = plt.figure(figsize=(224/72, 224/72))
img = librosa.display.specshow(librosa.power_to_db(tt[0, :, :, 0], ref=np.max), y_axis='mel', fmax=8000, x_axis='time', cmap='gray_r')
plt.tight_layout()
ax.set_ylim([0,5000])
#ax.set_adjustable('datalim')
plt.axis('off')


warped_masked_spectrogram = spec_augment_tensorflow.spec_augment(mel_spectrogram=mel_spectrogram)


# Show time warped & masked spectrogram
spec_augment_tensorflow.visualization_tensor_spectrogram(mel_spectrogram=spec_augment_tensorflow.spec_augment(mel_spectrogram),title="tensorflow Warped & Masked Mel Spectrogram")


warped_masked_spectrogram = spec_augment_tensorflow.spec_augment(mel_spectrogram=mel_spectrogram)






audio = os.listdir(positives_path)[125]
audio_path = os.path.join(positives_path,audio)

with open(audio_path, 'rb') as f:
    sample_rate, waveform = wavfile.read(f)
    frequencies, times, spectrogram = signal.spectrogram(waveform, sample_rate,nfft=1024)
    _spectrogram = np.array(pd.DataFrame(spectrogram).replace(0.0, value=10**-5))
    dBS = 10*np.log10(_spectrogram)

    plt.pcolormesh(times, frequencies, dBS, cmap = 'gray_r')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.ylim([0,10000])
    plt.show()

plt.specgram(waveform, sample_rate)

2**11


plt.specgram(waveform, sample_rate)





spectrograms_positives_path = positives_path + 'spectrograms/'
if not os.path.exists(spectrograms_positives_path):
    os.makedirs(spectrograms_positives_path)

def wav_to_spectrogram(wav_file_path, height = 224, width = 224):
    dpi = pylab.gcf().get_dpi()
    name_file = os.path.basename(wav_file_path).split('.')[0] + '.jpg'
    output_path = os.path.join(spectrograms_positives_path, name_file)

    with open(wav_file_path, 'rb') as f:
        waveform, sample_rate = librosa.load(f, sr = None)

        fig, ax = plt.subplots(figsize=(height/dpi, width/dpi))
        M = librosa.feature.melspectrogram(y=waveform, sr=sample_rate)
        M_db = librosa.power_to_db(M, ref=np.max)
        img = librosa.display.specshow(M_db, y_axis='mel', x_axis='time', ax=ax, cmap='gray_r')
        ax.set_ylim([0,5000])
        plt.axis('off')
        plt.savefig(output_path)


waveform, sample_rate = librosa.load(audio_path, sr = None)
D = librosa.stft(waveform)
S_db = librosa.amplitude_to_db(np.abs(D), ref = np.max)

fig, ax = plt.subplots()
img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax, cmap='gray_r')
ax.set(title='Now with labeled axes!')
fig.colorbar(img, ax=ax, format="%+2.f dB")
ax.set_ylim([0,4096])


dpi = pylab.gcf().get_dpi()
height = 224
width = 224

fig, ax = plt.subplots(figsize=(height/dpi, width/dpi))
M = librosa.feature.melspectrogram(y=waveform, sr=sample_rate)
M_db = librosa.power_to_db(M, ref=np.max)
img = librosa.display.specshow(M_db, y_axis='mel', x_axis='time', ax=ax, cmap='gray_r')
#img = librosa.display.specshow(M_db, ax=ax, cmap='gray_r')
#ax.set(title='Mel spectrogram display')
#fig.colorbar(img, ax=ax, format="%+2.f dB")
ax.set_ylim([0,5000])
plt.axis('off')
plt.savefig('C:/Users/Guillem/Desktop/prueba.jpg')

'''
