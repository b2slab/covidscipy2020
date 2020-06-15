#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 07:05:49 2020

@author: andrine
"""

import os
import numpy as np
from pydub import AudioSegment
from pyAudioAnalysis import MidTermFeatures as mF 
from pyAudioAnalysis import ShortTermFeatures as sF 
from pyAudioAnalysis import audioBasicIO
import matplotlib.pyplot as plt
import shutil
import math
from statistics import mean 
from statistics import median 

'''
For creating spektogram of signal, and save
'''
import librosa
import librosa.display
import skimage
from PIL import Image


cwd = os.getcwd()
data_dir = cwd + '/cough_data/'
trimmed_data_dir = cwd + '/trimmed_cough_data/'

'''
Cleaning up the filenames in the original untrimmed folders, to have all filenames on the same format
'''
def rename_files():
    for sub_dir in os.listdir(data_dir):
        i = 0
        for cough in os.listdir(data_dir + sub_dir):
            path = (data_dir + sub_dir)
            new_name = path + '/' + sub_dir+str(i)+'.wav'
            i = i+1
            old_name = path + '/' + cough
            os.rename(old_name,new_name)

'''
Detecting silence in the start of a audio-file in ms
'''
def detect_silence(sound, silence_threshold=-50.0, chunk_size=10):
    '''
    sound is a pydub.AudioSegment
    silence_threshold in dB
    chunk_size in ms

    iterate over chunks until you find the first one with sound
    '''
    trim_ms = 0 # ms

    assert chunk_size > 0 # to avoid infinite loop
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size

    return trim_ms

def trim(sound):
    #sound = AudioSegment.from_file(file, format="wav")
    
    start_trim = detect_silence(sound)
    end_trim = detect_silence(sound.reverse())
    
    duration = len(sound)    
    trimmed_sound = sound[start_trim:duration-end_trim]
    return trimmed_sound

def trim_files():
    wet = []
    dry = []
    wet_name = []
    dry_name = []
    for sub_dir in os.listdir(data_dir):
        i = 0
        for cough in os.listdir(data_dir + sub_dir):
            path = (data_dir + sub_dir)
            new_path  = (cwd + '/trimmed_cough_data/' + sub_dir)
            new_name = new_path + '/' + 'trimmed_' + sub_dir+str(i)
            i = i+1
            
            sound = AudioSegment.from_file((path + '/' + cough), format="wav")
            test_file = trim(sound)
            if sub_dir == 'dry':
                dry.append(test_file)
                dry_name.append(new_name)
            elif sub_dir == 'wet':
                wet.append(test_file)
                wet_name.append(new_name)
            #test_file.export(new_name, format="wav")
    return wet, dry, wet_name, dry_name    
 
def split_sound(sound, name , mini):
    new_sounds = []
    new_names = []
    length = len(sound)
    if (length > 2*mini):
        num_splits = math.floor(length/mini) 
        for i in range(num_splits - 1):
            if i == (num_splits-1):
               new_sound = sound[i*mini:-1] 
            else:
                new_sound = sound[i*mini:(i + 1)*mini]

            new_sounds.append(new_sound)
            new_names.append((name + '_' + str(i)))
    else:
        new_sounds = sound
        new_names = name
    return new_sounds, new_names  
     
def export_timmed_files(split = None):
    print('Split setting: ',split)
    remove_trimmed_data()
    wet,dry, wet_name, dry_name = trim_files()

    stats = get_stats_length(wet,dry)
    
    new_sounds = []
    new_names = []
    
    if split == 'mini': 
        split_on = stats[MINIMUM]
    elif split == 'mean' :
        split_on = stats[MEAN]
    elif split == 'median' :
        split_on = stats[MEDIAN]
        
        
    if split: 
        for sound, name in zip((wet + dry), (wet_name + dry_name)): 
            new_sound, new_name = split_sound(sound, name, split_on)
            
            if type(new_sound) == list:
                new_sounds = new_sounds + new_sound
                new_names = new_names + new_name
            else:
                new_sounds.append(new_sound)
                new_names.append(new_name)
     
        for sound, name in zip(new_sounds, new_names): 
            sound = trim(sound)
            
            if len(sound) >= stats[MINIMUM]:
                sound.export(name+'.wav', format="wav")
    else:
        for sound, name in zip((wet + dry), (wet_name + dry_name)): 
            sound.export(name+'.wav', format="wav")
    
    
    
  

def scale_minmax(X, mini=0.0, maxi=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (maxi - mini) + mini
    return X_scaled

def spectrogram_image(y, sr,  hop_length, n_mels):
    # use log-melspectrogram
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                            n_fft=hop_length*2, hop_length=hop_length)
    mels = np.log(mels + 1e-9) # add small number to avoid log(0)

    # min-max scale to fit inside 8-bit range
    img = scale_minmax(mels, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0) # put low frequencies at the bottom in image
    img = 255-img # invert. make black==more energy
    return img


def get_img(file,maxi, padd = True):
    [sampling_rate, signal] = audioBasicIO.read_audio_file(file)
    signal = audioBasicIO.stereo_to_mono(signal)
    signal = signal.astype('float64') 
    if padd:
        signal = add_padding(signal,maxi)  # Will add a padding so that all images matches 
    #plt.plot(signal)
 
    X = librosa.stft(signal)
    Xdb = librosa.amplitude_to_db(abs(X))
    
    #fig = plt.figure(figsize=(14, 5))
    #spec = librosa.display.specshow(Xdb, sr=sampling_rate, x_axis='time', y_axis='log')
    #plt.colorbar()
    
    
    hop_length = 512 # number of samples per time-step in spectrogram
    n_mels = 128 # number of bins in spectrogram. Height of image
    time_steps = 384 # number of time-steps. Width of image


    img = spectrogram_image(signal, sr=sampling_rate, hop_length=hop_length, n_mels=n_mels)
    print(np.shape(img))

    pil_img = Image.fromarray(img)
    #im.save("your_file1.jpeg")
    #display(pil_img)

    return pil_img

def save_images(maxi, padd = True):
    remove_spec_images()
    path = cwd + '/spectrogram'
    
    for sub_dir in os.listdir(trimmed_data_dir):
        for cough in os.listdir(trimmed_data_dir + '/' +  sub_dir):
            file_path = trimmed_data_dir + '/' +  sub_dir + '/' + cough
            img = get_img(file_path, maxi, padd)
            
            name = cough.split('.')[0] + '.jpeg'
            destination = ''
            if sub_dir == 'dry':
                
                destination = path + '/dry/' + name
            else:
                destination = path + '/wet/' + name
            img.save(destination, "JPEG", quality=80, optimize=True, progressive=True)


def load_data(folder):
    dry = []
    wet = []
    data_path = cwd + '/' + folder
    for sub_dir in os.listdir(data_path):
        for cough in os.listdir(data_path + '/' +  sub_dir):
            file_path = data_path + '/' +  sub_dir + '/' + cough
            [sampling_rate, signal] = audioBasicIO.read_audio_file(file_path)
            signal = audioBasicIO.stereo_to_mono(signal)
            signal = signal.astype('float64') 
            if sub_dir == 'dry':
                dry.append(signal)
            else:
                wet.append(signal)
    wet = np.asarray(wet)
    dry = np.asarray(dry)
    return wet, dry

MINIMUM = 0
MAXIMUM = 1
MEAN = 2
MEDIAN = 3
def get_stats_length(wet,dry):
    tot = np.append(wet,dry)
    maxi = max([len(elm) for elm in tot])
    mini = min([len(elm) for elm in tot])
    me = mean([len(elm) for elm in tot])
    med = median([len(elm) for elm in tot])
    return [mini, maxi, me, med]

def add_padding(signal, maxi):
    if len(signal)%2 !=0:
        signal = signal[:-1]
        
    if maxi%2 != 0:
        maxi = maxi -1 #If the max length is odd
    
    diff = maxi - len(signal)
    pad = int(diff/2)
    
    return np.pad(signal,pad_width = pad, mode = 'maximum')

def padding_all_signals(maxi):
    new_signal_dry= []
    for signal in dry: 
        new = add_padding(signal, maxi)
        new_signal_dry.append(new)
        plt.plot(new)
    return np.asarray(new_signal_dry)
  
def remove_spec_images():
    path = cwd + '/spectrogram/'
    for f in os.listdir(path):
        shutil.rmtree(path + f)   
    os.mkdir(path + 'dry/')
    os.mkdir(path + 'wet/')
    
def remove_trimmed_data():
    path = cwd + '/trimmed_cough_data/'
    for f in os.listdir(path):
        shutil.rmtree(path + f)   
    os.mkdir(path + 'dry/')
    os.mkdir(path + 'wet/')

        
if __name__=="__main__":
    #export_timmed_files('mini') # Split on selections: 'mini', 'mean', 'median'
    
    wet, dry = load_data('trimmed_cough_data')
    stats = get_stats_length(wet,dry)
    
    save_images(stats[MAXIMUM], False)
    
 
