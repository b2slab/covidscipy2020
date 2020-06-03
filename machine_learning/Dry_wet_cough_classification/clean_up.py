#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 07:05:49 2020

@author: andrine
"""

import os
from pydub import AudioSegment


# Clean up the files in a dataset

def rename_files():
    cwd = os.getcwd()
    data_dir = cwd + '/cough_data/'
    
    for sub_dir in os.listdir(data_dir):
        i = 0
        for cough in os.listdir(data_dir + sub_dir):
            path = (data_dir + sub_dir)
            new_name = path + '/' + sub_dir+str(i)+'.wav'
            i = i+1
            old_name = path + '/' + cough
            os.rename(old_name,new_name)



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

os.makedirs(os.getcwd() + '/test')