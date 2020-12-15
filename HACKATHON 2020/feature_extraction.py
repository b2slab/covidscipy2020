'''
TABULAR DATA -- FEATURE EXTRACTION
The pyAudioAnalysis has two functions in order to extract a bunch of useful
features from a wav file.
'''

from pyAudioAnalysis import MidTermFeatures as mF
import numpy as np
import pandas as pd
import os

basepath_train_cough = 'C:/Users/Guillem/Desktop/HACKATHON 2020/Unlabeled audio/TRAIN/Cough/'
basepath_train_nocough = 'C:/Users/Guillem/Desktop/HACKATHON 2020/Unlabeled audio/TRAIN/No_Cough/'

[mid_term_features_cough, wav_file_list_cough, mid_feature_names] =  mF.directory_feature_extraction(basepath_train_cough, 0.1,0.1, 0.01, 0.01, compute_beat=False)
[mid_term_features_nocough, wav_file_list_nocough, mid_feature_names] =  mF.directory_feature_extraction(basepath_train_nocough, 0.1,0.1, 0.01, 0.01, compute_beat=False)

label_nocough = np.zeros(np.shape(mid_term_features_nocough)[0])
label_cough = np.ones(np.shape(mid_term_features_cough)[0])

features = np.concatenate((mid_term_features_cough, mid_term_features_nocough))  # Equivalent to rbind() in R
labels = np.concatenate((label_cough, label_nocough))
mid_feature_names = np.array(mid_feature_names)

df = pd.DataFrame(features, columns = mid_feature_names)
df['Label'] = pd.Series(labels)

df.to_csv('C:/Users/Guillem/Desktop/HACKATHON 2020/Unlabeled audio/TRAIN/features_extracted.csv', index=False, header=True)
 
df = pd.read_csv('C:/Users/Guillem/Desktop/HACKATHON 2020/Unlabeled audio/TRAIN/features_extracted.csv')
print(df)

df['Label'].groupby(df['Label']).count()
