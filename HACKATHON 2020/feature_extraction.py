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

filenames_cough = []
for i in range(len(wav_file_list_cough)):
    filenames_cough.append(os.path.split(os.path.abspath(wav_file_list_cough[i]))[1].split('.')[0])

filenames_nocough = []
for i in range(len(wav_file_list_nocough)):
    filenames_nocough.append(os.path.split(os.path.abspath(wav_file_list_nocough[i]))[1].split('.')[0])

filenames = np.concatenate((filenames_cough, filenames_nocough))

df = pd.DataFrame(features, columns = mid_feature_names)
df['Label'] = pd.Series(labels)
df['Filenames'] = pd.Series(filenames)

df.to_excel('C:/Users/Guillem/Desktop/HACKATHON 2020/Unlabeled audio/TRAIN/features_extracted.xlsx', index=False, header=True)

df = pd.read_excel('C:/Users/Guillem/Desktop/HACKATHON 2020/Unlabeled audio/TRAIN/features_extracted.xlsx')
print(df)

df['Label'].groupby(df['Label']).count()



'''
Let's extract the features from some Positive cough audios.
We know in advance that some Cough-Shallow audios have too short duration
'''

import soundfile as sf
basepath = 'C:/Users/Guillem/Desktop/HACKATHON 2020/Labeled audio/Positives_audios/'

original_len = len(os.listdir(basepath))
for i in os.listdir(basepath):

    wav_file_path = os.path.join(basepath, i)

    with sf.SoundFile(wav_file_path) as f:
        duration = (len(f) / f.samplerate)

    if (duration < 1):
        os.remove(wav_file_path)

remove_len = len(os.listdir(basepath))
print('In total, {} audios have been removed due to short duration'.format(original_len-remove_len))


# Let's extract the features

[mid_term_features_pos, wav_file_list_pos, mid_feature_names] =  mF.directory_feature_extraction(basepath, 0.5,0.5, 0.05, 0.05, compute_beat=False)

label = np.ones(np.shape(mid_term_features_pos)[0])
features = np.array(mid_term_features_pos)
np.shape(features)

mid_feature_names = np.array(mid_feature_names)

filenames_pos = []
for i in range(len(wav_file_list_pos)):
    filenames_pos.append(os.path.split(wav_file_list_pos[i])[1].split('_')[0])

filenames_pos = np.array(filenames_pos)

df = pd.DataFrame(features, columns = mid_feature_names)
df['Label'] = pd.Series(label)
df['patient_id'] = pd.Series(filenames_pos)

'''
INNER JOIN OF BOTH TABLES
'''

metadata = pd.read_excel('C:/Users/Guillem/Desktop/HACKATHON 2020/Labeled audio/metadata.xlsx')

result = pd.merge(df,metadata,how='inner')
result.to_excel(basepath+'features_extracted.xlsx')
