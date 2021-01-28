from pyAudioAnalysis import MidTermFeatures as mF
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import opensmile
import os

basepath = 'C:/Users/Guillem/Desktop/Anomaly detection (autoencoders)/audios/'
pos_path = basepath + 'Positives_audios/'
neg_path = basepath + 'Negatives_audios/'

basepath = 'C:/Users/Guillem/Desktop/Anomaly detection (autoencoders)/New Coswara data/'
pos_path = basepath + 'pos/'
neg_path = basepath + 'neg/'


def check_duration(directory):
    original_len = len(os.listdir(directory))
    for audio in os.listdir(directory):
        wav_file_path = os.path.join(directory, audio)

        with sf.SoundFile(wav_file_path) as f:
            duration = (len(f) / f.samplerate)

        if (duration < 1):
            os.remove(wav_file_path)

    remove_len = len(os.listdir(directory))
    print('In total, {} audios have been removed due to short duration'.format(original_len-remove_len))

check_duration(pos_path)
check_duration(neg_path)

[mid_term_features_pos, wav_file_list_pos, mid_feature_names] =  mF.directory_feature_extraction(pos_path, 0.5,0.5, 0.05, 0.05, compute_beat=False)
[mid_term_features_neg, wav_file_list_neg, mid_feature_names] =  mF.directory_feature_extraction(neg_path, 0.5,0.5, 0.05, 0.05, compute_beat=False)

filenames_pos = []
for file in wav_file_list_pos:
    filenames_pos.append(file.split('/')[-1].split('\\')[-1].split('.')[0])

filenames_neg = []
for file in wav_file_list_neg:
    filenames_neg.append(file.split('/')[-1].split('\\')[-1].split('.')[0])

df_pos = pd.DataFrame(mid_term_features_pos, columns = mid_feature_names)
df_pos['filename'] = filenames_pos
df_pos['label'] = np.ones(len(df_pos))

df_neg = pd.DataFrame(mid_term_features_neg, columns = mid_feature_names)
df_neg['filename'] = filenames_neg
df_neg['label'] = np.zeros(len(df_neg))

df = pd.concat([df_pos, df_neg], axis = 0, ignore_index=True)











from pyAudioAnalysis import audioBasicIO
from tqdm import tqdm

smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv01b,
    feature_level=opensmile.FeatureLevel.Functionals,
)

data = pd.DataFrame([])
for wave_file in tqdm(os.listdir(pos_path)):

    wav_file_path = os.path.join(pos_path,wave_file)
    sampling_rate, signal = audioBasicIO.read_audio_file(wav_file_path)
    signal = audioBasicIO.stereo_to_mono(signal)

    ps = smile.process_signal(signal,sampling_rate)
    ps = pd.DataFrame(ps).reset_index().iloc[:,2:]
    ps['filename'] = wave_file.split('.')[0]
    ps['label'] = np.ones(len(ps))

    data = pd.concat([data, ps])

data_pos = data.reset_index().iloc[:,1:]

data = pd.DataFrame([])
for idx, wave_file in enumerate(os.listdir(neg_path)):

    wav_file_path = os.path.join(neg_path,wave_file)
    sampling_rate, signal = audioBasicIO.read_audio_file(wav_file_path)
    signal = audioBasicIO.stereo_to_mono(signal)

    ps = smile.process_signal(signal,sampling_rate)
    ps = pd.DataFrame(ps).reset_index().iloc[:,2:]
    ps['filename'] = wave_file.split('.')[0]
    ps['label'] = np.zeros(len(ps))
    data = pd.concat([data, ps])

data_neg = data.reset_index().iloc[:,1:]
data = pd.concat([data_pos, data_neg], axis = 0, ignore_index=True)


### Let's merge both datasets

final_df = df.merge(data, on = ['filename', 'label'])
final_df.to_excel(basepath + 'features_extracted.xlsx')
