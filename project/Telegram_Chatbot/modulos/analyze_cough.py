from pydub import AudioSegment
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import MidTermFeatures
import numpy as np
import pandas as pd
import opensmile
import warnings
import joblib
import os

# Analyze audio cough
def analyze_cough(ogg_path, data):
    wav_path = convert_ogg_to_wav(ogg_path)
    X_new = mid_term_feat_extraction(wav_path)

    cough = cough_prediction(X_new)
    if cough == True:
        metadata = convert_metadata(data)
        covid = covid_prediction(X_new, metadata)
        if covid == True:
            return True
        else:
            return False
    else:
        return None

# Convertion from .ogg to .wav
def convert_ogg_to_wav(original_path):
    converted_path = os.path.splitext(original_path)[0] + '.wav'
    audio = AudioSegment.from_ogg(original_path)
    audio.export(converted_path, format="wav")
    return converted_path

# Extract mid-term features from wav
def mid_term_feat_extraction(wav_file_path):

    sampling_rate, signal = audioBasicIO.read_audio_file(wav_file_path)
    if sampling_rate == 0:
        print('Sampling rate not correct.')
        return None

    signal = audioBasicIO.stereo_to_mono(signal)
    if signal.shape[0] < float(sampling_rate)/5:
        print("The duration of the audio is too short.")
        return None

    mid_window, mid_step, short_window, short_step = 0.5, 0.5, 0.05, 0.05
    mid_features, _, mid_feature_names = MidTermFeatures.mid_feature_extraction(signal, sampling_rate,
                                                                                round(mid_window * sampling_rate),
                                                                                round(mid_step * sampling_rate),
                                                                                round(sampling_rate * short_window),
                                                                                round(sampling_rate * short_step))
    mid_features = np.transpose(mid_features)
    mid_features = mid_features.mean(axis=0)
    # long term averaging of mid-term statistics
    if (not np.isnan(mid_features).any()) and (not np.isinf(mid_features).any()):
        #print('Mid-Terms features extracted correctly.')
        mid_dict = dict(zip(mid_feature_names,mid_features))
        mid_df = pd.DataFrame([mid_dict.values()], columns = mid_dict.keys())

        # Smile library audio extraction
        smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv01b,
                                feature_level=opensmile.FeatureLevel.Functionals,)
        smile_features = smile.process_signal(signal, sampling_rate)
        smile_df = pd.DataFrame(smile_features).reset_index().iloc[:,2:]

        final_df = pd.concat([mid_df, smile_df], axis=1)

        #excel_path = wav_file_path.strip('.') + 'features_extracted.xlsx'
        #final_df.to_excel(excel_path)
        return final_df
    else:
        #print('Mid-Terms features extracted incorrectly.')
        return None


# Load the cough recognition model and predict whether the audio is cough
def cough_prediction(X_new, opt_thresh = 0.6953):
    # Load the cough recognition model
    joblib_file = "/app/project/Telegram_Chatbot/modulos/random_forest_classifier.pkl"
    #joblib_file = "C:/Users/Guillem/Desktop/Bot_Telegram/classification_covid/predict_cough_covid/cough_nocough/gradient_boosting_classifier.pkl"
    cough_classifier = joblib.load(joblib_file)

    # Predict if audio is cough
    y_pred = cough_classifier.predict_proba(X_new)[:,1]

    if y_pred >= opt_thresh:
        #print('The audio has been recognised as COUGH')
        return True
    else:
        #print('The audio has been recognised as NO COUGH')
        return False

# Convert the data extracted by chatbot into input metadata for the model
def convert_metadata(data):
    metadata_keys = ['age','gender_female','gender_male','asthma_True','cough_True',
                    'smoker_True','hypertension_True','cold_True','diabetes_True',
                    'ihd_True','bd_True','st_True','fever_True','ftg_True','mp_True',
                    'loss_of_smell_True','pneumonia_True','diarrhoea_True','cld_True']

    metadata_values = []
    metadata_values.append(data['age'])

    if (data['gender'] == 'Female') | (data['gender'] == 'Mujer') | (data['gender'] == 'Dona'):
        metadata_values.append(1) # Gender_female = True
        metadata_values.append(0) # Gender_male = False
    else:
        metadata_values.append(0) # Gender_female = True
        metadata_values.append(1) # Gender_male = False

    metadata_values.append(int(data['symptoms']['asthma']))
    metadata_values.append(int(data['symptoms']['dry cough']))
    metadata_values.append(int(data['symptoms']['smoker']))
    metadata_values.append(int(data['symptoms']['hypertension']))
    metadata_values.append(int(data['symptoms']['cold']))
    metadata_values.append(int(data['symptoms']['diabetes']))
    metadata_values.append(int(data['symptoms']['IHD']))
    metadata_values.append(int(data['symptoms']['res_difficult']))
    metadata_values.append(int(data['symptoms']['sore_throat']))
    metadata_values.append(int(data['symptoms']['fever']))
    metadata_values.append(int(data['symptoms']['fatigue']))
    metadata_values.append(int(data['symptoms']['muscular_pain']))
    metadata_values.append(int(data['symptoms']['smell_loss']))
    metadata_values.append(int(data['symptoms']['pneumonia']))
    metadata_values.append(int(data['symptoms']['diarrhea']))
    metadata_values.append(int(data['symptoms']['CLD']))

    metadata = dict(zip(metadata_keys, metadata_values))
    metadata_df = pd.DataFrame([metadata.values()], columns = metadata.keys())
    return metadata_df


# Predict if cough audio is POSTIVE in COVID-19
def covid_prediction(X_new, metadata, optimal_threshold = 0.8):

    # optimal_threshold = 0.2397
    X = pd.concat([X_new, metadata], axis = 1)

    # Load the model
    warnings.filterwarnings("ignore")

    # Load the model
    joblib_file = "/app/project/Telegram_Chatbot/modulos/extratree_classifier.pkl"
    # joblib_file = "C:/Users/Guillem/Desktop/Bot_Telegram/classification_covid/predict_cough_covid/extratree_classifier.pkl"
    extratree_classifier = joblib.load(joblib_file)

    # Predictions
    prediction = extratree_classifier.predict_proba(X_new)[:,1]
    if prediction >= optimal_threshold:
        #print('Cough POSITIVE in COVID-19')
        return True
    else:
        #print('Cough NEGATIVE in COVID-19')
        return False

'''
def wav_to_binary(oga_path):
    wav_path = os.path.splitext(oga_path)[0] + '.wav'
    sampling_rate, signal = audioBasicIO.read_audio_file(wav_path)
    # dumped = pickle.dumps(signal, protocol=2)
    dumped = signal.tobytes()
    return dumped, sampling_rate
'''

def check_audio_duration(filepath):
    audio = AudioSegment.from_ogg(filepath)
    duration = audio.duration_seconds
    return duration
