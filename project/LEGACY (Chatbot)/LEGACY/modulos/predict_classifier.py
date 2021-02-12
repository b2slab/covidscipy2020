from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import MidTermFeatures
import numpy as np
import pandas as pd
import opensmile
import warnings
import joblib
import os


'''
HOW TO RUN THE FUNCTIONS
'''

# wav_file_path = 'C:/Users/Guillem/Desktop/predict_cough_covid/cough_samples/AwACAgQAAxkBAAIL3V_ToANs-f0tr9J1pNlYun2ZFz1YAAL8CAAC-eqgUkEigE4AAVoCsB4E.wav'
# directory_model = 'C:/Users/Guillem/Desktop/predict_cough_covid/'

# predict = extratree_prediction(wav_file_path, directory_model)
# predict


'''
DEFINITION OF FUNCTIONS
'''

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
        print('Mid-Terms features extracted correctly.')
    else:
        print('Mid-Terms features extracted incorrectly.')
        return None

    mid_dict = dict(zip(mid_feature_names,mid_features))
    mid_df = pd.DataFrame([mid_dict.values()], columns = mid_dict.keys())

    # Smile library audio extraction
    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv01b,
                            feature_level=opensmile.FeatureLevel.Functionals,)
    smile_features = smile.process_signal(signal, sampling_rate)
    smile_df = pd.DataFrame(smile_features).reset_index().iloc[:,2:]

    final_df = pd.concat([mid_df, smile_df], axis=1)
    # final_df.to_excel('features_extracted.xlsx')
    return final_df


'''
METADATA from PATIENTS

Note: By the moment, some of the boolean variables are not implemented yet in the Chatbot.

age -- edad del paciente
gender_female -- 1 si el paciente es mujer / 0 si es hombre
gender_male -- 1 si el paciente es hombre / 0 si es mujer
asthma_True -- 1 si el paciente sufre asma
cough_True -- 1 si el paciente sufre tos seca
smoker_True -- 1 si el paciente es fumador
hypertension_True -- 1 si el paciente sufre de hipertensión
cold_True -- 1 si el paciente sufre un catarro / resfriado
diabetes_True -- 1 si el paciente tiene algún tipo de diabetes
ihd_True -- 1 si el paciente sufre Ischemic Heart Disease (enfermedad de las arterias coronarias)
bd_True -- 1 si el paciente sufre dificultat respiratoria
st_True -- 1 si el paciente sufre sore throat (dolor de garganta)
fever_True -- 1 si el paciente tiene fiebre
ftg_True -- 1 si el paciente sufre fatiga
mp_True -- 1 si el paciente sufre dolor muscular
loss_of_smell_True -- 1 si el paciente sufre pérdida de olfato
pneumonia_True -- 1 si el paciente sufre neumonia
diarrhoea_True -- 1 si el paciente sufre diarrea
cld_True -- 1 si el paciente sufre Chronic Lung Disease (enfermedad crónica pulmonar)
'''

def random_metadata_generator():
    metadata_keys = ['age','gender_female','gender_male','asthma_True','cough_True',
                    'smoker_True','hypertension_True','cold_True','diabetes_True',
                    'ihd_True','bd_True','st_True','fever_True','ftg_True','mp_True',
                    'loss_of_smell_True','pneumonia_True','diarrhoea_True','cld_True']

    metadata_values = []
    age = int(np.random.normal(loc=30,scale=7,size=1).item())
    metadata_values.append(age)
    gender_female = np.random.choice(np.arange(0,2), size=1, p=[0.5, 0.5]).item()
    metadata_values.append(gender_female)
    if gender_female == 0:
        gender_male = 1
    else:
        gender_male = 0
    metadata_values.append(gender_male)

    for i in np.arange(len(metadata_values),len(metadata_keys)):
        rd = np.random.choice(np.arange(0,2), size=1, p=[0.95, 0.05]).item()
        metadata_values.append(rd)

    metadata = dict(zip(metadata_keys, metadata_values))
    metadata_df = pd.DataFrame([metadata.values()], columns = metadata.keys())
    return metadata_df

def extratree_prediction(wav_file_path, directory_model, optimal_threshold = 0.2397):

    features_extracted = mid_term_feat_extraction(wav_file_path)
    metadata = random_metadata_generator()
    X_new = pd.concat([features_extracted, metadata], axis = 1)

    # Load the model
    warnings.filterwarnings("ignore")

    joblib_file = "extratree_classifier.pkl"
    extratree_classifier = joblib.load(os.path.join(directory_model, joblib_file))

    # Predictions
    prediction = extratree_classifier.predict_proba(X_new)[:,1]
    if prediction >= optimal_threshold:
        print('Cough POSITIVE in COVID-19')
        return True
    else:
        print('Cough NEGATIVE in COVID-19')
        return False
