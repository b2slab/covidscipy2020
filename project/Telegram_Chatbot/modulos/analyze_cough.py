from pydub import AudioSegment
from pyAudioAnalysis import MidTermFeatures
import numpy as np
import pandas as pd
import opensmile
import warnings
import joblib
import os

## New added dependencies
import librosa
from scipy import signal, fft
# from pyAudioAnalysis import audioBasicIO   ---> No longer needed


# Analyze audio cough
def analyze_cough(ogg_path, data):
    '''
    The function is in charge of calling all the defined functions to maintain a clean script
    and a clear workflow. First, the audio is converted to wav. Then, the long-term features
    (based on mid-term and, therefore, short-term) are extracted from the converted audio.
    In this point, the audio is analyze. Only if the audio contains a cough (accordingly to our
    cough recognition model), the metadata of the chatbot user is extracted and converted to
    a proper format. Additionally, we predict whether the audio has been recorded by a user
    that has COVID-19 or not, accordingly to our covid recognition model.

    Note that if our first cough recognition model does not predict that the audio contains cough,
    the function returns None automatically (without computing the metadata or the covid prediction).

    @input:
        - ogg_path: absolute path where the original .ogg audio file has been downloaded from the Telegram API.
        - data: metadata extracted from the chatbot user. Only is converted and used in a proper format if
                the audio contains cough.
    @output:
        - True: if the cough audio has been recognised as POSITIVE in COVID-19.
        - False: if the cough audio has been recognised as NEGATIVE in COVID-19.
        - None: if the model has not detected cough in the audio.
    '''
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
    '''
    Convertion of original audio from .ogg format to .wav. The .ogg format is the default used by Telegram
    because its size. However, in order to analyze properly the audios, a transformation to a higher quality
    format such as .wav is needed.

    @input:
        - original_path: the absolute path where the .ogg audio has been downloaded from the Telegram API.
    @output:
        - converted_path: the absolute path where the converted .wav audio has been stored.
    '''
    converted_path = os.path.splitext(original_path)[0] + '.wav'
    audio = AudioSegment.from_ogg(original_path)
    audio.export(converted_path, format="wav")
    return converted_path

# We define 3 functions in order to create the following:
# **Lowpass-filter**: the cutoff frequency defines the limit at which frequencies are masked.
# **Amplification**: the audio gain is defined in such a way that the maximum value of the signal is equal to 1.
#                    In this way, the entire signal is amplified but its standardization is maintained.

def butter_lowpass(cutoff, fs, order=5):
    '''
    The function implements a low-pass filter of order 5 (using Butterworth digital filter). The purpose of the
    filter is to attenuate the frequencies whoose values are higher than the defined cutoff.
    The order 5 has been choosen because it provides a compromise between stability and sharpness of
    the transition between preserved and attenuated frequencies.
    Additionally, the cutoff frequency is expressed as the fraction of the Nyquist frequency,
    which at the same time is half the sampling rate of the signal.
    The filter is applied as a consequence that iOS devices record audios with greater spectral information
    (more high-frequency information) than the ones recorded by Android devices.

    @input:
        - cutoff: the cutoff frequency. Defines the boundary from which the frequencies will be attenuated.
        - fs: sampling rate of the signal
        - order: order of the filter

    @output:
        - b: numerator polynomials of the IIR filter (filter coefficients)
        - a: denominator polynomials of the IIR filter (filter coefficients)
    '''
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    '''
    The function extracts the 5-order low-pass filter coefficients and filters the original signal.
    In this way, the frequencies whoose values are higher than the cutoff frequency are masked (attenuated).

    @input:
        - data: original signal (numpy array)
        - cutoff: cutoff frequency
        - fs: sampling rate of the signal
        - order: order of the Butterworth filter
    @output:
        - y: filtered signal
    '''
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def increase_amplitude(data):
    '''
    The function normalize the amplitude of the signal by increasing its gain. Basically, it multiplies
    the signal by a gain factor. In this way, the signal gain ranges from -1 to +1. This is useful because
    the audios recorded by iOS tend to have greater gains than the ones recorded by Android.

    @input:
        - data: original signal
    @output:
        - data*factor: amplified signal
    '''
    max_original_signal = max(data)
    max_desired = 1
    factor = max_desired/max_original_signal
    return data*factor

def extract_features_audio(filename, low_pass_filt = False, cutoff_freq = 4096, amplification = False, compute_FFT = False):
    '''
    The function loads the signal and the sampling rate of the audio by using the Librosa library.
    Note that the audio is loaded as a mono signal (not stereo). Then it verifies if the sampling rate is
    correct. Furthermore, it filters the signal by using the low-pass Butterworth filter and amplify it, too.
    Lastly, if it is necessary, it computes the 1-D discrete Fourier Transform in order to work with the signal
    in the Frequency domain. As the signal is symmetric, it only extracts the positive part.
    The Mel-Spectrogram of the signal is also extracted and then transformed to decibels. As we are working
    with human sounds, the mel-spectrogram is better as it scales the original signal accordingly to the human
    audition.

    @input:
        - filename: absolute path where the audio is stored
        - low_pass_filt: if True, the signal is filtered
        - cutoff_freq: the cutoff frequency. As default, is defined at 4096Hz as we have seen that Android
                       audios reach its maximum peak at this frequency.
        - amplification: if True, the signal is amplified.
        - compute_FFT: if True, then the 1D DFT and the Mel-Spectrogram are computed.

    @output:
        - signal: the signal filtered and amplified (if applicable)
        - sampling_rate: the sampling rate of the signal
        - xf: frequencies [Hz] of the 1D DFT. Note that the maximum frequency is the Nyquist one.
        - yf: Gain of each frequency of the signal in each Frequency bin.
        - S_dB: Mel-Spectrogram in decibels
    '''
    # Read the audio
    signal, sampling_rate = librosa.load(filename, sr = None, mono=True) # Load the audio as Mono (not stereo)

    if sampling_rate == 0:
        print('Sampling rate not correct.')
        return None

    if (low_pass_filt == True):
        signal = butter_lowpass_filter(signal, cutoff_freq, sampling_rate)

        if (amplification == True):
            signal = increase_amplitude(signal)

    # FFT

    if (compute_FFT == True):
        duration = librosa.core.get_duration(y = signal, sr = sampling_rate)
        n = int(sampling_rate * duration) # Number of samples of audios
        yf = fft.rfft(signal)
        xf = fft.rfftfreq(n, 1 / sampling_rate)

        if len(xf) != len(yf):
            yf = yf[0:len(xf)]

        # Mel-Spectrogram
        S = librosa.feature.melspectrogram(y=signal, sr=sampling_rate)
        S_dB = librosa.power_to_db(S, ref=np.max)
        return signal, sampling_rate, xf, yf, S_dB

    else:
        return signal, sampling_rate


# Extract mid-term features from wav
def mid_term_feat_extraction(wav_file_path):
    '''
    This function is the core of the script. It find a way to go from low-level audio data samples
    to a higher-level representation of the audio content. We are interested to extract higher-level audio
    features that are capable of discriminating between different audio classes (cough/no-cough, covid/no-covid).
    Basically, the function extracts the mid-term features (based on the short-term ones) from the .wav audio and
    computes the long term averaging of the mid-term statistics.

    The most important concept of audio feature extraction is short-term windowing (or framing): this simply means
    that the audio signal is split into short-term windows (or frames). In this case, we have defined the short-term
    windows as well as the short-term steps equal to 10 msecs. Consequently, there is no-overlapping between windows (or frames).
    For each frame (whoose length is defined by the short-term windows parameter), we extract a set of short-term audio features.
    Those features are extracted directly from the audio sample values (Time domain) as well as from the FFT values (Frequency domain).

    Then, we extract 2 statistics, namely the mean and the std of each short-term feature sequence, using the provided
    mid-term window size. These statistics represents the mid-term features of the audio. Finally, we perform a long-term averaging
    in order to obtain a single large mean feature vector per each audio.

    Additionally, we use the OpenSmile library to extract cepstral features based on the cepstrum for each audio. Then, we concatenate
    both vectors. In this way, for each input audio, we extract a bunch of almost 150 features contained in a row vector. Hopefully, these
    features have enough discriminative information to classify correctly the audios.

    @input:
        - wav_file_path: the absolute path where the converted-to-wav audio is located.
    @output:
        - final_df: pandas DataFrame which contains almost 150 features extracted from the raw audio in a tabular way.
    '''

    # sampling_rate, signal = audioBasicIO.read_audio_file(wav_file_path)
    # if sampling_rate == 0:
    #     print('Sampling rate not correct.')
    #     return None

    # signal = audioBasicIO.stereo_to_mono(signal)
    # if signal.shape[0] < float(sampling_rate)/5:
    #     print("The duration of the audio is too short.")
    #     return None

     # Filtering and Amplification
    signal, sampling_rate = extract_features_audio(wav_file_path, low_pass_filt = True, cutoff_freq = 4096, amplification = True, compute_FFT = False)

    mid_window, mid_step, short_window, short_step = 0.1, 0.1, 0.01, 0.01
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
def cough_prediction(X_new, opt_thresh = 0.5):
    '''
    This function loads the cough recognition model and predicts whether the audio is cough based on the long-term
    averaging of its mid-term (and therefore short-term) features.

    It uses a classification model trained with 500 audios divided in two halves; the first half contains cough audios. The second half contains audios of sneezing, clearing
    throat and other human sounds. As the dataset has been splitted in training/testing, we have been able to compute the
    optimal threshold (0.6) of the model from the ROC curves. The optimal threshold divides the output probability in a way
    that provides the best compromise between specificity and sensibility. The model is stored as a pickle (binary data).

    Finally, if the outputed probability of the model is equal or larger than the optimal threshold, the input audio is
    classified as a cough. Otherwise, the audio is classified as a no-cough.

    @input:
        - X_new: pandas DataFrame which contains a row vector of features from the raw audio.
        - opt_thresh: the optimal threshold defined as 0.6.
    @output:
        - True/False: boolean output depending whether the audio is classified as cough or no-cough respectively.
    '''
    # Load the cough recognition model
    #joblib_file = "/app/project/Telegram_Chatbot/modulos/random_forest_classifier.pkl"
    joblib_file = "/home/dani/covidscipy2020/covidscipy2020/project/Telegram_Chatbot/modulos/random_forest_classifier.pkl"
    #joblib_file = "C:/Users/Guillem/Desktop/pruebas_audio_telegram/random_forest_classifier.pkl"
    cough_classifier = joblib.load(joblib_file)

    # Predict if audio is cough
    y_pred = cough_classifier.predict_proba(X_new)[:,1]
    print(y_pred)

    if y_pred >= opt_thresh:
        #print('The audio has been recognised as COUGH')
        return True
    else:
        #print('The audio has been recognised as NO COUGH')
        return False

# Convert the data extracted by chatbot into input metadata for the model
def convert_metadata(data):
    '''
    The function converts the data extracted by the chatbot into input metadata for the
    covid recognition model. Basically, all data is boolean except the age. The metadata
    is first builded in a key:value way (dictionary) and then transformed to a pandas
    DataFrame.

    @input:
        - data: list of data from the user extracted by the chatbot
    @output:
        - metadata_df: pandas DataFrame containing the converted metadata
    '''
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
    '''
    The function tries to predict whether a cough audio is recorded by a user that have COVID-19 or not.
    The covid recognition model has been trained with the Coswara dataset. Although several approaches
    has been tried (Transfer Learning of CNN based on spectrogram analysis, Convolutional autoencoders, etc.)
    the one which has provided the best results is the model implemented here (Extreme Randomized Tree classifier).

    It uses not only the bunch of long-term features extracted from an audio in a tabular way, but also the metadata
    of each patient as an input (because the Coswara dataset contained these meta-information). The model is stored
    in binary as a pickle, too.

    @input:
        - X_new: pandas DataFrame which contains a row vector of long-term features from the raw audio.
        - metadata: pandas DataFrame which contains the metadata (symptomatology) of the user who has cought.
        - optimal_threshold: the optimal threshold defined as 0.8 (note that the model is not calibrated).
    @output:
        - True/False: boolean output depending whether the audio is classified as covid cough or no-covid cough respectively.
    '''

    # optimal_threshold = 0.2397
    X = pd.concat([X_new, metadata], axis = 1)

    # Load the model
    warnings.filterwarnings("ignore")

    # Load the model
    joblib_file = "/app/project/Telegram_Chatbot/modulos/extratree_classifier.pkl"
    joblib_file = "/home/dani/covidscipy2020/covidscipy2020/project/Telegram_Chatbot/modulos/extratree_classifier.pkl"
    #joblib_file = "C:/Users/Guillem/Desktop/Bot_Telegram/classification_covid/predict_cough_covid/extratree_classifier.pkl"
    extratree_classifier = joblib.load(joblib_file)

    # Predictions
    prediction = extratree_classifier.predict_proba(X_new)[:,1]
    print(prediction)

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
    '''
    The function just quickly verifies the duration of the audio. If the duration is shorter than 1 second
    or larger than 7 seconds, a new cough recording is requested from the user.

    @input:
        - filepath: absolute path where the original .ogg file is located
    @output:
        - duration: duration of the audio in seconds (it has decimals)
    '''
    audio = AudioSegment.from_ogg(filepath)
    duration = audio.duration_seconds
    return duration
