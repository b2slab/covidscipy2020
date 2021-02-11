import pickle
from database_connection import *
from scipy.io import wavfile
import numpy as np
from scipy.io.wavfile import write

def get_binary_wav(user_name, output_file_name):

    database = DataBase()

    try:
        dict = database.db['Cough Audio'].find_one({'username':user_name}, {'cough_recording':1, '_id':0})
        wav_binary = dict['cough_recording']['blob_audio']
        sample_rate = dict['cough_recording']['sample_rate']
        restored = pickle.loads(wav_binary)

    except TypeError:
        print('The username you have introduced does not exist in the database')

    try:
        write(output_file_name + '.wav', sample_rate, restored)
    except TypeError:
        print('The output file name must be a string')


get_binary_wav('MariaTeresa', 'mariateresa_audio')
