import pickle
import numpy as np
from scipy.io import wavfile
from scipy.io.wavfile import write
import os
from bson.objectid import ObjectId
from modulos.database_connection import *

def wav_to_blob(file_path):

    file_dir, filename = os.path.split(os.path.abspath(file_path))
    input_file_path = os.path.abspath(file_path)
    basename = filename.split('.')[0]
    wav_file_path = os.path.join(file_dir, '{}.wav'.format(basename))

    sample_rate, wav_data = wavfile.read(wav_file_path)
    dumped = pickle.dumps(wav_data)

    return dumped, sample_rate

def store_blob_mongodb(file_path):
    database = DataBase()
    last_inserted_document = database.collection.find_one(sort=[( '_id', pymongo.DESCENDING )])
    last_id = str(last_inserted_document['_id'])
    last_username = str(last_inserted_document['username'])

    wav_binary, sample_rate = wav_to_blob(file_path)

    data = {}
    data['_id'] = ObjectId(last_id)
    data['username'] = last_username
    data['cough_recording'] = {}
    data['cough_recording']['blob_audio'] = wav_binary
    data['cough_recording']['sample_rate'] = sample_rate

    database.db['Cough Audio'].insert_one(data)
    print('El documento se ha insertado correctamente')

def delete_audios(file_path):
    file_dir, filename = os.path.split(os.path.abspath(file_path))
    input_file_path = os.path.abspath(file_path)
    basename = filename.split('.')[0]
    wav_file_path = os.path.join(file_dir, '{}.wav'.format(basename))

    if (os.path.exists(file_path) and os.path.exists(wav_file_path)):
        os.remove(file_path)
        os.remove(wav_file_path)
    else:
        print("The file does not exist")


'''
AÑADIR FUNCIONALIDAD ---> BORRAR TANTO ARCHIVO .OGA COMO WAV ORIGINAL DEL SISTEMA

La idea es guardar el archivo wav en binary y el sample rate en una colección diferente a Patients.
Lo que hacemos es pillar el id y username del ultimo documento insertado
para guardarlo en el nuevo documento que contiene:
Object # id
username
blob_audio
sample_rate
'''
