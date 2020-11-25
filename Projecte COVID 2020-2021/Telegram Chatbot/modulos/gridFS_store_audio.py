from pymongo import MongoClient
from gridfs import GridFS
from bson import objectid
from database_connection import *


import json

database = DataBase()

database.collection.update({'username': "Guillem"}, {'$set': {"blob_audio": binary_data}}, multi=False, upsert=False)
dict = database.collection.find_one({'username':'Guillem'}, {'blob_audio':1, '_id':0})

dict['blob_audio']

import wave

w = wave.open('C:/Users/Guillem/Desktop/Bot_Telegram/Cough_recordings/AwACAgQAAxkBAAIJDl-3rCJPscvgB0PMsbhQQUEzKQpEAAIvCAACiO3BUasyAfUIPHkbHgQ.wav', "rb")
binary_data = w.readframes(w.getnframes())
w.close()



import wave
import struct

def signal_to_wav(signal, fname, Fs):
    """Convert a numpy array into a wav file.

     Args
     ----
     signal : 1-D numpy array
         An array containing the audio signal.
     fname : str
         Name of the audio file where the signal will be saved.
     Fs: int
        Sampling rate of the signal.

    """
    data = struct.pack('<' + ('h'*len(signal)), *signal)
    wav_file = wave.open(fname, 'wb')
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(Fs)
    wav_file.writeframes(data)
    wav_file.close()


signal_to_wav(binary_data, 'prueba.wav', 16000)




binary_data





from scipy.io import wavfile

wav_file_path = 'C:/Users/Guillem/Desktop/Bot_Telegram/Cough_recordings/AwACAgQAAxkBAAIJDl-3rCJPscvgB0PMsbhQQUEzKQpEAAIvCAACiO3BUasyAfUIPHkbHgQ.wav'
sample_rate, wav_data = wavfile.read(wav_file_path)





from pydub import AudioSegment
from pydub.playback import play
import io

recording = AudioSegment.from_file(io.BytesIO(binary_data), format = "wav")
recording.export('prueba.wav', format='wav') # for export
play(recording) # for play





from scipy.io import wavfile
import io

## This may look a bit intricate/useless, considering the fact that scipy's read() and write() function already return a
## numpy ndarray, but the BytesIO "hack" may be useful in case you get the wav not through a file, but trough some websocket or
## HTTP Post request. This should obviously work with any other sound format, as long as you have the proper decoding function

with open(wav_file_path, "rb") as wavfile:
    input_wav = wavfile.read()

# here, input_wav is a bytes object representing the wav object
rate, data = read(io.BytesIO(input_wav))

# data is a numpy ND array representing the audio data. Let's do some stuff with it
reversed_data = data[::-1] #reversing it

#then, let's save it to a BytesIO object, which is a buffer for bytes object
bytes_wav = bytes()
byte_io = io.BytesIO(bytes_wav)
write(byte_io, rate, data)

output_wav = byte_io.read() # and back to bytes, tadaaa


write('prueba.wav',output_wav)




wav_file = wave.open(filename, 'wb')
wav_file.setnchannels(1)
wav_file.setsampwidth(2)
wav_file.setframerate(Fs)
wav_file.writeframes(data)
wav_file.close()




import pyaudio
import wave
import io

# response.audio_content is a byte string
with wave.open(io.BytesIO(binary_data), 'rb') as f:
    width = f.getsampwidth()
    channels = f.getnchannels()
    rate = f.getframerate()
pa = pyaudio.PyAudio()
pa_stream = pa.open(
    format=pyaudio.get_format_from_width(width),
    channels=channels,
    rate=rate,
    output=True
)
pa_stream.write(response.audio_content)
