"""
WE LOAD THE TENSORFLOW TRAINED MODEL CALLED YAMNET
"""

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
import io
from scipy.io import wavfile
from scipy import signal


# Load the model.
model = hub.load('https://tfhub.dev/google/yamnet/1')



#model = tf.saved_model.load('/home/dani/covidscipy2020/covidscipy2020/Projecte%20COVID%202020-2021/Telegram%20Chatbot/yamnet')
#model = tf.keras.models.load_model("/Bot_Telegram/yamnet/")

def class_names_from_csv(class_map_csv_text):
  """Returns list of class names corresponding to score vector."""
  class_names = []
  with open(class_map_csv_text, newline='\r\n') as csvfile:
      reader = csv.DictReader(csvfile)
      for row in reader:
          class_names.append(row['display_name'])

  return class_names

class_map_path = model.class_map_path().numpy()
class_names = class_names_from_csv(class_map_path)


def ensure_sample_rate(original_sample_rate, waveform,desired_sample_rate=16000):
    """Resample waveform if required."""
    if original_sample_rate != desired_sample_rate:
        desired_length = int(round(float(len(waveform)) / original_sample_rate * desired_sample_rate))
        waveform = signal.resample(waveform, desired_length)

    return desired_sample_rate, waveform


"""
END OF YAMNET IMPORTATION
"""
