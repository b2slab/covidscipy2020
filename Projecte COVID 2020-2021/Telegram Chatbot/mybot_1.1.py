import logging
import requests
import aiogram.utils.markdown as md
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters import Text
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.types import ParseMode
from aiogram.utils import executor
from pydub import AudioSegment
import os
import json
#from scipy.io import wavfile
import requests

logging.basicConfig(level=logging.INFO)
API_TOKEN = '1370389029:AAFIaYXbnHLCkNYIb5azZ2iOg5BWoRdOUC8'
bot = Bot(token=API_TOKEN)
save_path = '/home/dani/covidscipy2020/data/'
# For example use simple MemoryStorage for Dispatcher.
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)
#language



# States
class Form(StatesGroup):
    username = State()
    age = State()
    gender = State()
    location = State()
    #country = State()
    #postcode = State()
    has_corona = State()
    cough = State()
    dry_cough = State()
    fever = State()
    tiredness = State()
    smell_loss = State()
    head_ache = State()
    shortness_breath = State()
    chest_pain = State()
    others = State()
    upload_Data = State()


'''
DATABASE CONNECTION
'''

import pymongo
from sshtunnel import SSHTunnelForwarder

class DataBase:

    def __init__(self):

        '''
        Inicializamos la clase con este CONSTRUCTOR. De esta forma, cada vez que instanciemos
        la clase 'DataBase', nos conectaremos a la BBDD de MongoDB ubicada en el servidor
        remoto de la UPC.

        Para realizar la conexión remota, utilizaremos un túnel SSH mediante la función
        'SSHTunnelForwarder' del módulo 'sshtunnel'.

        Posteriormente, nos connectaremos a la base de datos 'Project_COVID'. De momento,
        accederemos a la única Colección de la base de datos, llamada 'Patients'. Las colecciones
        en MongoDB son el equivalente a las tablas en MySQL.
        '''

        sql_hostname = '127.0.0.1'
        sql_username = 'guillembonilla'
        sql_password = 'B2SLab2020!!!!'
        sql_main_database = 'Project_COVID'
        sql_port = 27017
        ssh_host = 'covidbot.upc.edu'
        ssh_user = 'covidbot'
        ssh_pass = 'B2SLab2020!!!!'
        ssh_port = 2244


        self.server = SSHTunnelForwarder(
            (ssh_host, ssh_port),
            ssh_username=ssh_user,
            ssh_password=ssh_pass,
            remote_bind_address=(sql_hostname, sql_port))

        self.server.start()

        self.client = pymongo.MongoClient(sql_hostname, self.server.local_bind_port)
        self.db = self.client[sql_main_database]
        self.collection = self.db['Patients']
        print('Colecciones de la BBDD: ', self.db.list_collection_names())
        print('Conexión establecida correctamente')
    def close(self):

        '''
        Cerramos connexión con el servidor
        '''

        self.server.stop()
        print("Hemos realizado correctamente la desconexión de la BBDD")

database = DataBase()

'''
DATABASE CONNECTED
'''






'''
START CHATBOT
'''

@dp.message_handler(commands='start')
async def cmd_start(message: types.Message):
    """
    Conversation's entry point
    """
    # Set state
    await Form.username.set()
    await message.reply(lang["en"]["q1"])


# You can use state '*' if you need to handle all states
@dp.message_handler(state='*', commands='cancel')
@dp.message_handler(Text(equals='cancel', ignore_case=True), state='*')
async def cancel_handler(message: types.Message, state: FSMContext):
    """
    Allow user to cancel any action
    """
    current_state = await state.get_state()
    if current_state is None:
        return

    await state.finish()
    # And remove keyboard (just in case)
    await message.reply('Cancelled.', reply_markup=types.ReplyKeyboardRemove())

@dp.message_handler(state='*', commands='stop')
@dp.message_handler(Text(equals='stop', ignore_case=True), state='*')
async def stop_handler(message: types.Message, state: FSMContext):
    """
    Allow user to stop the questions. Still saves everything answered. For test purposes.
    """
    current_state = await state.get_state()
    if current_state is None:
        return

    async with state.proxy() as data:

        await bot.send_message(
            message.chat.id,
            "Allright we'll stop here\n"
            "Please, give me a second while I upload the data."
        )
        #save_features(data.as_dict())

        '''
        Insertamos los datos con formato diccionario. No hace falta
        convertirlos a JSON ya que la propia BBDD de MongoDB los convierte
        a BSON al insertar el documento

        Un hecho relevante es que la propia Colección le agrega un ID único
        a cada Documento (a cada paciente)
        '''
        database.collection.insert_one(data.as_dict())

        await bot.send_message(
            message.chat.id,
            "That's it!"
        )
    await state.finish()
    await message.reply('Process stopped.', reply_markup=types.ReplyKeyboardRemove())


# Check username.
@dp.message_handler(lambda message: not message.text.isalpha(), state=Form.username)
async def process_user_invalid(message: types.Message):
    """
    If user is invalid
    """
    return await message.reply("Username is invalid. Please enter a correct username.")


@dp.message_handler(state=Form.username)
async def process_username(message: types.Message, state: FSMContext):
    """
    Process user name
    """
    async with state.proxy() as data:
        data['username'] = message.text

    await Form.next()
    await message.reply("How old are you?")


# Check age. Age has to be a digit
@dp.message_handler(lambda message: not message.text.isdigit(), state=Form.age)
async def process_age_invalid(message: types.Message):
    """
    If age is invalid
    """
    return await message.reply("Age has to be a number.\nHow old are you? (digits only)")

@dp.message_handler(lambda message: message.text.isdigit(), state=Form.age)
async def process_age(message: types.Message, state: FSMContext):
    # Update state and data
    await Form.next()
    await state.update_data(age=int(message.text))

    # Configure ReplyKeyboardMarkup
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add("male", "female")
    markup.add("other")

    await message.reply("What is your gender?", reply_markup=markup)

@dp.message_handler(lambda message: message.text not in ["male", "female", "other"], state=Form.gender)
async def process_gender_invalid(message: types.Message):
    """
    In this example gender has to be one of: Male, Female, Other.
    """
    return await message.reply("Bad gender name. Choose your gender from the keyboard.")

@dp.message_handler(state=Form.gender)
async def process_gender(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['gender'] = message.text
        #markup = types.ReplyKeyboardRemove()

        location_keyboard  = types.KeyboardButton(text="Send Current Location", request_location=True)
        reply_markup = types.ReplyKeyboardMarkup([[location_keyboard]], resize_keyboard=True)

    await Form.next()
    #await message.reply("In which country are you right now?", reply_markup=markup)
    return await message.reply("Would you mind to send us your current location?\n\nPlease activate location on your device.", reply_markup=reply_markup)


# Message handler if a non location message is received
@dp.message_handler(lambda message: types.message.ContentType not in ['location'], state=Form.location)
async def process_location_invalid(message: types.Message):
    """
    Filter.
    """

    location_keyboard  = types.KeyboardButton(text="Send Current Location", request_location=True)
    reply_markup = types.ReplyKeyboardMarkup([[location_keyboard]], resize_keyboard=True)

    return await message.reply("Please activate location on your device and send it to us.", reply_markup=reply_markup)


@dp.message_handler(state=Form.location, content_types=['location'])
async def process_location(message, state: FSMContext):
    async with state.proxy() as data:
        data['location'] = {}
        data['location']['latitude'] = message.location.latitude
        data['location']['longitude'] = message.location.longitude
        #print("{0}, {1}".format(message.location.latitude, message.location.longitude))

    await Form.next()
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add("positive")
    markup.add("negative")
    markup.add("unknown")
    await message.reply("Do you have Covid-19?", reply_markup=markup)

'''
@dp.message_handler(lambda message: not message.text.isalpha(), state=Form.country)
async def process_country_invalid(message: types.Message):
    """
    In this example country must not contain numbers
    """
    print(message.text)
    return await message.reply("Bad country name. Country should only contain letters")

@dp.message_handler(state=Form.country)
async def process_country(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['location'] = {}
        data['location']['country'] = message.text

    await Form.next()
    await message.reply("What is your zip code?")

@dp.message_handler(lambda message: not message.text.isalnum(), state=Form.postcode)
async def process_postcode_invalid(message: types.Message):
    """
    Postcode filter
    """
    return await message.reply("Bad postcode. Postcode should be alphanumeric.")

@dp.message_handler(state=Form.postcode)
async def process_postcode(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['location']['postcode'] = message.text

    await Form.next()
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add("positive")
    markup.add("negative")
    markup.add("unknown")
    await message.reply("Do you have Covid-19?", reply_markup=markup)
'''

@dp.message_handler(lambda message: message.text not in ["positive", "negative", "unknown"], state=Form.has_corona)
async def process_has_corona_invalid(message: types.Message):
    """
    Filter.
    """
    return await message.reply("Bad answer. Please, choose between the keyboard options.")

@dp.message_handler(state=Form.has_corona)
async def process_has_corona(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['diagnosis'] = message.text
        markup = types.ReplyKeyboardRemove()

    await Form.next()
    #await Form.next() #this line is to be removed after cough implementation
    #markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    #markup.add("yes", "no")
    #await message.reply("Thank you. Now let us ask you some questions about your symptoms."
                        #"Do you have a dry cough?", reply_markup=markup)
    await message.reply("Could you send us a recording of your cough?", reply_markup=markup)


# Message handler if a non voice message is received
@dp.message_handler(lambda message: types.message.ContentType not in ['voice'], state=Form.cough)
async def process_cough_invalid(message: types.Message):
    """
    Filter.
    """
    return await message.reply("Please send us a recording of your cough.")


@dp.message_handler(state=Form.cough, content_types=types.message.ContentType.VOICE)
async def process_cough(message: types.voice.Voice, state: FSMContext):
    # Update state and data
    await bot.send_message(message.chat.id,"Please, give me a second while I annalyze you cough...")
    file_id = message.voice.file_id
    file = await bot.get_file(file_id)
    file_path = file.file_path
    await bot.download_file(file_path, "cough_data/cough.ogg")
    accepted = is_cough(message.voice.file_id)

    if accepted == False:
        return await bot.send_message(message.chat.id,"Sorry, we didn't recognize this as cough. Please, cough again")

    else:
        await bot.send_message(message.chat.id,"Thanks for your cough")
        await Form.next()
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
        markup.add("yes", "no")
        return await message.reply("Do you have a dry cough?", reply_markup=markup)


"""
cough yet to be implemented
@dp.message_handler(state=Form.cough, content_types=types.message.ContentType.VOICE)
async def process_cough(message: types.voice.Voice, state: FSMContext):
    Update state and data
    await bot.send_message(
        message.chat.id,
        "Please, give me a second while I annalyze you cough..."
    )
    accepted, wav_file = is_cough(message.voice.file_id)
    if not accepted:
        return await bot.send_message(
            message.chat.id,
            "Sorry, we didn't recognize this as cough. Please, cough again"
        )

    else:
        async with state.proxy() as data:
            username = data['username']
            label = data['diagnosis']
            audio_features, audio_numpy, sample_rate = create_feature_from_audio(wav_file, label)
            data['audio_features'] = audio_features
            upload_audio(audio_numpy, sample_rate, username, label)

        await Form.next()
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
        markup.add("yes", "no")
        await message.reply("Do you have a dry cough?", reply_markup=markup)
"""

@dp.message_handler(lambda message: message.text not in ["yes", "no"], state=Form.dry_cough)
async def process_dry_cough_invalid(message: types.Message):
    """
    Text filter.
    """
    return await message.reply("Bad answer. Please, choose between the keyboard options.")


@dp.message_handler(state=Form.dry_cough)
async def process_dry_cough(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['symptoms'] = {}
        data['symptoms']['dry cough'] = (message.text == "yes")

    await Form.next()
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add("yes", "no")
    await message.reply("Do you have fever?", reply_markup=markup)

@dp.message_handler(lambda message: message.text not in ["yes", "no"], state=Form.fever)
async def process_fever_invalid(message: types.Message):
    """
    In this example gender has to be one of: Male, Female, Other.
    """
    return await message.reply("Bad answer. Please, choose between the keyboard options.")


@dp.message_handler(state=Form.fever)
async def process_fever(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['symptoms']['fever'] = (message.text == "yes")

    await Form.next()
    await message.reply("Do you feel more tired than usual?")


@dp.message_handler(lambda message: message.text not in ["yes", "no"], state=Form.tiredness)
async def process_tiredness_invalid(message: types.Message):
    return await message.reply("Bad answer. Please, choose between the keyboard options.")


@dp.message_handler(state=Form.tiredness)
async def process_tiredness(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['symptoms']['tiredness'] = (message.text == "yes")

    await Form.next()
    await message.reply("Do you feel that you have lost/diminished your sense of smell?")


@dp.message_handler(lambda message: message.text not in ["yes", "no"], state=Form.smell_loss)
async def process_loss_smell_invalid(message: types.Message):
    return await message.reply("Bad answer. Please, choose between the keyboard options.")


@dp.message_handler(state=Form.smell_loss)
async def process_loss_smell(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['symptoms']['loss of taste or smell'] = (message.text == "yes")

    await Form.next()
    await message.reply("Do you have a headache?")


@dp.message_handler(lambda message: message.text not in ["yes", "no"], state=Form.head_ache)
async def process_headache_invalid(message: types.Message):
    return await message.reply("Bad answer. Please, choose between the keyboard options.")


@dp.message_handler(state=Form.head_ache)
async def process_headache(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['symptoms']['headache'] = (message.text == "yes")

    await Form.next()
    await message.reply("Do you have difficulty breathing or shortness of breath?")


@dp.message_handler(lambda message: message.text not in ["yes", "no"], state=Form.shortness_breath)
async def process_shortness_breath_invalid(message: types.Message):
    return await message.reply("Bad answer. Please, choose between the keyboard options.")


@dp.message_handler(state=Form.shortness_breath)
async def process_shortness_breath(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['symptoms']['difficulty breathing or shortness of breath'] = (message.text == "yes")

    await Form.next()
    await message.reply("Do you have chest pain or pressure?")


@dp.message_handler(lambda message: message.text not in ["yes", "no"], state=Form.chest_pain)
async def process_chest_pain_invalid(message: types.Message):
    return await message.reply("Bad answer. Please, choose between the keyboard options.")


@dp.message_handler(state=Form.chest_pain)
async def process_chest_pain(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['symptoms']['chest pain or pressure'] = (message.text == "yes")
        markup = types.ReplyKeyboardRemove()

    await Form.next()
    await message.reply("Do you have any other information you would like to add?", reply_markup=markup)


@dp.message_handler(state=Form.others)
async def process_others(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['symptoms']['others'] = message.text

        await bot.send_message(
            message.chat.id,
            "Thank you very much for you collaboration!\n"
            "Please, give me a second while I upload the data."
        )
        #save_features(data.as_dict())

        '''
        Insertamos los datos con formato diccionario. No hace falta
        convertirlos a JSON ya que la propia BBDD de MongoDB los convierte
        a BSON al insertar el documento

        Un hecho relevante es que la propia Colección le agrega un ID único
        a cada Documento (a cada paciente). Este ID es del tipo Object_id el
        cual también almacena el momento en el que el usuario se ha registrado.
        '''
        database.collection.insert_one(data.as_dict())
        #this should be the used method:
        #requests.post('https://localhost:5001/users', data=data)


        await bot.send_message(
            message.chat.id,
            "That's it!"
        )


#functions

#def save_features(data_object):
#    outputFileName = "Patient #.txt"
#    outputVersion = 1
#    while os.path.isfile(save_path + outputFileName.replace("#", str(outputVersion))):
#        outputVersion += 1
#    outputFileName = outputFileName.replace("#", str(outputVersion))
#    filepath = os.path.join(save_path, outputFileName)
#    with open(filepath, 'w') as outfile:
#        json.dump(data_object, outfile)
#     data_object_json = json.dumps(data_object)


def is_cough():

    outputFileName = "Cough #.wav"
    outputVersion = 1
    while os.path.isfile(audio_path + outputFileName.replace("#", str(outputVersion))):
        outputVersion += 1
    outputFileName = outputFileName.replace("#", str(outputVersion))
    filepath = os.path.join(audio_path, outputFileName)
    src = "cough_data/cough.ogg"

    # convert wav to mp3
    sound = AudioSegment.from_ogg(src)
    sound.export(filepath, format="wav")

    accepted = yamnet_classifier(wav_file_path)

    return accepted


def yamnet_classifier(wav_file_path):
    sample_rate, wav_data = wavfile.read(wav_file_path)
    sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)

    waveform = wav_data / tf.int16.max
    #waveform2 = np.mean(waveform, axis = 1)   # If the audio is stereo and not mono

    # Run the model, check the output.
    scores, embeddings, spectrogram = model(waveform)

    scores_np = scores.numpy()
    spectrogram_np = spectrogram.numpy()
    infered_class = class_names[scores_np.mean(axis=0).argmax()]

    if infered_class == 'Cough':
        return True
    else:
        return False



'''
WE LOAD THE TENSORFLOW TRAINED MODEL CALLED YAMNET
'''

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
import io

import matplotlib.pyplot as plt
from IPython.display import Audio
from scipy.io import wavfile
from scipy import signal

# Load the model.
model = hub.load('https://tfhub.dev/google/yamnet/1')

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


'''
END OF YAMNET IMPORTATION
'''




if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
