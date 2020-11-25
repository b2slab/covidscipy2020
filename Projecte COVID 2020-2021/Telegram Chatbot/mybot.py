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
#<<<<<<< HEAD
from pydub import AudioSegment
#=======
from pyAudioAnalysis import audioTrainTest as aT
#>>>>>>> f100371b2e13c59e1fb6d1a6b87ac1836f395796


from modulos.database_connection import *    # Importamos clase para instanciar base de datos
from modulos.yamnet_importation import *
from modulos.cough_classification import *
from modulos.languages_chatbot import *

import os
import json

database = DataBase()   # Conexión a base de datos


logging.basicConfig(level=logging.INFO)
API_TOKEN = '1370389029:AAFIaYXbnHLCkNYIb5azZ2iOg5BWoRdOUC8'
bot = Bot(token=API_TOKEN)
save_path = '/home/dani/covidscipy2020/data/'
# For example use simple MemoryStorage for Dispatcher.
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)
#language_text

questions = import_languages() # Importamos preguntas en tres idiomas

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
START CHATBOT
'''
global lang
@dp.message_handler(commands='start')
async def cmd_start(message: types.Message):
    """
    Conversation's entry point
    """
    # Set state and language
    global lang
    locale = message.from_user.locale
    lang = locale.language

    # Si no reconoce idioma, por defecto activa el español
    if lang not in ["en","es","ca"]:
        lang = "es"

    await Form.username.set()
    await message.reply(questions[lang]["q1"])


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
    await message.reply(questions[lang]["q2"], reply_markup=types.ReplyKeyboardRemove())

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
            questions[lang]["q3"]
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
            questions[lang]["q4"]
        )
    await state.finish()
    await message.reply(questions[lang]["q5"], reply_markup=types.ReplyKeyboardRemove())


# Check username.
@dp.message_handler(lambda message: not message.text.isalpha(), state=Form.username)
async def process_user_invalid(message: types.Message):
    """
    If user is invalid
    """
    return await message.reply(questions[lang]["q6"])


@dp.message_handler(state=Form.username)
async def process_username(message: types.Message, state: FSMContext):
    """
    Process user name
    """
    async with state.proxy() as data:
        data['username'] = message.text

    await Form.next()
    await message.reply(questions[lang]["q7"])


# Check age. Age has to be a digit
@dp.message_handler(lambda message: not message.text.isdigit(), state=Form.age)
async def process_age_invalid(message: types.Message):
    """
    If age is invalid
    """
    return await message.reply(questions[lang]["q8"])

@dp.message_handler(lambda message: message.text.isdigit(), state=Form.age)
async def process_age(message: types.Message, state: FSMContext):
    # Update state and data
    async with state.proxy() as data:
        data['age'] = int(message.text)

    await Form.next()
    #await state.update_data(age=int(message.text))

    # Configure ReplyKeyboardMarkup
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add(questions[lang]["q9"], questions[lang]["q10"])
    markup.add(questions[lang]["q11"])

    await message.reply(questions[lang]["q36"], reply_markup=markup)

@dp.message_handler(lambda message: message.text not in [questions[lang]["q9"], questions[lang]["q10"], questions[lang]["q11"]], state=Form.gender)
async def process_gender_invalid(message: types.Message):
    """
    In this example gender has to be one of: Male, Female, Other.
    """

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add(questions[lang]["q9"], questions[lang]["q10"])
    markup.add(questions[lang]["q11"])

    return await message.reply(questions[lang]["q12"], reply_markup=markup)

@dp.message_handler(state=Form.gender)
async def process_gender(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['gender'] = message.text
        #markup = types.ReplyKeyboardRemove()

        location_keyboard  = types.KeyboardButton(text=questions[lang]["q13"], request_location=True)
        reply_markup = types.ReplyKeyboardMarkup([[location_keyboard]], resize_keyboard=True)

    await Form.next()
    #await message.reply("In which country are you right now?", reply_markup=markup)
    return await message.reply(questions[lang]["q14"], reply_markup=reply_markup)


# Message handler if a non location message is received
@dp.message_handler(lambda message: types.message.ContentType not in ['location'], state=Form.location)
async def process_location_invalid(message: types.Message):
    """
    Filter.
    """

    location_keyboard  = types.KeyboardButton(text=questions[lang]["q13"], request_location=True)
    reply_markup = types.ReplyKeyboardMarkup([[location_keyboard]], resize_keyboard=True)

    return await message.reply(questions[lang]["q15"], reply_markup=reply_markup)


@dp.message_handler(state=Form.location, content_types=['location'])
async def process_location(message, state: FSMContext):
    async with state.proxy() as data:
        data['location'] = {}
        data['location']['latitude'] = message.location.latitude
        data['location']['longitude'] = message.location.longitude
        #print("{0}, {1}".format(message.location.latitude, message.location.longitude))

    await Form.next()
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add(questions[lang]["q16"])
    markup.add(questions[lang]["q17"])
    markup.add(questions[lang]["q18"])
    await message.reply(questions[lang]["q19"], reply_markup=markup)

@dp.message_handler(lambda message: message.text not in [questions[lang]["q16"], questions[lang]["q17"], questions[lang]["q18"]], state=Form.has_corona)
async def process_has_corona_invalid(message: types.Message):
    """
    Filter.
    """
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add(questions[lang]["q16"])
    markup.add(questions[lang]["q17"])
    markup.add(questions[lang]["q18"])
    return await message.reply(questions[lang]["q12"], reply_markup=markup)

@dp.message_handler(state=Form.has_corona)
async def process_has_corona(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['diagnosis'] = message.text
        markup = types.ReplyKeyboardRemove()

    await Form.next()
    await message.reply(questions[lang]["q20"], reply_markup=markup)


# Message handler if a non voice message is received
@dp.message_handler(lambda message: types.message.ContentType not in ['voice'], state=Form.cough)
async def process_cough_invalid(message: types.Message):
    """
    Filter.
    """
    return await message.reply(questions[lang]["q21"])


@dp.message_handler(state=Form.cough, content_types=types.message.ContentType.VOICE)
async def process_cough(message: types.voice.Voice, state: FSMContext):
    # Update state and data
    await bot.send_message(message.chat.id,questions[lang]["q22"])

    file_id = message.voice.file_id
    file = await bot.get_file(file_id)
    file_path_URL = file.file_path
    file_path = 'C:/Users/Guillem/Desktop/Bot_Telegram/Cough_recordings/{}.oga'.format(file_id) #Aquí deberemos indicar el directorio dónce guardemos el archivo en el servidor
    await bot.download_file(file_path_URL, file_path)

    #accepted = is_cough(message.voice.file_id)
    accepted = is_cough(file_path)

    if (accepted == True):
        await bot.send_message(message.chat.id,questions[lang]["q24"])
        await Form.next()
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
        markup.add(questions[lang]["q26"], questions[lang]["q27"])
        return await message.reply(questions[lang]["q25"], reply_markup=markup)

    else:
        return await bot.send_message(message.chat.id,questions[lang]["q23"])


@dp.message_handler(lambda message: message.text not in [questions[lang]["q26"], questions[lang]["q27"]], state=Form.dry_cough)
async def process_dry_cough_invalid(message: types.Message):
    """
    Text filter.
    """
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add(questions[lang]["q26"], questions[lang]["q27"])

    return await message.reply(questions[lang]["q12"], reply_markup=markup)


@dp.message_handler(state=Form.dry_cough)
async def process_dry_cough(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['symptoms'] = {}
        data['symptoms']['dry cough'] = (message.text == questions[lang]["q26"])

    await Form.next()
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add(questions[lang]["q26"], questions[lang]["q27"])
    await message.reply(questions[lang]["q28"], reply_markup=markup)

@dp.message_handler(lambda message: message.text not in [questions[lang]["q26"], questions[lang]["q27"]], state=Form.fever)
async def process_fever_invalid(message: types.Message):

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add(questions[lang]["q26"], questions[lang]["q27"])

    return await message.reply(questions[lang]["q12"], reply_markup=markup)


@dp.message_handler(state=Form.fever)
async def process_fever(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['symptoms']['fever'] = (message.text == questions[lang]["q26"])

    await Form.next()
    await message.reply(questions[lang]["q29"])


@dp.message_handler(lambda message: message.text not in [questions[lang]["q26"], questions[lang]["q27"]], state=Form.tiredness)
async def process_tiredness_invalid(message: types.Message):

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add(questions[lang]["q26"], questions[lang]["q27"])

    return await message.reply(questions[lang]["q12"], reply_markup=markup)


@dp.message_handler(state=Form.tiredness)
async def process_tiredness(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['symptoms']['tiredness'] = (message.text == questions[lang]["q26"])

    await Form.next()
    await message.reply(questions[lang]["q30"])


@dp.message_handler(lambda message: message.text not in [questions[lang]["q26"], questions[lang]["q27"]], state=Form.smell_loss)
async def process_loss_smell_invalid(message: types.Message):

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add(questions[lang]["q26"], questions[lang]["q27"])

    return await message.reply(questions[lang]["q12"], reply_markup=markup)


@dp.message_handler(state=Form.smell_loss)
async def process_loss_smell(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['symptoms']['loss of taste or smell'] = (message.text == questions[lang]["q26"])

    await Form.next()
    await message.reply(questions[lang]["q31"])


@dp.message_handler(lambda message: message.text not in [questions[lang]["q26"], questions[lang]["q27"]], state=Form.head_ache)
async def process_headache_invalid(message: types.Message):

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add(questions[lang]["q26"], questions[lang]["q27"])

    return await message.reply(questions[lang]["q12"], reply_markup=markup)


@dp.message_handler(state=Form.head_ache)
async def process_headache(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['symptoms']['headache'] = (message.text == questions[lang]["q26"])

    await Form.next()
    await message.reply(questions[lang]["q33"])


@dp.message_handler(lambda message: message.text not in [questions[lang]["q26"], questions[lang]["q27"]], state=Form.shortness_breath)
async def process_shortness_breath_invalid(message: types.Message):

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add(questions[lang]["q26"], questions[lang]["q27"])

    return await message.reply(questions[lang]["q12"], reply_markup=markup)


@dp.message_handler(state=Form.shortness_breath)
async def process_shortness_breath(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['symptoms']['difficulty breathing or shortness of breath'] = (message.text == questions[lang]["q26"])

    await Form.next()
    await message.reply(questions[lang]["q32"])


@dp.message_handler(lambda message: message.text not in [questions[lang]["q26"], questions[lang]["q27"]], state=Form.chest_pain)
async def process_chest_pain_invalid(message: types.Message):

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add(questions[lang]["q26"], questions[lang]["q27"])

    return await message.reply(questions[lang]["q12"], reply_markup=markup)


@dp.message_handler(state=Form.chest_pain)
async def process_chest_pain(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['symptoms']['chest pain or pressure'] = (message.text == questions[lang]["q26"])
        markup = types.ReplyKeyboardRemove()

    await Form.next()
    await message.reply(questions[lang]["q34"], reply_markup=markup)


@dp.message_handler(state=Form.others)
async def process_others(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['symptoms']['others'] = message.text

        await bot.send_message(
            message.chat.id,
            questions[lang]["q35"]
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

        requests.post('http://0.0.0.0:5001/users', json=data)
        #database.collection.insert_one(data.as_dict())



        await bot.send_message(
            message.chat.id,
            questions[lang]["q4"]
        )

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
