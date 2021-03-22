import logging
import requests
from aiogram.utils.executor import start_webhook
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters import Text
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.utils import executor


'''
Añadido para que funcione en LOCAL
'''

#import nest_asyncio
#nest_asyncio.apply()
#__import__('IPython').embed()
from project.Telegram_Chatbot.modulos.analyze_cough import *

import numpy as np






from project.Telegram_Chatbot.modulos.database_connection import *    # Importamos clase para instanciar base de datos
from project.Telegram_Chatbot.modulos.languages_chatbot import *
from project.Telegram_Chatbot.settings import (BOT_TOKEN, HEROKU_APP_NAME,
                          WEBHOOK_URL, WEBHOOK_PATH,
                          WEBAPP_HOST, API_HOST)#, WEBAPP_PORT)
import os
import json

database = DataBase()   # Conexión a base de datos

logging.basicConfig(level=logging.INFO)
# For example use simple MemoryStorage for Dispatcher.
storage = MemoryStorage()
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot, storage=storage)
dp.middleware.setup(LoggingMiddleware())
#language_text

questions = import_languages() # Importamos preguntas en tres idiomas

# States
class Form(StatesGroup):

    start = State()
    menu = State()
    delete = State()
    username = State()
    age = State()
    gender = State()
    location = State()
    #country = State()
    #postcode = State()

    has_corona = State()
    vaccine = State()
    #cough = State()
    dry_cough = State()
    smoker = State()
    cold = State()
    res_difficult = State()
    sore_throat = State()
    fever = State()
    fatigue = State()
    muscular = State()
    smell = State()
    pneumonia = State()
    diarrhea = State()

    hypertension = State()
    asma = State()
    diabetes = State()
    CLD = State()
    IHD = State()

    cough = State()

    others = State()
    upload_Data = State()


'''
START CHATBOT
'''


#lang = 'es'
@dp.message_handler(state = None)
@dp.message_handler(state = Form.start)
@dp.message_handler(state='*', commands='cancel')
@dp.message_handler(lambda message: message.text == 'No', state=Form.menu)
@dp.message_handler(lambda message: message.text in ['CANCEL', 'CANCELAR'], state=Form.delete)
@dp.message_handler(lambda message: message.text not in ["About", "Acerca de", "Sobre nosaltres",
                                                         "Delete data", "Eliminar datos", "Eliminar dades",
                                                         "Add data", "Añadir datos", "Afegir dades",
                                                         "Exit", "Salir", "Sortir"], state=Form.menu)
async def cmd_start(message: types.Message):
    """
    Conversation's entry point
    """
    # Set state and language

    lang = message.from_user.locale.language
    id = message.from_user.id
    name = message.from_user.first_name
    # Si no reconoce idioma, por defecto activa el español
    #if lang not in ["en","es","ca"]:
     #   lang = "es"
    #await bot.set_webhook(WEBHOOK_URL, drop_pending_updates=False)
    await Form.menu.set()

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add(questions[lang]["q56"], questions[lang]["q57"])
    markup.add(questions[lang]["q58"], questions[lang]["q59"])
    await message.reply(questions[lang]["q60"] %(name, id), reply_markup=markup)


@dp.message_handler(lambda message: message.text in ["About", "Acerca de", "Sobre nosaltres"], state=Form.menu)
async def about(message: types.Message):
    lang = message.from_user.locale.language
    return await message.reply(questions[lang]["q37"])

@dp.message_handler(lambda message: message.text in ["Add data", "Añadir datos", "Afegir dades"], state=Form.menu)
async def add_my_data(message: types.Message):
    await Form.username.set()
    lang = message.from_user.locale.language
    return await message.reply(questions[lang]["q38"], reply_markup=types.ReplyKeyboardRemove())



@dp.message_handler(lambda message: message.text in ["No", "Delete data", "Eliminar datos", "Eliminar dades"], state=Form.menu)
async def delete_data(message: types.Message):
    await Form.delete.set()
    lang = message.from_user.locale.language
    id = message.from_user.id
    response = requests.get(API_HOST+'users/%s'%id)
    data_delete = json.loads(response.content)
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    if len(data_delete) == 0:
        markup.add(questions[lang]["q61"])
        return await message.reply(questions[lang]["q67"], reply_markup=markup)
    
    else:
        for i in data_delete:
            markup.add(i["username"])
        markup.add(questions[lang]["q61"])
        return await message.reply(questions[lang]["q62"], reply_markup=markup)

@dp.message_handler(lambda message: message.text not in ['CANCEL', 'CANCELAR'], state=Form.delete)
async def deleting_data(message: types.Message):
    await Form.menu.set()
    lang = message.from_user.locale.language
    id = message.from_user.id
    response = requests.delete(API_HOST+'users/%s/%s'%(id, message.text))
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add(questions[lang]["q26"], questions[lang]["q27"])
    return await message.reply(questions[lang]["q63"] % json.loads(response.content)['Status'], reply_markup=markup)

@dp.message_handler(lambda message: message.text in ["Exit", "Salir", "Sortir"], state=Form.menu)
async def exit(message: types.Message):
    lang = message.from_user.locale.language
    await Form.start.set()
    return await message.reply(questions[lang]["q64"], reply_markup=types.ReplyKeyboardRemove())
# You can use state '*' if you need to handle all states

@dp.message_handler(state='*', commands='cancel')
@dp.message_handler(Text(equals='cancel', ignore_case=True), state='*')
async def cancel_handler(message: types.Message, state: FSMContext):
    """
    Allow user to cancel any action
    """
    lang = message.from_user.locale.language
    current_state = await state.get_state()
    if current_state is None:
        return

    await Form.start.set()
    # And remove keyboard (just in case)
    await message.reply(questions[lang]["q2"], reply_markup=types.ReplyKeyboardRemove())

@dp.message_handler(state='*', commands='stop')
@dp.message_handler(Text(equals='stop', ignore_case=True), state='*')
async def stop_handler(message: types.Message, state: FSMContext):
    """
    Allow user to stop the questions. Still saves everything answered. For test purposes.
    """
    lang = message.from_user.locale.language
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
    lang = message.from_user.locale.language
    return await message.reply(questions[lang]["q6"])


@dp.message_handler(state=Form.username)
async def process_username(message: types.Message, state: FSMContext):
    lang = message.from_user.locale.language
    id = message.from_user.id
    async with state.proxy() as data:
        data['id'] = id
        data['username'] = message.text

    await Form.next()
    await message.reply(questions[lang]["q7"])


# Check age. Age has to be a digit
@dp.message_handler(lambda message: (not message.text.isdigit()) or (int(message.text) not in np.arange(0,121)), state=Form.age)
async def process_age_invalid(message: types.Message):
    lang = message.from_user.locale.language
    """
    If age is invalid
    """
    return await message.reply(questions[lang]["q8"])

@dp.message_handler(lambda message: message.text.isdigit(), state=Form.age)
async def process_age(message: types.Message, state: FSMContext):
    lang = message.from_user.locale.language
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

@dp.message_handler(lambda message: message.text not in ['Male', 'Female', 'Other', 'Hombre', 'Mujer', 'Otro', 'Home', 'Dona', 'Altre'], state=Form.gender)
async def process_gender_invalid(message: types.Message):
    lang = message.from_user.locale.language
    """
    In this example gender has to be one of: Male, Female, Other.
    """

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add(questions[lang]["q9"], questions[lang]["q10"])
    markup.add(questions[lang]["q11"])

    return await message.reply(questions[lang]["q12"], reply_markup=markup)

@dp.message_handler(state=Form.gender)
async def process_gender(message: types.Message, state: FSMContext):
    lang = message.from_user.locale.language
    async with state.proxy() as data:
        data['gender'] = message.text
        #markup = types.ReplyKeyboardRemove()

        location_keyboard  = types.KeyboardButton(text=questions[lang]["q13"], request_location=True)
        reply_markup = types.ReplyKeyboardMarkup([[location_keyboard]], resize_keyboard=True)
        reply_markup.add(questions[lang]["q66"])


    await Form.next()
    #await message.reply("In which country are you right now?", reply_markup=markup)
    return await message.reply(questions[lang]["q14"], reply_markup=reply_markup)


@dp.message_handler(state=Form.location, content_types=['location'])
async def process_location(message, state: FSMContext):
    lang = message.from_user.locale.language
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

@dp.message_handler(lambda message: "Skip", state=Form.location)
async def process_location(message, state: FSMContext):
    lang = message.from_user.locale.language

    await Form.next()
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add(questions[lang]["q16"])
    markup.add(questions[lang]["q17"])
    markup.add(questions[lang]["q18"])
    await message.reply(questions[lang]["q19"], reply_markup=markup)

@dp.message_handler(lambda message: message.text not in ["Currently positive", "Had covid in the past", "Never been diagnosed",
                                                         "Positivo","Negativo", "Desconocido",
                                                         "Positiu","Negatiu","Desconegut"], state=Form.has_corona)
async def process_has_corona_invalid(message: types.Message):
    lang = message.from_user.locale.language
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
    lang = message.from_user.locale.language
    async with state.proxy() as data:
        data['diagnosis'] = message.text
        markup =types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
        markup.add(questions[lang]["q26"], questions[lang]["q27"])
    await Form.next()
    await message.reply(questions[lang]["q39"], reply_markup=markup)

@dp.message_handler(state=Form.vaccine)
async def process_tiredness(message: types.Message, state: FSMContext):
    lang = message.from_user.locale.language
    async with state.proxy() as data:
        data['vaccine'] = str(message.text == questions[lang]["q26"])

    #markup = types.ReplyKeyboardRemove()
    #await Form.next()
    # await message.reply(questions[lang]["q20"], reply_markup=markup)
    await Form.next()
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add(questions[lang]["q26"], questions[lang]["q27"])
    await message.reply(questions[lang]["q25"], reply_markup=markup)


@dp.message_handler(lambda message: message.text not in ['Sí', 'Yes', 'No'], state=Form.dry_cough)
async def process_binary_invalid(message: types.Message):
    lang = message.from_user.locale.language
    """
    Text filter.
    """
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add(questions[lang]["q26"], questions[lang]["q27"])

    return await message.reply(questions[lang]["q12"], reply_markup=markup)


@dp.message_handler(state=Form.dry_cough)
async def process_dry_cough(message: types.Message, state: FSMContext):
    lang = message.from_user.locale.language
    async with state.proxy() as data:
        data['symptoms'] = {}
        data['symptoms']['dry cough'] = str(message.text == questions[lang]["q26"])

    await Form.next()
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add(questions[lang]["q26"], questions[lang]["q27"])
    await message.reply(questions[lang]["q28"], reply_markup=markup)

@dp.message_handler(lambda message: message.text not in ['Sí', 'Yes', 'No'], state=Form.smoker)
async def process_smoker_invalid(message: types.Message):
    lang = message.from_user.locale.language

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add(questions[lang]["q26"], questions[lang]["q27"])

    return await message.reply(questions[lang]["q12"], reply_markup=markup)


@dp.message_handler(state=Form.smoker)
async def process_smoker(message: types.Message, state: FSMContext):
    lang = message.from_user.locale.language
    async with state.proxy() as data:
        data['symptoms']['smoker'] = str(message.text == questions[lang]["q26"])

    await Form.next()
    await message.reply(questions[lang]["q29"])


@dp.message_handler(lambda message: message.text not in ['Sí', 'Yes', 'No'], state=Form.cold)
async def process_tiredness_invalid(message: types.Message):
    lang = message.from_user.locale.language

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add(questions[lang]["q26"], questions[lang]["q27"])

    return await message.reply(questions[lang]["q12"], reply_markup=markup)


@dp.message_handler(state=Form.cold)
async def process_tiredness(message: types.Message, state: FSMContext):
    lang = message.from_user.locale.language
    async with state.proxy() as data:
        data['symptoms']['cold'] = str(message.text == questions[lang]["q26"])

    await Form.next()
    await message.reply(questions[lang]["q30"])


@dp.message_handler(lambda message: message.text not in ['Sí', 'Yes', 'No'], state=Form.res_difficult)
async def process_res_difficult_invalid(message: types.Message):
    lang = message.from_user.locale.language

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add(questions[lang]["q26"], questions[lang]["q27"])

    return await message.reply(questions[lang]["q12"], reply_markup=markup)


@dp.message_handler(state=Form.res_difficult)
async def process_loss_smell(message: types.Message, state: FSMContext):
    lang = message.from_user.locale.language
    async with state.proxy() as data:
        data['symptoms']['res_difficult'] = str(message.text == questions[lang]["q26"])

    await Form.next()
    await message.reply(questions[lang]["q31"])


@dp.message_handler(lambda message: message.text not in ['Sí', 'Yes', 'No'], state=Form.sore_throat)
async def process_headache_invalid(message: types.Message):
    lang = message.from_user.locale.language

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add(questions[lang]["q26"], questions[lang]["q27"])

    return await message.reply(questions[lang]["q12"], reply_markup=markup)


@dp.message_handler(state=Form.sore_throat)
async def process_headache(message: types.Message, state: FSMContext):
    lang = message.from_user.locale.language
    async with state.proxy() as data:
        data['symptoms']['sore_throat'] = str(message.text == questions[lang]["q26"])

    await Form.next()
    await message.reply(questions[lang]["q40"])


@dp.message_handler(lambda message: message.text not in ['Sí', 'Yes', 'No'], state=Form.fever)
async def process_fever_invalid(message: types.Message):
    lang = message.from_user.locale.language

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add(questions[lang]["q26"], questions[lang]["q27"])

    return await message.reply(questions[lang]["q12"], reply_markup=markup)


@dp.message_handler(state=Form.fever)
async def process_shortness_breath(message: types.Message, state: FSMContext):
    lang = message.from_user.locale.language
    async with state.proxy() as data:
        data['symptoms']['fever'] = str(message.text == questions[lang]["q26"])

    await Form.next()
    await message.reply(questions[lang]["q41"])


@dp.message_handler(lambda message: message.text not in ['Sí', 'Yes', 'No'], state=Form.fatigue)
async def process_chest_pain_invalid(message: types.Message):
    lang = message.from_user.locale.language

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add(questions[lang]["q26"], questions[lang]["q27"])

    return await message.reply(questions[lang]["q12"], reply_markup=markup)


@dp.message_handler(lambda message: message.text in ['Sí', 'Yes', 'No'],state=Form.fatigue)
async def process_chest_pain(message: types.Message, state: FSMContext):
    lang = message.from_user.locale.language
    async with state.proxy() as data:
        data['symptoms']['fatigue'] = str(message.text == questions[lang]["q26"])
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
        markup.add(questions[lang]["q26"], questions[lang]["q27"])

    await Form.next()
    await message.reply(questions[lang]["q42"], reply_markup=markup)


@dp.message_handler(lambda message: message.text in ['Sí', 'Yes', 'No'],state=Form.muscular)
async def process_chest_pain(message: types.Message, state: FSMContext):
    lang = message.from_user.locale.language
    async with state.proxy() as data:
        data['symptoms']['muscular_pain'] = str(message.text == questions[lang]["q26"])
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
        markup.add(questions[lang]["q26"], questions[lang]["q27"])

    await Form.next()
    await message.reply(questions[lang]["q43"], reply_markup=markup)


@dp.message_handler(lambda message: message.text in ['Sí', 'Yes', 'No'],state=Form.smell)
async def process_chest_pain(message: types.Message, state: FSMContext):
    lang = message.from_user.locale.language
    async with state.proxy() as data:
        data['symptoms']['smell_loss'] = str(message.text == questions[lang]["q26"])
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
        markup.add(questions[lang]["q26"], questions[lang]["q27"])

    await Form.next()
    await message.reply(questions[lang]["q44"], reply_markup=markup)

@dp.message_handler(lambda message: message.text in ['Sí', 'Yes', 'No'],state=Form.pneumonia)
async def process_chest_pain(message: types.Message, state: FSMContext):
    lang = message.from_user.locale.language
    async with state.proxy() as data:
        data['symptoms']['pneumonia'] = str(message.text == questions[lang]["q26"])
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
        markup.add(questions[lang]["q26"], questions[lang]["q27"])

    await Form.next()
    await message.reply(questions[lang]["q45"], reply_markup=markup)

@dp.message_handler(lambda message: message.text in ['Sí', 'Yes', 'No'],state=Form.diarrhea)
async def process_chest_pain(message: types.Message, state: FSMContext):
    lang = message.from_user.locale.language
    async with state.proxy() as data:
        data['symptoms']['diarrhea'] = str(message.text == questions[lang]["q26"])
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
        markup.add(questions[lang]["q26"], questions[lang]["q27"])

    await Form.next()
    await message.reply(questions[lang]["q46"], reply_markup=markup)

@dp.message_handler(lambda message: message.text in ['Sí', 'Yes', 'No'],state=Form.hypertension)
async def process_chest_pain(message: types.Message, state: FSMContext):
    lang = message.from_user.locale.language
    async with state.proxy() as data:
        data['symptoms']['hypertension'] = str(message.text == questions[lang]["q26"])
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
        markup.add(questions[lang]["q26"], questions[lang]["q27"])

    await Form.next()
    await message.reply(questions[lang]["q47"], reply_markup=markup)

@dp.message_handler(lambda message: message.text in ['Sí', 'Yes', 'No'],state=Form.asma)
async def process_chest_pain(message: types.Message, state: FSMContext):
    lang = message.from_user.locale.language
    async with state.proxy() as data:
        data['symptoms']['asthma'] = str(message.text == questions[lang]["q26"])
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
        markup.add(questions[lang]["q26"], questions[lang]["q27"])

    await Form.next()
    await message.reply(questions[lang]["q48"], reply_markup=markup)

@dp.message_handler(lambda message: message.text in ['Sí', 'Yes', 'No'],state=Form.diabetes)
async def process_chest_pain(message: types.Message, state: FSMContext):
    lang = message.from_user.locale.language
    async with state.proxy() as data:
        data['symptoms']['diabetes'] = str(message.text == questions[lang]["q26"])
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
        markup.add(questions[lang]["q26"], questions[lang]["q27"])

    await Form.next()
    await message.reply(questions[lang]["q49"], reply_markup=markup)

@dp.message_handler(lambda message: message.text in ['Sí', 'Yes', 'No'],state=Form.CLD)
async def process_chest_pain(message: types.Message, state: FSMContext):
    lang = message.from_user.locale.language
    async with state.proxy() as data:
        data['symptoms']['CLD'] = str(message.text == questions[lang]["q26"])
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
        markup.add(questions[lang]["q26"], questions[lang]["q27"])

    await Form.next()
    await message.reply(questions[lang]["q50"], reply_markup=markup)

@dp.message_handler(lambda message: message.text in ['Sí', 'Yes', 'No'],state=Form.IHD)
async def process_chest_pain(message: types.Message, state: FSMContext):
    lang = message.from_user.locale.language
    async with state.proxy() as data:
        data['symptoms']['IHD'] = str(message.text == questions[lang]["q26"])
        #markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
        #markup.add(questions[lang]["q26"], questions[lang]["q27"])

    #await Form.next()
    #await message.reply("This is the end of the form. Do you want to add any extra information?"), reply_markup=markup)
    markup = types.ReplyKeyboardRemove()
    await Form.next()
    await message.reply(questions[lang]["q20"], reply_markup=markup)




# Message handler if a non voice message is received
@dp.message_handler(lambda message: types.message.ContentType not in ['voice'], state=Form.cough)
async def process_cough_invalid(message: types.Message):
    lang = message.from_user.locale.language
    """
    Filter.
    """
    return await message.reply(questions[lang]["q21"])


@dp.message_handler(state=Form.cough, content_types=types.message.ContentType.VOICE)
async def process_cough(message: types.voice.Voice, state: FSMContext):
    lang = message.from_user.locale.language
    # Update state and data
    await bot.send_message(message.chat.id,questions[lang]["q22"])

    file_id = message.voice.file_id
    file = await bot.get_file(file_id)
    file_path_URL = file.file_path

    file_path = '/tmp/{}.oga'.format(file_id)
    #file_path = '/home/dani/covidscipy2020/{}.oga'.format(file_id)
    # file_path = 'C:/Users/Guillem/Desktop/prueba_audio/{}.oga'.format(file_id)
    #Aquí deberemos indicar el directorio dónce guardemos el archivo en el servidor
    async with state.proxy() as data:
        data['file_path'] = file_path

    await bot.download_file(file_path_URL, file_path)

    ###test area
    #await bot.send_message(message.chat.id, questions[lang]["q24"])
    #await Form.next()

    duration = check_audio_duration(file_path)
    if duration < 1.0:
        try:
            f = open(file_path)

        except IOError:
            print("File not accessible")

        finally:
            f.close()
            os.remove(file_path)
        await bot.send_message(message.chat.id, questions[lang]["q51"])

    elif duration >= 7.0:
        try:
            f = open(file_path)

        except IOError:
            print("File not accessible")

        finally:
            f.close()
            os.remove(file_path)
        return await bot.send_message(message.chat.id, questions[lang]["q52"])

    else:
        async with state.proxy() as data:
            veredict = analyze_cough(file_path, data)
            

    if veredict == None:
        # The audio is not recognised as COUGH
        return await bot.send_message(message.chat.id, questions[lang]["q23"])

    else:
        if veredict == True:
            await bot.send_message(message.chat.id, questions[lang]["q53"])
        elif veredict == False:
            await bot.send_message(message.chat.id, questions[lang]["q54"])

        async with state.proxy() as data:
            #objectID = database.store_oga_GridFS(file_path, data['diagnosis'], veredict)
            data['audio_file'] = {}
            data['audio_file']['filename'] = file_id
            #data['audio_file']['ObjectID'] = objectID
            data['audio_file']['covid_positive'] = veredict




        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
        markup.add(questions[lang]["q26"], questions[lang]["q27"])
        await Form.next()
        return await message.reply(questions[lang]["q55"], reply_markup=markup)



@dp.message_handler(lambda message: message.text == "No",state=Form.others)
async def process_others(message: types.Message, state: FSMContext):
    lang = message.from_user.locale.language
    async with state.proxy() as data:
        data['symptoms']['others'] = message.text

        markup = types.ReplyKeyboardRemove()
        await message.reply(questions[lang]["q35"], reply_markup=markup)

        file_path = data['file_path']
        print(str(file_path))
        del data['file_path']
        file = {'upload_file': open(file_path, 'rb'),
                'json': (None, json.dumps(data.as_dict()), 'application/json')}

        requests.post(API_HOST+'users', json=data.as_dict())

    await bot.send_message(
        message.chat.id,
        questions[lang]["q4"]
    )
    await Form.start.set()

@dp.message_handler(lambda message: message.text in ['Yes', 'Sí'], state=Form.others)
async def process_others_write(message: types.Message, state: FSMContext):
    lang = message.from_user.locale.language
    await message.reply(questions[lang]["q65"])


@dp.message_handler(lambda message: message.text not in ['Sí', 'Yes', 'No'], state=Form.others)
async def process_others(message: types.Message, state: FSMContext):
    lang = message.from_user.locale.language
    async with state.proxy() as data:
        data['symptoms']['others'] = message.text
        file_path = data['file_path']
        markup = types.ReplyKeyboardRemove()
        await message.reply(questions[lang]["q35"], reply_markup=markup)
        #await message.reply(file_path, reply_markup=markup)


        #save_features(data.as_dict())

        '''
        Insertamos los datos con formato diccionario. No hace falta
        convertirlos a JSON ya que la propia BBDD de MongoDB los convierte
        a BSON al insertar el documento

        Un hecho relevante es que la propia Colección le agrega un ID único
        a cada Documento (a cada paciente). Este ID es del tipo Object_id el
        cual también almacena el momento en el que el usuario se ha registrado.
        '''
        file_path = data['file_path']
        print(str(file_path))
        del data['file_path']
        file = {'upload_file': open(file_path, 'rb'),
                 'json': (None, json.dumps(data.as_dict()), 'application/json')}

        requests.post(API_HOST+'users', files=file)

    await bot.send_message(
            message.chat.id,
            questions[lang]["q4"])

        #requests.post(API_HOST+'users', json=data.as_dict())


    await Form.start.set()


def main():
    executor.start_polling(dp, skip_updates=True)
    logging.basicConfig(level=logging.INFO)

    start_webhook(
        dispatcher=dp,
        webhook_path=WEBHOOK_PATH,
        skip_updates=True,
        host=WEBAPP_HOST,
        #port=WEBAPP_PORT
    )
main()
