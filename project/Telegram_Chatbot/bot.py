import logging
import requests
import sys
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
from project.Telegram_Chatbot.modulos.json_tools import *
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
    """
    .. note::
        **State machine:**
        This bot works as a state machine. At each message, the bot will be in one state. Depending on which one, the bot will react accordingly. Languaje (lang) is set on every function.

    """
    start = State()
    menu = State()
    delete = State()
    deleting = State()
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
@dp.message_handler(lambda message: message.text == 'No', state=Form.deleting)
@dp.message_handler(lambda message: message.text in ['CANCEL', 'CANCELAR'], state=Form.delete)
@dp.message_handler(lambda message: message.text not in ["About", "Acerca de", "Sobre nosaltres",
                                                         "Delete data", "Eliminar datos", "Eliminar dades",
                                                         "Add data", "Añadir datos", "Afegir dades",
                                                         "Exit", "Salir", "Sortir"], state=Form.menu)
async def cmd_start(message: types.Message):
    """
    Starting point.

    +-------------+---------------+---------------+
    |State        |Message        |Command        |
    +=============+===============+===============+
    |None         |               |               |
    +-------------+---------------+---------------+
    |start        |               |               |
    +-------------+---------------+---------------+
    |'*'  (any)   |               |Cancel         |
    +-------------+---------------+---------------+
    |deleting     |No             |               |
    +-------------+---------------+---------------+
    |delete       |CANCEL         |               |
    +-------------+---------------+---------------+
    |menu         |NOT IN:        |               |
    |             | - About       |               |
    |             | - Delete data |               |
    |             | - Add data    |               |
    |             | - Exit        |               |
    +-------------+---------------+---------------+

    --Output state             menu
    """

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
    """
    --Input state              menu
    --Input message            "About"
    --Output state             menu
    """
    lang = message.from_user.locale.language
    return await message.reply(questions[lang]["q37"])

@dp.message_handler(lambda message: message.text in ["Add data", "Añadir datos", "Afegir dades"], state=Form.menu)
async def add_my_data(message: types.Message):
    """
    --Input state              menu
    --Input message            "Add data"
    --Output state             username
    """
    await Form.username.set()
    lang = message.from_user.locale.language
    return await message.reply(questions[lang]["q38"], reply_markup=types.ReplyKeyboardRemove())


@dp.message_handler(lambda message: message.text in ['Sí', 'Yes'], state=Form.deleting)
@dp.message_handler(lambda message: message.text in ["No", "Delete data", "Eliminar datos", "Eliminar dades"], state=Form.menu)
async def delete_data(message: types.Message):
    """
    Performs a **GET** with user's *id*. Returns all entries from that *id*.
        +-------------+---------------+
        |State        |Message        |
        +=============+===============+
        |deleting     |Yes            |
        +-------------+---------------+
        |menu         | - No          |
        |             | - Delete data |
        +-------------+---------------+

        --Output state             delete
    """
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
    """
    Perfoms a **DELETE** on the entry seleected by *username* variable.
    --Input state              delete
    --Input message            NOT IN: 'Add data'
    --Output state             deleting
    """
    lang = message.from_user.locale.language
    await Form.deleting.set()
    id = message.from_user.id
    response = requests.delete(API_HOST+'users/%s/%s'%(id, message.text))
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add(questions[lang]["q26"], questions[lang]["q27"])
    return await message.reply(questions[lang]["q63"] % json.loads(response.content)['Status'], reply_markup=markup)

@dp.message_handler(lambda message: message.text in ["Exit", "Salir", "Sortir"], state=Form.menu)
async def exit(message: types.Message):
    """
    --Input state              menu
    --Input message            "Exit"
    --Output state             start
    """
    lang = message.from_user.locale.language
    await Form.start.set()
    return await message.reply(questions[lang]["q64"], reply_markup=types.ReplyKeyboardRemove())
# You can use state '*' if you need to handle all states

@dp.message_handler(state='*', commands='cancel')
@dp.message_handler(Text(equals='cancel', ignore_case=True), state='*')
async def cancel_handler(message: types.Message, state: FSMContext):
    """
    --Input state              '*'
    --Input command            /cancel
    --Output state             start
    """
    lang = message.from_user.locale.language
    current_state = await state.get_state()
    if current_state is None:
        return

    await Form.start.set()
    # And remove keyboard (just in case)
    await message.reply(questions[lang]["q2"], reply_markup=types.ReplyKeyboardRemove())


# Check username.
@dp.message_handler(lambda message: not message.text.isalpha(), state=Form.username)
async def process_user_invalid(message: types.Message):
    """
    --Input state              username
    --Input message            NOT IN: alpha
    --Output state             username
    Philter so the username is only alpha.
    """
    lang = message.from_user.locale.language
    return await message.reply(questions[lang]["q6"])


@dp.message_handler(state=Form.username)
async def process_username(message: types.Message, state: FSMContext):
    """
    --Input state              username
    --Input message            alpha
    --Output state             age

    Stores *Input message* as ['username'] and the id as ['id'] in the metadata.
    """
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
    """
    --Input state              age
    --Input message            NOT IN: digit
    --Output state             age

    Philter so age is digit only.
    """
    lang = message.from_user.locale.language
    """
    If age is invalid
    """
    return await message.reply(questions[lang]["q8"])

@dp.message_handler(lambda message: message.text.isdigit(), state=Form.age)
async def process_age(message: types.Message, state: FSMContext):
    """
    --Input state              age
    --Input message            digit
    --Output state             gender

    Stores *Input message* as ['age'] in the metadata.
    """

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
    """
    --Input state              gender
    --Input message            NOT IN: 'Male', 'Female', 'Other'
    --Output state             gender

    Philter so gender is a correct input.
    """
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
    """
    --Input state              gender
    --Input message            IN: 'Male', 'Female', 'Other'
    --Output state             location

    Stores *Input message* as ['gender'] in the metadata.
    """
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
    """
    --Input state              location
    --Input message            location
    --Output state             has_corona

    Stores *Input message* as ['location']['latitude'] / ['location']['longitude'] in the metadata. Only available on phone.
    """
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
    """
    --Input state              location
    --Input message            'Skip'
    --Output state             has_corona

    Skips location.
    """
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
    """
    --Input state              has_corona
    --Input message            NOT IN: "Currently positive", "Had covid in the past", "Never been diagnosed"
    --Output state             has_corona

    Philter so diagnosis is a correct input.
    """
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
    """
    --Input state              has_corona
    --Input message            IN: "Currently positive", "Had covid in the past", "Never been diagnosed"
    --Output state             vaccine

    Stores *Input message* as ['diagnosis'] in the metadata.
    """
    lang = message.from_user.locale.language
    async with state.proxy() as data:
        data['diagnosis'] = message.text
        markup =types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
        markup.add(questions[lang]["q26"], questions[lang]["q27"])
    await Form.next()
    await message.reply(questions[lang]["q39"], reply_markup=markup)

@dp.message_handler(lambda message: message.text in ['Sí', 'Yes', 'No'], state=Form.vaccine)
async def process_tiredness(message: types.Message, state: FSMContext):
    """
    --Input state              vaccine
    --Input message            IN: 'Yes', 'No'
    --Output state             dry_cough

    Stores *Input message* as ['vaccine'] in the metadata.
    """
    lang = message.from_user.locale.language
    async with state.proxy() as data:
        data['vaccine'] = (message.text == questions[lang]["q26"])

    #markup = types.ReplyKeyboardRemove()
    #await Form.next()
    # await message.reply(questions[lang]["q20"], reply_markup=markup)
    await Form.next()
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add(questions[lang]["q26"], questions[lang]["q27"])
    await message.reply(questions[lang]["q25"], reply_markup=markup)



@dp.message_handler(lambda message: message.text in ['Sí', 'Yes', 'No'], state=Form.dry_cough)
async def process_dry_cough(message: types.Message, state: FSMContext):
    """
    --Input state              dry_cough
    --Input message            IN: 'Yes', 'No'
    --Output state             smoker

    Stores *Input message* as ['symptoms']['dry cough'] in the metadata.
    """
    lang = message.from_user.locale.language
    async with state.proxy() as data:
        data['symptoms'] = {}
        data['symptoms']['dry cough'] = (message.text == questions[lang]["q26"])

    await Form.next()
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add(questions[lang]["q26"], questions[lang]["q27"])
    await message.reply(questions[lang]["q28"], reply_markup=markup)


@dp.message_handler(lambda message: message.text in ['Sí', 'Yes', 'No'], state=Form.smoker)
async def process_smoker(message: types.Message, state: FSMContext):
    """
    --Input state              smoker
    --Input message            IN: 'Yes', 'No'
    --Output state             cold

    Stores *Input message* as ['symptoms']['smoker'] in the metadata.
    """
    lang = message.from_user.locale.language
    async with state.proxy() as data:
        data['symptoms']['smoker'] = (message.text == questions[lang]["q26"])

    await Form.next()
    await message.reply(questions[lang]["q29"])



@dp.message_handler(lambda message: message.text in ['Sí', 'Yes', 'No'], state=Form.cold)
async def process_tiredness(message: types.Message, state: FSMContext):
    """
    --Input state              cold
    --Input message            IN: 'Yes', 'No'
    --Output state             res_difficult

    Stores *Input message* as ['symptoms']['cold'] in the metadata.
    """
    lang = message.from_user.locale.language
    async with state.proxy() as data:
        data['symptoms']['cold'] = (message.text == questions[lang]["q26"])

    await Form.next()
    await message.reply(questions[lang]["q30"])


@dp.message_handler(lambda message: message.text in ['Sí', 'Yes', 'No'], state=Form.res_difficult)
async def process_loss_smell(message: types.Message, state: FSMContext):
    """
    --Input state              res_difficult
    --Input message            IN: 'Yes', 'No'
    --Output state             sore_throat

    Stores *Input message* as ['symptoms']['res_difficult'] in the metadata.
    """
    lang = message.from_user.locale.language
    async with state.proxy() as data:
        data['symptoms']['res_difficult'] = (message.text == questions[lang]["q26"])

    await Form.next()
    await message.reply(questions[lang]["q31"])



@dp.message_handler(lambda message: message.text in ['Sí', 'Yes', 'No'], state=Form.sore_throat)
async def process_headache(message: types.Message, state: FSMContext):
    """
    --Input state              sore_throat
    --Input message            IN: 'Yes', 'No'
    --Output state             fever

    Stores *Input message* as ['symptoms']['sore_throat'] in the metadata.
    """
    lang = message.from_user.locale.language
    async with state.proxy() as data:
        data['symptoms']['sore_throat'] = (message.text == questions[lang]["q26"])

    await Form.next()
    await message.reply(questions[lang]["q40"])


@dp.message_handler(lambda message: message.text in ['Sí', 'Yes', 'No'], state=Form.fever)
async def process_shortness_breath(message: types.Message, state: FSMContext):
    """
    --Input state              fever
    --Input message            IN: 'Yes', 'No'
    --Output state             fatigue

    Stores *Input message* as ['symptoms']['fever'] in the metadata.
    """
    lang = message.from_user.locale.language
    async with state.proxy() as data:
        data['symptoms']['fever'] = (message.text == questions[lang]["q26"])

    await Form.next()
    await message.reply(questions[lang]["q41"])


@dp.message_handler(lambda message: message.text in ['Sí', 'Yes', 'No'],state=Form.fatigue)
async def process_chest_pain(message: types.Message, state: FSMContext):
    """
    --Input state              fatigue
    --Input message            IN: 'Yes', 'No'
    --Output state             muscular

    Stores *Input message* as ['symptoms']['fatigue'] in the metadata.
    """
    lang = message.from_user.locale.language
    async with state.proxy() as data:
        data['symptoms']['fatigue'] = (message.text == questions[lang]["q26"])
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
        markup.add(questions[lang]["q26"], questions[lang]["q27"])

    await Form.next()
    await message.reply(questions[lang]["q42"], reply_markup=markup)


@dp.message_handler(lambda message: message.text in ['Sí', 'Yes', 'No'],state=Form.muscular)
async def process_chest_pain(message: types.Message, state: FSMContext):
    """
    --Input state              muscular
    --Input message            IN: 'Yes', 'No'
    --Output state             smell

    Stores *Input message* as ['symptoms']['muscular_pain'] in the metadata.
    """
    lang = message.from_user.locale.language
    async with state.proxy() as data:
        data['symptoms']['muscular_pain'] = (message.text == questions[lang]["q26"])
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
        markup.add(questions[lang]["q26"], questions[lang]["q27"])

    await Form.next()
    await message.reply(questions[lang]["q43"], reply_markup=markup)


@dp.message_handler(lambda message: message.text in ['Sí', 'Yes', 'No'],state=Form.smell)
async def process_chest_pain(message: types.Message, state: FSMContext):
    """
    --Input state              smell
    --Input message            IN: 'Yes', 'No'
    --Output state             pneumonia

    Stores *Input message* as ['symptoms']['smell_loss'] in the metadata.
    """
    lang = message.from_user.locale.language
    async with state.proxy() as data:
        data['symptoms']['smell_loss'] = (message.text == questions[lang]["q26"])
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
        markup.add(questions[lang]["q26"], questions[lang]["q27"])

    await Form.next()
    await message.reply(questions[lang]["q44"], reply_markup=markup)

@dp.message_handler(lambda message: message.text in ['Sí', 'Yes', 'No'],state=Form.pneumonia)
async def process_chest_pain(message: types.Message, state: FSMContext):
    """
    --Input state              pneumonia
    --Input message            IN: 'Yes', 'No'
    --Output state             diarrhea

    Stores *Input message* as ['symptoms']['pneumonia'] in the metadata.
    """
    lang = message.from_user.locale.language
    async with state.proxy() as data:
        data['symptoms']['pneumonia'] = (message.text == questions[lang]["q26"])
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
        markup.add(questions[lang]["q26"], questions[lang]["q27"])

    await Form.next()
    await message.reply(questions[lang]["q45"], reply_markup=markup)

@dp.message_handler(lambda message: message.text in ['Sí', 'Yes', 'No'],state=Form.diarrhea)
async def process_chest_pain(message: types.Message, state: FSMContext):
    """
    --Input state              diarrhea
    --Input message            IN: 'Yes', 'No'
    --Output state             hypertension

    Stores *Input message* as ['symptoms']['diarrhea'] in the metadata.
    """
    lang = message.from_user.locale.language
    async with state.proxy() as data:
        data['symptoms']['diarrhea'] = (message.text == questions[lang]["q26"])
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
        markup.add(questions[lang]["q26"], questions[lang]["q27"])

    await Form.next()
    await message.reply(questions[lang]["q46"], reply_markup=markup)

@dp.message_handler(lambda message: message.text in ['Sí', 'Yes', 'No'],state=Form.hypertension)
async def process_chest_pain(message: types.Message, state: FSMContext):
    """
    --Input state              hypertension
    --Input message            IN: 'Yes', 'No'
    --Output state             asma

    Stores *Input message* as ['symptoms']['hypertension'] in the metadata.
    """
    lang = message.from_user.locale.language
    async with state.proxy() as data:
        data['symptoms']['hypertension'] = (message.text == questions[lang]["q26"])
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
        markup.add(questions[lang]["q26"], questions[lang]["q27"])

    await Form.next()
    await message.reply(questions[lang]["q47"], reply_markup=markup)

@dp.message_handler(lambda message: message.text in ['Sí', 'Yes', 'No'],state=Form.asma)
async def process_chest_pain(message: types.Message, state: FSMContext):
    """
    --Input state              asma
    --Input message            IN: 'Yes', 'No'
    --Output state             diabetes

    Stores *Input message* as ['symptoms']['asthma'] in the metadata.
    """
    lang = message.from_user.locale.language
    async with state.proxy() as data:
        data['symptoms']['asthma'] = (message.text == questions[lang]["q26"])
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
        markup.add(questions[lang]["q26"], questions[lang]["q27"])

    await Form.next()
    await message.reply(questions[lang]["q48"], reply_markup=markup)

@dp.message_handler(lambda message: message.text in ['Sí', 'Yes', 'No'],state=Form.diabetes)
async def process_chest_pain(message: types.Message, state: FSMContext):
    """
    --Input state              diabetes
    --Input message            IN: 'Yes', 'No'
    --Output state             CLD

    Stores *Input message* as ['symptoms']['diabetes'] in the metadata.
    """
    lang = message.from_user.locale.language
    async with state.proxy() as data:
        data['symptoms']['diabetes'] = (message.text == questions[lang]["q26"])
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
        markup.add(questions[lang]["q26"], questions[lang]["q27"])

    await Form.next()
    await message.reply(questions[lang]["q49"], reply_markup=markup)

@dp.message_handler(lambda message: message.text in ['Sí', 'Yes', 'No'],state=Form.CLD)
async def process_chest_pain(message: types.Message, state: FSMContext):
    """
    --Input state              CLD
    --Input message            IN: 'Yes', 'No'
    --Output state             IHD

    Stores *Input message* as ['symptoms']['CLD'] in the metadata.
    """
    lang = message.from_user.locale.language
    async with state.proxy() as data:
        data['symptoms']['CLD'] = (message.text == questions[lang]["q26"])
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
        markup.add(questions[lang]["q26"], questions[lang]["q27"])

    await Form.next()
    await message.reply(questions[lang]["q50"], reply_markup=markup)

@dp.message_handler(lambda message: message.text in ['Sí', 'Yes', 'No'],state=Form.IHD)
async def process_chest_pain(message: types.Message, state: FSMContext):
    """
    --Input state              IHD
    --Input message            IN: 'Yes', 'No'
    --Output state             cough

    Stores *Input message* as ['symptoms']['IHD'] in the metadata.
    """
    lang = message.from_user.locale.language
    async with state.proxy() as data:
        data['symptoms']['IHD'] = (message.text == questions[lang]["q26"])
        #markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
        #markup.add(questions[lang]["q26"], questions[lang]["q27"])

    markup = types.ReplyKeyboardRemove()
    await Form.next()
    await message.reply(questions[lang]["q20"], reply_markup=markup)




# Message handler if a non voice message is received
@dp.message_handler(lambda message: types.message.ContentType not in ['voice'], state=Form.cough)
async def process_cough_invalid(message: types.Message):
    """
    --Input state              cough
    --Input message            NOT IN: voice message
    --Output state             cough

    Philter so only a voice message is sent.
    """
    lang = message.from_user.locale.language
    return await message.reply(questions[lang]["q21"])


@dp.message_handler(state=Form.cough, content_types=types.message.ContentType.VOICE)
async def process_cough(message: types.voice.Voice, state: FSMContext):
    """
    --Input state              cough
    --Input message            IN: voice message
    --Output state             others

    Processes cough sample in two steps:

        - Checks for duration. If it is not comprissed between 1second and 7 second an error message will be displayed.
        - Analyzes the cough using *analyze_cough*. Returns *veredict*:

    +-------------+---------------+---------------+
    |Veredict                                     |
    +=============+===============+===============+
    |None    |Audio sample not recognized as cough|
    +--------+------------------------------------+
    |False   | Covid negative                     |
    +---------+-----------------------------------+
    |True    |  Covid positive                    |
    +--------+------------------------------------+

    .. note::
        The cough sample is stored in a temporary folder in the Heroku server. Since the audio is downloaded and processed during a single *request* this causes no trouble. The audio is deleted from this temporary folder after being stored in GridFS anyway.

    - Stores veredict as ['audio_file']['covid_positive'] in the metadata.
    """
    lang = message.from_user.locale.language
    # Update state and data
    await bot.send_message(message.chat.id,questions[lang]["q22"])

    file_id = message.voice.file_id
    file = await bot.get_file(file_id)
    file_path_URL = file.file_path

    file_path = '/tmp/{}.oga'.format(file_id)
    #file_path = '/home/dani/covidscipy2020/{}.oga'.format(file_id)
    #file_path = 'C:/Users/Guillem/Desktop/prueba_audio/{}.oga'.format(file_id)
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
            data['audio_file']['covid_positive'] = veredict




        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
        markup.add(questions[lang]["q26"], questions[lang]["q27"])
        await Form.next()
        return await message.reply(questions[lang]["q55"], reply_markup=markup)



@dp.message_handler(lambda message: message.text in ['Yes', 'Sí'], state=Form.others)
async def process_others_write(message: types.Message, state: FSMContext):
    """
    --Input state              others
    --Input message            'Yes'
    --Output state             others

    """
    lang = message.from_user.locale.language
    await message.reply(questions[lang]["q65"])


@dp.message_handler(lambda message: message.text not in ['Sí', 'Yes'], state=Form.others)
async def process_others(message: types.Message, state: FSMContext):
    """
    --Input state              others
    --Input message            NOT IN: 'Yes'
    --Output state             Start

        - Stores *Input message* as ['symptoms']['others'] in the metadata.
        - Deletes *file_path* from the metadata (json).
        - Performs a **POST** request on *API_HOST+'users'* with a body containing both the metadata and the audio sample in a *files* dictionary.
        - Resets chatbot.

    .. note::
        Metadata is stored in *dict* format. The DB converts it to BSON automatically when inserting the entry.
    """
    lang = message.from_user.locale.language
    async with state.proxy() as data:
        data['symptoms']['others'] = message.text
        markup = types.ReplyKeyboardRemove()
        await message.reply(questions[lang]["q35"], reply_markup=markup)
        #await message.reply(file_path, reply_markup=markup)


        #save_features(data.as_dict())

        file_path = data['file_path']
        print(str(file_path))
        del data['file_path']
        data = convert_bool(data.as_dict())
        file = {'upload_file': open(file_path, 'rb'),
                 'json': (None, json.dumps(data), 'application/json')}

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

