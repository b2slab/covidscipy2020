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
global lang


@dp.message_handler(state = None)
@dp.message_handler(state = Form.start)
@dp.message_handler(state='*', commands='cancel')
@dp.message_handler(lambda message: message.text == "No", state=Form.menu)
@dp.message_handler(lambda message: message.text == "CANCEL", state=Form.delete)
async def cmd_start(message: types.Message):
    """
    Conversation's entry point
    """
    # Set state and language
    global lang, id, name
    locale = message.from_user.locale
    lang = locale.language
    id = message.from_user.id
    name = message.from_user.first_name

    # Si no reconoce idioma, por defecto activa el español
    if lang not in ["en","es","ca"]:
        lang = "es"
    #await bot.set_webhook(WEBHOOK_URL, drop_pending_updates=False)
    await Form.menu.set()

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add("Add data", "Delete data")
    markup.add("About", "Exit")

    await message.reply("Welcome to covid scipy %s. Your id is %s Select one of the following" %(name, id), reply_markup=markup)

@dp.message_handler(lambda message: message.text == "About", state=Form.menu)
async def about(message: types.Message):
    return await message.reply(questions[lang]["q37"])

@dp.message_handler(lambda message: message.text == "Add data", state=Form.menu)
async def add_my_data(message: types.Message):
    await Form.username.set()
    return await message.reply(questions[lang]["q38"], reply_markup=types.ReplyKeyboardRemove())



@dp.message_handler(lambda message: message.text in ["Delete data","Yes"], state=Form.menu)
async def delete_data(message: types.Message):
    await Form.delete.set()
    response = requests.get(API_HOST+'users/%s'%id)
    data_delete = json.loads(response.content)
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)

    for i in data_delete:
        markup.add(i["username"])
    markup.add("CANCEL")
    return await message.reply("These are the entries you have uploaded. Which one do you want to delete?", reply_markup=markup)

@dp.message_handler(lambda message: message.text not in ["CANCEL"], state=Form.delete)
async def deleting_data(message: types.Message):
    await Form.menu.set()
    response = requests.delete(API_HOST+'users/%s/%s'%(id, message.text))
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add("Yes", "No")
    return await message.reply("%s. Do you want to delete more entries?" % json.loads(response.content)['Status'], reply_markup=markup)

@dp.message_handler(lambda message: "Exit", state=Form.menu)
async def exit(message: types.Message):
    await Form.start.set()
    return await message.reply("Bye!", reply_markup=types.ReplyKeyboardRemove())
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

    await Form.start.set()
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
        data['id'] = id
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
        reply_markup.add('Skip')


    await Form.next()
    #await message.reply("In which country are you right now?", reply_markup=markup)
    return await message.reply(questions[lang]["q14"], reply_markup=reply_markup)


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

@dp.message_handler(lambda message: "Skip", state=Form.location)
async def process_location(message, state: FSMContext):

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
        markup =types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
        markup.add(questions[lang]["q26"], questions[lang]["q27"])
    await Form.next()
    await message.reply(questions[lang]["q39"], reply_markup=markup)

@dp.message_handler(state=Form.vaccine)
async def process_tiredness(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['vaccine'] = (message.text == questions[lang]["q26"])

    #markup = types.ReplyKeyboardRemove()
    #await Form.next()
    # await message.reply(questions[lang]["q20"], reply_markup=markup)
    await Form.next()
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add(questions[lang]["q26"], questions[lang]["q27"])
    await message.reply(questions[lang]["q25"], reply_markup=markup)


@dp.message_handler(lambda message: message.text not in [questions[lang]["q26"], questions[lang]["q27"]], state=Form.dry_cough)
async def process_binary_invalid(message: types.Message):
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

@dp.message_handler(lambda message: message.text not in [questions[lang]["q26"], questions[lang]["q27"]], state=Form.smoker)
async def process_smoker_invalid(message: types.Message):

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add(questions[lang]["q26"], questions[lang]["q27"])

    return await message.reply(questions[lang]["q12"], reply_markup=markup)


@dp.message_handler(state=Form.smoker)
async def process_smoker(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['symptoms']['smoker'] = (message.text == questions[lang]["q26"])

    await Form.next()
    await message.reply(questions[lang]["q29"])


@dp.message_handler(lambda message: message.text not in [questions[lang]["q26"], questions[lang]["q27"]], state=Form.cold)
async def process_tiredness_invalid(message: types.Message):

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add(questions[lang]["q26"], questions[lang]["q27"])

    return await message.reply(questions[lang]["q12"], reply_markup=markup)


@dp.message_handler(state=Form.cold)
async def process_tiredness(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['symptoms']['cold'] = (message.text == questions[lang]["q26"])

    await Form.next()
    await message.reply(questions[lang]["q30"])


@dp.message_handler(lambda message: message.text not in [questions[lang]["q26"], questions[lang]["q27"]], state=Form.res_difficult)
async def process_res_difficult_invalid(message: types.Message):

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add(questions[lang]["q26"], questions[lang]["q27"])

    return await message.reply(questions[lang]["q12"], reply_markup=markup)


@dp.message_handler(state=Form.res_difficult)
async def process_loss_smell(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['symptoms']['res_difficult'] = (message.text == questions[lang]["q26"])

    await Form.next()
    await message.reply(questions[lang]["q31"])


@dp.message_handler(lambda message: message.text not in [questions[lang]["q26"], questions[lang]["q27"]], state=Form.sore_throat)
async def process_headache_invalid(message: types.Message):

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add(questions[lang]["q26"], questions[lang]["q27"])

    return await message.reply(questions[lang]["q12"], reply_markup=markup)


@dp.message_handler(state=Form.sore_throat)
async def process_headache(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['symptoms']['sore_throat'] = (message.text == questions[lang]["q26"])

    await Form.next()
    await message.reply(questions[lang]["q40"])


@dp.message_handler(lambda message: message.text not in [questions[lang]["q26"], questions[lang]["q27"]], state=Form.fever)
async def process_fever_invalid(message: types.Message):

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add(questions[lang]["q26"], questions[lang]["q27"])

    return await message.reply(questions[lang]["q12"], reply_markup=markup)


@dp.message_handler(state=Form.fever)
async def process_shortness_breath(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['symptoms']['fever'] = (message.text == questions[lang]["q26"])

    await Form.next()
    await message.reply(questions[lang]["q41"])


@dp.message_handler(lambda message: message.text not in [questions[lang]["q26"], questions[lang]["q27"]], state=Form.fatigue)
async def process_chest_pain_invalid(message: types.Message):

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add(questions[lang]["q26"], questions[lang]["q27"])

    return await message.reply(questions[lang]["q12"], reply_markup=markup)


@dp.message_handler(state=Form.fatigue)
async def process_chest_pain(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['symptoms']['fatigue'] = (message.text == questions[lang]["q26"])
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
        markup.add(questions[lang]["q26"], questions[lang]["q27"])

    await Form.next()
    await message.reply(questions[lang]["q42"], reply_markup=markup)


@dp.message_handler(state=Form.muscular)
async def process_chest_pain(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['symptoms']['muscular_pain'] = (message.text == questions[lang]["q26"])
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
        markup.add(questions[lang]["q26"], questions[lang]["q27"])

    await Form.next()
    await message.reply(questions[lang]["q43"], reply_markup=markup)


@dp.message_handler(state=Form.smell)
async def process_chest_pain(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['symptoms']['smell_loss'] = (message.text == questions[lang]["q26"])
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
        markup.add(questions[lang]["q26"], questions[lang]["q27"])

    await Form.next()
    await message.reply(questions[lang]["q44"], reply_markup=markup)

@dp.message_handler(state=Form.pneumonia)
async def process_chest_pain(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['symptoms']['pneumonia'] = (message.text == questions[lang]["q26"])
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
        markup.add(questions[lang]["q26"], questions[lang]["q27"])

    await Form.next()
    await message.reply(questions[lang]["q45"], reply_markup=markup)

@dp.message_handler(state=Form.diarrhea)
async def process_chest_pain(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['symptoms']['diarrhea'] = (message.text == questions[lang]["q26"])
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
        markup.add(questions[lang]["q26"], questions[lang]["q27"])

    await Form.next()
    await message.reply(questions[lang]["q46"], reply_markup=markup)

@dp.message_handler(state=Form.hypertension)
async def process_chest_pain(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['symptoms']['hypertension'] = (message.text == questions[lang]["q26"])
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
        markup.add(questions[lang]["q26"], questions[lang]["q27"])

    await Form.next()
    await message.reply(questions[lang]["q47"], reply_markup=markup)

@dp.message_handler(state=Form.asma)
async def process_chest_pain(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['symptoms']['asthma'] = (message.text == questions[lang]["q26"])
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
        markup.add(questions[lang]["q26"], questions[lang]["q27"])

    await Form.next()
    await message.reply(questions[lang]["q48"], reply_markup=markup)

@dp.message_handler(state=Form.diabetes)
async def process_chest_pain(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['symptoms']['diabetes'] = (message.text == questions[lang]["q26"])
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
        markup.add(questions[lang]["q26"], questions[lang]["q27"])

    await Form.next()
    await message.reply(questions[lang]["q49"], reply_markup=markup)

@dp.message_handler(state=Form.CLD)
async def process_chest_pain(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['symptoms']['CLD'] = (message.text == questions[lang]["q26"])
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
        markup.add(questions[lang]["q26"], questions[lang]["q27"])

    await Form.next()
    await message.reply(questions[lang]["q50"], reply_markup=markup)

@dp.message_handler(state=Form.IHD)
async def process_chest_pain(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['symptoms']['IHD'] = (message.text == questions[lang]["q26"])
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

    global file_path
    # file_path = '/tmp/{}.oga'.format(file_id)
    file_path = '/tmp/{}.oga'.format(file_id)
    #Aquí deberemos indicar el directorio dónce guardemos el archivo en el servidor

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
        return await bot.send_message(message.chat.id, questions[lang]["q51"])
    elif duration >= 5.0:
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
            objectID = database.store_oga_GridFS(file_path)
            data['audio_file'] = {}
            data['audio_file']['filename'] = file_id
            data['audio_file']['ObjectID'] = objectID
            data['audio_file']['covid_positive'] = veredict

        try:
            f = open(file_path)

        except IOError:
            print("File not accessible")

        finally:
            f.close()
            os.remove(file_path)

        try:
            wav_path = file_path.strip('.oga') + '.wav'
            w = open(wav_path)

        except IOError:
            print("File not accessible")

        finally:
            w.close()
            os.remove(wav_path)

        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
        markup.add(questions[lang]["q26"], questions[lang]["q27"])
        await Form.next()
        return await message.reply(questions[lang]["q55"], reply_markup=markup)



@dp.message_handler(lambda message: message.text == "No",state=Form.others)
async def process_others(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['symptoms']['others'] = message.text

        markup = types.ReplyKeyboardRemove()
        await message.reply(questions[lang]["q35"], reply_markup=markup)

        '''
        await bot.send_message(
            message.chat.id,
            questions[lang]["q35"]
        )
        '''

        requests.post(API_HOST+'users', json=data.as_dict())

    await bot.send_message(
        message.chat.id,
        questions[lang]["q4"]
    )
    await Form.start.set()

@dp.message_handler(lambda message: message.text == "Yes", state=Form.others)
async def process_others_write(message: types.Message, state: FSMContext):
    await message.reply("Please write it here:")


@dp.message_handler(lambda message: message.text not in ['Yes', 'No'], state=Form.others)
async def process_others(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['symptoms']['others'] = message.text

        markup = types.ReplyKeyboardRemove()
        await message.reply(questions[lang]["q35"], reply_markup=markup)

        '''
        await bot.send_message(
            message.chat.id,
            questions[lang]["q35"]
        )
        '''

        #save_features(data.as_dict())

        '''
        Insertamos los datos con formato diccionario. No hace falta
        convertirlos a JSON ya que la propia BBDD de MongoDB los convierte
        a BSON al insertar el documento

        Un hecho relevante es que la propia Colección le agrega un ID único
        a cada Documento (a cada paciente). Este ID es del tipo Object_id el
        cual también almacena el momento en el que el usuario se ha registrado.
        '''
        #database.collection.insert_one(data.as_dict())
        #requests.get('http://0.0.0.0:5001/users', json = data)


        #requests.post('http://0.0.0.0:5001/users', json=data.as_dict())
        requests.post(API_HOST+'users', json=data.as_dict())
        #database.collection.insert_one(data.as_dict())



    await bot.send_message(
        message.chat.id,
        questions[lang]["q4"]
    )
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
