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
import os
import json


logging.basicConfig(level=logging.INFO)
API_TOKEN = '1370389029:AAFIaYXbnHLCkNYIb5azZ2iOg5BWoRdOUC8'
bot = Bot(token=API_TOKEN)
save_path = '/home/dani/covidscipy2020/data/'
# For example use simple MemoryStorage for Dispatcher.
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)
#language_text

questions = {
    "en":{
        "q1":"Hi there! Please, enter your name.",
        "q2":"Cancelled",
        "q3":"Allright we'll stop here...\n\n Please, give me a second while I upload the data.",
        "q4":"That's it!",
        "q5":"Process stopped",
        "q6":"Username is invalid. Please enter a correct username.",
        "q7":"How old are you?",
        "q8":"Age has to be a number.\n\n How old are you? (digits only)",
        "q9":"Male",
        "q10":"Female",
        "q11":"Other",
        "q12":"Inadequate answer. Please choose one of the provided options",
        "q13":"Send Current Location",
        "q14":"Would you mind to send us your current location?\n\nPlease activate location on your device.",
        "q15":"Please activate location on your device and send it to us.",
        "q16":"Positive",
        "q17":"Negative",
        "q18":"Unknown",
        "q19":"Do you have Covid-19?",
        "q20":"Could you send us a recording of your cough?",
        "q21":"Just use the audio message option from telegram and cough to the microphone.",
        "q22":"Please, give me a second while I annalyze you cough...",
        "q23":"Sorry, we didn't recognize this as cough. Please, cough again",
        "q24":"Thanks for your cough",
        "q25":"Do you have dry cough?",
        "q26":"Yes",
        "q27":"No",
        "q28":"Do you have fever?",
        "q29":"Do you feel more tired than usual?",
        "q30":"Do you feel that you have lost/diminished your sense of smell?",
        "q31":"Do you have a headache?",
        "q32":"Do you have chest pain or pressure?",
        "q33":"Do you have difficulty breathing or shortness of breath?",
        "q34":"Do you have any other information you would like to add?",
        "q35":"Thank you very much for you collaboration!\n\n Please, give me a second while I upload the data.",

        "q36":"What is your gender?"

    },
    "es":{
        "q1":"Hola! Por favor, introduzca su nombre.",
        "q2":"Cancelado",
        "q3":"Esta bien, lo dejamos aquí...\n\n Deme un segundo mientras almaceno los datos.",
        "q4":"Ya esta!",
        "q5":"Proceso detenido",
        "q6":"Usuario inválido. Por favor, introduzca un nombre correcto.",
        "q7":"¿Cuántos años tiene?",
        "q8":"Su edad debe ser un número.\n\n ¿Cuántos años tiene?",
        "q9":"Hombre",
        "q10":"Mujer",
        "q11":"Otro",
        "q12":"Respuesta incorrecta. Por favor, elija una de las opciones del teclado",
        "q13":"Enviar su ubicación",
        "q14":"¿Le importaría enviarnos su ubicación?\n\n Por favor, active la ubicación en su dispositivo.",
        "q15":"Por favor, active la ubicación en su dispositivo y envíenosla",
        "q16":"Positivo",
        "q17":"Negativo",
        "q18":"Desconocido",
        "q19":"¿Tiene usted covid-19?",
        "q20":"¿Podría enviarnos una grabación de audio de su tos?",
        "q21":"Por favor, envíenos una nota de voz de su tos.",
        "q22":"Analizando audio...",
        "q23":"Lo sentimos, no hemos reconocido su audio como tos. ¿Podría volverlo a intentar?",
        "q24":"Audio aceptado, gracias por su tos.",
        "q25":"¿Padece usted tos seca?",
        "q26":"Sí",
        "q27":"No",
        "q28":"¿Padece usted fiebre?",
        "q29":"¿Se siente más cansado de lo normal?",
        "q30":"¿Siente un sentido del olfato reducido?",
        "q31":"¿Padece dolor de cabeza?",
        "q32":"¿Padece dolor o presión en el pecho?",
        "q33":"¿Tiene dificultades para respirar?",
        "q34":"¿Hay algo más que quiera añadir?",
        "q35":"¡Muchas gracias por su colaboración!\n\n Aguarde un segundo mientras se guardan los datos.",

        "q36":"¿Cuál es tu género?"

    },
    "ca":{
        "q1":"Hola! Si us plau, introdueixi el seu nom.",
        "q2":"Cancel·lat",
        "q3":"Esta bé, ho deixem aquí...\n\n Doni'm un segon mentre emmagatzemo les dades.",
        "q4":"Ja esta!",
        "q5":"Procés detingut",
        "q6":"Usuari invàlid. Si us plau, introdueixi un nom correcte.",
        "q7":"Quants anys té?",
        "q8":"La seva edat ha de ser un número. \n\n Quants anys té?",
        "q9":"Home",
        "q10":"Dona",
        "q11":"Altre",
        "q12":"Resposta incorrecta. Si us plau, triï una de les opcions de teclat",
        "q13":"Enviar la seva ubicació",
        "q14":"Li faria res enviar-nos la seva ubicació?\n\n Si us plau, activi la ubicació en el seu dispositiu.",
        "q15":"Si us plau, activa la ubicació en el seu dispositiu i envieu-nos-la",
        "q16":"Positiu",
        "q17":"Negatiu",
        "q18":"Desconegut",
        "q19":"Té vostè covid-19?",
        "q20":"Podria enviar-nos un enregistrament d'àudio de la seva tos?",
        "q21":"Si us plau, envieu-nos una nota de veu de la seva tos.",
        "q22":"Analitzant àudio ...",
        "q23":"Ho sentim, no hem reconegut el seu àudio com a tos. Podria tornar-ho a intentar?",
        "q24":"Audio acceptat, gràcies per la seva tos.",
        "q25":"Pateix vostè tos seca?",
        "q26":"Si",
        "q27":"No",
        "q28":"Pateix vostè febre?",
        "q29":"¿Se sent més cansat del normal?",
        "q30":"¿Sent un sentit de l'olfacte reduït?",
        "q31":"Pateix mal de cap?",
        "q32":"Pateix dolor o pressió al pit?",
        "q33":"Té dificultats per respirar?",
        "q34":"Hi ha alguna cosa més que vulgui afegir?",
        "q35":"Moltes gràcies per la seva col·laboració! \n\n Esperi un segon mentre s'emmagatzemen les dades.",

        "q36":"Quin és el teu gènere?"
    }
}

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
    await Form.next()
    await state.update_data(age=int(message.text))

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
    return await message.reply(questions[lang]["q12"])

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

@dp.message_handler(lambda message: message.text not in [questions[lang]["q16"], questions[lang]["q17"], questions[lang]["q18"]], state=Form.has_corona)
async def process_has_corona_invalid(message: types.Message):
    """
    Filter.
    """
    return await message.reply(questions[lang]["q12"])

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
    file_path = 'C:/Users/Guillem/Desktop/Bot_Telegram/Prueba/{}.oga'.format(file_id) #Aquí deberemos indicar el directorio dónce guardemos el archivo en el servidor
    await bot.download_file(file_path_URL, file_path)

    #accepted = is_cough(message.voice.file_id)
    accepted = is_cough(file_path)

    if (accepted[0] == False and accepted[1]<=0.5):
        return await bot.send_message(message.chat.id,questions[lang]["q23"])

    elif (accepted[0] == True and accepted[1]>0.5):
        await bot.send_message(message.chat.id,questions[lang]["q24"])
        await Form.next()
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
        markup.add(questions[lang]["q26"], questions[lang]["q27"])
        return await message.reply(questions[lang]["q25"], reply_markup=markup)

    else:
        return await bot.send_message(message.chat.id,questions[lang]["q23"])



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

@dp.message_handler(lambda message: message.text not in [questions[lang]["q26"], questions[lang]["q27"]], state=Form.dry_cough)
async def process_dry_cough_invalid(message: types.Message):
    """
    Text filter.
    """
    return await message.reply(questions[lang]["q12"])


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
    """
    In this example gender has to be one of: Male, Female, Other.
    """
    return await message.reply(questions[lang]["q12"])


@dp.message_handler(state=Form.fever)
async def process_fever(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['symptoms']['fever'] = (message.text == questions[lang]["q26"])

    await Form.next()
    await message.reply(questions[lang]["q29"])


@dp.message_handler(lambda message: message.text not in [questions[lang]["q26"], questions[lang]["q27"]], state=Form.tiredness)
async def process_tiredness_invalid(message: types.Message):
    return await message.reply(questions[lang]["q12"])


@dp.message_handler(state=Form.tiredness)
async def process_tiredness(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['symptoms']['tiredness'] = (message.text == questions[lang]["q26"])

    await Form.next()
    await message.reply(questions[lang]["q30"])


@dp.message_handler(lambda message: message.text not in [questions[lang]["q26"], questions[lang]["q27"]], state=Form.smell_loss)
async def process_loss_smell_invalid(message: types.Message):
    return await message.reply(questions[lang]["q12"])


@dp.message_handler(state=Form.smell_loss)
async def process_loss_smell(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['symptoms']['loss of taste or smell'] = (message.text == questions[lang]["q26"])

    await Form.next()
    await message.reply(questions[lang]["q31"])


@dp.message_handler(lambda message: message.text not in [questions[lang]["q26"], questions[lang]["q27"]], state=Form.head_ache)
async def process_headache_invalid(message: types.Message):
    return await message.reply(questions[lang]["q12"])


@dp.message_handler(state=Form.head_ache)
async def process_headache(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['symptoms']['headache'] = (message.text == questions[lang]["q26"])

    await Form.next()
    await message.reply(questions[lang]["q33"])


@dp.message_handler(lambda message: message.text not in [questions[lang]["q26"], questions[lang]["q27"]], state=Form.shortness_breath)
async def process_shortness_breath_invalid(message: types.Message):
    return await message.reply(questions[lang]["q12"])


@dp.message_handler(state=Form.shortness_breath)
async def process_shortness_breath(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['symptoms']['difficulty breathing or shortness of breath'] = (message.text == questions[lang]["q26"])

    await Form.next()
    await message.reply(questions[lang]["q32"])


@dp.message_handler(lambda message: message.text not in [questions[lang]["q26"], questions[lang]["q27"]], state=Form.chest_pain)
async def process_chest_pain_invalid(message: types.Message):
    return await message.reply(questions[lang]["q12"])


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
        database.collection.insert_one(data.as_dict())


        await bot.send_message(
            message.chat.id,
            questions[lang]["q4"]
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
'''
def is_cough(file_id):
    file = await bot.get_file(file_id)
    file_path_URL = file.file_path
    file_path = 'C:/Users/Guillem/Desktop/Bot_Telegram/Prueba/{}.oga'.format(file_id)
    await bot.download_file(file_path_URL, file_path)

    wav_file_path = convert_to_wav(file_path)
    accepted = yamnet_classifier(wav_file_path)

    return accepted
'''

'''
def convert_to_wav(input_file):

    from pydub import AudioSegment

    file_dir, filename = os.path.split(os.path.abspath(input_file))
    input_file_path = os.path.abspath(input_file)
    basename = filename.split('.')[0]
    output_file = os.path.join(file_dir, '{}.wav'.format(basename))

    sound = AudioSegment.from_ogg(input_file)
    sound.export(output_file, format="wav")

    return output_file
'''

'''
def is_cough(file_id):
    url = 'https://api.telegram.org/bot{}/getFile?file_id={}'.format(API_TOKEN, file_id)
    r = requests.get(url)
    file_path = r.json()["result"]["file_path"]
    url = 'https://api.telegram.org/file/bot{}/{}'.format(API_TOKEN, file_path)
    r = requests.get(url)  # Descargamos el archivo de audio

    #file_dir = SYSTEM_PATH
    #os.makedirs(file_dir, exist_ok=True)

    filename = 'C:/Users/Guillem/Desktop/Bot_Telegram/Prueba/{}.oga'.format(file_id)
    with open(filename, 'wb') as f:
        f.write(r.content)
    wav_file_path = convert_to_wav(filename)
    accepted = yamnet_classifier(wav_file_path)

    return accepted


def convert_to_wav(input_file):
    file_dir, filename = os.path.split(os.path.abspath(input_file))
    input_file_path = os.path.abspath(input_file)
    basename = filename.split('.')[0]
    output_file = os.path.join(file_dir, '{}.wav'.format(basename))
    #output_file = 'C:/Users/Guillem/Desktop/Bot Telegram/Prueba/test.wav'
    ffmpeg_instruction = 'ffmpeg -y -i {} {}'.format(input_file_path,output_file)
    os.system(ffmpeg_instruction)
    return output_file
'''


def is_cough(file_path):
    wav_file_path = convert_to_wav(file_path)
    yamnet_veredict = yamnet_classifier(wav_file_path)
    svm_veredict = aT.file_classification(wav_file_path, "cough_classifier/svm_cough", "svm")
    svm_predict = svm_veredict[1][0]

    accepted = [yamnet_veredict, svm_predict]
    return accepted

def convert_to_wav(input_file):
    file_dir, filename = os.path.split(os.path.abspath(input_file))
    input_file_path = os.path.abspath(input_file)
    basename = filename.split('.')[0]
    output_file = os.path.join(file_dir, '{}.wav'.format(basename))

    ffmpeg_instruction = 'ffmpeg -y -i {} {}'.format(input_file_path,output_file)
    os.system(ffmpeg_instruction)
    return output_file


def yamnet_classifier(wav_file_path, visualization = False):
    sample_rate, wav_data = wavfile.read(wav_file_path)
    sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)

    waveform = wav_data / tf.int16.max
    #waveform2 = np.mean(waveform, axis = 1)   # If the audio is stereo and not mono

    # Run the model, check the output.
    scores, embeddings, spectrogram = model(waveform)

    scores_np = scores.numpy()
    spectrogram_np = spectrogram.numpy()
    infered_class = class_names[scores_np.mean(axis=0).argmax()]

    if (visualization):

        plt.figure(figsize=(10, 6))

        # Plot the waveform.
        plt.subplot(2, 1, 1)
        plt.plot(waveform)
        plt.xlim([0, len(waveform)])

        # Plot the log-mel spectrogram (returned by the model).
        plt.subplot(2, 1, 2)
        plt.imshow(spectrogram_np.T, aspect='auto', interpolation='nearest', origin='lower')


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
