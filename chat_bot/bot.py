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
import os
import numpy as np
import io
from io import BytesIO
import sys

#sys.path.append(
#    'C:\\Users\\matth\\Dropbox\\1UNI\\Erasmus\\ETSEIB\\FINAL_Auswahl\\240AR064 Scientific Phyton for Engineers\\Project')
from covidscipy2020.machine_learning.Cough_NoCough_classification.yamnet import classifier

logging.basicConfig(level=logging.INFO)

API_TOKEN = '995583036:AAGmrBpgnGvI0tXccH1bIf9xaQZ5i9mWLdk'

DB_URL = 'http://127.0.0.1:5000'  # url where db is hosted
# SYSTEM_PATH = os.environ['HOME'] + './audio-files' # path on system to save audio files
SYSTEM_PATH = './audio-files'  # path on system to save audio files
bot = Bot(token=API_TOKEN)

# For example use simple MemoryStorage for Dispatcher.
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)


# States
class Form(StatesGroup):
    username = State()
    age = State()
    gender = State()
    height = State()
    cough = State()
    has_corona = State()
    coughing_frecuency = State()
    temperature = State()
    health_issue = State()


@dp.message_handler(commands='start')
async def cmd_start(message: types.Message):
    """
    Conversation's entry point
    """
    # Set state
    await Form.username.set()

    await message.reply("Hi there! Please, enter your username.")


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

    logging.info('Cancelling state %r', current_state)
    # Cancel state and inform user about it
    await state.finish()
    # And remove keyboard (just in case)
    await message.reply('Cancelled.', reply_markup=types.ReplyKeyboardRemove())


@dp.message_handler(state=Form.username)
async def process_username(message: types.Message, state: FSMContext):
    """
    Process user name
    """
    async with state.proxy() as data:
        data['username'] = message.text

    await Form.next()
    await message.reply("How old are you?")


# Check age. Age gotta be digit
@dp.message_handler(lambda message: not message.text.isdigit(), state=Form.age)
async def process_age_invalid(message: types.Message):
    """
    If age is invalid
    """
    return await message.reply("Age gotta be a number.\nHow old are you? (digits only)")


@dp.message_handler(lambda message: message.text.isdigit(), state=Form.age)
async def process_age(message: types.Message, state: FSMContext):
    # Update state and data
    await Form.next()
    await state.update_data(age=int(message.text))

    # Configure ReplyKeyboardMarkup
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add("Male", "Female")
    markup.add("Other")

    await message.reply("What is your gender?", reply_markup=markup)


@dp.message_handler(lambda message: message.text not in ["Male", "Female", "Other"], state=Form.gender)
async def process_gender_invalid(message: types.Message):
    """
    In this example gender has to be one of: Male, Female, Other.
    """
    return await message.reply("Bad gender name. Choose your gender from the keyboard.")


@dp.message_handler(state=Form.gender)
async def process_gender(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['gender'] = message.text

        # Remove keyboard
        markup = types.ReplyKeyboardRemove()

    await Form.next()
    await message.reply("What is your height?", reply_markup=markup)


# Check age. Height gotta be digit
@dp.message_handler(lambda message: not message.text.isdigit(), state=Form.height)
async def process_height_invalid(message: types.Message):
    """
    If height is invalid
    """
    return await message.reply("Height gotta be a number.\nWhat is your height? (digits only)")


@dp.message_handler(lambda message: message.text.isdigit(), state=Form.height)
async def process_age(message: types.Message, state: FSMContext):
    # Update state and data
    await Form.next()
    await state.update_data(height=int(message.text))
    await message.reply("Please cough")


@dp.message_handler(state=Form.cough, content_types=types.message.ContentType.VOICE)
async def process_cough(message: types.voice.Voice, state: FSMContext):
    # Update state and data
    await bot.send_message(
        message.chat.id,
        "Please, give me a second while I annalyze you cough..."
    )
    if not is_cough(message.voice.file_id):
        return await bot.send_message(
            message.chat.id,
            "Sorry, we didn't recognize this as cough. Please, cough again"
        )

    else:
        await bot.send_message(message.chat.id, "Thank you!")
        await Form.next()
        # Configure ReplyKeyboardMarkup
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
        markup.add("Yes", "No")
        markup.add("Not certain")

        await message.reply("Do you have corona?", reply_markup=markup)


@dp.message_handler(state=Form.has_corona)
async def process_has_corona(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['has_corona'] = message.text

        corona_states = {
            "Yes": "do",
            "No": "do not",
            "Not certain": "do not know if you"
        }

        # Remove keyboard
        markup = types.ReplyKeyboardRemove()

        # And send message
        await bot.send_message(
            message.chat.id,
            md.text(
                md.text('Hi! Nice to meet you,', data['username']),
                md.text('Age:', data['age']),
                md.text('Gender:', data['gender']),
                md.text('Height:', data['height']),
                md.text(f'You {corona_states[data["has_corona"]]} have corona'),
                sep='\n',
            ),
            reply_markup=markup,
            parse_mode=ParseMode.MARKDOWN,
        )
    await state.finish()


#
# @dp.message_handler(content_types=types.message.ContentType.VOICE)
# async def echo(message: types.voice.Voice):
#     # old style:
#     # await bot.send_message(message.chat.id, message.text)
#     logging.info(message)


def is_cough(file_id):
    url = f"https://api.telegram.org/bot{API_TOKEN}/getFile?file_id={file_id}"
    r = requests.get(url)
    file_path = r.json()["result"]["file_path"]
    url = f"https://api.telegram.org/file/bot{API_TOKEN}/{file_path}"
    r = requests.get(url)
    audio_blob = r.content

    # audio_json = {'name', 'audio': audio_blob in str format}
    # try:
    #     response = requests.post(DB_URL + '/audio', data=audio_blob)
    #     r.raise_for_status()
    # except:
    """
        An error has been encounter while uploading audio to db.
        We save the file to the system as a temporary solution
    """
    file_dir = SYSTEM_PATH
    os.makedirs(file_dir, exist_ok=True)
    filename = os.path.join(file_dir, f"{file_id}.ogg")
    with open(filename, 'wb') as f:
        f.write(r.content)

    wav_file_path = convert_to_wav(filename)
    top_labels = classifier.classify(wav_file_path)
    accepted = "Cough" in top_labels
    print("TOP LABELS: ", top_labels)
    return accepted


def convert_to_wav(input_file):
    file_dir, filename = os.path.split(os.path.abspath(input_file))
    basename = filename.split('.')[0]
    output_file = os.path.join(file_dir, f"{basename}.wav")
    os.system(f'ffmpeg -y -i {input_file} {output_file}')
    return output_file


def upload_to_database(filename, data):
    # pass
    if data["has_corona"] == "Yes":
        label = 0
    else:
        label = 1
    feature_dictionary = create_feature_from_audio(filename, label)


def create_feature_from_audio(filename, label):
    import pyogg
    import numpy as np
    import ctypes, numpy, pyogg
    import matplotlib.pyplot as plt
    import scipy.io.wavfile

    # https://github.com/Zuzu-Typ/PyOgg/issues/19
    # file = pyogg.OpusFile(filename)  # stereo
    # audio_path_opus = "./"
    file = pyogg.OpusFile(filename)
    target_datatype = ctypes.c_short * (file.buffer_length // 2)  # always divide by 2 for some reason
    buffer_as_array = ctypes.cast(file.buffer,
                                  ctypes.POINTER(target_datatype)).contents
    if file.channels == 1:
        wav = numpy.array(buffer_as_array)
    elif file.channels == 2:
        wav = numpy.array((wav[0::2],
                           wav[1::2]))
    else:
        raise NotImplementedError()
    # This is the final numpy array
    signal = numpy.transpose(wav)
    sampling_rate = 48000
    print(numpy.shape(wav))

    # plt.figure
    # plt.title("Signal Wave...")
    # plt.plot(signal)
    # plt.show()

    # Calculating features from final_data
    from pyAudioAnalysis import MidTermFeatures as mF
    from pyAudioAnalysis import ShortTermFeatures as sF
    from pyAudioAnalysis import audioBasicIO

    mid_window = round(0.1 * sampling_rate)
    mid_step = round(0.1 * sampling_rate)
    short_window = round(sampling_rate * 0.01)
    short_step = round(sampling_rate * 0.01)

    signal = audioBasicIO.stereo_to_mono(signal)
    print(type(signal))
    # print(np.shape(signal))
    signal = signal.astype('float64')  # this line is because librosa was making an error - need floats

    [mid_features, short_features, mid_feature_names] = mF.mid_feature_extraction(signal, sampling_rate, mid_window,
                                                                                  mid_step, short_window, short_step);
    mid_features = np.transpose(mid_features)
    mid_term_features = mid_features.mean(axis=0)
    mid_term_features = np.reshape(mid_term_features, (-1, 1))
    mid_term_features = np.transpose(mid_term_features)

    import json
    features_list = features.tolist()
    feature_dict = dict(zip(['label'] + mid_feature_names, [label] + features_list))

    # print(np.shape(mid_term_features))
    # len(mid_feature_names)

    # Getting the classification result with Cough=0, No_Cough=1
    # from joblib import dump, load
    # from sklearn import preprocessing
    # cough_classifier = load('Cough_NoCough_classifier.joblib')
    # features = preprocessing.StandardScaler().fit_transform(mid_term_features)
    # prediction = cough_classifier.predict(features)  # coughs=0 , no_cough = 1
    return feature_dictionary


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
