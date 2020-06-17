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
import json
from scipy.io import wavfile
import requests

from Cough_NoCough_classification.yamnet import classifier

logging.basicConfig(level=logging.INFO)
API_TOKEN = '995583036:AAGmrBpgnGvI0tXccH1bIf9xaQZ5i9mWLdk'

#DB_URL = 'http://127.0.0.1:5000' # url where db is hosted
#DB_DATA_URL = f"{DB_URL}/data" # url where db is hosted
#HEADERS = {'content-type': 'application/json'}

#SYSTEM_PATH = os.environ['HOME'] + './audio-files' # path on system to save audio files
SYSTEM_PATH = './audio-files' # path on system to save audio files

bot = Bot(token=API_TOKEN)

# For example use simple MemoryStorage for Dispatcher.
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)


# States
class Form(StatesGroup):
    username = State()
    age = State()
    gender = State()
    country = State()
    city = State()
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

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add("Spain")
    markup.add("France")
    markup.add("Sweden")
    await Form.next()
    await message.reply("In which country are you righ now?", reply_markup=markup)



@dp.message_handler(lambda message: message.text not in ["Spain", "France", "Sweden"], state=Form.country)
async def process_country_invalid(message: types.Message):
    """
    In this example gender has to be one of: Male, Female, Other.
    """
    return await message.reply("Bad country name. Choose country from the options.")


@dp.message_handler(state=Form.country)
async def process_country(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['location'] = {}
        data['location']['country'] = message.text

        # Remove keyboard
        types.ReplyKeyboardRemove()

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add("Barcelona")
    markup.add("Madrid")
    await Form.next()
    await message.reply("In which city are you right now?", reply_markup=markup)


@dp.message_handler(lambda message: message.text not in ["Barcelona", "Madrid"], state=Form.city)
async def process_city_invalid(message: types.Message):
    """
    In this example gender has to be one of: Male, Female, Other.
    """
    return await message.reply("Bad city name. Choose country from the options.")


@dp.message_handler(state=Form.city)
async def process_city(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['location']['city'] = message.text

    await Form.next()
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add("positive")
    markup.add("negative")
    markup.add("unknown")
    await message.reply("Do you have Covid-19?", reply_markup=markup)


@dp.message_handler(lambda message: message.text not in ["positive", "negative", "unknown"], state=Form.has_corona)
async def process_has_corona_invalid(message: types.Message):
    """
    In this example gender has to be one of: Male, Female, Other.
    """
    return await message.reply("Bad answer. Please, choose between the keyboard options.")


@dp.message_handler(state=Form.has_corona)
async def process_has_corona(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['diagnosis'] = message.text
        markup = types.ReplyKeyboardRemove()

    await Form.next()
    await message.reply("Please, cough", reply_markup=markup)


@dp.message_handler(state=Form.cough, content_types=types.message.ContentType.VOICE)
async def process_cough(message: types.voice.Voice, state: FSMContext):
    # Update state and data
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


@dp.message_handler(lambda message: message.text not in ["yes", "no"], state=Form.dry_cough)
async def process_dry_cough_invalid(message: types.Message):
    """
    In this example gender has to be one of: Male, Female, Other.
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
    await message.reply("Thank you very much! Now let us ask you some questions about your"
                        "symptoms.\n Do you have fever?", reply_markup=markup)


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
        resp = upload_features(data.as_dict())
        if resp.json()['_status'] == "OK":
            await bot.send_message(message.chat.id, "Upload successful! Have a great day")
        else:
            await bot.send_message(
                message.chat.id,
                "Sorry, something went wrong..."
            )
        
    await state.finish()


def is_cough(file_id):
    url = f"https://api.telegram.org/bot{API_TOKEN}/getFile?file_id={file_id}"
    r = requests.get(url)
    file_path = r.json()["result"]["file_path"]
    url = f"https://api.telegram.org/file/bot{API_TOKEN}/{file_path}"
    r = requests.get(url)

    file_dir = SYSTEM_PATH
    os.makedirs(file_dir, exist_ok=True)

    filename = './audio-files/test.ogg'
    with open(filename, 'wb') as f:
        f.write(r.content)
    wav_file_path = convert_to_wav(filename)
    top_labels = classifier.classify(wav_file_path)
    accepted = "Cough" in top_labels
    print("TOP LABELS: ", top_labels)
    return accepted, wav_file_path


def convert_to_wav(input_file):
    file_dir, filename = os.path.split(os.path.abspath(input_file))
    basename = filename.split('.')[0]
    #output_file = os.path.join(file_dir, f"{basename}.wav")
    output_file = './audio-files/test.wav'
    os.system(f'ffmpeg -y -i {input_file} {output_file}')
    return output_file


def upload_features(data_object):
    import requests
    url = 'http://127.0.0.1:5000/data'
    headers = {'content-type': 'application/json'}
    print('object data:')
    print(data_object)
    # data_object_json = json.dumps(data_object)
    x = requests.post(url, json=data_object, headers=headers)
    print(x.json())
    return x


def upload_audio(audio_numpy, sample_rate, username, label):
    url = 'http://127.0.0.1:5000/rawAudio'
    headers = {'content-type': 'application/json'}
    audio_numpy_str = json.dumps(audio_numpy.tolist())
    data_audio = {"username": username, "audio_file": audio_numpy_str, "sample_rate": str(sample_rate), "label": str(label)}
    x = requests.post(url, json=data_audio, headers=headers)
    print(x.json())
    return x


def create_feature_from_audio(filename, label):
    import numpy as np
    import numpy

    # This is the final numpy array
    sampling_rate, wav = wavfile.read(filename)
    wavfile.write('testing.wav', sampling_rate, wav)
    signal = numpy.transpose(wav)
    print(numpy.shape(wav))

    # Calculating features from final_data
    from pyAudioAnalysis import MidTermFeatures as mF
    from pyAudioAnalysis import ShortTermFeatures as sF
    from pyAudioAnalysis import audioBasicIO

    mid_window = round(0.1 * sampling_rate)
    mid_step = round(0.1 * sampling_rate)
    short_window = round(sampling_rate * 0.01)
    short_step = round(sampling_rate * 0.01)

    signal = audioBasicIO.stereo_to_mono(signal)
    signal = signal.astype('float64')  # this line is because librosa was making an error - need floats

    [mid_features, short_features, mid_feature_names] = mF.mid_feature_extraction(signal, sampling_rate, mid_window,
                                                                                  mid_step, short_window, short_step);
    mid_features = np.transpose(mid_features)
    mid_term_features = mid_features.mean(axis=0)
    mid_term_features_list = mid_term_features.tolist()
    mid_term_features_list = list(map(str, mid_term_features_list))
    feature_dict = dict(zip(['label'] + mid_feature_names, [label] + mid_term_features_list))

    return feature_dict, signal, sampling_rate


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
