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

# States
class Form(StatesGroup):
    username = State()
    age = State()
    gender = State()
    country = State()
    postcode = State()
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
        save_features(data.as_dict())
        await bot.send_message(
            message.chat.id,
            "That's it!"
        )
    await state.finish()
    await message.reply('Process stopped.', reply_markup=types.ReplyKeyboardRemove())

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
        markup = types.ReplyKeyboardRemove()

    await Form.next()
    await message.reply("In which country are you right now?", reply_markup=markup)

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
    await Form.next() #this line is to be removed after cough implementation
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add("yes", "no")
    await message.reply("Thank you. Now let us ask you some questions about your symptoms."
                        "Do you have a dry cough?", reply_markup=markup)

#cough yet to be implemented
"""
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
        save_features(data.as_dict())
        await bot.send_message(
            message.chat.id,
            "That's it!"
        )
#functions

def save_features(data_object):
    outputFileName = "Patient #.txt"
    outputVersion = 1
    while os.path.isfile(save_path + outputFileName.replace("#", str(outputVersion))):
        outputVersion += 1
    outputFileName = outputFileName.replace("#", str(outputVersion))
    filepath = os.path.join(save_path, outputFileName)
    with open(filepath, 'w') as outfile:
        json.dump(data_object, outfile)
    # data_object_json = json.dumps(data_object)


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)




