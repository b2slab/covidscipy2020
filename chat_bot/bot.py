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

logging.basicConfig(level=logging.INFO)

API_TOKEN = '995583036:AAGmrBpgnGvI0tXccH1bIf9xaQZ5i9mWLdk'

DB_URL = 'http://127.0.0.1:5000' # url where db is hosted
SYSTEM_PATH = os.environ['HOME'] + '/audio-files' # path on system to save audio files

bot = Bot(token=API_TOKEN)

# For example use simple MemoryStorage for Dispatcher.
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)


# States
class Form(StatesGroup):
    firstname = State()
    surname = State()
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
    await Form.firstname.set()

    await message.reply("Hi there! What's your name?")


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


@dp.message_handler(state=Form.firstname)
async def process_name(message: types.Message, state: FSMContext):
    """
    Process user name
    """
    async with state.proxy() as data:
        data['firstname'] = message.text

    await Form.next()
    await message.reply("What's your surname?")


@dp.message_handler(state=Form.surname)
async def process_name(message: types.Message, state: FSMContext):
    """
    Process user name
    """
    async with state.proxy() as data:
        data['surname'] = message.text

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


@dp.message_handler(state=Form.cough,content_types=types.message.ContentType.VOICE)
async def process_cough(message: types.voice.Voice, state: FSMContext):
    # Update state and data
    download_voice(message.voice.file_id)
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
                md.text('Hi! Nice to meet you,', md.bold(data['firstname'])),
                md.text('Age:', md.code(data['age'])),
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


def download_voice(file_id):
    url = f"https://api.telegram.org/bot{API_TOKEN}/getFile?file_id={file_id}"
    r = requests.get(url)
    file_path = r.json()["result"]["file_path"]
    url = f"https://api.telegram.org/file/bot{API_TOKEN}/{file_path}"
    r = requests.get(url)


    audio_blob = r.content
    # audio_json = {'name', 'audio': audio_blob in str format}
    try:
        response = requests.post(DB_URL + '/audio', data=audio_blob)
        r.raise_for_status()
    except:
        """
            An error has been encounter while uploading audio to db.
            We save the file to the system as a temporary solution
        """
        filename = f'{SYSTEM_PATH}/{file_id}.oga'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            f.write(r.content)


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
