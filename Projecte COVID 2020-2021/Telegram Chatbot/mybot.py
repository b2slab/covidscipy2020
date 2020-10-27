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

@dp.message_handler(state=Form.username)
async def process_username(message: types.Message, state: FSMContext):
    """
    Process user name
    """
    async with state.proxy() as data:
        data['username'] = message.text

    await Form.next()
    await message.reply("How old are you?")


@dp.message_handler(state=Form.age)
async def process_age(message: types.Message, state: FSMContext):
    # Update state and data
    await state.update_data(age=int(message.text))
    async with state.proxy() as data:
        data['age'] = message.text
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

    await state.finish()

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






