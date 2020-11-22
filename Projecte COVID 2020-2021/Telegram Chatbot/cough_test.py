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
from pyAudioAnalysis import audioTrainTest as aT
from os import path
from pydub import AudioSegment
import os


logging.basicConfig(level=logging.INFO)
API_TOKEN = '1370389029:AAFIaYXbnHLCkNYIb5azZ2iOg5BWoRdOUC8'
bot = Bot(token=API_TOKEN)
save_path = '/home/dani/covidscipy2020/data/'
audio_path = 'cough_data/'
# For example use simple MemoryStorage for Dispatcher.
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)

# States
class Form(StatesGroup):
    cough = State()



@dp.message_handler(commands='start')
async def cmd_start(message: types.Message):
    """
    Conversation's entry point
    """
    # Set state
    await Form.cough.set()

    await message.reply("Cough.")

@dp.message_handler(state=Form.cough, content_types=types.message.ContentType.VOICE)
async def process_cough(message: types.voice.Voice, state: FSMContext):
    # Update state and data
    file_id=message.voice.file_id
    file = await bot.get_file(file_id)
    file_path = file.file_path
    await bot.download_file(file_path, "cough_data/cough.ogg")

    veredict = analyze_cough()
    if veredict<=0.5:
        return await bot.send_message(
            message.chat.id,
            "Sorry, we did not recognize this as cough. Can you try i again? (prob = %s)" % (round(veredict, 2))
        )

    else:
        await Form.next()
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
        markup.add("yes", "no")
        await message.reply("Nice, cough bro (prob = %s). Do you have a dry cough?" % (round(veredict, 2)), reply_markup=markup)


def analyze_cough():
    outputFileName = "Cough #.wav"
    outputVersion = 1
    while os.path.isfile(audio_path + outputFileName.replace("#", str(outputVersion))):
        outputVersion += 1
    outputFileName = outputFileName.replace("#", str(outputVersion))
    filepath = os.path.join(audio_path, outputFileName)
    src = "cough_data/cough.ogg"

    # convert wav to mp3
    sound = AudioSegment.from_ogg(src)
    sound.export(filepath, format="wav")
    # %%
    veredict = aT.file_classification(filepath, "cough_classifier/svm_cough", "svm")
    return veredict[1][0]

def save_audio(file_id):
    file_info = bot.get_file(file_id)
    downloaded_file = bot.download_file(file_info)
    with open('new_file.ogg', 'wb') as new_file:
        new_file.write(downloaded_file)

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)