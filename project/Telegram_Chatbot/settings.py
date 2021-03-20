import os


BOT_TOKEN = '1370389029:AAFIaYXbnHLCkNYIb5azZ2iOg5BWoRdOUC8' #test
#BOT_TOKEN = '1648913260:AAH9HkGOKozj6FxMPdJucc0uYkpQJEpem8I' #master
if not BOT_TOKEN:
    print('You have forgot to set BOT_TOKEN')
    quit()

HEROKU_APP_NAME = os.getenv('covidscipy2020')


# webhook settings
WEBHOOK_HOST = f'https://{HEROKU_APP_NAME}.herokuapp.com'
WEBHOOK_PATH = f'/webhook/{BOT_TOKEN}'
WEBHOOK_URL = f'{WEBHOOK_HOST}{WEBHOOK_PATH}'

# webserver settings
WEBAPP_HOST = '0.0.0.0'
#WEBAPP_PORT = int(os.getenv('PORT'))

API_HOST = 'https://covidscipy2020.herokuapp.com/'
