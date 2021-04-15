import requests
import json
id = 111212212
response = requests.get('https://covidscipy2020.herokuapp.com/'+'users/%s'%id)
print(type(response.status_code))