import requests
import json

import sys
from flask import jsonify
#url='http://127.0.0.1:5000/users'
url ='https://covidscipy2020.herokuapp.com/users'

#headers = {'Authorization': 'my-api-key'}
dct = {
	"id": 1000000001,
	"username": "Daniel",
	"age": 25,
	"gender": "Male",
	"diagnosis": "Never been diagnosed",
	"vaccine": "False",
	"symptoms": {
		"dry cough": False,
		"smoker": "False",
		"cold": "False",
		"res_difficult": "False",
		"sore_throat": "False",
		"fever": "False",
		"fatigue": "False",
		"muscular_pain": "False",
		"smell_loss": "False",
		"pneumonia": "False",
		"diarrhea": "False",
		"hypertension": "False",
		"asthma": "False",
		"diabetes": "False",
		"CLD": "False",
		"IHD": "False",
		"others": "No"
	},
	"audio_file": {
		"filename": "test",
		"ObjectID": "",
		"covid_positive": "False"
	}
}
def convert(obj):
    if isinstance(obj, bool):
        return str(obj).lower()
    if isinstance(obj, (list, tuple)):
        return [convert(item) for item in obj]
    if isinstance(obj, dict):
        return {convert(key):convert(value) for key, value in obj.items()}
    return obj

dct = {
  "is_open": True
}
print (json.dumps(dct))
print (json.dumps(convert(dct)))


'''
print(image_metadata)

data = json.dumps(image_metadata)
print(data)
print(type(image_metadata))

#data = {'name': 'test.oga', 'data': json.dumps(image_metadata)}
files = {'upload_file': open('/home/dani/covidscipy2020/AwACAgQAAxkBAAIS0WBWZuf_3naQ-Jr7VbizaqmDGv9JAAIQCQACoYSwUquh9mgwQEDlHgQ.oga','rb'),
         'json': (None, json.dumps(image_metadata), 'application/json')}
#files = {'file': ('test.oga', open('/home/dani/covidscipy2020/test.oga', 'rb'), 'audio/oga', {'Expires': '0'})}
r = requests.post(url, files=files)
print(r.status)
'''