import requests
import json
from flask import jsonify
url='http://127.0.0.1:5000/users'

#headers = {'Authorization': 'my-api-key'}
image_metadata = {
	"id": 1000000001,
	"username": "Daniel",
	"age": 25,
	"gender": "Male",
	"diagnosis": "Never been diagnosed",
	"vaccine": "False",
	"symptoms": {
		"dry cough": "False",
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
del image_metadata["username"]
print(image_metadata)
'''
data = json.dumps(image_metadata)
print(data)
print(type(image_metadata))

#data = {'name': 'test.oga', 'data': json.dumps(image_metadata)}
files = {'upload_file': open('/home/dani/covidscipy2020/test.oga','rb'),
         'json': (None, json.dumps(image_metadata), 'application/json')}
#files = {'file': ('test.oga', open('/home/dani/covidscipy2020/test.oga', 'rb'), 'audio/oga', {'Expires': '0'})}
r = requests.post(url, files=files)
'''