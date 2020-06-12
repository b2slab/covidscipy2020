import requests

url = 'http://127.0.0.1:5000/rawAudio'

headers = {'content-type': 'application/json'}

testObject = {
    "username": "TestUser112"
    #"audio_file": None
} 


x = requests.post(url, json=testObject, headers=headers)
print(x.json())