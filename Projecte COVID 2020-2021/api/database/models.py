import json
import requests
import os
import requests

os.environ['NO_PROXY'] = 'localhost'

student = {
        "username": "Tami",
        "age": 24,
        "gender": "female",
        "location": {
            "latitude": 41.432236,
            "longitude": 2.14043
        },
        "diagnosis": "negative",
        "symptoms": {
            "dry cough": "True",
            "fever": "True",
            "tiredness": "False",
            "loss of taste or smell": "False",
            "headache": "True",
            "difficulty breathing or shortness of breath": "False",
            "chest pain or pressure": "True",
            "others": "Sometimes I feel really sick"
        }
    }

requests.post('https://localhost:5001/users', data = student)