import pymongo
import re
from sshtunnel import SSHTunnelForwarder
from flask import Flask, request, json, Response
from bson import json_util
from bson.objectid import ObjectId
from pymongo import MongoClient


app = Flask(__name__)


class DataBase:

    def __init__(self):

        sql_hostname = '127.0.0.1'
        sql_main_database = 'Project_COVID'
        sql_port = 27017
        ssh_host = 'covidbot.upc.edu'
        ssh_user = 'covidbot'
        ssh_pass = 'B2SLab2020!!!!'
        ssh_port = 2244

        self.server = SSHTunnelForwarder(
            (ssh_host, ssh_port),
            ssh_username=ssh_user,
            ssh_password=ssh_pass,
            remote_bind_address=(sql_hostname, sql_port))

        self.server.start()

        self.client = pymongo.MongoClient(sql_hostname, self.server.local_bind_port)
        self.db = self.client[sql_main_database]
        self.collection = self.db['Patients']
        print('Colecciones de la BBDD: ', self.db.list_collection_names())
        print('Conexión establecida correctamente')

    def close(self):
        self.server.stop()
        print("Hemos realizado correctamente la desconexión de la BBDD")


    def get_all_users(self):
        documents = self.collection.find()
        output = [{item: data[item] for item in data} for data in documents]
        return output

    def get_user_by_id(self,id):
        documents = self.collection.find({"_id": ObjectId(id)})
        output = [{item: data[item] for item in data} for data in documents]
        return output

    def write_user(self, data):
        response = self.collection.insert_one(data)
        output = {'Status': 'Successfully Inserted',
              'Document_ID': str(response.inserted_id)}
        return output


class location(object):
    def __init__(self, latitude, longitude, *args, **kwargs):
        self.latitude = latitude
        self.longitude = longitude

class symptoms(object):
    def __init__(self, cough, dry_cough, fever, tiredness, smell_loss, head_ache, shortness_breath,
                 chest_pain, others, *args, **kwargs):
        self.cough = cough
        self.dry_cough = dry_cough
        self.fever = fever
        self.tiredness = tiredness
        self.smell_loss = smell_loss
        self.head_ache = head_ache
        self.shortness_breath = shortness_breath
        self.chest_pain = chest_pain
        self.others = others


class Patient(object):
    def __init__(self, username, age, gender, location, diagnosis, symptoms, *args, **kwargs):
        self.username = username
        self.age = age
        self.gender = gender
        self.location = location
        self.diagnosis = diagnosis
        self.symptoms = symptoms



@app.route('/')
def base():
    return Response(response=json.dumps({"Status": "UP"}),
                    status=200,
                    mimetype='application/json')


@app.route('/users', methods=['GET'])
def get_users():
    response = DataBase().get_all_users()
    return Response(response=json_util.dumps(response),
                    status=200,
                    mimetype='application/json')

@app.route('/users/<id>', methods=['GET'])
def get_user_by_id(id):
    data = DataBase().get_user_by_id(id)
    pattern = re.compile("^[A-Za-z0-9]{24}$")
    if not pattern.match(id):
        return Response(response = 'huh?', status=400)
    else:
        if data == []:
            response = Response(status=404)
        else:
            response = Response(response=json_util.dumps(data),
                     status=200,
                         mimetype='application/json')
        return response

@app.route('/users', methods=['POST'])
def add_user():
    data = request.json
    print(data)
### studentJson = '{"rollNumber": 1, "name": "Emma", "age":"18"}'
    try:
        Patient(**data)

    except Exception as inst:
        print(inst)
        return Response(response=json.dumps({"Error": str(inst)}),
                        status=400,
                        mimetype='application/json')

    ##
    response = DataBase().write_user(data)
    return Response(response=json.dumps(response),
                    status=200,
                    mimetype='application/json')

if __name__ == '__main__':
    database = DataBase()
    app.run(debug=True, port=5001, host='0.0.0.0')