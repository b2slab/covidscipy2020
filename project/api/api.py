import pymongo
import re
from sshtunnel import SSHTunnelForwarder
from flask import Flask, request, json, Response
from bson import json_util
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import gridfs
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
        documents = self.collection.find({"id": int(id)})
        output = [{item: data[item] for item in data} for data in documents]
        return output

    def write_user(self, data):
        response = self.collection.insert_one(data)
        output = {'Status': 'Successfully Inserted',
              'Document_ID': str(response.inserted_id)}
        return output

    def delete_entry(self, id, username):
        filt = {"$and":[{ "id": int(id) }, { "username": username}]}
        #documents = self.collection.find(filt)
        #audio_id = [{item: data[item] for item in data} for data in documents]
        #audioDB = gridfs.GridFS(self.db)
        #audioDB.delete(audio_id)
        response = self.collection.delete_one(filt)
        output = {'Status': 'Successfully deleted' if response.deleted_count > 0 else "Document not found."}
        return output

#class location(object):
 #   def __init__(self, latitude, longitude, *args, **kwargs):
   #     self.latitude = latitude
    #    self.longitude = longitude

class symptoms(object):
    def __init__(self, dry_cough, smoker, cold, res_difficult, sore_throat, fever,fatigue, muscular_pain, smell_loss,
                 pneumonia, diarrhea, hypertension, asthma, diabetes, CLD, IHD, *args, **kwargs):
        #self.cough = cough
        self.dry_cough = dry_cough
        self.smoker = smoker
        self.cold = cold
        self.res_difficult = res_difficult
        self.sore_throat = sore_throat
        self.fever = fever
        self.fatigue = fatigue
        self.muscular_pain = muscular_pain
        self.smell_loss = smell_loss
        self.pneumonnia = pneumonia
        self.diarrhea = diarrhea
        self.hypertension = hypertension
        self.asthma = asthma
        self.diabtes = diabetes
        self.CLD = CLD
        self.IHD = IHD
        #self.others = others


class Patient(object):
    def __init__(self, id, age, gender,  diagnosis, vaccine, symptoms, *args, **kwargs):
        self.id = id
        self.age = age
        self.gender = gender
        #self.location = location
        self.diagnosis = diagnosis
        self.vaccine = vaccine
        self.symptoms = symptoms

limiter = Limiter(
    app,
    key_func=get_remote_address
)

@app.route('/')
def base():
    return Response(response=json.dumps({"Status": "UP"}),
                    status=200,
                    mimetype='application/json')


@app.route('/users', methods=['GET'])
def get_users():
    response = DataBase().get_all_users()
    print (json_util.dumps(response))
    return Response(response=json_util.dumps(response),
                    status=200,
                    mimetype='application/json')

@app.route('/users/<id>', methods=['GET'])
def get_user_by_id(id):
    data = DataBase().get_user_by_id(id)

    if data == []:
        response = Response(status=404)
    else:
        response = Response(response=json_util.dumps(data),
                     status=200,
                         mimetype='application/json')
    return response

@app.route('/users', methods=['POST'])
@limiter.limit("1 per 10 second")
def add_user():
    data = request.json
    print(data)

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
"""
@app.route('/users', methods=['PUT'])
def update_user():
    data = request.json
    print(data)
    response = DataBase().update(data)
    return Response(response=json.dumps(response),
                    status=200,
                    mimetype='application/json')
"""
@app.route('/users/<id>/<username>', methods=['DELETE'])
def delete_user(id, username):
    response = DataBase().delete_entry(id, username)
    print(response)
    if response["Status"] == "Document not found.":
        return Response(response = 'You have no entry with that username', status=404)
    else:
        return Response(response=json.dumps(response),
                        status=200,
                        mimetype='application/json')

if __name__ == '__main__':
    database = DataBase()
    app.run()

    #app.run(debug=True, port=5001, host='0.0.0.0')
    #app.run(debug=True, port=2244, host='covidbot.upc.edu')