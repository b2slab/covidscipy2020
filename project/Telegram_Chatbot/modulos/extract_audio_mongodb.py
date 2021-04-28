import pymongo
import gridfs
import os
from sshtunnel import SSHTunnelForwarder
from bson.objectid import ObjectId

class DataBase:

    def __init__(self):

        '''
        Inicializamos la clase con este CONSTRUCTOR. De esta forma, cada vez que instanciemos
        la clase 'DataBase', nos conectaremos a la BBDD de MongoDB ubicada en el servidor
        remoto de la UPC.

        Para realizar la conexión remota, utilizaremos un túnel SSH mediante la función
        'SSHTunnelForwarder' del módulo 'sshtunnel'.

        Posteriormente, nos connectaremos a la base de datos 'Project_COVID'. De momento,
        accederemos a la única Colección de la base de datos, llamada 'Patients'. Las colecciones
        en MongoDB son el equivalente a las tablas en MySQL.
        '''

        sql_hostname = '127.0.0.1'
        sql_username = 'guillembonilla'
        sql_password = 'B2SLab2020!!!!'
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
        '''
        Closes server connection
        '''
        self.server.stop()
        print("Hemos realizado correctamente la desconexión de la BBDD")

    def connect_GridFS(self):
        db = self.db
        audioDB = gridfs.GridFS(db)
        return audioDB

    def find_ObjectID(self, username):
        resp = self.collection.find_one({'username': username}, {"audio_file.ObjectID": 1, '_id':0})
        if (resp is None):
            return "Username not found"
        objectid_file = list(resp['audio_file'].values())[0]
        return objectid_file

    def find_store_audio(self, username, output_path, filename):
        audioDB = self.connect_GridFS()
        objectid_file = self.find_ObjectID(username)
        if (audioDB.exists({'_id': ObjectId(objectid_file)})==False):
            return "Audio not found"
        gout = audioDB.get(ObjectId(objectid_file))
        fout = open(output_path + filename + '.oga', 'wb')
        fout.write(gout.read())
        fout.close()
        gout.close()
        return "Audio successfully stored in {}".format(output_path)


# Connection to database
MongoDB_ = DataBase()

# Find and store audios by specifying username
MongoDB_.find_store_audio('Berta', 'C:/Users/Guillem/Desktop/', 'audio_berta')
