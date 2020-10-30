import pymongo
from sshtunnel import SSHTunnelForwarder

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
        #self.collection = self.db['Patients']
        self.collection = self.db['TEST']
        print('Colecciones de la BBDD: ', self.db.collection_names())
        print('Conexión establecida correctamente')

    def insert_document(self, username, age, gender):

        '''
        Mediante esta función añadiremos los datos de un nuevo paciente a la
        colección Patients. La información de cada paciente es almacenada en un
        documento JSON.

        Al insertar los datos del paciente, un ID del documento único será generado
        automáticamente (podemos asignar el valor que queramos, pero por defecto se
        genera automáticamente)
        '''

        document = {'username': username, 'age': age, 'gender': gender}
        record = self.collection.insert_one(document)
        print("ID of the inserted document: ", record.inserted_id)

    def Find_All(self):

        '''
        Mediante esta función imprimimos en pantalla todos los documentos (rows)
        almacenados en la Colección 'Patients'
        '''

        for i in self.collection.find():
            print(i)

    def Query_filter(self, Key, Value):

        '''
        Mediante esta función podemos filtrar los documentos imprimidos en pantalla
        por una Key -- nombre de la columna -- y un Value -- Valor de dicha columna--
        '''

        query = {'{}'.format(Key):'{}'.format(Value)}
        docs = self.collection.find(query)

        for i in docs:
            print(i)

    def Delete_One(self, Key, Value):

        '''
        Mediante esta función eliminamos un documento en el que coincida
        la query {Key:Value}
        '''

        query = {'{}'.format(Key):'{}'.format(Value)}
        self.collection.delete_one(query)

    def Update_One(self, Key, Value, New_Value):

        '''
        Mediante esta función actualizamos el documento con parámetro Key de valor Value
        a un nuevo valor New_Value
        '''

        query = {'{}'.format(Key):'{}'.format(Value)}
        newvalues = { "$set" : {'{}'.format(Key):'{}'.format(New_Value)} }
        self.collection.update_one(query, newvalues)

    def close(self):

        '''
        Cerramos connexión con el servidor
        '''

        self.server.stop()
        print("Hemos realizado correctamente la desconexión de la BBDD")

database = DataBase()
database.Find_All()
database.Query_filter('gender', 'female')
database.Delete_One('username', 'Abuelo')
database.Update_One('username', 'Guillem', 'Guillermo')

database.insert_document('Maria',67,'female')
