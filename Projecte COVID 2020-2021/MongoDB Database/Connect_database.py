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
        self.collection = self.db['Patients']
        print('Colecciones de la BBDD: ', self.db.collection_names())
        print('Conexión establecida correctamente')
    def close(self):

        '''
        Cerramos connexión con el servidor
        '''

        self.server.stop()
        print("Hemos realizado correctamente la desconexión de la BBDD")



import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import mymodule
