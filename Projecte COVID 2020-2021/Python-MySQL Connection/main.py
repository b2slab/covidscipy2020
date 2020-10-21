import pymysql
import numpy as np
from multipledispatch import dispatch

class DataBase:
    def __init__(self):
        self.connection = pymysql.connect(
            host = 'localhost',
            user = 'root',
            password = 'projectecovid',
            database = 'project_covid_creb'
        )

        self.cursor = self.connection.cursor()
        print('Conexión establecida correctamente')

    def select_user(self, id):

        # sql = 'SELECT * FROM patients WHERE patient_id = %s' %id
        sql = "SELECT * FROM patients WHERE patient_id = '{}'".format(id)

        try:
            self.cursor.execute(sql)
            user = self.cursor.fetchone()

            # print('patient_id: ', user[0], 'gender: ', user[1])
            print(user)

        except Exception as e:
            raise

    def select_all_users(self):

        sql = 'SELECT * FROM patients'

        try:
            self.cursor.execute(sql)
            users = self.cursor.fetchall()

            for user in users:
                print(user)
                print('_________\n')

        except Exception as e:
            raise

    def update_user(self, id, age):
        '''
        Esta función actualiza el campo 'age' de un usuario según el 'id' que introduzcamos
        Cabe destacar que el método acaba aplicando la sentencia 'commit()', por lo que el
        update del cambio es PERMANENTE. Si quisieramos realizar un seguimiento de los cambios
        realizados, deberíamos programar 'TRIGGERS de actualización' en nuestra base de datos
        '''
        sql = "UPDATE patients SET age = '{}' WHERE patient_id = '{}'".format(age,id)

        try:
            self.cursor.execute(sql)
            self.connection.commit()
            print("Hemos realizado un cambio permanente")
        except Exception as e:
            raise

    def insert_user(self, gender, age, height, weight, hometown, num_people):

        self.gender = gender
        self.age = age
        self.height = height
        self.weight = weight
        self.hometown = hometown
        self.num_people = num_people

        def check_correct_inputs(gender, age, height, weight, hometown, num_people):
            if (self.gender != 'male') and (self.gender != 'female') or (type(self.gender) != str):
                print("Debes introducir un género válido")
                sql = ''
            elif (self.age <= 0) or (self.age>=130) or (type(self.age)!= int):
                print("Debes introducir una edad válida")
                sql = ''
            elif (self.height <= 0) or (self.height >= 250) or (type(self.height)!= int):
                print("Debes introducir una altura válida")
                sql = ''
            elif (self.weight <= 0) or (self.weight >= 300) or (type(self.weight)!= int):
                print("Debes introducir un peso válido")
                sql = ''
            elif (type(self.hometown)!=str):
                print("Debes introducir una ciudad válida")
                sql = ''
            elif (self.num_people < 0) or (self.num_people>20) or (type(self.num_people)!= int):
                print("Debes introducir un # de personas con las que vives válido")
                sql = ''
            else:
                print("Inputs correctos")
                sql = "INSERT INTO patients(gender, age, height, weight, hometown, num_people, date_creation) VALUES ('{}','{}','{}','{}','{}','{}',NOW())".format(self.gender, self.age, self.height, self.weight, self.hometown, self.num_people)
            return sql

        self.sql = check_correct_inputs(gender, age, height, weight, hometown, num_people)

        try:
            self.cursor.execute(self.sql)
            self.connection.commit()
            print('Se ha realizado el INSERTAR correctamente')
        except Exception as e:
            raise

    def close(self):
        self.connection.close()
        print("Hemos realizado correctamente la desconexión de la BBDD")

    def insert_patient_symptoms(self, suffer):

        '''
        Esta función nos servirá para guardar el id del paciente con todos
        los posibles síntomas del covid y si los sufre o no
        '''
        def get_patient_ID(self):

            '''
            Esta función se encargará de descubrir el id del último paciente registrado
            para que de esta forma podamos saber el id que se le atribuirá al paciente
            que se está registrando en este momento
            '''

            sql = "SELECT MAX(patient_id) FROM project_covid_creb.patients"

            try:
                self.cursor.execute(sql)
                id = self.cursor.fetchone()
                return id+1

            except Exception as e:
                raise

        def get_num_symptoms(self):

            '''
            Esta función se encargará de descubrir cuántos síntomas en total tenemos
            alojados en la base de datos. Por supuesto, deberá coincidir con la largada
            del input array 'suffer' de variables binarias
            '''

            sql = "SELECT COUNT(symptom_id) FROM project_covid_creb.symptoms"

            try:
                self.cursor.execute(sql)
                count_symp = self.cursor.fetchone()
                return count_symp

            except Exception as e:
                raise

        patient_id = get_patient_ID(self)
        num_symptoms = get_num_symptoms(self)

        def check_input_length(self, suffer, num_symptoms):

            '''
            Esta función se encargará de comprobar si la longitud del array input 'suffer'
            coincide con la longitud total de los síntomas. De esta forma, sabemos que
            el paciente ha respondido a todas las preguntas sobre síntomas
            '''
            if len(suffer != num_symptoms):
                print("Incorrect input length")
            else:
                print("Correct Input")
                return suffer

        self.suffer = check_input_length(suffer)
        self.patient_id = patient_id
        self.symptoms_id = np.arange(1,num_symptoms+1)

        print("Paciente: ", self.patient_id)
        print("Síntomas: ", self.symptoms_id)
        print("Los sufre?: ", self.suffer)


###########################################################################################
###########################################################################################
###########################################################################################

database = DataBase()
database.select_user(1)
database.select_all_users()
database.update_user(3,99)
database.select_user(1) # IT WORKED -- watch out with 'commit()' statement as it upload permanently our BBDD
database.insert_user('female', 43, 190, 90,'Cádiz', 1)
database.close()



'''
SUGERENCIA:
Podemos intentar crear un DECORATOR que tome como input el 'data type'
de los argumentos de entrada de la función a decorar.
Si el tipo de los argumentos no concuerda con el insertado en el decorador,
que salte un error especificando el problema.
'''



suffering = np.array([1,0,0,0,1,1,0,0,0,1,0,1,0,1,0,1,0,0,1,0,1,1,0,1,0])
len(suffering)

database.insert_patient_symptoms(suffering)
