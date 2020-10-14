import pymysql
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
