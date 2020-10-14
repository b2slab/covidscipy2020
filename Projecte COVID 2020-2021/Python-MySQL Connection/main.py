import pymysql

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

    def close(self):
        self.connection.close()
        print("Hemos realizado correctamente la desconexión de la BBDD")




database = DataBase()
database.select_user(1)
database.select_all_users()
database.update_user(3,99)
database.select_user(3) # IT WORKED -- watch out with 'commit()' statement as it upload permanently our BBDD
database.close()
