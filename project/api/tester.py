from datetime import datetime

now = datetime.now()
print(type(now))
print(now)
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print(dt_string)