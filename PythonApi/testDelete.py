import requests as req
import json

#Delete specific user (can perform url request with either
# /data/<username> or /data/<_id>)

#First get the ID of the user we want to delete
url = 'http://127.0.0.1:5000/data/TestUser112'
resp = req.get(url)

#Parse JSON response to python dictionary
user_data = json.loads(resp.text)
print(user_data)

#Get user id's. Required to delete
user_id = user_data["_id"]
user_etag = user_data["_etag"]

print("_id: "+str(user_id))
print("_etag: "+str(user_etag))

#We can now delete the user using the _id and _etag variables
#In order to delete we must supply the etag in the request header
url = "http://127.0.0.1:5000/data/" + str(user_id)
print("Deleting user at "+url)
headers = {'content-type': 'application/json', 'If-Match': user_etag}
x = req.delete(url, headers=headers)
print(x)