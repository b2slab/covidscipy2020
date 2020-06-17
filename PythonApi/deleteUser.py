import requests as req
import json

def delete(username):
  url = 'http://127.0.0.1:5000/data/' + username
  resp = req.get(url)
  user_data = json.loads(resp.text)
  
  #print(user_data)
  
  user_id = user_data["_id"]
  user_etag = user_data["_etag"]
  
  #print("_id: "+str(user_id))
  #print("_etag: "+str(user_etag))

  url = "http://127.0.0.1:5000/data/" + str(user_id)
  print("Deleting user: " + username +  at " + url)
  headers = {'content-type': 'application/json', 'If-Match': user_etag}
  x = req.delete(url, headers=headers)
  print(x)

if __name__ == "__main__":
    user = input('Enter the username: ')
    delete(user)
    
