import requests as req
import json    # or `import simplejson as json` if on Python < 2.6

def userEntry(username):
  url = 'http://127.0.0.1:5000/data/' + username + '?pretty'
  resp = req.get(url)
  jsondata=resp.text
  obj = json.loads(jsondata) 
  print(obj) #for demonstration purposes
  #print(resp.text) # Printing response
  return obj
  
if __name__ == "__main__":
  #Get all
  user = input('Enter the username: ')  #for demonstration purposes
  userEntry(user)   #for demonstration purposes, outside the program just call function userEntry and pass it the username
