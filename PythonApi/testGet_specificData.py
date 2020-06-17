import requests as req
import json    # or `import simplejson as json` if on Python < 2.6




if __name__ == "__main__":
  #Get all
  user = input('Enter the username: ')

  url = 'http://127.0.0.1:5000/data/' + user + '?pretty'
  resp = req.get(url)
  jsondata=resp.text
  obj = json.loads(jsondata) 
  #print(obj["age"])
  #print(resp.text) # Printing response

#Get specific user (can perform url request with either
# /data/<username> or /data/<_id>)
#url = 'http://127.0.0.1:5000/data/TestUser130?pretty'
#resp = req.get(url)
#print(resp.text) # Printing response

