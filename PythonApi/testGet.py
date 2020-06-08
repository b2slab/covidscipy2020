import requests as req

#Get all
url = 'http://127.0.0.1:5000/data?pretty'
resp = req.get(url)
print(resp.text) # Printing response

#Get specific user (can perform url request with either
# /data/<username> or /data/<_id>)
#url = 'http://127.0.0.1:5000/data/TestUser130?pretty'
#resp = req.get(url)
#print(resp.text) # Printing response