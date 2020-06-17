import requests as req
import numpy as np
import json

#Get all                                                                               
url = 'http://127.0.0.1:5000/rawAudio/ericeric?pretty'
resp = req.get(url).json()
#print(resp.text) # Printing response

audio = np.array(json.loads(resp["audio_file"]))
sample_rate = int(resp["sample_rate"])

from wavio import write
write("example2.wav", audio.astype('float'), sample_rate, sampwidth=3)

#Get specific user (can perform url request with either                                
# /data/<username> or /data/<_id>)                                                     
#url = 'http://127.0.0.1:5000/data/TestUser130?pretty'                                 
#resp = req.get(url)                                                                   
#print(resp.text) # Printing response

