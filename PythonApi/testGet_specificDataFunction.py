import requests as req
import json    # or `import simplejson as json` if on Python < 2.6

def getSpecific(username, values): 
    url = 'http://127.0.0.1:5000/data/'+username+'?pretty'
    resp = req.get(url)
    jsondata=resp.text
    obj = json.loads(jsondata) 
#    print(username, values, ":", obj[values])
#    return obj[values]

    specificValues = []
    print(username, ":")
    for element in values:
#        print(element, ": ", obj[element])
#        specificValues.append(element)
        specificValues.append(obj[element])
    return specificValues

if __name__ == "__main__":
    print(getSpecific("Christian",["age", "gender"])) 
    # or simply getSpecific("Christian",["age", "gender"]) if print statement in function is uncommented
    
    #getSpecific("Christian","age")

    #print(resp.text) # Printing response

    #Get specific user (can perform url request with either
    # /data/<username> or /data/<_id>)
    #url = 'http://127.0.0.1:5000/data/TestUser130?pretty'
    #resp = req.get(url)
    #print(resp.text) # Printing response
