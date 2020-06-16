import requests as req
import json    # or `import simplejson as json` if on Python < 2.6

def getSpecific(username, values): 
    url = 'http://127.0.0.1:5000/data/'+username+'?pretty'
    resp = req.get(url)
    jsondata=resp.text
    obj = json.loads(jsondata) 
#    print(username, values, ":", obj[values])
#    return obj[values]

    # Pseudocode 1: check if username exists
    # if username doesn't exist:
    #     print("No data for this username")
    #     return None
    
    specificValues = []
    print("Username: ", username)
    for element in values:
        
        # Pseudocode 2: check if element exists as a key in obj
        # if element isn't part of obj:
        #   print(element, " is not a valid key for this username")
        #   pass
        
        print(element, ": ", obj[element])
#        specificValues.append(element)
        specificValues.append(obj[element])
    return specificValues

if __name__ == "__main__":
    user = input('Enter the username: ')
    dataentries = []
    while True:
        dataentry = input('Enter a key for a desired data entry or finish by pressing enter : ')
        if dataentry == "":
            break
        dataentries.append(dataentry)
    print(getSpecific(user,dataentries))
    
    #print(resp.text) # Printing response

    #Get specific user (can perform url request with either
    # /data/<username> or /data/<_id>)
    #url = 'http://127.0.0.1:5000/data/TestUser130?pretty'
    #resp = req.get(url)
    #print(resp.text) # Printing response
