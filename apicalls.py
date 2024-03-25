import requests
import json
import os

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

with open('config.json','r') as f:
    config = json.load(f) 
model_output_path = os.path.join(config['output_model_path'])

#Call each API endpoint and store the responses
response1 = requests.get(URL+'/prediction').content.decode('utf-8') #put an API call here
response2 = requests.get(URL+'/scoring').content.decode('utf-8') #put an API call here
response3 = requests.get(URL+'/summarystats').content.decode('utf-8') #put an API call here
response4 = requests.get(URL+'/diagnostics').content.decode('utf-8') #put an API call here

# #combine all API responses
responses = 'model prediction: '+ str(response1) +'\n\n'  #combine reponses here
responses += 'model scoring: '+str(response2) + '\n\n'
responses +=  'model summary: '+str(response3) + '\n\n'
responses +=  'diagnostics: '+str(response4) + '\n'

#write the responses to your workspace
output_file_path = os.getcwd()+'\\'+model_output_path+'\\'+'apireturens.txt'
with open(output_file_path,'w') as f:
    f.writelines(responses)



