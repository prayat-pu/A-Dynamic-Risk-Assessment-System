

import training
import scoring
import deployment
import diagnostics
import reporting
import os
import json
import ingestion

#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
prod_path = config['prod_deployment_path']
ingest_folder_path = config['output_folder_path']

##################Check and read new data
#first, read ingestedfiles.txt
ingested_file_path = os.getcwd()+'\\'+prod_path+'\\'+'ingestedfiles.txt'
with open(ingested_file_path,'r') as f:
    ingested_info_dict = eval(f.readlines()[0])
ingested_filenames = ingested_info_dict['filenames']
# print(ingested_filenames)

#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
filenames = os.listdir(os.getcwd()+'\\'+input_folder_path)
for filename in filenames:
    ##################Deciding whether to proceed, part 1
    #if you found new data, you should proceed. otherwise, do end the process here
    if filename not in ingested_filenames:
        ingestion.merge_multiple_dataframe()
        break


##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
lastestscore_file_path = os.getcwd()+'\\'+prod_path+'\\'+'latestscore.txt'
with open(lastestscore_file_path,'r') as f:
    latestscore_info_dict = eval(f.readlines()[0])
latest_f1score = latestscore_info_dict['latest_f1']

# make prediction with new data from the previous steps
model = scoring.load_model(prod_path)
X_test,y = scoring.read_testdata(ingest_folder_path,'finaldata.csv')
new_f1score = scoring.score_model(X_test,y,model)['latest_f1']

##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
# model drift -> new score worst than latest score
if new_f1score - latest_f1score < 0:
    print('model drift occured.')
    ##################Re-training
    os.system('python training.py')
    os.system('python scoring.py')
    ##################Re-deployment
    #if you found evidence for model drift, re-run the deployment.py script
    os.system('python deployment.py')
else:
    print(f'new f1: {new_f1score}')
    print(f'latest f1: {latest_f1score}')
    print('your model is not drift.')


##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model
os.system('python reporting.py')
os.system('python apicalls.py')






