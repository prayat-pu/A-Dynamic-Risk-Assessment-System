import os
import json
import shutil



##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
model_path = os.path.join(config['output_model_path']) 



####################function for deployment
def store_model_into_pickle():
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    
    output_path = os.getcwd()+'\\'+prod_deployment_path 
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # copy model into deployment folder
    model_src_path = os.getcwd()+'\\'+ model_path + '\\' + 'trainedmodel.pkl'
    model_dst_path = os.getcwd()+'\\'+ prod_deployment_path + '\\' + 'trainedmodel.pkl'
    shutil.copyfile(model_src_path, model_dst_path)

    # copy latesscore.txt  into deployment folder
    latestscore_src_path = os.getcwd()+'\\'+ model_path + '\\' + 'latestscore.txt'
    latestscore_dst_path = os.getcwd()+'\\'+ prod_deployment_path + '\\' + 'latestscore.txt'
    shutil.copyfile(latestscore_src_path, latestscore_dst_path)

    # copy ingestedfiles.txt into deployment folder
    ingesteddata_src_path = os.getcwd()+'\\'+ dataset_csv_path + '\\' + 'ingestedfiles.txt'
    ingesteddata_dst_path = os.getcwd()+'\\'+ prod_deployment_path + '\\' + 'ingestedfiles.txt'
    shutil.copyfile(ingesteddata_src_path, ingesteddata_dst_path)


    

if __name__ == '__main__':
    store_model_into_pickle()
        
        

