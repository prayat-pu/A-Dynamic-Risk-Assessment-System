
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess


##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
deploy_path = os.path.join(config['prod_deployment_path'])
output_ingest_path = os.path.join(config['output_folder_path'])


def read_test_df(test_data_path):
    test_df = pd.read_csv(os.getcwd()+'\\'+test_data_path+'\\'+'testdata.csv')
    return test_df

##################Function to get model predictions
def model_predictions():
    #read the deployed model and a test dataset, calculate predictions
    # read deployed model
    model_filepath = os.getcwd()+'\\'+deploy_path+'\\'+'trainedmodel.pkl'
    with open(model_filepath,'rb') as file:
        model = pickle.load(file)

    # read testdata.csv
    test_df = read_test_df(test_data_path)
    X_test = test_df.loc[:,['lastmonth_activity','lastyear_activity','number_of_employees']].values.reshape(-1,3)

    predict = model.predict(X_test)
    
    return list(predict) #return value should be a list containing all predictions

##################Function to get summary statistics
def dataframe_summary():
    test_df = read_test_df(test_data_path)
    #calculate summary statistics here
    #return #return value should be a list containing all summary statistics
    numerical_features = ['lastmonth_activity','lastyear_activity','number_of_employees']
    descriptive_dict = dict()

    for feature in numerical_features:
        descriptive_dict[feature] = {
            'mean':test_df[feature].mean(),
            'median':test_df[feature].median(),
            'std':test_df[feature].std(),
        }
    
    
    # write stat summery file
    record_filename='dataset statistic summery.txt'
    record_file_path = os.getcwd()+'\\'+output_ingest_path + '\\'+record_filename
    with open(record_file_path, "w") as fp:
        json.dump(descriptive_dict, fp)

    return descriptive_dict
          
##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    time_list = []

    ingest_starttime = timeit.default_timer()
    os.system('python ingestion.py')
    ingest_time = timeit.default_timer() - ingest_starttime
    time_list.append(ingest_time)

    training_starttime = timeit.default_timer()
    os.system('python training.py')
    training_time = timeit.default_timer() - training_starttime
    time_list.append(training_time)

    return time_list #return a list of 2 timing values in seconds


##################Function to check missing value
def missing_value_percentage():
    test_df = read_test_df(test_data_path)
    total = len(test_df)
    nas = list(test_df.isna().sum())
    percentage_nas = [item/total for item in nas]
    return percentage_nas


##################Function to check dependencies
def outdated_packages_list():
    #get a list of 
    outdated = subprocess.check_output(['pip','list','--outdated'])
    return outdated


if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    execution_time()
    missing_value_percentage()
    outdated_packages_list()





    
