from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json





#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
model_path = os.path.join(config['output_model_path'])

#################Function for model scoring
def score_model(X_test,y,model):
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    predict = model.predict(X_test)
    f1 = metrics.f1_score(y,predict,average='binary')

    # save the score into output model path
    all_score_records = {
        'latest_f1':f1
    }

    return all_score_records

def score_model_and_save(X_test,y,model):
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    all_score_records = score_model(X_test,y,model)

    record_filename='latestscore.txt'
    record_file_path = os.getcwd()+'\\'+model_path + '\\'+record_filename
    with open(record_file_path, "w") as fp:
        json.dump(all_score_records, fp)
        
    return all_score_records

def read_testdata(test_data_path,filenames):
    test_df = pd.read_csv(os.getcwd()+'\\'+test_data_path+'\\'+filenames)
    X_test = test_df.loc[:,['lastmonth_activity','lastyear_activity','number_of_employees']].values.reshape(-1,3)
    y = test_df['exited'].values.reshape(-1,1).ravel()
    return X_test,y

def load_model(model_path):
    model_filepath = os.getcwd()+'\\'+model_path+'\\'+'trainedmodel.pkl'
    with open(model_filepath,'rb') as file:
        model = pickle.load(file)
    return model

if __name__ == '__main__':
    X_test,y = read_testdata(test_data_path,'testdata.csv')
    model = load_model(model_path)
    score_model_and_save(X_test,y,model)