from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 


#################Function for training the model
def train_model(X,y,model_path):
    
    #use this logistic regression for training
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    #fit the logistic regression to your data
    model.fit(X,y)

    output_path = os.getcwd()+'\\'+model_path 
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #write the trained model to your workspace in a file called trainedmodel.pkl
    model_filepath = output_path+'\\'+'trainedmodel.pkl'
    pickle.dump(model, open(model_filepath,'wb'))

def read_data(dataset_path):
    # read the data in csv format
    df = pd.read_csv(os.getcwd()+'\\'+dataset_path+'\\'+'finaldata.csv')
    X = df.loc[:,['lastmonth_activity','lastyear_activity','number_of_employees']].values.reshape(-1,3)
    y = df['exited'].values.reshape(-1,1).ravel()
    return X,y

if __name__ == '__main__':
    X,y = read_data(dataset_csv_path)
    train_model(X,y,model_path)
