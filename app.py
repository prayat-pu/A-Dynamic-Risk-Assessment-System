from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import diagnostics 
import json
import os
from sklearn import metrics



######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
model_path = os.path.join(config['prod_deployment_path'])

prediction_model = None

def load_model(model_path):
    model_filepath = os.getcwd()+'\\'+model_path+'\\'+'trainedmodel.pkl'
    with open(model_filepath,'rb') as file:
        model = pickle.load(file)
    return model

def read_testdata(test_data_path):
    test_df = pd.read_csv(os.getcwd()+'\\'+test_data_path+'\\'+'testdata.csv')
    X_test = test_df.loc[:,['lastmonth_activity','lastyear_activity','number_of_employees']].values.reshape(-1,3)
    y = test_df['exited'].values.reshape(-1,1).ravel()
    return X_test,y


#######################Prediction Endpoint
@app.route("/prediction", methods=['GET'])
def predict():        
    #call the prediction function you created in Step 3
    predict = diagnostics.model_predictions() 
    return str(predict) #add return value for prediction outputs

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET'])
def scoring():        
    #check the score of the deployed model
    X_test,y_true = read_testdata(test_data_path)
    model = load_model(model_path)
    y_pred = model.predict(X_test)
    f1 = metrics.f1_score(y_true,y_pred,average='binary')
    return str(f1) #add return value (a single F1 score number)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET'])
def stats():        
    #check means, medians, and modes for each column
    summary_dict = diagnostics.dataframe_summary()
    return summary_dict #return a list of all calculated summary statistics

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET'])
def diagnostics_function():        
    #check timing and percent NA values
    nas_percentage = diagnostics.missing_value_percentage()
    lib_outdated = diagnostics.outdated_packages_list().decode('utf-8')
    time_processing = diagnostics.execution_time()

    return 'percentages of missing value for each features: {},\n\n outdated library: {},\n\n time_processing: {}'.format(nas_percentage, lib_outdated, time_processing) #add return value for all diagnostics

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
