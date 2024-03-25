import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix 
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os



###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
model_path = os.path.join(config['output_model_path'])



##############Function for reporting
def score_model(test_filename):
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace

    model_filepath = os.getcwd()+'\\'+model_path+'\\'+'trainedmodel.pkl'
    with open(model_filepath,'rb') as file:
        model = pickle.load(file)

    test_df = pd.read_csv(os.getcwd()+'\\'+test_data_path+'\\'+test_filename)
    X_test = test_df.loc[:,['lastmonth_activity','lastyear_activity','number_of_employees']].values.reshape(-1,3)
    y_true = test_df['exited'].values.reshape(-1,1).ravel()

    y_pred = model.predict(X_test)
    class_names = ['no exit','exit']

    disp = plot_confusion_matrix(model, X_test, y_true,
                                display_labels=class_names,
                                  cmap=plt.cm.Blues,
                                  normalize=None)
    
    plt.savefig(os.getcwd()+'\\'+model_path+'\\'+'confusionmatrix.png')





if __name__ == '__main__':
    score_model('testdata.csv')
