import pandas as pd
import numpy as np
import os
import json
from datetime import datetime




#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']



#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    template_df = pd.DataFrame(columns=['corporation','lastmonth_activity',
                                        'lastyear_activity','number_of_employees',
                                        'exited'])

    filenames = os.listdir(os.getcwd()+'\\'+input_folder_path)
    for each_filename in filenames:
        df1 = pd.read_csv(os.getcwd()+'\\'+input_folder_path+'\\'+each_filename)
        template_df = template_df.append(df1)

    output_path = os.getcwd()+'\\'+output_folder_path 
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # de-duplicate rows
    template_df.drop_duplicates(inplace=True)

    # writing final data into output folder
    output_filename = 'finaldata.csv' 
    template_df.to_csv(output_path+ '\\'+output_filename,index=False)

    # record information of ingested data
    dateTimeObj = datetime.now()
    timenow = str(dateTimeObj.year) +'/'+str(dateTimeObj.month)+'/'+str(dateTimeObj.day)

    all_records = {
        'source_location': input_folder_path,
        'data_size': len(template_df),
        'time': timenow,
        'filenames':filenames
    }

    # write record file
    record_filename='ingestedfiles.txt'
    record_file_path = os.getcwd()+'\\'+output_folder_path + '\\'+record_filename
    with open(record_file_path, "w") as fp:
        json.dump(all_records, fp)


if __name__ == '__main__':
    merge_multiple_dataframe()
