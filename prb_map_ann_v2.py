#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as pylot

from datetime import  tzinfo, timezone
from datetime import datetime as dt
from google.colab import drive, files

#%%
class PreProcess:
    #reads csv file at the given path
    def read_file(self, path: str):
        if path.startswith('/'):
            path = '.' + path
        elif not path.startswith('/') or not path.startswith('.'):
            path = './' + path
        else: pass
        df = pd.read_csv(path)
        return df

    #converts the date time to timestamp
    def convert_timestamps(self, data_frame: pd.DataFrame):
        try:
            tm = dt.strptime(data_frame, '%Y-%m-%d %H:%M:%S.%f %Z')
        except ValueError:
            tm = dt.strptime(data_frame, '%Y-%m-%d %H:%M:%S %Z')
        converted = tm.timestamp() * 1000 
        return converted
    
    def sort_values(self, data_frame: pd.DataFrame, value: str = 'user_id', asc: bool = False):
        data = data_frame.sort_values([i for i in value], ascending=asc).reset_index(drop=True)
        return data
    
    def split_to_trip(self, data_frame: pd.DataFrame):
        data_frame['trip'] = (data_frame['timestamp'] - data_frame[i]['timestamp'].shift(1) > 60000).cumsum()
        return data_frame

#%%
class GoogleDrive:
    #mounts the drive and change the file path to google drive's path
    def mount_drive(self):
        drive.mount('/content/gdrive')
        os.chdir('./gdrive/My Drive')
        return
    
    #creates file at the given path
    def create_file(self, path: str):
        os.mkdir(path)
        return
    
    #deletes the file at the given path
    def remove_file(self, path: str):
        os.rmdir(path)
        return

    #it allows you to upload files to google drive
    def upload_files(self):
        uploaded = files.upload()
        return
#%%
class Model:

#%%
gdrive = GoogleDrive()
pre_process = PreProcess()

#%%
gdrive.mount_drive()
path = 'Colab Notebooks/bq-results-20190628-104852-awu4v6ig6fyq.csv'
data_frame = pre_process.read_file(path)

#%%
data_frame['timestamp'] = data_frame['timestamp'].apply(pre_process.convert_timestamps)
sort_parameter = ['user_id', 'timestamp']
data_frame = pre_process.sort_values(data_frame, sort_parameter, False)

#%%
