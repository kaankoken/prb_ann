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
    
    def sort_values(self, data_frame: pd.DataFrame):
        x = data_frame.groupby('user_id', sort=True)
        t = []
        [t.append(value) for key, value in x]
        x = []
        [x.append(i.sort_values('timestamp',ascending=True).reset_index(drop=True)) for i in t]
        
        return x
    
    def split_to_trip(self, data_frame: pd.DataFrame):
        data_frame['trip'] = (data_frame['timestamp'] - data_frame[i]['timestamp'].shift(1) > 60000).cumsum()
        return data_frame

    def divide_to_trip(self, data: pd.DataFrame):
        trip = data.index[np.where(data['trip'] - data['trip'].shift(1) > 0)]
        tripCount = trip.values.size
        data = data.drop('trip', axis=1)

        changedData = []

        endpoint = 0
        if (tripCount > 0):
            for i in range(0, tripCount):
                if (i == 0):
                    changedData.append(data.loc[0: trip[i] -1].reset_index(drop=True))
                else:
                    changedData.append(data.loc[trip[i - 1]: trip[i] -1].reset_index(drop=True))

                endpoint = i
            changedData.append(data.loc[trip[endpoint]: len(data)].reset_index(drop=True))
        else:
            changedData.append(data.loc[0: len(data)].reset_index(drop=True))

        return changedData
    
    def flatten(self, data: pd.DataFrame):
        x = [e for sl in data for e in sl]
        return x
    
    def drop_if_small(self, data: pd.DataFrame, size: int):
        y = []
        for i in data:
            if len(i) > size:
                y.append(i)

        return y
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
data_frame = pre_process.sort_values(data_frame)
data_frame = pre_process.drop_if_small(data_frame, 4)

#%%
for i in range(0,len(data_frame)):
  data_frame[i]['trip'] = (data_frame[i]['timestamp'] - data_frame[i]['timestamp'].shift(1) > 60000).cumsum()
#%%
for i in range(0,len(data_frame)):
  data_frame[i] = pre_process.divide_to_trip(data_frame[i])
data_frame = pre_process.flatten(data_frame)

#%%
data_frame = pre_process.drop_if_small(data_frame, 4)