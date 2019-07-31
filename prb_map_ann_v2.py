#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as pylot

from datetime import  tzinfo, timezone
from datetime import datetime as dt
from google.colab import drive, files

from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, TimeDistributed, RepeatVector, Flatten

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
    
    def find_min_max(self, value: pd.DataFrame):
        max = value.max()
        min = value.min()
        return min, max

    def divide_by_angle_norm(self, data: pd.DataFrame, d_type: str):
        if (d_type == 'lat'):
            normalized_data = data / 180
        else:
            normalized_data = data / 360
        return normalized_data

    def denorm_angle(self, result):
        denorm_x = result[0][0][1] * 180
        denorm_y = result[0][0][2] * 360

        return denorm_x, denorm_y

    def min_max_norm(self, data: pd.DataFrame, min: int, max: int):
        if (max - min == 0):
            max = 1 
        normalized_data = ((data - min) / ((max - min)))
        return normalized_data

    def denorm_min_max(self, result, min: int, max: int, key: str):
        if (key == 'lat'):
            denormalized = result[0][0][1] * max - result[0][0][1] * min + min
        else:
            denormalized = result[0][0][2] * max - result[0][0][2] * min + min
        return denormalized

    def create_dataset(self, data: pd.DataFrame, n_steps: int):
        X, y = list(), list()
        for i in range(len(data)):
          # find the end of this pattern
          end_ix = i + n_steps
          # check if we are beyond the dataset
          if end_ix > len(data)-1:
            break
          # gather input and output parts of the pattern
          seq_x, seq_y = data.iloc[i:end_ix, :], data.iloc[end_ix, :]
          X.append(np.array(seq_x))
          y.append(np.array(seq_y))
        return np.array(X), np.array(y)

    '''def fitData(self, value: pd.DataFrame, min: int, max: int):
        fitedData = np.interp(value, (min, max), (0, +1))
        return fitedData'''

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
    def set_the_model(self, look_back):
        self.model = Sequential()
        self.model.add(LSTM(32, activation="tanh", input_shape=(look_back, 3),return_sequences=True)) #, stateful=True
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(64, activation="tanh", return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(128, activation="tanh", return_sequences=True))
        self.model.add(LSTM(64, activation="tanh", return_sequences=True))
        self.model.add(LSTM(32, activation="tanh", return_sequences=True)) #, stateful=True
        self.model.add(TimeDistributed(Dense(look_back, activation="relu")))
        #self.model.add(Dense(3))
        self.model.compile(loss="mse", optimizer="nadam", metrics=['acc']) #mse
        self.model.summary()
    
    def start_train(self, trainD1, trainD2):
        es = EarlyStopping(monitor='loss', patience = 2, mode='min')
        self.model.fit(trainD1, trainD1, epochs=300, batch_size = 32, verbose=1,  callbacks=[es])
    
    def predict_result(self, test_case):
        value = self.model.predict(test_case, verbose=0)
        return value
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
data_frame = pre_process.drop_if_small(data_frame, 4)

#%%
for i in data_frame:
    min_lat, max_lat = pre_process.find_min_max(i['latitude'])
    min_long, max_long = pre_process.find_min_max(i['longitude'])
    min_time, max_time = pre_process.find_min_max(i['timestamp'])
    i['latitude'] = pre_process.min_max_norm(i['latitude'], min_lat, max_lat)
    i['longitude'] = pre_process.min_max_norm(i['longitude'], min_long, max_long)
    i['timestamp'] = pre_process.min_max_norm(i['timestamp'], min_time, max_time)
    i.drop('user_id', axis=1, inplace=True)

#%%
look_back = 3
data_frame_copy = data_frame.copy()
#data_frame_copy[0]['latitude'][0] = pre_process.denorm_min_max(data_frame_copy[0]['latitude'][0], lat[0][0], lat[0][1])

#%%
#work on single dataset

for i in data_frame_copy:
  if (len(i) == 19082):
    x = i

train_size = int(len(x) * 0.67)
test_size = len(x) - train_size

train = x[0: train_size]
test = x[train_size: len(x)]

train_x, train_y = pre_process.create_dataset(train, 3)
test_x, test_y = pre_process.create_dataset(test, 3)

train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], 3)
test_x = test_x.reshape(test_x.shape[0], test_x.shape[1], 3)


#%%
model = Model()
model.set_the_model(look_back)
model.start_train(train_x, train_y)

#%%
test_res = model.predict_result(test_x[0:1])

plt.scatter(test_res[0][0][1], test_res[0][0][2], color='red', alpha=0.2)
plt.scatter(testY[0][1], testY[0][2], color='blue', alpha=0.2)
#%%

import folium

m = folium.Map(location=[39.9334, 32.8597], zoom_start=15)
# I can add marker one by one on the map
folium.Marker([denormalizedX, denormalizedY], popup='prediction').add_to(m)
for i, j in zip(x, y):
  folium.CircleMarker(location=[i, j], popup='real1', radius=2, icon=folium.Icon(color='rbg', angle=76)).add_to(m)

#m.save(outfile='gdrive/My Drive/Colab Notebooks/map.html')
m