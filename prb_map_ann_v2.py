#%%
import os
import pandas as pd
import numpy as np
from google.colab import drive, files
import matplotlib.pyplot as pylot

#%%
class PreProcess:
    __csv_path: str
    def __init__(self, csv_path: str = 'Colab Notebooks/bq-results-20190628-104852-awu4v6ig6fyq.csv'):
        if csv_path.startswith('/'):
            self.__csv_path = '.' + csv_path
        else:
            self.__csv_path = './' + csv_path

    def read_file(self):
        df = pd.read_csv(self.__csv_path)
        return df

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