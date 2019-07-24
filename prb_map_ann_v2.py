#%%
import os
import pandas as pd
import numpy as np
from google.colab import drive, files
import matplotlib.pylot as pylot

#%% 
class PreProcess:
#%%
class GoogleDrive:
    #mounts the drive and change the file path to google drive's path
    def mount_drive(self):
        drive.mount('/content/gdrive')
        os.chdir('/gdrive')
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

class Model: