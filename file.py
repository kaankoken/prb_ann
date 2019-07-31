#%%
import shutil, os, glob
import pandas as pd
from datetime import datetime as dt
#%%

def move_file(src: str, dest:str, file_name: str):
    if not os.path.isdir(src):
        print('not a directory')
        return
    if not os.path.isdir(dest):
        os.mkdir(dest)
    if glob.glob(src + file_name + '*'):
        for file in glob.glob(src + file_name + '*'):
            shutil.move(file, dest)
    else:
        print('file does not exist')
        return

def combine_file_as_csv(src_dir: str, dest_dir:str, src_file_name: str, dest_file_name: str):
    if not os.path.isdir(dest_dir):
        os.mkdir(dest_dir)
    if not os.path.isdir(src_dir):
        print('directory does not exist')
        return
    if not glob.glob(src_dir + src_file_name + '*'):
        print('file does not exist')
        return
    with open(dest_dir + dest_file_name + '.csv', 'w') as outfile:
        for fname in glob.glob(src_dir + src_file_name + '*'):
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
def dateparse(timestamp):
    try:
        date = dt.strptime(timestamp, '%Y%m%d%H%M%S%f%Z')
    except ValueError:
        date = dt.strptime(timestamp, '%Y%m%d%H%M%S')
    return date 

def parse_files(src: str, src_file_name: str):
    data = pd.read_csv(src + src_file_name, sep=';', parse_dates=['timestamp'], date_parser=dateparse, names=["user_id", "timestamp", "longitude", "latitude", "Speed", "Course", "Altitude"])
    data = data.drop(labels=['Speed', 'Course', 'Altitude'], axis=1)
    data['timestamp'] = data['timestamp'].apply(lambda x: str(x) + ' UTC')
    data.to_csv(src + src_file_name, sep=',', header=True, index=False)
#%%
file_loc = '/home/legolas/Downloads/'
file_to = '/home/legolas/Desktop/prb_proj/txt-file/'
create_csv = '/home/legolas/Desktop/prb_proj/csv-file/'
#%%
move_file(file_loc, file_to, file_name='one_day_data')
#combine_file_as_csv(file_to, create_csv, src_file_name='one_day_data', dest_file_name='gps_data')
#%%

#parse_files(create_csv, src_file_name='gps_data.csv')