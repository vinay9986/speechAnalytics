import os
import sys
import pandas as pd
import warnings
import librosa
warnings.filterwarnings('ignore')

data_main_path = <PATH TO PARENT FOLDER CONTAINING BELOW SUB-FOLDERS>
folders = ['f-h/', 'f-p/', 'm-h/', 'm-p/']
csv_file = "svd_w2v2_dataset.csv"
df = pd.DataFrame(columns = ['path', 'sex', 'duration', 'target'])
try:
    for folder in folders:
        current_folder = data_main_path+folder
        files = os.listdir(current_folder)
        for file in files:
            path = current_folder+file
            duration = librosa.get_duration(filename=path)
            if folder == 'f-h/':
                sex = 0
                target = 'healthy'
            elif folder == 'f-p/':
                sex = 0
                target = 'pathology'
            elif folder == 'm-h/':
                sex = 1
                target = 'healthy'
            else:
                sex = 1
                target = 'pathology'
            df.loc[len(df.index)] = [path, sex, duration, target]
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv(path_or_buf=data_main_path+csv_file, index=False, header=True)
    print('Initial files created, please keep note of processed file location: ', data_main_path+csv_file)
except IOError as e:
    print("Unable to copy file. %s" % e)
    exit(1)
except:
    print("Unexpected error:", sys.exc_info())
    exit(1)