#SVD database download limits the max number of downloadable samples to 864
#Download samples in groups as discribed
# download all samples where speaker = female and recording session = healthy and place the downloaded files in folder named f-h
# download all samples where speaker = female and recording session = pathological and place the downloaded files in folder named f-p
# download all samples where speaker = male and recording session = healthy and place the downloaded files in folder named m-h
# download all samples where speaker = male and recording session = pathological and place the downloaded files in folder named m-p

import shutil
import os
import sys
import random as rand
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

rand.seed(12)

data_main_path = <PATH TO PARENT FOLDER CONTAINING GROUPED SAMPLES>
folders = ['f-h/', 'f-p/', 'm-h/', 'm-p/']
all_files_path = <PATH WHERE YOU WANT PROCESSED FILES>
csv_file = "dataset.csv"
df = pd.DataFrame(columns = ['filename', 'sex', 'fold', 'target'])
try:
    for folder in folders:
        current_folder = data_main_path+folder
        files = os.listdir(current_folder)
        for file in files:
            voice_sample = current_folder+file
            shutil.copyfile(voice_sample, all_files_path+file)
            fold = rand.choice([0, 1])
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
            df.loc[len(df.index)] = [file, sex, fold, target]
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv(path_or_buf=all_files_path+csv_file, index=False, header=True)
    print('Initial files created, please keep note of processed file location: ', all_files_path)
except IOError as e:
    print("Unable to copy file. %s" % e)
    exit(1)
except:
    print("Unexpected error:", sys.exc_info())
    exit(1)