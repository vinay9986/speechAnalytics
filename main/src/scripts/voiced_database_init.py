import shutil
import pandas as pd
import random as rand
import re
import warnings
warnings.filterwarnings('ignore')

rand.seed(12)
data_path = <PATH TO UNZIPPED VOICED FOLDER TILL 1.0.0>
audio_path = <PATH WHERE YOU WANT PROCESSED FILES>
dat_ext = ".dat"
wav_ext = ".wav"
hea_ext = ".hea"
target_regex = re.compile('<diagnoses>: .* <')
sex_regex = re.compile('<sex>: .* <')
df = pd.DataFrame(columns = ['filename', 'sex', 'fold', 'target'])
csv_file = "dataset.csv"
try:
    with open(data_path+"RECORDS") as fp:
        Lines = fp.readlines()
        for _, file_base in enumerate(Lines):
            file_name = file_base.strip()+dat_ext
            fold = rand.choice([0, 1])
            target = 'healthy'
            voice_sample = data_path+file_name
            voice_hea = data_path+file_base.strip()+hea_ext
            shutil.copyfile(voice_sample, audio_path+file_name)
            with open(voice_hea) as hea:
                line = hea.readlines()[-1]
                diagnoses = target_regex.findall(line)[0].split(':')[1].split('<')[0]
                sex = sex_regex.findall(line)[0].split(':')[1].split('<')[0]
                if "healthy" != diagnoses.strip().lower():
                    target = 'pathological'
                if "m" == sex.strip().lower():
                    sex = 1
                else:
                    sex = 0
            df.loc[len(df.index)] = [file_base.strip()+wav_ext, sex, fold, target]
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv(path_or_buf=audio_path+csv_file, index=False, header=True)
    print('Initial files created, please run shell commands provided in datToWavConvertion.txt')
    print('Please keep note of processed file location: ', audio_path)
except IOError as e:
    print("Unable to copy file. %s" % e)
    exit(1)
except:
    print("Unexpected error:", sys.exc_info())
    exit(1)