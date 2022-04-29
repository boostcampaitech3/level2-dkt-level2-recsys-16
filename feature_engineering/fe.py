import os
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from utils import cal_sec
import configparser

# config
config = configparser.ConfigParser()
config.read('config.ini')
bins = eval(config["ELAPSED"]['bins'])
fill_with_mean = config["ELAPSED"].getboolean('fill_with_mean')
window_size = int(config["MOMENTUM"]['window_size'])
log = ["train_data", "test_data"]

PATH = '../../data'

df_train = pd.read_csv(os.path.join(PATH,'train_data.csv'), parse_dates=['Timestamp'])
df_test = pd.read_csv(os.path.join(PATH,'test_data.csv'), parse_dates=['Timestamp'])
df_train = df_train.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)
df_test = df_test.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)

### train ###
for k, df in enumerate([df_train, df_test]):
    s = log[k]
    print(f"making {s}_adjusted...")
    # class
    df['class'] = pd.Series(df['assessmentItemID'].apply(lambda x: int(x[2])),dtype='int8')

    # elpased
    diff = df.loc[:, ['userID', 'testId', 'Timestamp']].groupby(['userID', 'testId']).diff()
    diff = diff['Timestamp'].apply(cal_sec)
    df['elapsed'] = diff.shift(-1).fillna(-1)

    if bins:
        df['elapsed'] = pd.cut(df['elapsed'], bins = bins, labels = list(range(len(bins)-1)))
    elif fill_with_mean:
        df.loc[df.elapsed == -1, "elapsed"] = df.loc[df.elapsed != -1, "elapsed"].mean()

    # momentum
    tmp = df.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift(1))

    i = 0
    l = len(tmp)
    momoentum = []
    while i<l:
        if math.isnan(tmp[i]):
            momoentum.extend([0]+list(tmp[i+1:i+window_size]))
            i+=window_size
        else:
            momoentum.append(tmp[i]-tmp[i-window_size])
            i+=1

    df["momentum"] = momoentum
    df["momentum"] = df["momentum"].fillna(0)

    df.to_csv(os.path.join(PATH,f'{s}_adjusted.csv'), index=False)
    print(f"saved -- {s}_adjusted.csv")


# ### test ###
# print("making test_data_adjusted...")
# # class
# df_test['class'] = pd.Series(df_test['assessmentItemID'].apply(lambda x: int(x[2])),dtype='int8')

# # elpased
# diff = df_test.loc[:, ['userID', 'testId', 'Timestamp']].groupby(['userID', 'testId']).diff()
# diff = diff['Timestamp'].apply(cal_sec)
# df_test['elapsed'] = diff.shift(-1).fillna(-1)
# if bins:
#     df_test['elapsed'] = pd.cut(df_test['elapsed'], bins = bins, labels = list(range(len(bins)-1)))
# elif fill_with_mean:
#     df_test.loc[df_test.elapsed == -1, "elapsed"] = df_test.loc[df_test.elapsed != -1, "elapsed"].mean()

# # momentum
# tmp = df_test.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift(1))

# i = 0
# l = len(tmp)
# momoentum = []
# while i<l:
#     if math.isnan(tmp[i]):
#         momoentum.extend([0]+list(tmp[i+1:i+window_size]))
#         i+=window_size
#     else:
#         momoentum.append(tmp[i]-tmp[i-window_size])
#         i+=1

# df_test["momentum"] = momoentum
# df_test["momentum"] = df_test["momentum"].fillna(0)

# df_test.to_csv(os.path.join(PATH,'test_data_adjusted.csv'), index=False)
# print("saved -- test_data_adjusted.csv")