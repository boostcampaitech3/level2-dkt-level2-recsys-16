import os
import pandas as pd
import configparser

# config
config = configparser.ConfigParser()
config.read('config.ini')
DATA_PATH = config['PATH']["data_path"]
DATA_PATH_TRAIN = config['PATH']["train_data_path"]
wrong_in_a_row = int(config["ABNORMAL"]['wrong_in_a_row'])

df_train = pd.read_csv(DATA_PATH_TRAIN)
tmp = df_train.groupby(['userID'])['answerCode'].transform(lambda x: x.cumsum().shift(1))
cnt = 0
abnormal = []
for i in range(wrong_in_a_row,len(tmp)):
    if tmp[i]-tmp[i-wrong_in_a_row]==0:
        cnt+=1
    else:
        cnt=0
    if cnt>=wrong_in_a_row:
        abnormal.append(i)

abnormal = set(map(lambda x: df_train["userID"][x], abnormal))

drop_index = []
for i in range(len(df_train)):
    if df_train['userID'][i] in abnormal:
        drop_index.append(i)

df_train.drop(drop_index, inplace=True)
df_train.reset_index(inplace=True)
df_train.to_csv(os.path.join(DATA_PATH,'train_abnormal_del.csv'), index=False)
print(f'{len(abnormal)}명의 유저 정보 {len(drop_index)}개 삭제, {len(df_train)}개의 row로 변경')