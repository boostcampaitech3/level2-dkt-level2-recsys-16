import os
import math
import numpy as np
import pandas as pd
from collections import Counter
import configparser

from torch import seed

# config
config = configparser.ConfigParser()
config.read('config.ini')
DATA_PATH = config['PATH']["data_path"]
DATA_PATH_TRAIN = config['PATH']["train_data_path"]
DATA_PATH_TEST = config['PATH']["test_data_path"]
length = int(config["ARGS"]['length'])
seed_ = int(config["ARGS"]['seed'])
mode = config["ARGS"]['mode']
with_test_data = config["ARGS"].getboolean('with_test_data')

def augmentation(df, length, mode='split', seed=42):
    print(f'augmentation starts on mode <<<{mode}>>>... ')
    
    df = df.copy()
    num_counts = Counter(df['userID'])
    users = df['userID'].unique()
    n = df['userID'].nunique()
    s = 0
    
    if mode =='split':
        user_augmentation = []
        for user in users:
            if user_augmentation:
                s = user_augmentation[-1]+1
            if num_counts[user]>= 2*length:
                for i in range(s, s+num_counts[user]//length):
                    user_augmentation.extend([i]*length)
                user_augmentation.extend([s+num_counts[user]//length]*(num_counts[user]%length))
            else:
                user_augmentation.extend([s]*num_counts[user])

        df['userID']=user_augmentation
        
    elif mode=='crop':
        max_ = max(users)
        df_new = pd.DataFrame()
        np.random.seed(seed_)
        for user in users:
            if num_counts[user]>= 2*length:
                k = np.random.randint(s,s+num_counts[user]-length)
                df_tmp = df.iloc[k:k+length,:].copy()
                df_tmp['userID'] = [max_+1]*length
                df_new = pd.concat([df_new,df_tmp])
            s+=num_counts[user]
            max_+=1
        df = pd.concat([df,df_new])
        df.reset_index()
    
    m = df['userID'].nunique()
    print(f'Done: {n}명에서 {m}명으로 augmentation')
    
    return df

if __name__ == "__main__":
    save_dir = os.path.join(DATA_PATH, 'augmented_csv')
    os.makedirs(save_dir, exist_ok=True)
    df_train = pd.read_csv(DATA_PATH_TRAIN)
    df_test = pd.read_csv(DATA_PATH_TEST)
    
    if with_test_data:
        df = pd.concat([df_train, df_test])
        df = augmentation(df,length, mode=mode, seed=seed_)
        df.to_csv(os.path.join(save_dir, f'train+test_augment_{mode}.csv'), index=False)

    else:
        df_train = augmentation(df_train,length, mode=mode, seed=seed_)
        df_test = augmentation(df_test,length, mode=mode, seed=seed_)
        df_train.to_csv(os.path.join(save_dir, f'train_augment_{mode}.csv'), index=False)
        df_test.to_csv(os.path.join(save_dir, f'test_augment_{mode}.csv'), index=False)