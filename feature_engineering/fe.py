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
time_bins = eval(config["ELAPSED"]['time_bins'])
fill_with_mean = config["ELAPSED"].getboolean('fill_with_mean')
window_size = int(config["MOMENTUM"]['window_size'])
log = ["train_data", "test_data"]
with_test_data = config["DATA"].getboolean('with_test_data')

PATH = config['PATH']['data_path']
PATH_TRAIN = config['PATH']['train_data_path']
PATH_TEST = config['PATH']['test_data_path']

df_train = pd.read_csv(PATH_TRAIN, parse_dates=['Timestamp'])
df_test = pd.read_csv(PATH_TEST, parse_dates=['Timestamp'])
df_train = df_train.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)
df_test = df_test.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)

print('augmentation starts ...')

if with_test_data:
    new_df = df_train
else:    
    new_df = pd.concat([df_train, df_test])
    new_df = new_df[new_df['answerCode'] != -1]
    new_df = new_df.sort_values(['userID', 'Timestamp'])

# elpased
diff = df_test.loc[:, ['userID', 'testId', 'Timestamp']].groupby(['userID', 'testId']).diff()
diff = diff['Timestamp'].apply(cal_sec)
df_train['elapsed'] = diff.shift(-1).fillna(-1)

diff = df_test.loc[:, ['userID', 'testId', 'Timestamp']].groupby(['userID', 'testId']).diff()
diff = diff['Timestamp'].apply(cal_sec)
df_test['elapsed'] = diff.shift(-1).fillna(-1)

diff = new_df.loc[:, ['userID', 'testId', 'Timestamp']].groupby(['userID', 'testId']).diff()
diff = diff['Timestamp'].apply(cal_sec)
new_df['elapsed'] = diff.shift(-1).fillna(-1)

if bins:
    df_train['elapsed'] = pd.cut(df_train['elapsed'], bins = bins, labels = list(range(len(bins)-1)))
    df_test['elapsed'] = pd.cut(df_test['elapsed'], bins = bins, labels = list(range(len(bins)-1)))
    new_df['elapsed'] = pd.cut(new_df['elapsed'], bins = bins, labels = list(range(len(bins)-1)))
elif fill_with_mean:
    df_train.loc[df_train.elapsed == -1, "elapsed"] = df_train.loc[df_train.elapsed != -1, "elapsed"].mean()
    df_test.loc[df_train.elapsed == -1, "elapsed"] = df_test.loc[df_test.elapsed != -1, "elapsed"].mean()
    new_df.loc[df_train.elapsed == -1, "elapsed"] = new_df.loc[new_df.elapsed != -1, "elapsed"].mean()

# hour, day of week
new_df['hour'] = new_df['Timestamp'].map(lambda x: x.hour)
df_train['hour'] = df_train['Timestamp'].map(lambda x: x.hour)
df_test['hour'] = df_test['Timestamp'].map(lambda x: x.hour)
new_df['dow'] = new_df['Timestamp'].map(lambda x: x.dayofweek)
df_train['dow'] = df_train['Timestamp'].map(lambda x: x.dayofweek)
df_test['dow'] = df_test['Timestamp'].map(lambda x: x.dayofweek)
if time_bins:
    new_df['hour'] = pd.cut(new_df['hour'], bins = time_bins, labels = list(range(len(time_bins)-1)))
    new_df['hour'] = new_df['hour'].map(lambda x: list(range(len(time_bins)-1))[0] if x==list(range(len(time_bins)-1))[-1] else x)
    df_train['hour'] = pd.cut(df_train['hour'], bins = time_bins, labels = list(range(len(time_bins)-1)))
    df_train['hour'] = df_train['hour'].map(lambda x: list(range(len(time_bins)-1))[0] if x==list(range(len(time_bins)-1))[-1] else x)
    df_test['hour'] = pd.cut(df_test['hour'], bins = time_bins, labels = list(range(len(time_bins)-1)))
    df_test['hour'] = df_test['hour'].map(lambda x: list(range(len(time_bins)-1))[0] if x==list(range(len(time_bins)-1))[-1] else x)

new_df['class'] = new_df['testId'].str[2]
df_train['class'] = df_train['testId'].str[2]
df_test['class'] = df_test['testId'].str[2]

df_train['problem_num'] = df_train['assessmentItemID'].str[-2:]
df_test['problem_num'] = df_test['assessmentItemID'].str[-2:]

# class별 정답률
class_groupby = new_df.groupby('class').agg({
    'answerCode': lambda x: np.mean(x)
}).rename(columns = {'answerCode' : 'class_rate'})

df_train = df_train.merge(class_groupby.reset_index()[['class', 'class_rate']], on=['class'])
df_test = df_test.merge(class_groupby.reset_index()[['class', 'class_rate']], on=['class'])

# elapsed별 정답률
elapsed_groupby = new_df.groupby('elapsed').agg({
    'answerCode': lambda x: np.mean(x)
}).rename(columns = {'answerCode' : 'elapsed_rate'})

df_train = df_train.merge(elapsed_groupby.reset_index()[['elapsed', 'elapsed_rate']], on=['elapsed'])
df_test = df_test.merge(elapsed_groupby.reset_index()[['elapsed', 'elapsed_rate']], on=['elapsed'])

# problem_num 정답률
problem_num_groupby = new_df.groupby('problem_num').agg({
    'answerCode': lambda x: np.mean(x)
}).rename(columns = {'answerCode' : 'problem_num_rate'})

df_train = df_train.merge(problem_num_groupby.reset_index()[['problem_num', 'problem_num_rate']], on=['problem_num'])
df_test = df_test.merge(problem_num_groupby.reset_index()[['problem_num', 'problem_num_rate']], on=['problem_num'])

# knowledgeTag 정답률
knowledge_groupby = new_df.groupby('KnowledgeTag').agg({
    'answerCode': lambda x: np.mean(x)
}).rename(columns = {'answerCode' : 'knowledge_rate'})

df_train = df_train.merge(knowledge_groupby.reset_index()[['KnowledgeTag', 'knowledge_rate']], on=['KnowledgeTag'])
df_test = df_test.merge(knowledge_groupby.reset_index()[['KnowledgeTag', 'knowledge_rate']], on=['KnowledgeTag'])

# test 정답률
test_groupby = new_df.groupby('testId').agg({
    'answerCode': lambda x: np.mean(x)
}).rename(columns = {'answerCode' : 'test_rate'})

df_train = df_train.merge(test_groupby.reset_index()[['testId', 'test_rate']], on=['testId'])
df_test = df_test.merge(test_groupby.reset_index()[['testId', 'test_rate']], on=['testId'])

# assessmentItemID 정답률
assessment_groupby = new_df.groupby('assessmentItemID').agg({
    'answerCode': lambda x: np.mean(x)
}).rename(columns = {'answerCode' : 'assessment_rate'})

df_train = df_train.merge(assessment_groupby.reset_index()[['assessmentItemID', 'assessment_rate']], on=['assessmentItemID'])
df_test = df_test.merge(assessment_groupby.reset_index()[['assessmentItemID', 'assessment_rate']], on=['assessmentItemID'])

# user별 푼 문제수
df_train['solve_count'] = 1
df_test['solve_count'] = 1
df_train['solve_count'] = df_test.groupby('userID')['solve_count'].cumsum() - 1
df_test['solve_count'] = df_test.groupby('userID')['solve_count'].cumsum() - 1

# user의 hour별 정답률
hour_groupby = new_df.groupby('hour').agg({
    'answerCode': lambda x: np.mean(x)
}).rename(columns = {'answerCode' : 'hour_rate'})

df_train = df_train.merge(hour_groupby.reset_index()[['hour', 'hour_rate']], on=['hour'])
df_test = df_test.merge(hour_groupby.reset_index()[['hour', 'hour_rate']], on=['hour'])

# dow별 정답률
dow_groupby = new_df.groupby('dow').agg({
    'answerCode': lambda x: np.mean(x)
}).rename(columns = {'answerCode' : 'dow_rate'})

df_train = df_train.merge(dow_groupby.reset_index()[['dow', 'dow_rate']], on=['dow'])
df_test = df_test.merge(dow_groupby.reset_index()[['dow', 'dow_rate']], on=['dow'])

df_train = df_train.sort_values(['userID', 'Timestamp'])
df_test = df_test.sort_values(['userID', 'Timestamp'])

# user가 해당 문제를 전에 푼 횟수
df_train['num_solve_before'] = 1
df_test['num_solve_before'] = 1
df_train['num_solve_before'] = df_train.groupby(['userID', 'assessmentItemID'])['num_solve_before'].cumsum() - 1
df_test['num_solve_before'] = df_test.groupby(['userID', 'assessmentItemID'])['num_solve_before'].cumsum() - 1

# user_acc
df_train['user_correct_answer'] = df_train.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift(1))
df_train['user_total_answer'] = df_train.groupby('userID')['answerCode'].cumcount()
df_train['user_acc'] = df_train['user_correct_answer']/df_train['user_total_answer']
df_train['user_acc'] = df_train['user_acc'].fillna(0)
df_test['user_correct_answer'] = df_test.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift(1))
df_test['user_total_answer'] = df_test.groupby('userID')['answerCode'].cumcount()
df_test['user_acc'] = df_test['user_correct_answer']/df_test['user_total_answer']
df_test['user_acc'] = df_test['user_acc'].fillna(0)

df_train = df_train.sort_values(['userID', 'Timestamp'])
df_test = df_test.sort_values(['userID', 'Timestamp'])

# user단위 knowledge_rate별 정답률
df_train['user_knowledge'] = df_train.groupby(['userID', 'KnowledgeTag'])['answerCode'].transform(lambda x: x.cumsum().shift(1))
df_train['user_total_knowledge'] = df_train.groupby(['userID', 'KnowledgeTag'])['answerCode'].cumcount()
df_train['user_knowledge_rate'] = df_train['user_knowledge']/df_train['user_total_knowledge']
df_train['user_knowledge_rate'] = df_train['user_knowledge_rate'].fillna(0)
df_test['user_knowledge'] = df_test.groupby(['userID', 'KnowledgeTag'])['answerCode'].transform(lambda x: x.cumsum().shift(1))
df_test['user_total_knowledge'] = df_test.groupby(['userID', 'KnowledgeTag'])['answerCode'].cumcount()
df_test['user_knowledge_rate'] = df_test['user_knowledge']/df_test['user_total_knowledge']
df_test['user_knowledge_rate'] = df_test['user_knowledge_rate'].fillna(0)

df_train = df_train.sort_values(['userID', 'Timestamp'])
df_test = df_test.sort_values(['userID', 'Timestamp'])

# momentum
train_group = df_train.groupby('userID')['answerCode']
test_group = df_test.groupby('userID')['answerCode']
df_train['momentum'] = 0
df_test['momentum'] = 0
for i in range(1, window_size+1):
    df_train['momentum'] += train_group.shift(i).fillna(0)
    df_test['momentum'] += test_group.shift(i).fillna(0)
df_train['momentum'] = df_train['momentum'] / window_size
df_test['momentum'] = df_test['momentum'] / window_size

df_train = df_train.sort_values(['userID', 'Timestamp'])
df_test = df_test.sort_values(['userID', 'Timestamp'])

df_train.to_csv('/opt/ml/input/data/train_FE.csv')
df_test.to_csv('/opt/ml/input/data/test_FE.csv')

print('Done!')