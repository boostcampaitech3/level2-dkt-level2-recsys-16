import os
import random

import numpy as np
import pandas as pd
import torch


def setSeeds(seed=42):

    # 랜덤 시드를 설정하여 매 코드를 실행할 때마다 동일한 결과를 얻게 합니다.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def lgbm_feature_engineering(df):
    print('lgbm feature engineering...')
    # 유저별 시퀀스를 고려하기 위해 아래와 같이 정렬
    df.sort_values(by=['userID', 'Timestamp'], inplace=True)

    # 유저들의 문제 풀이수, 정답 수, 정답률을 시간순으로 누적해서 계산
    df['user_correct_answer'] = df.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift(1))
    df['user_total_answer'] = df.groupby('userID')['answerCode'].cumcount()
    df['user_acc'] = df['user_correct_answer'] / df['user_total_answer']
    df['head_tag'] = df['assessmentItemID'].apply(lambda x: x[1:4])  # 학년 tag 만들기
    df['q_tag'] = df['assessmentItemID'].apply(lambda x: x[-2:])  # 소소분류 tag 만들기

    # testId와 KnowledgeTag의 전체 정답률은 한번에 계산
    # 아래 데이터는 제출용 데이터셋에 대해서도 재사용

    # 대분류 mean(010, 020, ...) 도 추가해도 좋을듯 --> 했음
    # 문제분류(1번~12번) 에 대한 분류 --> 했음
    # 문제풀이에 걸린 시간 태그
    # 평균 문제 풀이 태그
    correct_t = df.groupby(['testId'])['answerCode'].agg(['mean', 'sum'])
    correct_t.columns = ["test_mean", 'test_sum']
    correct_k = df.groupby(['KnowledgeTag'])['answerCode'].agg(['mean', 'sum'])
    correct_k.columns = ["tag_mean", 'tag_sum']
    correct_h = df.groupby(['head_tag'])['answerCode'].agg(['mean', 'sum'])
    correct_h.columns = ["head_mean", 'head_sum']
    correct_q = df.groupby(['q_tag'])['answerCode'].agg(['mean', 'sum'])
    correct_q.columns = ["q_mean", 'q_sum']

    # print(correct_k.head(5))

    df = pd.merge(df, correct_t, on=['testId'], how="left")
    df = pd.merge(df, correct_k, on=['KnowledgeTag'], how="left")
    df = pd.merge(df, correct_h, on=['head_tag'], how='left')  # 헤드태그별 분류 붙이기
    df = pd.merge(df, correct_q, on=['q_tag'], how='left')  # 문제번호별 분류 붙이기

    # LGBM 에서 사용하기 위해 카테고리컬 데이터를 int type 으로 변환
    df['q_tag'] = df['q_tag'].astype('int64')
    df['head_tag'] = df['head_tag'].astype('int64').apply(lambda x: x // 10)

    # 카테고리컬 데이터 renumbering 작업
    know = df['KnowledgeTag'].unique()
    df['KnowledgeTag'] = df['KnowledgeTag'].apply(lambda x: np.where(know == x)[0][0])

    # 사용할 Feature 설정
    FEATS = ['user_correct_answer', 'user_total_answer',
             'user_acc', 'test_mean', 'test_sum', 'tag_mean', 'tag_sum',
             'head_mean', 'head_sum', 'q_mean', 'q_sum',
             'KnowledgeTag', 'head_tag', 'q_tag']
    CATEGORICAL_FEATS = ['KnowledgeTag', 'head_tag', 'q_tag', ]

    return df, FEATS, CATEGORICAL_FEATS
