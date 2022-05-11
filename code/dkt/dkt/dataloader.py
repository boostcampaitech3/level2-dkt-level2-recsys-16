import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.preprocessing import LabelEncoder


class Preprocess:
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def split_data(self, data, ratio=0.85, shuffle=True, seed=0):
        """
        split data into two parts with a given ratio.
        """
        if shuffle:
            random.seed(seed)  # fix to default seed 0
            random.shuffle(data)

        size = int(len(data) * ratio)
        data_1 = data[:size]
        data_2 = data[size:]

        return data_1, data_2


    def __save_labels(self, encoder, name):
        le_path = os.path.join(self.args.asset_dir, name + "_classes.npy")
        np.save(le_path, encoder.classes_)

    def __preprocessing(self, df, is_train=True):
        cate_cols = self.args.FEAT_COLUMN
        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)

        for col in cate_cols:
            le = LabelEncoder()

            if col in ['user_order', 'item_order']:
                continue

            elif is_train:
                # For UNKNOWN class
                a = df[col].unique().tolist() + ["unknown"]
                le.fit(a)
                self.__save_labels(le, col)
            else:
                label_path = os.path.join(self.args.asset_dir, col + "_classes.npy")
                le.classes_ = np.load(label_path)

                df[col] = df[col].apply(
                    lambda x: x if str(x) in le.classes_ else "unknown"
                )

            # 모든 컬럼이 범주형이라고 가정
            # print(df[col])
            df[col] = df[col].astype(str)
            test = le.transform(df[col])
            df[col] = test
            # print(df[col])
            # cate_col 의 item 개수만큼 좁혀서 정수로 뱉어주는 구간
            # user_id 와 assessment_id 는 order 를 사용해서 정수로 뱉어서 넣어줄수 잇음
        return df

    def __feature_engineering(self, df):
        self.args.USERID_COLUMN = ['userID']

        ##### 'userID' 대신에 다른 피쳐로 붙여야 댐, 'assessmentID' 대신에도 다른걸로 붙여놓아용
        ##### nn.Embbedding_from_pretrained 를 사용해서 lookup_table의 순서에 맞추어 embbedding 을 가져오는 식으로 짜고 싶다
        # self.args.FEAT_COLUMN = ['problem_num', 'KnowledgeTag', 'testId', 'class'] # , 'user_order', 'assessmentID_order'
        self.args.FEAT_COLUMN = ['testId', 'KnowledgeTag', 'user_order', 'item_order'] # 순서 두개를 넣고

        # self.args.CONT_FEAT_COLUMN = ['user_knowledge_rate', 'user_acc', 'momentum', 'knowledge_rate', 'class_rate', 'assessment_rate', 'hour_rate','elapsed_rate']
        self.args.CONT_FEAT_COLUMN = [] # 값은 유지되서 그대로 넘겨지는 부분
        # self.args.EXCLUDE_COLUMN = ['Timestamp','user_total_answer','user_correct_answer','momentum']
        self.args.ANSWER_COLUMN = ['answerCode']

        self.args.n_cate_feat = len(self.args.FEAT_COLUMN)
        self.args.n_cont_feat = len(self.args.CONT_FEAT_COLUMN)

        # assert df.head().shape[1] == len(self.args.USERID_COLUMN) + len(self.args.ANSWER_COLUMN) \
        #                              + len(self.args.FEAT_COLUMN) + len(self.args.CONT_FEAT_COLUMN) \
        #                              + len(self.args.EXCLUDE_COLUMN)

        print(f"using category columns: {self.args.FEAT_COLUMN}")
        print(f"using continuous columns: {self.args.CONT_FEAT_COLUMN}")

        return df

    def df_to_tuple(self, r):
        # print(r)
        # userid별로 끊어서 df가 r 로 넘어옴 (user의 iteraction 개수 x columns들의 개수 합) -> (a, 14)
        return [r[x].values for x in self.args.FEAT_COLUMN] \
               + [r[x].values for x in self.args.CONT_FEAT_COLUMN] \
               + [r[x].values for x in self.args.ANSWER_COLUMN]

    def load_data_from_file(self, file_name, is_train=True):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path)

        user = pd.read_csv(os.path.join(self.args.data_dir, self.args.user_emb))
        item = pd.read_csv(os.path.join(self.args.data_dir, self.args.item_emb))

        print(item.columns)

        ## process_batch-> masking 예외처리 부분
        item = item[['assessmentItemID', 'order']]
        item['order'] = item['order'] - 1

        user = user[['userID', 'order']]
        user['order'] = user['order'] - 1


        df = df.merge(item, on=['assessmentItemID'])
        df = df.merge(user, on=['userID'])


        df.columns = list(df.columns)[:-2] + ['item_order','user_order']
        # --------------


        df = self.__feature_engineering(df)
        print(f'after __feature_engineering\n{df.head()}')
        df = self.__preprocessing(df, is_train)
        print(f'after __preprocessing\n{df.head()}')

        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용
        self.args.n_embedding_layers = []

        # asset 가져오고
        for val in self.args.FEAT_COLUMN[:-2]:
            self.args.n_embedding_layers.append(len(np.load(os.path.join(self.args.asset_dir, val+'_classes.npy'))))

        # print(self.args.n_embedding_layers) # [1538, 913]

        # print(df)
        # df 불러올때 순서대로 안불리는 경우 있어서
        df = df.sort_values(by=["userID", "Timestamp"], axis=0)
        # print(df)
        columns = self.args.USERID_COLUMN + self.args.FEAT_COLUMN \
                  + self.args.CONT_FEAT_COLUMN + self.args.ANSWER_COLUMN

        group = df[columns].groupby('userID').apply(self.df_to_tuple)

        # print(len(group[0])) #3 ( feat col 2개 + answer 1개 )
        # print(len(group[1])) #3
        # print(len(group.values)) # 7442 유저 id 별로 따라감

        return group.values  # 여기까지는 아직 embedding 되지 않음

    def load_train_data(self, file_name):
        self.train_data = self.load_data_from_file(file_name)

    def load_test_data(self, file_name):
        self.test_data = self.load_data_from_file(file_name, is_train=False)


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args

    def __getitem__(self, index):
        row = self.data[index]

        # 각 data의 sequence length
        seq_len = len(row[0])

        cate_cols = row[:self.args.n_cate_feat]
        cont_cols = row[self.args.n_cate_feat:-1]
        answer_col = row[-1]

        # max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        if seq_len > self.args.max_seq_len:
            for i, col in enumerate(cate_cols):
                cate_cols[i] = col[-self.args.max_seq_len:]

            
            for i, col in enumerate(cont_cols):
                cont_cols[i] = col[-self.args.max_seq_len:]

            answer_col = answer_col[-self.args.max_seq_len:]
            

            mask = np.ones(self.args.max_seq_len, dtype=np.int16)
        else:
            mask = np.zeros(self.args.max_seq_len, dtype=np.int16)
            mask[-seq_len:] = 1

        # np.array -> torch.tensor 형변환
        for i, col in enumerate(cate_cols):
            cate_cols[i] = torch.tensor(col)
        
        # cont_cols = torch.tensor(cont_cols) if self.args.n_cont_feat > 0 else None
        for i, col in enumerate(cont_cols):
            cont_cols[i] = torch.tensor(col)

        mask = torch.tensor(mask)
        answer_col = torch.tensor(answer_col)
        
        return *cate_cols, *cont_cols, mask, answer_col

    def __len__(self):
        return len(self.data)


from torch.nn.utils.rnn import pad_sequence


def collate(batch):
    col_n = len(batch[0])
    col_list = [[] for _ in range(col_n)]
    max_seq_len = len(batch[0][-2])

    # batch의 값들을 각 column끼리 그룹화
    for row in batch:
        for i, col in enumerate(row):
            pre_padded = torch.zeros(max_seq_len).to(col.dtype)
            pre_padded[-len(col) :] = col
            col_list[i].append(pre_padded)

    for i, _ in enumerate(col_list):
        col_list[i] = torch.stack(col_list[i])

    return col_list


def get_loaders(args, train, valid):

    pin_memory = False
    train_loader, valid_loader = None, None

    if train is not None:
        trainset = DKTDataset(train, args)
        train_loader = torch.utils.data.DataLoader(
            trainset,
            num_workers=args.num_workers,
            shuffle=True,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
            collate_fn=collate,
        )
    if valid is not None:
        valset = DKTDataset(valid, args)
        valid_loader = torch.utils.data.DataLoader(
            valset,
            num_workers=args.num_workers,
            shuffle=False,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
            collate_fn=collate,
        )

    return train_loader, valid_loader


def lgbm_custom_k_fold_split(df, k):
    users = list(
        zip(df['userID'].value_counts().index, df['userID'].value_counts()))  # (유저의 인덱스, 유저아이템row 개수)들로 구성된 리스트
    random.shuffle(users)
    step = len(users) // k + 1

    users = list(users[i:i + step] for i in range(0, len(users), step))

    dfs = []

    for user in users:
        user_id = []
        # print(user[0])
        for user in user:
            user_id.append(user[0])

        # print(user_id)

        fold = df[df['userID'].isin(user_id)]

        # print(fold.head(5))
        # print(fold)

        dfs.append(fold.copy())  # 객체자체를 복제

    return dfs


# train과 test 데이터셋은 사용자 별로 묶어서 분리를 해주어야함
seed = random.seed(42)


def lgbm_custom_train_test_split(df, ratio=0.7, split=True):
    users = list(zip(df['userID'].value_counts().index, df['userID'].value_counts()))
    random.shuffle(users)

    max_train_data_len = ratio * len(df)
    sum_of_train_data = 0
    user_ids = []

    for user_id, count in users:
        sum_of_train_data += count
        if max_train_data_len < sum_of_train_data:
            break
        user_ids.append(user_id)

    train = df[df['userID'].isin(user_ids)]
    test = df[df['userID'].isin(user_ids) == False]

    # test데이터셋은 각 유저의 마지막 interaction만 추출
    test = test[test['userID'] != test['userID'].shift(-1)]
    return train, test