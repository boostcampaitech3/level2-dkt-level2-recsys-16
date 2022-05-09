import torch
import torch.optim
import torch.nn as nn
import pandas as pd
import os
from dkt.dataloader import lgbm_custom_k_fold_split, lgbm_custom_train_test_split
from dkt.utils import lgbm_feature_engineering
import wandb
import random
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import numpy as np

try:
    from transformers.modeling_bert import BertConfig, BertEncoder, BertModel
except:
    from transformers.models.bert.modeling_bert import (BertConfig,
                                                        BertEncoder, BertModel)


class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // self.args.dim_div)
        self.embedding_features = nn.ModuleList([])
        for value in self.args.n_embedding_layers:
            self.embedding_features.append(nn.Embedding(value + 1, self.hidden_dim // self.args.dim_div))

        self.has_cont_emb = self.args.n_cont_feat != 0
        emb_dim = self.hidden_dim if self.args.n_cont_feat == 0 else self.hidden_dim // 2

        self.embedding_cont_features = nn.Sequential(
                nn.Linear(self.args.n_cont_feat, emb_dim),
                nn.LayerNorm(emb_dim)
        )

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim//self.args.dim_div)*(len(self.args.n_embedding_layers)+1), emb_dim)

        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True
        )

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def init_hidden(self, batch_size):
        h = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        c = c.to(self.device)

        return (h, c)

    def forward(self, input):

       

        # *cate_features, cont_features, mask, interaction, gather_index, correct
        cont_features, mask, interaction = input[-5:-2]


        print(interaction)
        batch_size = interaction.size(0)

        # Embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_features = []
        for _input, _embedding_feature in zip(input[:-5], self.embedding_features):
            value = _embedding_feature(_input)
            embed_features.append(value)

        embed_features = [embed_interaction] + embed_features

        embed = torch.cat(embed_features, 2)
        X = self.comb_proj(embed)

        if self.has_cont_emb:
            cont_emb = self.embedding_cont_features(cont_features)
            X = torch.cat([X, cont_emb], 2)

        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(X, hidden)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds


class LSTMATTN(nn.Module):
    def __init__(self, args):
        super(LSTMATTN, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.n_heads = self.args.n_heads
        self.drop_out = self.args.drop_out

        # Embedding
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // self.args.dim_div)
        self.embedding_features = nn.ModuleList([])
        for value in self.args.n_embedding_layers:
            self.embedding_features.append(nn.Embedding(value + 1, self.hidden_dim // self.args.dim_div))
        
        self.has_cont_emb = self.args.n_cont_feat != 0
        emb_dim = self.hidden_dim if self.args.n_cont_feat == 0 else self.hidden_dim // 2
        
        self.embedding_cont_features = nn.Sequential(
                nn.Linear(self.args.n_cont_feat, emb_dim),
                nn.LayerNorm(emb_dim)
        )
        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim//self.args.dim_div)*(len(self.args.n_embedding_layers)+1), emb_dim)


        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True
        )

        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=1,
            num_attention_heads=self.n_heads,
            intermediate_size=self.hidden_dim,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
        )
        self.attn = BertEncoder(self.config)

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def init_hidden(self, batch_size):
        h = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        c = c.to(self.device)

        return (h, c)

    def forward(self, input):
        cont_features, mask, interaction = input[-5:-2]

        batch_size = interaction.size(0)

        # Embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_features = []
        for _input, _embedding_feature in zip(input[:-5], self.embedding_features):
            value = _embedding_feature(_input)
            embed_features.append(value)

        embed_features = [embed_interaction] + embed_features

        embed = torch.cat(embed_features, 2)
        X = self.comb_proj(embed)

        if self.has_cont_emb:
            cont_emb = self.embedding_cont_features(cont_features)
            X = torch.cat([X, cont_emb], 2)

        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(X, hidden)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.n_layers

        encoded_layers = self.attn(out, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoded_layers[-1]

        out = self.fc(sequence_output)

        preds = self.activation(out).view(batch_size, -1)

        return preds


class Bert(nn.Module):
    def __init__(self, args):
        super(Bert, self).__init__()
        self.args = args
        self.device = args.device

        # Defining some parameters
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)



        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // self.args.dim_div)
        self.embedding_features = nn.ModuleList([])
        for value in self.args.n_embedding_layers:
            self.embedding_features.append(nn.Embedding(value + 1, self.hidden_dim // self.args.dim_div))

        self.has_cont_emb = self.args.n_cont_feat != 0
        emb_dim = self.hidden_dim if self.args.n_cont_feat == 0 else self.hidden_dim // 2
        
        self.embedding_cont_features = nn.Sequential(
                nn.Linear(self.args.n_cont_feat, emb_dim),
                nn.LayerNorm(emb_dim)

        )
        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim//self.args.dim_div)*(len(self.args.n_embedding_layers)+1), emb_dim)

        # Bert config
        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.args.n_layers,
            num_attention_heads=self.args.n_heads,
            max_position_embeddings=self.args.max_seq_len,
        )

        # Defining the layers
        # Bert Layer
        self.encoder = BertModel(self.config)

        # Fully connected layer
        self.fc = nn.Linear(self.args.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def forward(self, input):
        cont_features, mask, interaction = input[-5:-2]
        batch_size = interaction.size(0)

        # 신나는 embedding
        embed_interaction = self.embedding_interaction(interaction)

        embed_features = []
        for _input, _embedding_feature in zip(input[:-5], self.embedding_features):
            value = _embedding_feature(_input)
            embed_features.append(value)

        embed_features = [embed_interaction] + embed_features

        embed = torch.cat(embed_features, 2)
        X = self.comb_proj(embed)

        if self.has_cont_emb:
            cont_emb = self.embedding_cont_features(cont_features)
            X = torch.cat([X, cont_emb], 2)

        # Bert
        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=mask)
        out = encoded_layers[0]

        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds


class LGBM:
    def __init__(self, args):
        self.args = args

    def train(self):
        df = pd.read_csv(os.path.join(self.args.data_dir, 'train_data.csv'))
        df, FEATS, CATEGORICAL_FEATS = lgbm_feature_engineering(df)
        print(df.sample(3))

        train, test = lgbm_custom_train_test_split(df, ratio=self.args.split_ratio)

        y_train = train['answerCode']
        train = train.drop(['answerCode'], axis=1)
        y_test = test['answerCode']
        test = test.drop(['answerCode'], axis=1)

        lgb_train = lgb.Dataset(train[FEATS], y_train, feature_name=FEATS, categorical_feature=CATEGORICAL_FEATS)
        lgb_test = lgb.Dataset(test[FEATS], y_test, feature_name=FEATS, categorical_feature=CATEGORICAL_FEATS)

        model = lgb.train(  # args 로 받게끔 하는 부분
            params={'objective': 'binary',
                    'metric': ['binary_logloss', 'auc'],
                    'boosting': self.args.boosting, #'dart', # default: gbdt(gradient boosting decision tree)
                    'max_depth': self.args.max_dep, #12, # handle overfitting, lowering will do, 3~12 recommended
                    'num_leaves': self.args.num_leaves, #512, # default: 31
                    'min_data_in_leaf': self.args.mdil, #200, # handle overfitting, minimum number of records a leaf may have
                    'feature_fraction': self.args.ff, #0.8, # randomly choose fraction of parameters when building tree in each iteration
                    'bagging_fraction': self.args.bf, #0.8, # use fraction of data for each iteration, speed up and avoid overfitting
                    'lambda': self.args.lmda, #0.2, # specifies regularization
                    'min_gain_to_split': self.args.mgts, #20, # describe the minimum gain to make a split, used to control number of useful splits in tree
                    'max_cat_group': self.args.mcg, #64, #When the number of category is large, finding the split point on it is easily over-fitting
                    'tree_learner': self.args.tl #'feature',  # default: serial, [data, feature]
                    },
            train_set=lgb_train,
            valid_sets=[lgb_train, lgb_test],  # 자동으로 훈련데이터는 빼고 나머지를 모두 사용해 valid
            # learning_rate = 0.001, # default: 0.1
            verbose_eval=100,
            num_boost_round=1000,  # 최대 epoch 비슷, alias: num_iteration, n_estimators, num_trees
            early_stopping_rounds=100  # stop training if one metric of one validation data doesn’t improve.
        )

        preds = model.predict(test[FEATS], num_iteration=model.best_iteration)  # num_iteration 은 early_stopping 했을때 사용
        acc = accuracy_score(y_test, np.where(preds >= 0.5, 1, 0))
        auc = roc_auc_score(y_test, preds)

        wandb.log(
            {
                "valid_auc": auc,
                "valid_acc": acc,
            }
        )
        print(f'VALID AUC : {auc} ACC : {acc}\n')

        # SAVE OUTPUT
        output_dir = 'lgbm-model/'

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model.save_model(os.path.join(output_dir, self.args.model_name))
        print('model saved!')


    def infer(self):
        output_dir = 'lgbm-model/'
        model = lgb.Booster(model_file=os.path.join(output_dir, self.args.model_name))  # 모델 불러오기

        # LOAD TESTDATA
        test_csv_file_path = os.path.join(self.args.data_dir, 'test_data.csv')
        test_df = pd.read_csv(test_csv_file_path)

        # FEATURE ENGINEERING
        test_df, FEATS, CATEGORICAL_FEATS = lgbm_feature_engineering(test_df)

        # LEAVE LAST INTERACTION ONLY
        test_df = test_df[test_df['userID'] != test_df['userID'].shift(-1)]

        # DROP ANSWERCODE
        test_df = test_df.drop(['answerCode'], axis=1)

        total_preds = model.predict(test_df[FEATS], num_iteration=model.best_iteration)

        # SAVE OUTPUT
        output_dir = 'lgbm-output/'
        write_path = os.path.join(output_dir, self.args.model_name.split('.')[0]+'_submission.csv')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(write_path, 'w', encoding='utf8') as w:
            print("writing prediction : {}".format(write_path))
            w.write("id,prediction\n")
            for id, p in enumerate(total_preds):
                w.write('{},{}\n'.format(id, p))

        print('inference completed!')
