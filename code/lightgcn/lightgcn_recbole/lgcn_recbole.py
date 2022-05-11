from logging import getLogger
import os
import json
import pandas as pd
import time, datetime

from recbole.model.general_recommender.lightgcn import LightGCN

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, get_trainer, init_seed, set_color

from recbole.config import Config
from recbole.data import create_dataset

from sklearn.metrics import accuracy_score, roc_auc_score

import torch


logger = getLogger()

# configurations initialization
config = Config(model='LightGCN', dataset="train_data", config_file_list=[f'train_data.yaml'])
config['epochs'] = 1
config['show_progress'] = False
config['device'] = "cuda" if torch.cuda.is_available() else "cpu"
config['save_dataloaders'] = True
config['save_dataset'] = True
config['embedding_size'] = 85




init_seed(config['seed'], config['reproducibility'])
# logger initialization
init_logger(config)

logger.info(config)

# dataset filtering
dataset = create_dataset(config)
user_id = config['USER_ID_FIELD'] # user_id 를 idx 로 바꾼애들
item_id = config['ITEM_ID_FIELD'] # assessment_id 를 idx 로 바꾼애들

user_id2token = dataset.field2id_token[user_id]
item_id2token = dataset.field2id_token[item_id]

user_token2id = {val: idx for idx, val in enumerate(user_id2token)}
item_token2id = {val: idx for idx, val in enumerate(item_id2token)}

# print(user_token2id)

u_data = {'U-id': user_token2id.keys(), 'users': user_token2id.values()}
i_data = {'I-id':item_token2id.keys(), 'assess':item_token2id.values()}
# print(u_data)

p = pd.DataFrame.from_dict(u_data)
d = pd.DataFrame.from_dict(i_data)

p.to_csv('user_token2id.csv', sep=',', index=False) # 'U-id': assessmentID indexing, 'users': 'U-id'의 embedding 순서
d.to_csv('item_token2id.csv', sep=',', index=False)  # 'I-id': assessmentID indexing, 'assess': 'I-id'의 embedding 순서

# d.to_csv('item_token2id.csv', sep=',')

# print(user_token2id)
# print(item_token2id)

logger.info(dataset)

# dataset splitting
train_data, valid_data, test_data = data_preparation(config, dataset)

# model loading and initialization
init_seed(config['seed'], config['reproducibility'])
model = LightGCN(config, train_data.dataset).to(config['device'])
logger.info(model)

# trainer loading and initialization
trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

# model training
best_valid_score, best_valid_result = trainer.fit(
    train_data, valid_data, saved=True, show_progress=config['show_progress']
)

# model evaluation
test_result = trainer.evaluate(test_data, load_best_model="True", show_progress=config['show_progress'])

logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
logger.info(set_color('test result', 'yellow') + f': {test_result}')

result = {
    'best_valid_score': best_valid_score,
    'valid_score_bigger': config['valid_metric_bigger'],
    'best_valid_result': best_valid_result,
    'test_result': test_result
}

print(json.dumps(result, indent=4))

print(model)