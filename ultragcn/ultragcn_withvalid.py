import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from IPython import embed
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import torch.utils.data as data
import scipy.sparse as sp
import os
import gc
import configparser
import time
import argparse
import csv
import pandas as pd
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter


def data_param_prepare(config_file):

    config = configparser.ConfigParser()
    config.read(config_file)

    params = {}

    embedding_dim = config.getint('Model', 'embedding_dim')
    params['embedding_dim'] = embedding_dim
    ii_neighbor_num = config.getint('Model', 'ii_neighbor_num')
    params['ii_neighbor_num'] = ii_neighbor_num
    model_save_path = config['Model']['model_save_path']
    params['model_save_path'] = model_save_path
    max_epoch = config.getint('Model', 'max_epoch')
    params['max_epoch'] = max_epoch

    params['enable_tensorboard'] = config.getboolean('Model', 'enable_tensorboard')
    
 
    initial_weight = config.getfloat('Model', 'initial_weight')
    params['initial_weight'] = initial_weight

    dataset = config['Training']['dataset']
    params['dataset'] = dataset
    train_file_path = config['Training']['train_file_path']
    neg_train_file_path = config['Training']['neg_train_file_path']
    gpu = config['Training']['gpu']
    params['gpu'] = gpu
    device = torch.device('cuda:'+ params['gpu'] if torch.cuda.is_available() else "cpu")
    params['device'] = device
    lr = config.getfloat('Training', 'learning_rate')
    params['lr'] = lr
    batch_size = config.getint('Training', 'batch_size')
    params['batch_size'] = batch_size
    early_stop_epoch = config.getint('Training', 'early_stop_epoch')
    params['early_stop_epoch'] = early_stop_epoch
    w1 = config.getfloat('Training', 'w1')
    w2 = config.getfloat('Training', 'w2')
    w3 = config.getfloat('Training', 'w3')
    w4 = config.getfloat('Training', 'w4')
    params['w1'] = w1
    params['w2'] = w2
    params['w3'] = w3
    params['w4'] = w4
    negative_num = config.getint('Training', 'negative_num')
    negative_weight = config.getfloat('Training', 'negative_weight')
    params['negative_num'] = negative_num
    params['negative_weight'] = negative_weight


    gamma = config.getfloat('Training', 'gamma')
    params['gamma'] = gamma
    lambda_ = config.getfloat('Training', 'lambda')
    params['lambda'] = lambda_
    sampling_sift_pos = config.getboolean('Training', 'sampling_sift_pos')
    params['sampling_sift_pos'] = sampling_sift_pos
    
    

    test_batch_size = config.getint('Testing', 'test_batch_size')
    params['test_batch_size'] = test_batch_size
    topk = config.getint('Testing', 'topk') 
    params['topk'] = topk

    test_file_path = config['Testing']['test_file_path']

    # dataset processing
    train_data, neg_train_data, test_data, train_mat, user_num, item_num, constraint_mat = load_data(train_file_path, neg_train_file_path, test_file_path)
    train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle = True, num_workers=5)
    # neg_train_loader = data.DataLoader(neg_train_data, batch_size=batch_size, shuffle = True, num_workers=5)
    test_loader = data.DataLoader(list(range(user_num)), batch_size=test_batch_size, shuffle=False, num_workers=5)
    validation_loader = data.DataLoader(list(range(user_num)), batch_size=test_batch_size, shuffle=False, num_workers=5)
    # test_loader = data.DataLoader(test_data, batch_size=test_batch_size, shuffle=False, num_workers=5)

    params['user_num'] = user_num
    params['item_num'] = item_num


    # mask matrix for testing to accelarate testing speed
    mask = torch.zeros(user_num, item_num)
    # negative로 변환
    interacted_items = [[] for _ in range(user_num)]
    for (u, i) in train_data:
        mask[u][i] = -np.inf
        interacted_items[u].append(i)
    neg_interacted_items = [[] for _ in range(user_num)]
    for (u, i) in neg_train_data:
        neg_interacted_items[u].append(i)

    # test user-item interaction, which is ground truth
    test_ground_truth_list = [[] for _ in range(user_num)]
    for (u, i) in test_data:
        test_ground_truth_list[u].append(i)

    
    # Compute \Omega to extend UltraGCN to the item-item occurrence graph
    ii_cons_mat_path = './' + dataset + '_ii_constraint_mat'
    ii_neigh_mat_path = './' + dataset + '_ii_neighbor_mat'
    
    if os.path.exists(ii_cons_mat_path):
        ii_constraint_mat = pload(ii_cons_mat_path)
        ii_neighbor_mat = pload(ii_neigh_mat_path)
    else:
        ii_neighbor_mat, ii_constraint_mat = get_ii_constraint_mat(train_mat, ii_neighbor_num)
        pstore(ii_neighbor_mat, ii_neigh_mat_path)
        pstore(ii_constraint_mat, ii_cons_mat_path)

    return params, constraint_mat, ii_constraint_mat, ii_neighbor_mat, train_loader, test_loader,validation_loader, mask, test_ground_truth_list, interacted_items, neg_interacted_items




def get_ii_constraint_mat(train_mat, num_neighbors, ii_diagonal_zero = False):
    
    print('Computing \\Omega for the item-item graph... ')
    A = train_mat.T.dot(train_mat)	# I * I
    n_items = A.shape[0]
    res_mat = torch.zeros((n_items, num_neighbors))
    res_sim_mat = torch.zeros((n_items, num_neighbors))
    # diagonal을 0으로
    if ii_diagonal_zero:
        A[range(n_items), range(n_items)] = 0
    items_D = np.sum(A, axis = 0).reshape(-1)
    users_D = np.sum(A, axis = 1).reshape(-1)

    beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
    beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)
    all_ii_constraint_mat = torch.from_numpy(beta_uD.dot(beta_iD))
    for i in range(n_items):
        row = all_ii_constraint_mat[i] * torch.from_numpy(A.getrow(i).toarray()[0])
        row_sims, row_idxs = torch.topk(row, num_neighbors)
        res_mat[i] = row_idxs
        res_sim_mat[i] = row_sims
        if i % 15000 == 0:
            print('i-i constraint matrix {} ok'.format(i))

    print('Computation \\Omega OK!')
    return res_mat.long(), res_sim_mat.float()

    

def load_data(train_file, neg_train_file, test_file):
    trainUniqueUsers, trainItem, trainUser = [], [], []
    neg_trainUniqueUsers, neg_trainItem, neg_trainUser = [], [], []
    testUniqueUsers, testItem, testUser = [], [], []
    n_user, m_item = 0, 0
    trainDataSize, neg_DataSize,testDataSize = 0, 0, 0
    with open(train_file, 'r') as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                items = [int(i) for i in l[1:]]
                uid = int(l[0])
                trainUniqueUsers.append(uid)
                trainUser.extend([uid] * len(items))
                trainItem.extend(items)
                m_item = max(m_item, max(items))
                n_user = max(n_user, uid)
                trainDataSize += len(items)
    trainUniqueUsers = np.array(trainUniqueUsers)
    trainUser = np.array(trainUser)
    trainItem = np.array(trainItem)
    # negsample
    with open(neg_train_file, 'r') as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                items = [int(i) for i in l[1:]]
                uid = int(l[0])
                neg_trainUniqueUsers.append(uid)
                neg_trainUser.extend([uid] * len(items))
                neg_trainItem.extend(items)
                neg_DataSize += len(items)
    neg_trainUniqueUsers = np.array(neg_trainUniqueUsers)
    neg_trainUser = np.array(neg_trainUser)
    neg_trainItem = np.array(neg_trainItem)

    with open(test_file) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                try:
                    items = [int(i) for i in l[1:]]
                except:
                    items = []
                uid = int(l[0])
                testUniqueUsers.append(uid)
                testUser.extend([uid] * len(items))
                testItem.extend(items)
                try:
                    m_item = max(m_item, max(items))
                except:
                    m_item = m_item
                n_user = max(n_user, uid)
                testDataSize += len(items)


    train_data = []
    neg_train_data = []
    test_data = []

    n_user += 1
    m_item += 1

    for i in range(len(trainUser)):
        train_data.append([trainUser[i], trainItem[i]])
    # neg
    for i in range(len(neg_trainUser)):
        neg_train_data.append([neg_trainUser[i], neg_trainItem[i]])
    for i in range(len(testUser)):
        test_data.append([testUser[i], testItem[i]])
    train_mat = sp.dok_matrix((n_user, m_item), dtype=np.float32)
    # neg
    neg_train_mat = sp.dok_matrix((n_user, m_item), dtype=np.float32)

    for x in train_data:
        train_mat[x[0], x[1]] = 1.0
    # neg
    for x in neg_train_data:
        neg_train_mat[x[0], x[1]] = 1.0


    # construct degree matrix for graphmf

    items_D = np.sum(train_mat, axis = 0).reshape(-1)
    users_D = np.sum(train_mat, axis = 1).reshape(-1)

    beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
    beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)

    constraint_mat = {"beta_uD": torch.from_numpy(beta_uD).reshape(-1),
                      "beta_iD": torch.from_numpy(beta_iD).reshape(-1)}
    print("n_user is {}".format(n_user))
    print("n_item is {}".format(m_item))
    return train_data, neg_train_data, test_data, train_mat, n_user, m_item, constraint_mat



'''
Useful functions
'''

def pload(path):
	with open(path, 'rb') as f:
		res = pickle.load(f)
	print('load path = {} object'.format(path))
	return res

def pstore(x, path):
	with open(path, 'wb') as f:
		pickle.dump(x, f)
	print('store object in path = {} ok'.format(path))


def Sampling(pos_train_data, item_num, neg_ratio, interacted_items, neg_interacted_items):
    neg_candidates = np.arange(item_num)
    neg_items = []
    for u in pos_train_data[0]:
        if neg_interacted_items[u]==[]:
            probs=np.ones(item_num)
            probs[interacted_items[u]] = 1
        else:
            probs = np.zeros(item_num)
            probs[neg_interacted_items[u]] = 1
        probs /= np.sum(probs)           
        u_neg_items = np.random.choice(neg_candidates, size = neg_ratio, p = probs, replace = True).reshape(1, -1)

        neg_items.append(u_neg_items)

    neg_items = np.concatenate(neg_items, axis = 0) 
    neg_items = torch.from_numpy(neg_items)
    return pos_train_data[0], pos_train_data[1], neg_items	# users, pos_items, neg_items



'''
Model Definition
'''

class UltraGCN(nn.Module):
    def __init__(self, params, constraint_mat, ii_constraint_mat, ii_neighbor_mat):
        super(UltraGCN, self).__init__()
        self.user_num = params['user_num']
        self.item_num = params['item_num']
        self.embedding_dim = params['embedding_dim']
        self.w1 = params['w1']
        self.w2 = params['w2']
        self.w3 = params['w3']
        self.w4 = params['w4']

        self.negative_weight = params['negative_weight']
        self.gamma = params['gamma']
        self.lambda_ = params['lambda']

        self.user_embeds = nn.Embedding(self.user_num, self.embedding_dim)
        self.item_embeds = nn.Embedding(self.item_num, self.embedding_dim)

        self.constraint_mat = constraint_mat
        self.ii_constraint_mat = ii_constraint_mat
        self.ii_neighbor_mat = ii_neighbor_mat

        self.initial_weight = params['initial_weight']


        self.initial_weights()

    def initial_weights(self):
        nn.init.normal_(self.user_embeds.weight, std=self.initial_weight)
        nn.init.normal_(self.item_embeds.weight, std=self.initial_weight)


    def get_omegas(self, users, pos_items, neg_items):
        device = self.get_device()
        if self.w2 > 0:
            pos_weight = torch.mul(self.constraint_mat['beta_uD'][users], self.constraint_mat['beta_iD'][pos_items]).to(device)
            pow_weight = self.w1 + self.w2 * pos_weight
        else:
            pos_weight = self.w1 * torch.ones(len(pos_items)).to(device)
        
        # users = (users * self.item_num).unsqueeze(0)
        
        if self.w4 > 0:
            neg_weight = torch.mul(torch.repeat_interleave(self.constraint_mat['beta_uD'][users], neg_items.size(1)), self.constraint_mat['beta_iD'][neg_items.flatten()]).to(device)
            neg_weight = self.w3 + self.w4 * neg_weight
        else:
            neg_weight = self.w3 * torch.ones(neg_items.size(0) * neg_items.size(1)).to(device)


        weight = torch.cat((pow_weight, neg_weight))
        return weight



    def cal_loss_L(self, users, pos_items, neg_items, omega_weight):
        device = self.get_device()
        user_embeds = self.user_embeds(users)
        pos_embeds = self.item_embeds(pos_items)
        neg_embeds = self.item_embeds(neg_items)
      
        pos_scores = (user_embeds * pos_embeds).sum(dim=-1) # batch_size
        user_embeds = user_embeds.unsqueeze(1)
        neg_scores = (user_embeds * neg_embeds).sum(dim=-1) # batch_size * negative_num

        neg_labels = torch.zeros(neg_scores.size()).to(device)
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, neg_labels, weight = omega_weight[len(pos_scores):].view(neg_scores.size()), reduction='none').mean(dim = -1)
        
        pos_labels = torch.ones(pos_scores.size()).to(device)
        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, pos_labels, weight = omega_weight[:len(pos_scores)], reduction='none')

        loss = pos_loss + neg_loss * self.negative_weight
      
        return loss.sum()



    def cal_loss_I(self, users, pos_items):
        device = self.get_device()
        neighbor_embeds = self.item_embeds(self.ii_neighbor_mat[pos_items].to(device))    # len(pos_items) * num_neighbors * dim
        sim_scores = self.ii_constraint_mat[pos_items].to(device)     # len(pos_items) * num_neighbors
        user_embeds = self.user_embeds(users).unsqueeze(1)
        
        loss = -sim_scores * (user_embeds * neighbor_embeds).sum(dim=-1).sigmoid().log()
      
        # loss = loss.sum(-1)
        return loss.sum()

    def norm_loss(self):
        loss = 0.0
        for parameter in self.parameters():
            loss += torch.sum(parameter ** 2)
        return loss / 2


    def forward(self, users, pos_items, neg_items):
        omega_weight = self.get_omegas(users, pos_items, neg_items)
        
        loss = self.cal_loss_L(users, pos_items, neg_items, omega_weight)
        loss += self.gamma * self.norm_loss()
        loss += self.lambda_ * self.cal_loss_I(users, pos_items)
        return loss


    def test_foward(self, users):
        items = torch.arange(self.item_num).to(users.device)
        user_embeds = self.user_embeds(users)
        item_embeds = self.item_embeds(items)
         
        return user_embeds.mm(item_embeds.t())


    def get_device(self):
        return self.user_embeds.weight.device



'''
Train
'''
########################### TRAINING #####################################
def train(model, optimizer, train_loader, test_loader,validation_loader, mask, test_ground_truth_list, interacted_items,neg_interacted_items, user_mem, item_mem, val_item_mem, answercode, params): 
    device = params['device']
    best_epoch, best_recall, best_ndcg, best_acc, best_auc = 0, 0, 0, 0, 0
    early_stop_count = 0
    early_stop = False

    batches = len(train_loader.dataset) // params['batch_size']
    if len(train_loader.dataset) % params['batch_size'] != 0:
        batches += 1
    print('Total training batches = {}'.format(batches))
    
    if params['enable_tensorboard']:
        writer = SummaryWriter()
    

    for epoch in range(params['max_epoch']):
        model.train() 
        start_time = time.time()

        for batch, x in enumerate(train_loader): # x: tensor:[users, pos_items]
            users, pos_items, neg_items = Sampling(x, params['item_num'], params['negative_num'], interacted_items, neg_interacted_items)
            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)
            model.zero_grad()
            loss = model(users, pos_items, neg_items)
            if params['enable_tensorboard']:
                writer.add_scalar("Loss/train_batch", loss, batches * epoch + batch)
            loss.backward()
            optimizer.step()
        print("train_{} is completed".format(epoch))
        
        train_time = time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time))
        if params['enable_tensorboard']:
            writer.add_scalar("Loss/train_epoch", loss, epoch)
        # validation_loader, val_item_mem, answercode 만들기
        need_val = True
        if epoch % 5 != 0:
            need_val = False
        # if need_val:
        # if True:
        #     start_time = time.time()
        #     acc, auc, auc2 = validation(model, validation_loader, user_mem, val_item_mem, answercode, epoch)
        #     if params['enable_tensorboard']:
        #         writer.add_scalar('Results/acc', acc, epoch)
        #         writer.add_scalar('Results/auc', auc, epoch)
        #     test_time = time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time))
            
        #     print('The time for epoch {} is: train time = {}, test time = {}'.format(epoch, train_time, test_time))
        #     print("Loss = {:.5f}, acc: {:5f} \t auc: {:.5f}, auc2: {:.5f}".format(loss.item(), acc, auc, auc2))

        #     if acc > best_acc:
        #         best_epoch, best_acc, best_auc = acc, auc, epoch
        #         early_stop_count = 0   

        #     else:
        #         early_stop_count += 1
        #         if early_stop_count == params['early_stop_epoch']:
        #             early_stop = True
        #inference 항상 만들기
        inference(model, test_loader,user_mem,item_mem,epoch)
        torch.save(model.state_dict(), params['model_save_path']+'256'+str(epoch))        
        # if epoch % 5 == 0 or epoch<=1:
        #     inference(model, test_loader,user_mem,item_mem,epoch)
        #     torch.save(model.state_dict(), params['model_save_path']+'256'+str(epoch))
        # if early_stop:
        #     print('##########################################')
        #     print('Early stop is triggered at {} epochs.'.format(epoch))
        #     print('Results:')
        #     print('best epoch = {}, best acc = {}, best auc = {}'.format(best_epoch, best_acc, best_auc))
        #     break

    writer.flush()

    print('Training end!')


# The below 7 functions (hit, ndcg, RecallPrecision_ATk, MRRatK_r, NDCGatK_r, test_one_batch, getLabel) follow this license.
# MIT License

# Copyright (c) 2020 Xiang Wang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
########################### TESTING #####################################
'''
Test and metrics
'''

def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	return 0


def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return np.reciprocal(np.log2(index+2))
	return 0


def RecallPrecision_ATk(test_data, r, k):
	"""
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
	right_pred = r[:, :k].sum(1)
	precis_n = k
	
	recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
	recall_n = np.where(recall_n != 0, recall_n, 1)
	recall = np.sum(right_pred / recall_n)
	precis = np.sum(right_pred) / precis_n
	return {'recall': recall, 'precision': precis}


def MRRatK_r(r, k):
	"""
    Mean Reciprocal Rank
    """
	pred_data = r[:, :k]
	scores = np.log2(1. / np.arange(1, k + 1))
	pred_data = pred_data / scores
	pred_data = pred_data.sum(1)
	return np.sum(pred_data)


def NDCGatK_r(test_data, r, k):
	"""
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
	assert len(r) == len(test_data)
	pred_data = r[:, :k]

	test_matrix = np.zeros((len(pred_data), k))
	for i, items in enumerate(test_data):
		length = k if k <= len(items) else len(items)
		test_matrix[i, :length] = 1
	max_r = test_matrix
	idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
	dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
	dcg = np.sum(dcg, axis=1)
	idcg[idcg == 0.] = 1.
	ndcg = dcg / idcg
	ndcg[np.isnan(ndcg)] = 0.
	return np.sum(ndcg)



def test_one_batch(X, k):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = getLabel(groundTrue, sorted_items)
    ret = RecallPrecision_ATk(groundTrue, r, k)
    return ret['precision'], ret['recall'], NDCGatK_r(groundTrue,r,k)

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

def acc_fun(rating_list, answercode):
    for i in range(len(rating_list)):
        if rating_list[i]>0.5:
            rating_list[i]=1
        else:
            rating_list[i]=0
    den=0
    nom=0
    for i in range(len(rating_list)):
        nom+=1
        if rating_list[i]==answercode[i]:
            den+=1
    acc=den/nom
    return acc


def validation(model, validation_loader, user_mem, val_item_mem,answercode, epoch):
    rating_list = []
    with torch.no_grad():
        model.eval()
        r_user=0
        for idx, batch_users in enumerate(validation_loader):
            batch_users = batch_users.to(model.get_device())
            rating = model.test_foward(batch_users) 
            rating = rating.cpu()
            for k in user_mem:
                rating_K= float(rating[k][int(val_item_mem[r_user])])
                rating_list.append(rating_K)
                r_user+=1
    maxnum=max(rating_list)
    minnum=min(rating_list)
    for i in range(len(rating_list)):
        rating_list[i]=1/(maxnum-minnum+1/1000000000)*(rating_list[i]-minnum)
    np_answercode=np.array(answercode)
    np_rating_list=np.array(rating_list)
    acc=acc_fun(rating_list, answercode)
    auc=roc_auc_score(np_answercode, np_rating_list)
    auc2=acc_fun(rating_list, answercode)
    print("{}th validation is done".format(epoch))
    return acc, auc, auc2

def aucfun(answercode, rating_list):
    index = [i for i in range(1, len(answercode)+1)]
    inp1 = pd.DataFrame({'index' : index, 'label' : answercode, 'probability' : rating_list})

    FPR = []
    TPR = []
    P = len(inp1[inp1['label'] == 1])
    N = len(inp1[inp1['label'] == 0])
    # for i in inp1['probability']:
    setp=set(rating_list)
    listp=list(setp)
    listp.sort(reverse=True)
    for i in listp:
        tmp_p = data[data['rating_list'] >= i]
        TP = len(tmp_p[tmp_p['label'] == 1])
        tmp_TPR = TP/P
        tmp_n = data[data['rating_list'] >= i]
        FP = len(tmp_n[tmp_n['label'] == 0])
        tmp_FPR = FP/N
        TPR.append(tmp_TPR)
        FPR.append(tmp_FPR)
    AUC_TPR = [0] + TPR
    AUC_FPR = [0] + FPR
    AUC = 0
    for i in range(1, len(AUC_TPR)):
        tmp_AUC = (AUC_TPR[i - 1] + AUC_TPR[i]) * (AUC_FPR[i] - AUC_FPR[i - 1]) / 2
        AUC += tmp_AUC
    return AUC

def inference(model, test_loader,user_mem, item_mem, epoch):

    rating_list = []
    with torch.no_grad():
        model.eval()
        r_user=0
        for idx, batch_users in enumerate(test_loader):
            batch_users = batch_users.to(model.get_device())
            rating = model.test_foward(batch_users) 
            rating = rating.cpu()
            for k in user_mem:
                rating_K= float(rating[k][int(item_mem[r_user])])
                rating_list.append(rating_K)
                r_user+=1
    maxnum=max(rating_list)
    minnum=min(rating_list)
    # for i in range(len(rating_list)):
    #     rating_list[i]=[1/(maxnum-minnum+1/1000000000)*(rating_list[i]-minnum)] 
    list_df = pd.DataFrame(rating_list, columns=['prediction'])
    list_df.to_csv('inference_withvalid_1024_{}_lambda10.csv'.format(epoch))        
    print("<making inference_withvalid_{}.csv> is done".format(epoch))           
    return rating_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str,default='dkt_config.ini', help='config file path')
    args = parser.parse_args()

    print('###################### UltraGCN ######################')
    PATH='data/dkt_withvalid'
    item_mem = pd.read_csv(os.path.join(PATH,'testilist.csv'))
    item_mem_temp=item_mem.values.tolist()
    user_mem = pd.read_csv(os.path.join(PATH,'userilist.csv'))
    user_mem_temp=user_mem.values.tolist()
    user_mem=[]
    item_mem=[]
    for i in range(len(user_mem_temp)):
        temp=int(user_mem_temp[i][0])
        user_mem.append(temp)
    for i in range(len(item_mem_temp)):
        temp=int(item_mem_temp[i][0])
        item_mem.append(temp)
    val_item_mem=[]
    answercode=[]
    vallist = pd.read_csv(os.path.join(PATH,'vallist.csv'))
    vallist_temp=vallist.values.tolist()
    for i in range(len(vallist_temp)):
        temp=int(vallist_temp[i][0])
        temp2=int(vallist_temp[i][1])
        val_item_mem.append(temp)
        answercode.append(temp2)

    print('1. Loading Configuration...')
    params, constraint_mat, ii_constraint_mat, ii_neighbor_mat, train_loader, test_loader,validation_loader, mask, test_ground_truth_list, interacted_items,neg_interacted_items = data_param_prepare(args.config_file)
    print('Load Configuration OK, show them below')
    print('Configuration:')
    print(params)

    ultragcn = UltraGCN(params, constraint_mat, ii_constraint_mat, ii_neighbor_mat)
    ultragcn = ultragcn.to(params['device'])
    optimizer = torch.optim.Adam(ultragcn.parameters(), lr=params['lr'])
    train(ultragcn, optimizer, train_loader, test_loader,validation_loader, mask, test_ground_truth_list, interacted_items, neg_interacted_items, user_mem, item_mem, val_item_mem, answercode, params)

    print('END')



