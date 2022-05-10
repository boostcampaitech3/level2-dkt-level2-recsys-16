import torch

# model = torch.load('../../lightgcn/lightgcn_recbole/saved/lgcn-emb-64.pth')

# print(model['state_dict'])
#
# print(model['state_dict']['user_embedding.weight'])
#
# print(len(model['state_dict']['user_embedding.weight']))
#
# print(len(model['state_dict']['item_embedding.weight'][0])) # 85

# k = torch.nn.Parameter(model['state_dict']['item_embedding.weight'])


# print(k)
# print(len(k))

print(torch.randn(20, 5, 10)) # batch, sentence_length, embedding_dim