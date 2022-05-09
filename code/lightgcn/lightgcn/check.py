import torch


model = torch.load('../weight/best_model.pt')

print(len(model['model']['embedding.weight'][0]))
# print(model[1])
