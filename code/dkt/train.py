import os
import sys

import torch
import wandb
from args import parse_args
from dkt import trainer
from dkt.dataloader import Preprocess
from dkt.utils import setSeeds
from dkt.model import LGBM

def main(args):
    wandb.login()

    setSeeds(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    args.device = device

    if args.model == 'lgbm':
        wandb.init(project="dkt-sweep2", config=vars(args))
        model = LGBM(args)
        print('model', type(model))
        model.train()
    else:
        preprocess = Preprocess(args)
        preprocess.load_train_data(args.file_name)
        train_data = preprocess.get_train_data()
        train_data, valid_data = preprocess.split_data(train_data)
        wandb.init(project="dkt-sweep2", config=vars(args))
        trainer.run(args, train_data, valid_data)




if __name__ == "__main__":
    args = parse_args(mode="train")
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
