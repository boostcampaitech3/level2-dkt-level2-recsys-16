import os
import pandas as pd

import torch
import wandb
from args import parse_args
from dkt import trainer
from dkt.dataloader import Preprocess, get_loaders
from dkt.utils import setSeeds
from dkt.trainer import load_model, get_optimizer, get_scheduler, train


def df_to_tuple(r):
    global args
    return [r[x].values for x in args.FEAT_COLUMN] \
            + [r[x].values for x in args.CONT_FEAT_COLUMN] \
            + [r[x].values for x in args.ANSWER_COLUMN]

def main(args):
    wandb.login()

    setSeeds(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    args.device = device

    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)
    train_data = preprocess.get_train_data()

    train_data, valid_data = preprocess.split_data(train_data)

    wandb.init(project="dkt", config=vars(args))
    trainer.run(args, train_data, valid_data)

    if args.pseudo_label_file:
        args.n_epochs = args.pseudo_epochs
        preprocess = Preprocess(args)
        df_test = pd.read_csv(os.path.join(args.data_dir, args.test_file_name))
        df_pseudo = pd.read_csv(os.path.join(args.data_dir,args.pseudo_label_file))
        cnt=0
        tmp = df_pseudo['prediction'].map(lambda x: 1 if x>=0.5 else 0)
        labels = []
        for x in df_test['answerCode']:
            if x!=-1:
                labels.append(x)
            else:
                labels.append(tmp[cnt])
                cnt+=1
        df_test['answerCode'] = labels
        df_test.fillna(0,inplace=True)
        df_test.to_csv(os.path.join(args.data_dir,'pseudo.csv'), index=False)
        preprocess = Preprocess(args)
        preprocess.load_train_data('pseudo.csv')
        pseudo_data = preprocess.get_train_data()
        trainer.run_pesudo(args, pseudo_data, valid_data)
          
if __name__ == "__main__":
    args = parse_args(mode="train")
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
