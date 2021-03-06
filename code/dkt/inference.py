import os

import torch
from args import parse_args
from dkt import trainer
from dkt.dataloader import Preprocess
from dkt.model import LGBM

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    if args.model == 'lgbm':
        model = LGBM(args)
        print('model', type(model))
        model.infer()
    else:
        preprocess = Preprocess(args)
        preprocess.load_test_data(args.test_file_name)
        test_data = preprocess.get_test_data()

        trainer.inference(args, test_data)


if __name__ == "__main__":
    args = parse_args(mode="train")
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
