import argparse


def parse_args(mode="train"):
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int, help="seed")

    parser.add_argument("--device", default="cpu", type=str, help="cpu or gpu")

    parser.add_argument(
        "--data_dir",
        default="../../data/",
        type=str,
        help="data directory",
    )
    parser.add_argument(
        "--asset_dir", default="asset/", type=str, help="data directory"
    )

    parser.add_argument(
        "--file_name", default="train_data.csv", type=str, help="train file name"
    )

    parser.add_argument(
        "--model_dir", default="models/", type=str, help="model directory"
    )
    parser.add_argument(
        "--model_name", default="model.pt", type=str, help="model file name"
    )

    parser.add_argument(
        "--output_dir", default="output/", type=str, help="output directory"
    )
    parser.add_argument(
        "--test_file_name", default="test_data.csv", type=str, help="test file name"
    )

    parser.add_argument(
        "--max_seq_len", default=20, type=int, help="max sequence length"
    )
    parser.add_argument("--num_workers", default=1, type=int, help="number of workers")

    # 모델
    parser.add_argument(
        "--hidden_dim", default=64, type=int, help="hidden dimension size"
    )
    parser.add_argument("--n_layers", default=2, type=int, help="number of layers")
    parser.add_argument("--n_heads", default=2, type=int, help="number of heads")
    parser.add_argument("--drop_out", default=0.2, type=float, help="drop out rate")

    # 훈련
    parser.add_argument("--n_epochs", default=20, type=int, help="number of epochs")
    parser.add_argument("--batch_size", default=64, type=int, help="batch size")
    parser.add_argument("--lr", default=0.0001, type=float, help="learning rate")
    parser.add_argument("--clip_grad", default=10, type=int, help="clip grad")
    parser.add_argument("--patience", default=5, type=int, help="for early stopping")

    parser.add_argument(
        "--log_steps", default=50, type=int, help="print log per n steps"
    )

    ### 중요 ###
    parser.add_argument("--model", default="lstm", type=str, help="model type")
    parser.add_argument("--optimizer", default="adam", type=str, help="optimizer type")
    parser.add_argument(
        "--scheduler", default="plateau", type=str, help="scheduler type"
    )


    #### LGBM argument
    parser.add_argument("--boosting", default='gbdt', type=str, help="default: gbdt(gradient boosting decision tree")
    parser.add_argument("--max_dep", default=12, type=int, help="handle overfitting, lowering will do, 3~12 recommended")
    parser.add_argument("--num_leaves", default=512, type=int, help="default: 31")
    parser.add_argument("--mdil", default=200, type=int, help="handle overfitting, minimum number of records a leaf may have")
    parser.add_argument("--ff", default=0.8, type=float, help="randomly choose fraction of parameters when building tree in each iteration")
    parser.add_argument("--bf", default=0.8, type=float, help="use fraction of data for each iteration, speed up and avoid overfitting")
    parser.add_argument("--lmda", default=0.2, type=float, help="specifies regularization")
    parser.add_argument("--mgts", default=20, type=int, help="describe the minimum gain to make a split, used to control number of useful splits in tree")
    parser.add_argument("--mcg", default=64, type=int, help="When the number of category is large, finding the split point on it is easily over-fitting")
    parser.add_argument("--tl", default='feature', type=str, help="default: serial, or data or feature")
    parser.add_argument("--split_ratio", default=0.7, type=float, help="train/test split ratio")
    args = parser.parse_args()

    return args

