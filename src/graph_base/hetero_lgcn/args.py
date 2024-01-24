import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument("--use_cuda_if_available", default=True, type=bool, help="Use GPU")
    
    parser.add_argument("--data_dir", default="/home/minseo/Naver_Ai/data", type=str, help="")
    
    parser.add_argument("--output_dir", default="./outputs__tt--like_tt/", type=str, help="")
    
    parser.add_argument("--hidden_dim", default=64, type=int, help="")
    parser.add_argument("--n_layers", default=3, type=int, help="")
    parser.add_argument("--alpha", default=None, type=float, help="")
    
    parser.add_argument("--n_epochs", default=70, type=int, help="")
    parser.add_argument("--lr", default=0.1, type=float, help="")
    parser.add_argument("--model_dir", default="./models/", type=str, help="")
    parser.add_argument("--model_name", default="best_model.pt", type=str, help="")
    parser.add_argument("--model",default="lightgcn",type=str,help="")
    parser.add_argument("--patience",default="1000",type=int,help="")
    parser.add_argument("--agreegation_method",default="1",type=int,help="") # sum, attn
    args = parser.parse_args()

    return args
