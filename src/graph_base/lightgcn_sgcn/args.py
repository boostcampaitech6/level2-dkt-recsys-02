import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument("--use_cuda_if_available", default=True, type=bool, help="Use GPU")
    
    parser.add_argument("--data_dir", default="/home/minseo/Naver_Ai/data", type=str, help="")
    
    parser.add_argument("--output_dir", default="./outputs/", type=str, help="")
    
    
    parser.add_argument("--input_dim", default=75, type=int, help="")
    parser.add_argument("--hidden_dim", default=32, type=int, help="")
    parser.add_argument("--n_layers", default=3, type=int, help="")
    
    parser.add_argument("--n_epochs", default=65, type=int, help="")
    parser.add_argument("--lr", default=0.01, type=float, help="")
    parser.add_argument("--model_dir", default="./models/", type=str, help="")
    parser.add_argument("--model_name", default="best_model.pt", type=str, help="")
    parser.add_argument("--model",default="Sgcn",type=str,help="")
    parser.add_argument("--patience",default="10",type=int,help="")
    parser.add_argument("--lamb", default=5, type=int, help="")
    parser.add_argument("--bias", default=True, type=bool, help="")
    
    
    args = parser.parse_args()

    return args
