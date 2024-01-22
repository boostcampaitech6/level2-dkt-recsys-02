import os

import numpy as np
import torch
import wandb

from module import trainer
from module.args import parse_args
from module.dataloader import Preprocess
from module.utils import get_logger, set_seeds, logging_conf, get_expname
from model_configs.default_config import lightGBMParams, tabNetParams, xgboostParams, catBoostParams
import json

args = parse_args()

# JSON 파일에서 sweep 설정 읽어오기
try:
    if args.model == 'lgbm':
        default_params = lightGBMParams
        with open('./sweep/lgbm_sweep.json', 'r') as file:
            sweep_config = json.load(file)
    elif args.model == 'xgb':
        default_params = xgboostParams
        with open('./sweep/xgb_sweep.json', 'r') as file:
            sweep_config = json.load(file)
    elif args.model == 'catboost':
        default_params = catBoostParams
        with open('./sweep/catboost_sweep.json', 'r') as file:
            sweep_config = json.load(file)
    elif args.model == 'tabnet':
        default_params = tabNetParams
        with open('./sweep/tabnet_sweep.json', 'r') as file:
            sweep_config = json.load(file)
    else:
        raise ValueError("Invalid model type. Supported types are: lgbm, xgb, catboost, tabnet")

except FileNotFoundError:
    print(f"Error: JSON file not found for {args.model} model. Make sure the sweep configuration file exists.")
    exit(1)
except json.JSONDecodeError:
    print(f"Error: JSON decoding failed for {args.model} model. Check the format of the sweep configuration file.")
    exit(1)


def main():
    os.makedirs(args.model_dir, exist_ok=True)
    set_seeds(args.seed)
     
    wandb.init(entity='raise_level2', project="dkt", config=default_params)
    trainer.run(args=args, w_config=wandb.config)
    
    wandb.finish()


if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep=sweep_config, project='dkt', entity='raise_level2')
    wandb.agent(sweep_id, function=main, count=1)

