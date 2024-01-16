import os

import numpy as np
import torch
import wandb

from module import trainer
from module.args import parse_args
from module.dataloader import Preprocess
from module.utils import get_logger, set_seeds, logging_conf, get_expname
from model_configs.default_config import lightGBMParams
import json

args = parse_args()

# JSON 파일에서 sweep 설정 읽어오기
if args.model == 'lgbm':
    with open('./sweep/lgbm_sweep.json', 'r') as file:
        sweep_config = json.load(file)
        default_params = lightGBMParams


def main():
    os.makedirs(args.model_dir, exist_ok=True)
    set_seeds(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    wandb.init(entity='raise_level2', project="dkt", config=default_params)
    trainer.run(args=args, w_config=wandb.config)
    
    wandb.finish()


if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep=sweep_config, project='dkt', entity='raise_level2')
    wandb.agent(sweep_id, function=main, count=2)

