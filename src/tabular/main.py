import os

import numpy as np
import torch
import wandb

from module import trainer
from module.args import parse_args
from module.dataloader import Preprocess
from module.utils import get_logger, set_seeds, logging_conf, get_expname

import yaml

# Load sweep configuration from YAML file
# sweep_config = wandb.sweep(yaml.safe_load(open('sweep.yaml', 'r')))

def main(args, model_params):
    set_seeds(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    
    wandb.init(project="dkt", config=vars(args))
    exp_name = get_expname(args)
    wandb.run.name = exp_name
    wandb.run.save()
    
    trainer.run(args=args, exp_name=exp_name, model_params=model_params)
    
    wandb.finish()


if __name__ == "__main__":
    args, model_parmas = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    main(args, model_parmas)