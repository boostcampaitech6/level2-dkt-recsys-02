import os

import numpy as np
import torch
import wandb

from dkt import trainer
from dkt.args import parse_args
from dkt.dataloader import Preprocess
from dkt.utils import get_logger, set_seeds, logging_conf, get_expname


logger = get_logger(logging_conf)


def main(args):
    wandb.login()
    set_seeds(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info("Preparing Train data ...")
    preprocess = Preprocess(args)
    preprocess.load_train_data(file_name=args.file_name)
    train_data: np.ndarray = preprocess.get_train_data()
    train_data, valid_data = preprocess.split_data(data=train_data)
    
    wandb.init(project="dkt", config=vars(args))
    exp_name = get_expname(args)
    wandb.run.name = exp_name
    wandb.run.save()
    
    logger.info("Building Model ...")
    model: torch.nn.Module = trainer.get_model(args=args).to(args.device)
    
    logger.info("Start Training ...")
    trainer.run(args=args, train_data=train_data, valid_data=valid_data, model=model, exp_name=exp_name)
    
    logger.info("Preparing Test data ...")
    preprocess.load_test_data(file_name=args.test_file_name)
    test_data: np.ndarray = preprocess.get_test_data()
    trainer.inference(args=args, test_data=test_data, model=model)
    
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
