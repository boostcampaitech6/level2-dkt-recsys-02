import os
import argparse

import torch
import wandb

from Sgcn.args import parse_args
from Sgcn.datasets import prepare_dataset
from Sgcn import trainer
from Sgcn.utils import get_logger, set_seeds, logging_conf, get_expname


logger = get_logger(logging_conf)


def main(args: argparse.Namespace):
    wandb.login()
    wandb.init(project="dkt", config=vars(args))
    set_seeds(args.seed)
    
    use_cuda: bool = torch.cuda.is_available() and args.use_cuda_if_available
    device = torch.device("cuda" if use_cuda else "cpu")

    logger.info("Preparing data ...")
    logger.info(device)
    train_data, valid_data, test_data, n_node = prepare_dataset(device=device, data_dir=args.data_dir, return_origin_train=True)
    wandb.run.name = get_expname(args)
    wandb.run.save()
    logger.info("Building Model ...")
    model = trainer.build(
        n_node = n_node,
        in_channels =  args.input_dim,
        hidden_channels = args.hidden_dim,
        num_layers = args.n_layers,
        lamb = args.lamb,
        bias = args.bias
    )
    model = model.to(device)
    
    logger.info("Start Training ...")
    trainer.run(
        model=model,
        train_data=train_data,
        num_nodes=n_node,
        valid_data=valid_data,
        test_data=test_data,
        n_epochs=args.n_epochs,
        learning_rate=args.lr,
        model_dir=args.model_dir,
        patience = args.patience
    )
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(name=args.model_dir, exist_ok=True)
    main(args=args)
