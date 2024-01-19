import os

import torch

from Sgcn.args import parse_args
from Sgcn.datasets import prepare_dataset
from Sgcn import trainer
from Sgcn.utils import get_logger, logging_conf, set_seeds


logger = get_logger(logging_conf)


def main(args):
    set_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Preparing data ...")
    train_data, val_data, test_data, n_node = prepare_dataset(device=device, data_dir=args.data_dir)

    logger.info("Loading Model ...")
    weight: str = os.path.join(args.model_dir, args.model_name)
    model: torch.nn.Module = trainer.build(
        n_node = n_node,
        in_channels =  args.input_dim,
        hidden_channels = args.hidden_dim,
        num_layers = args.n_layers,
        lamb = args.lamb,
        bias = args.bias
    )
    model = model.to(device)
    
    logger.info("Make Predictions & Save Submission ...")
    trainer.inference(model=model, test_data=val_data, train_data=train_data, output_dir=args.output_dir, n_nodes=n_node)

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(name=args.model_dir, exist_ok=True)
    main(args=args)
