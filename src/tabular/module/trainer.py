import math
import os
import wandb

import numpy as np
import wandb

from .dataloader import Preprocess, xy_data_split
from .metric import get_metric
from .model import LightGBMModel
from .utils import get_logger, logging_conf, get_expname

logger = get_logger(logger_conf=logging_conf)

def run(args, w_config):
    exp_name = get_expname(args)
    wandb.run.name = exp_name
    wandb.run.save()
    
    logger.info("Preparing Train data ...")
    preprocess = Preprocess(args)
    preprocess.load_train_data(file_name=args.file_name)
    train_data: np.ndarray = preprocess.get_train_data()
    train_data, valid_data = preprocess.split_data(df=train_data)
    
    x_train, y_train = xy_data_split(train_data)
    x_valid, y_valid = xy_data_split(valid_data)
    
    logger.info("Building Model ...")
    if args.model_name == 'lgbm':
        model = LightGBMModel(config=w_config)


    # TRAIN
    logger.info("Start Training ...")
    model.fit(x_train, y_train, x_valid, y_valid)
    
    # VALID
    preds = model.predict(x_valid)
    acc, auc = get_metric(y_valid, preds)
    logger.info("TRAIN AUC : %.4f ACC : %.4f", auc, acc)
    
    # WandB Logging
    wandb.log(dict(valid_acc=acc,
                   valid_auc=auc))
    
    # INFERENCE
    logger.info("Preparing Test data ...")
    preprocess.load_test_data(file_name=args.test_file_name)
    test_data = preprocess.get_test_data()
    inference(args=args, test_data=test_data, model=model, exp_name=exp_name)
    

def inference(args, test_data, model, exp_name):
    
    total_preds = model.predict(test_data)
    
    write_path = os.path.join(args.output_dir, f"{exp_name}_submission.csv")
    os.makedirs(name=args.output_dir, exist_ok=True)
    with open(write_path, "w", encoding="utf8") as w:
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            w.write("{},{}\n".format(id, p))
    logger.info("Successfully saved submission as %s", write_path)
    
    # WandB Artifact Logging
    submission_artifact = wandb.Artifact('submission', type='output')
    submission_artifact.add_file(local_path=write_path)
    wandb.log_artifact(submission_artifact)