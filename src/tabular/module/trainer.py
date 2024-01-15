import math
import os

import numpy as np
import wandb

from .dataloader import Preprocess, xy_data_split
from .metric import get_metric
from .model import LightGBMModel
from .utils import get_logger, logging_conf, get_expname

logger = get_logger(logger_conf=logging_conf)

# 사용할 Feature 설정
FEATS = ['KnowledgeTag', 'user_correct_answer', 'user_total_answer',
         'user_acc', 'test_mean', 'test_sum', 'tag_mean','tag_sum']

LightGBMConfig={
    'learning_rate':0.1,
    'n_estimators':10000,
    'max_depth':10,
    'num_leaves': 200,
    'colsample_bytree':0.7,
    'objective':'binary',
    'metric':'rmse',
    'early_stopping_rounds':50,
    'verbosity':10
}

def run(args, exp_name):
    
    logger.info("Preparing Train data ...")
    preprocess = Preprocess(args)
    preprocess.load_train_data(file_name=args.file_name)
    train_data: np.ndarray = preprocess.get_train_data()
    train_data, valid_data = preprocess.split_data(df=train_data)
    
    x_train, y_train = xy_data_split(train_data)
    x_valid, y_valid = xy_data_split(valid_data)
    
    logger.info("Building Model ...")
    if args.model_name == 'lgbm':
        config = LightGBMConfig
        model = LightGBMModel(feats=FEATS, config=config)


    logger.info("Start Training ...")
    # TRAIN
    model.fit(x_train, y_train, x_valid, y_valid)
    
    # VALID
    preds = model.predict(x_valid)
    acc, auc = get_metric(y_valid, preds)
    logger.info("TRAIN AUC : %.4f ACC : %.4f", auc, acc)
    
    # WandB Artifact Logging
    # model_artifact = wandb.Artifact(f'{exp_name}', type='model')
    # model_artifact.add_file(local_path=f'{args.model_dir}{exp_name}.pt')
    # wandb.log_artifact(model_artifact)
    # joblib.dump(model, f'{args.model_dir}{exp_name}_{args.model_name}.pkl')
    
    # INFERENCE
    logger.info("Preparing Test data ...")
    preprocess.load_test_data(file_name=args.test_file_name)
    test_data = preprocess.get_test_data()
    inference(args=args, test_data=test_data, model=model, exp_name=exp_name)
    

def inference(args, test_data, model, exp_name):
    
    total_preds = model.predict(test_data[FEATS])
    
    write_path = os.path.join(args.output_dir, f"{exp_name}_submission.csv")
    os.makedirs(name=args.output_dir, exist_ok=True)
    with open(write_path, "w", encoding="utf8") as w:
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            w.write("{},{}\n".format(id, p))
    logger.info("Successfully saved submission as %s", write_path)
    
    # WandB Artifact Logging
    # submission_artifact = wandb.Artifact('submission', type='output')
    # submission_artifact.add_file(local_path=write_path)
    # wandb.log_artifact(submission_artifact)