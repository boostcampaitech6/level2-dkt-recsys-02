import math
import os
import wandb
import torch
import numpy as np
import wandb
from sklearn.model_selection import GroupKFold

from .dataloader import Preprocess, xy_data_split
from .metric import get_metric
from .model import LightGBMModel, XGBoostModel, CatBoostModel, TabNetModel
from .utils import get_logger, logging_conf, get_expname
from model_configs.default_config import FEATS, cat_cols

logger = get_logger(logger_conf=logging_conf)


def run(args, w_config):
    exp_name = get_expname(args)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    wandb.run.name = exp_name
    wandb.run.save()
    
    logger.info("Preparing Train data ...")
    preprocess = Preprocess(args)
    preprocess.load_train_data(file_name=args.file_name)    
    train_data: np.ndarray = preprocess.get_train_data()
    
    logger.info("Building Model ...")
    if args.model == 'lgbm':
        model = LightGBMModel(config=w_config)
    elif args.model == 'xgb':
        model = XGBoostModel(config=w_config)
    elif args.model == 'catboost':
        train_data, cat_idxs, cat_dims = preprocess.label_encoding(df=train_data, is_train=True)
        model = CatBoostModel(config=w_config, cat_idxs=cat_idxs)
    elif args.model == 'tabnet':
        train_data, cat_idxs, cat_dims = preprocess.label_encoding(df=train_data, is_train=True)
        model = TabNetModel(config=w_config, cuda=args.device, cat_idxs=cat_idxs, cat_dims=cat_dims)
    
    # # OOF 예측 초기화
    # oof_predictions = np.zeros(len(train_data))
    
    # Test 데이터 예측 결과 초기화
    all_test_preds = []
    
    # Valid 데이터 예측 결과 초기화
    all_val_acc = []
    all_val_auc = []
    
    splitter = GroupKFold(n_splits=args.fold)
    for fold, (train_index, valid_index) in enumerate(splitter.split(train_data, groups=train_data["userID"])):
        x_train, y_train = xy_data_split(train_data.iloc[train_index])
        x_valid, y_valid = xy_data_split(train_data.iloc[valid_index])
        
        logger.info(f"Fold {fold + 1}: Training ...")
        model.fit(x_train, y_train, x_valid, y_valid)
        
        logger.info(f"Fold {fold + 1}: Validating ...")
        preds = model.predict(x_valid)
        
        # 평가 및 Logging
        auc, acc = get_metric(y_valid, preds)
        logger.info(f"Fold {fold + 1}: Validation AUC: {auc}, Accuracy: {acc}")
        
        all_val_acc.append(acc)
        all_val_auc.append(auc)
        
        wandb.log(dict(Val_AUC=auc, Val_Acc=acc))

        # INFERENCE FOR TEST DATA
        logger.info(f"Fold {fold + 1}: Inference on Test Data ...")
        preprocess.load_test_data(file_name=args.test_file_name)
        test_data = preprocess.get_test_data()
        if args.model in ['tabnet', 'catboost']:
            test_data = preprocess.label_encoding(df=test_data, is_train=False)
        test_preds = model.predict(test_data[FEATS])
        
        # 저장된 각 fold의 테스트 데이터 예측을 리스트에 추가
        all_test_preds.append(test_preds)

    # 각 fold에서의 테스트 데이터 예측 평균 계산
    avg_test_preds = np.mean(all_test_preds, axis=0)
    
    # 각 fold 평가 평균 계산
    overall_auc, overall_acc = np.mean(all_val_auc), np.mean(all_val_acc)
    logger.info(f"Overall Valid AUC: {overall_auc}, Valid Accuracy: {overall_acc}")
    
    wandb.log(dict(Overall_Val_AUC=overall_auc, Overall_Val_Acc=overall_acc))
    
    
    # 최종 테스트 데이터 예측 결과 활용
    logger.info(f"Out Of Fold Inference on Test Data ...")
    write_path = os.path.join(args.output_dir, f"{exp_name}_submission.csv")
    os.makedirs(name=args.output_dir, exist_ok=True)
    with open(write_path, "w", encoding="utf8") as w:
        w.write("id,prediction\n")
        for id, p in enumerate(avg_test_preds):
            w.write("{},{}\n".format(id, p))
    logger.info("Successfully saved submission as %s", write_path)
    
    # WandB Artifact Logging
    submission_artifact = wandb.Artifact('submission', type='output')
    submission_artifact.add_file(local_path=write_path)
    wandb.log_artifact(submission_artifact)
    

# def inference(args, test_data, model, exp_name):
    
#     total_preds = model.predict(test_data)
    
#     write_path = os.path.join(args.output_dir, f"{exp_name}_submission.csv")
#     os.makedirs(name=args.output_dir, exist_ok=True)
#     with open(write_path, "w", encoding="utf8") as w:
#         w.write("id,prediction\n")
#         for id, p in enumerate(total_preds):
#             w.write("{},{}\n".format(id, p))
#     logger.info("Successfully saved submission as %s", write_path)
    
#     # WandB Artifact Logging
#     submission_artifact = wandb.Artifact('submission', type='output')
#     submission_artifact.add_file(local_path=write_path)
#     wandb.log_artifact(submission_artifact)
