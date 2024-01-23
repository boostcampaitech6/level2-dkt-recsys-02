import torch
import wandb
import os
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from wandb.lightgbm import wandb_callback, log_summary
from wandb.xgboost import WandbCallback
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from model_configs.default_config import lightGBMParams, tabNetParams, xgboostParams, cat_cols


class LightGBMModel:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.params = lightGBMParams

    def setting(self):
        self.params['boosting'] = self.config['boosting']
        self.params['max_depth'] = self.config['max_depth']
        self.params['learning_rate'] = self.config['learning_rate']
        self.params['num_leaves'] = self.config['num_leaves']
        self.params['colsample_bytree'] = self.config['colsample_bytree']
        self.params['num_boost_round'] = self.config['num_boost_round']
        return self.params
        
    def fit(self, x_train, y_train, x_valid, y_valid):
        params = self.setting()
        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_valid = lgb.Dataset(x_valid, y_valid, reference=lgb_train)
        self.model = lgb.train(
            params,
            train_set=lgb_train,
            valid_sets=[lgb_train, lgb_valid],
            callbacks=[wandb_callback()]
        )
        log_summary(self.model, save_model_checkpoint=True)

    def predict(self, x_test):
        if self.model is not None:
            return self.model.predict(x_test)
        else:
            raise ValueError("Model has not been trained. Please call fit() first.")

class XGBoostModel:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.params = xgboostParams

    def setting(self):
        self.params['max_depth'] = self.config['max_depth']
        self.params['eta'] = self.config['eta']
        self.params['colsample_bytree'] = self.config['colsample_bytree']
        self.params['scale_pos_weight'] = self.config['scale_pos_weight']
        self.params['seed'] = self.config['seed']
        self.params['n_estimators'] = self.config['n_estimators']
        self.params['gamma'] = self.config['gamma']
        self.params['lambda'] = self.config['lambda']
        self.params['alpha'] = self.config['alpha']
        return self.params

    def fit(self, x_train, y_train, x_valid, y_valid):
        params = self.setting()
        self.model = XGBClassifier(**params)

        self.model.fit(
            x_train.values, 
            y_train.values.flatten(),
            eval_metric=['auc', 'error', 'logloss'],
            eval_set=[(x_train.values, y_train.values.flatten()),
                        (x_valid.values, y_valid.values.flatten())],
            early_stopping_rounds=self.config['early_stopping_rounds'],
            callbacks=[
                WandbCallback(log_model=True,
                            log_feature_importance=True,
                            )
            ],
            verbose=True)

    def predict(self, X_test):
        if self.model is not None:
            preds = self.model.predict_proba(X_test.values)
            return np.max(preds, axis=1)
        else:
            raise ValueError("Model has not been trained. Please call fit() first.")



class TabNetModel:
    def __init__(self, config, cuda, cat_idxs, cat_dims):
        self.config = config
        self.model = None
        self.params = tabNetParams
        self.cuda = cuda
        self.cat_idxs = cat_idxs
        self.cat_dims = cat_dims

    def setting(self):
        # Update the parameters based on the config
        if self.config['optimizer_fn'] == 'adam':
            self.params['optimizer_fn'] = torch.optim.Adam
        elif self.config['optimizer_fn'] == 'sgd':
            self.params['optimizer_fn'] == torch.optim.sgd
        elif self.config['optimizer_fn'] == 'adamw':
            self.params['optimizer_fn'] == torch.optim.AdamW
            
        self.params['n_d'] = self.config['n_d']
        self.params['n_a'] = self.config['n_a']
        self.params['cat_idxs'] = self.cat_idxs
        self.params['cat_dims'] = self.cat_dims
        self.params['cat_emb_dim'] = self.config['cat_emb_dim']
        self.params['optimizer_params']['lr'] = self.config['lr']
        self.params['optimizer_params']['weight_decay'] = self.config['weight_decay']
        self.params['seed'] = self.config['seed']
        self.params['device_name'] = self.cuda

        return self.params

    def fit(self, x_train, y_train, x_valid, y_valid):
        params = self.setting()
        # Initialize TabNetClassifier with the updated parameters
        self.model = TabNetClassifier(**params)

        # Fit the model
        self.model.fit(
            X_train=x_train.values,
            y_train=y_train.values.flatten(),
            eval_set=[(x_train.values, y_train.values.flatten()), 
                      (x_valid.values, y_valid.values.flatten())],
            eval_name=['train', 'valid'],
            eval_metric=['auc', 'accuracy', 'logloss'],
            max_epochs=self.config['max_epochs'],
            patience=self.config['patience'],
            batch_size=self.config['batch_size'],
            virtual_batch_size=128,
            drop_last=False,
        )
        self.make_plot(model = self.model)

    def predict(self, x_test):
        if self.model is not None:
            preds = self.model.predict_proba(x_test.values)
            return np.max(preds, axis=1)
        else:
            raise ValueError("Model has not been trained. Please call fit() first.")
    
    def make_plot(self, model):

        if not os.path.exists("plot"):
            os.makedirs("plot")

        loss_chart_artifact = wandb.Artifact("loss_chart", type="image")
        acc_chart_artifact = wandb.Artifact("acc_chart", type="image")
        auc_chart_artifact = wandb.Artifact("auc_chart", type="image")

        # Loss 그래프 생성 및 저장
        plt.plot(model.history['train_accuracy'], label='train')
        plt.plot(model.history['valid_accuracy'], label='val')
        plt.title('Acc per epoch')
        plt.ylabel('Acc')
        plt.xlabel('Epoch')
        plt.legend()

        acc_chart_path = "./plot/acc_chart.png"
        plt.savefig(acc_chart_path)
        acc_chart_artifact.add_file(acc_chart_path, name="acc_chart.png")
        wandb.log_artifact(acc_chart_artifact)

        # Accuracy 그래프 생성 및 저장
        plt.figure()  # 새로운 그래프를 생성
        plt.plot(model.history['loss'], label='train')
        plt.plot(model.history['valid_logloss'], label='val')
        plt.title('Loss per epoch')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()

        loss_chart_path = "./plot/loss_chart.png"
        plt.savefig(loss_chart_path)
        loss_chart_artifact.add_file(loss_chart_path, name="loss_chart.png")
        wandb.log_artifact(loss_chart_artifact)

        # AUC 그래프 생성 및 저장
        plt.figure()  # 새로운 그래프를 생성
        plt.plot(model.history['train_auc'], label='train')
        plt.plot(model.history['valid_auc'], label='val')
        plt.title('AUC per epoch')
        plt.ylabel('AUC')
        plt.xlabel('Epoch')
        plt.legend()

        auc_chart_path = "./plot/auc_chart.png"
        plt.savefig(auc_chart_path)
        auc_chart_artifact.add_file(auc_chart_path, name="auc_chart.png")
        wandb.log_artifact(auc_chart_artifact)

