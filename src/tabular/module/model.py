import torch
import wandb
import os
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from wandb.lightgbm import wandb_callback, log_summary
from wandb.xgboost import WandbCallback
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.callbacks import Callback
from model_configs.default_config import lightGBMParams, tabNetParams, xgboostParams, catBoostParams


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


class CatBoostModel:
    def __init__(self, config, cat_idxs=[]):
        self.config = config
        self.model = None
        self.params = catBoostParams
        self.cat_idxs = cat_idxs

    def setting(self):
        self.params['learning_rate'] = self.config['learning_rate']
        self.params['depth'] = self.config['depth']
        self.params['iterations'] = self.config['iterations']
        self.params['min_child_samples'] = self.config['min_child_samples']
        return self.params

    def fit(self, x_train, y_train, x_valid, y_valid):
        params = self.setting()
        train_data = Pool(x_train, label=y_train, cat_features=self.cat_idxs)
        valid_data = Pool(x_valid, label=y_valid, cat_features=self.cat_idxs)

        self.model = CatBoostClassifier(**params)

        self.model.fit(
            train_data,
            eval_set=[valid_data],
            early_stopping_rounds=self.config['early_stopping_rounds'],
            verbose=True,
            callbacks=[WandBCallback()]
        )

    def predict(self, X_test):
        if self.model is not None:
            preds = self.model.predict_proba(X_test)
            return np.max(preds, axis=1)  # Assuming binary classification
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
            callbacks=[WandBCallbackTabNet()]
        )
        # self.make_plot(model = self.model)

    def predict(self, x_test):
        if self.model is not None:
            preds = self.model.predict_proba(x_test.values)
            return np.max(preds, axis=1)
        else:
            raise ValueError("Model has not been trained. Please call fit() first.")
    

class WandBCallback:
    def after_iteration(self, info):
        iteration = info.iteration
        metrics = info.metrics
        for metric_name, metric_value in metrics.items():
            if metric_name == 'learn':
                wandb.log({'Train Logloss': np.mean(metric_value['Logloss'])})
                wandb.log({'Train Acc': np.mean(metric_value['Accuracy'])})
            elif metric_name == 'validation':
                wandb.log({'Valid Loss': np.mean(metric_value['Logloss'])})
                wandb.log({'Valid Acc': np.mean(metric_value['Accuracy'])})
                wandb.log({'Valid AUC': np.mean(metric_value['AUC'])})
        return True

class WandBCallbackTabNet(Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        wandb.log({"Train AUC": logs["train_auc"]})
        wandb.log({"Train AUC": logs["valid_auc"]})

        wandb.log({"Train Loss": logs["loss"]})
        wandb.log({"Valid Loss": logs["valid_logloss"]})
        
        wandb.log({"Train Acc": logs["train_accuracy"]})
        wandb.log({"Train Acc": logs["valid_accuracy"]})