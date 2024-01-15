import lightgbm as lgb
from wandb.lightgbm import wandb_callback, log_summary
from model_configs.default_config import lightGBMParams, FEATS


class LightGBMModel:
    def __init__(self, config):
        self.feats = FEATS
        self.config = config
        self.model = None
        self.params = lightGBMParams

    def setting(self):
        self.params['boosting'] = self.config['boosting']
        self.params['max_depth'] = self.config['max_depth']
        self.params['learning_rate'] = self.config['learning_rate']
        self.params['num_leaves'] = self.config['num_leaves']
        self.params['colsample_bytree'] = self.config['colsample_bytree']
        self.params['num_iterations'] = self.config['num_iterations']
        return self.params
        
    def fit(self, x_train, y_train, x_valid, y_valid):
        params = self.setting()
        lgb_train = lgb.Dataset(x_train[self.feats], y_train)
        lgb_valid = lgb.Dataset(x_valid[self.feats], y_valid, reference=lgb_train)
        self.model = lgb.train(
            params,
            train_set=lgb_train,
            valid_sets=lgb_valid,
            num_boost_round=500,
            callbacks=[wandb_callback()]
        )
        log_summary(self.model, save_model_checkpoint=True)

    def predict(self, x_valid):
        if self.model is not None:
            return self.model.predict(x_valid[self.feats])
        else:
            raise ValueError("Model has not been trained. Please call fit() first.")