import lightgbm as lgb

class LightGBMModel:
    def __init__(self, feats, config):
        self.feats = feats
        self.config = config
        self.model = None

    def fit(self, x_train, y_train, x_valid, y_valid):
        lgb_train = lgb.Dataset(x_train[self.feats], y_train)
        lgb_valid = lgb.Dataset(x_valid[self.feats], y_valid, reference=lgb_train)

        self.model = lgb.train(
            self.config,
            train_set=lgb_train,
            valid_sets=lgb_valid,
            num_boost_round=500
        )

    def predict(self, x_valid):
        if self.model is not None:
            return self.model.predict(x_valid[self.feats])
        else:
            raise ValueError("Model has not been trained. Please call fit() first.")