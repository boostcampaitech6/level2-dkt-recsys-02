# 사용할 Feature 설정
FEATS = ['KnowledgeTag', 'user_correct_answer', 'user_total_answer',
         'user_acc', 'test_mean', 'test_sum', 'tag_mean','tag_sum']

# LightGBM default parameters
lightGBMParams={
    'learning_rate':0.1,
    'n_estimators':10000,
    'max_depth':-1,
    'num_leaves': 200,
    'lambda_l1': 0,
    'lambda_l2': 0,
    'colsample_bytree':0.7,
    'objective':'binary',
    'metric':'binary_logloss',
    'early_stopping_rounds':50,
    'verbosity':10
}
