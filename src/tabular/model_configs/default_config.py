import torch
# 사용할 Feature 설정
FEATS = [
    # 'userID',
    # 'assessmentItemID',
    # 'testId',
    # 'answerCode',
    # 'Timestamp',
    'KnowledgeTag',
    'Dffclt',
    'Dscrmn',
    'Gussng',
    'testTag',
    'user_correct_answer',
    'user_total_answer',
    'user_acc',
    # 'user_sum',
    'user_mean',
    # 'assessment_sum',
    # 'assessment_mean',
    # 'test_sum',
    # 'test_mean',
    # 'knowledgeTag_sum',
    # 'knowledgeTag_mean',
    # 'testTag_sum',
    # 'testTag_mean',
    # 'relative_answer_assessment',
    'relative_answer_mean',
    'time_to_solve',
    'time_to_solve_mean',
    # 'prior_assessment_frequency',
    # 'prior_KnowledgeTag_frequency',
    'prior_testTag_frequency'
    ]

# 범주형 Feature 설정
cat_cols = [
    # 'userID',
    # 'assessmentItemID',
    # 'testId',
    'KnowledgeTag'
    ]

# LightGBM default parameters
lightGBMParams={
    'learning_rate':0.1,
    'num_boost_round':500,
    'max_depth':10,
    'num_leaves': 50,
    'lambda_l1': 0,
    'lambda_l2': 0,
    'colsample_bytree':0.7,
    'objective':'binary',
    'metric':'auc',
    'early_stopping_rounds':10,
    'verbosity':1
}

# TabNet default parameters
tabNetParams = {
    'n_d': 8,
    'n_a': 8,
    'cat_idxs': [],
    'cat_dims': [],
    'cat_emb_dim': 1,
    'optimizer_fn': torch.optim.Adam,
    'optimizer_params': {'lr': 1e-4, 'weight_decay': 1e-5},
    'scheduler_params': {'step_size': 5, 'gamma': 0.8},
    'scheduler_fn': torch.optim.lr_scheduler.StepLR,
    'mask_type': 'sparsemax',
    'verbose': 1,
    # 'max_epochs': 100,
    # 'batch_size': 128,
    # 'patience': 7,
    'seed': 42,
    'device_name': 'cpu'
}
