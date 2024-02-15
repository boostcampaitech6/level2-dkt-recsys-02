import os
import random
import time
import numpy as np
import torch


class process:
    def __init__(self, logger, name):
        self.logger = logger
        self.name = name

    def __enter__(self):
        self.logger.info(f"{self.name} - Started")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.info(f"{self.name} - Complete")


def set_seeds(seed: int = 42):
    # 랜덤 시드를 설정하여 매 코드를 실행할 때마다 동일한 결과를 얻게 합니다.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_logger(logger_conf: dict):
    import logging
    import logging.config

    logging.config.dictConfig(logger_conf)
    logger = logging.getLogger()
    return logger


logging_conf = {  # only used when 'user_wandb==False'
    "version": 1,
    "formatters": {
        "basic": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "basic",
            "stream": "ext://sys.stdout",
        },
        "file_handler": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "basic",
            "filename": "run.log",
        },
    },
    "root": {"level": "INFO", "handlers": ["console", "file_handler"]},
}


def get_expname(args):
    now = time.localtime()
    now_date = time.strftime('%Y%m%d', now)
    now_hour = time.strftime('%X', now)
    save_time = now_date + '_' + now_hour.replace(':', '')
    expname = save_time + '_' + args.model
    return expname


def dropout_edge(train_data: dict, p: float = 0.5):
    if p < 0. or p > 1.:
        raise ValueError(f'숫자 똑띠 안넣나 0~1이다 ~ 만약에, 네부캠 7기 이후 사람이 이 글을 본다면, 뒤로가기 하세요 실력 안늘어요 ㅎㅎ;'
                        f'(이거 넣으면 없는 엣지도 드랍 할려고 으휴 ..? {p}')
        
    user_problem_edge_index = train_data['edge_user_item']
    user_problem_label = train_data['label']
    user_test_edge_index = train_data['edge_user_test']
    user_tag_edge_index = train_data['edge_user_know']
    
    up = user_problem_edge_index.shape[1]
    mask = torch.rand(up, device=user_problem_edge_index.device) >= p

    user_problem_edge_index = user_problem_edge_index[:, mask]
    user_test_edge_index = user_test_edge_index[:, mask]
    user_tag_edge_index = user_tag_edge_index[:, mask]
    user_problem_label = user_problem_label[mask] 

    return {'edge_user_item': user_problem_edge_index, 'label': user_problem_label, 'edge_user_test': user_test_edge_index, 'edge_user_know': user_tag_edge_index}
