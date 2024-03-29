import os
from typing import Tuple

import pandas as pd
import torch

from lightgcn.utils import get_logger, logging_conf


logger = get_logger(logging_conf)


def prepare_dataset(device: str, data_dir: str, return_origin_train:bool = False) -> Tuple[dict, dict, int]:
    data = load_data(data_dir=data_dir)
    train_data, test_data = separate_data(data=data)
    id2index: dict = indexing_data(data=data)
    origin_train_data = process_data(train_data, id2index=id2index, device=device)
    train_data, val_data = train_to_tval_split(train_data,rule="last_percent")
    train_data_proc = process_data(data=train_data, id2index=id2index, device=device)
    val_data_proc = process_data(data=val_data,id2index=id2index,device=device)
    test_data_proc = process_data(data=test_data, id2index=id2index, device=device)
    print_data_stat(train_data, "Train")
    print_data_stat(val_data,"val")
    print_data_stat(test_data, "Test")
    if return_origin_train:
        return origin_train_data, val_data_proc ,test_data_proc, len(id2index) 
    else:
        return train_data_proc, val_data_proc ,test_data_proc, len(id2index)


def load_data(data_dir: str) -> pd.DataFrame: 
    path1 = os.path.join(data_dir, "train_data.csv")
    path2 = os.path.join(data_dir, "test_data.csv")
    data1 = pd.read_csv(path1)
    data2 = pd.read_csv(path2)

    data = pd.concat([data1, data2])
    data.drop_duplicates(subset=["userID", "assessmentItemID"], keep="last", inplace=True)
    return data


def separate_data(data: pd.DataFrame) -> Tuple[pd.DataFrame]:
    train_data = data[data.answerCode >= 0]
    test_data = data[data.answerCode < 0]
    return train_data, test_data


def indexing_data(data: pd.DataFrame) -> dict:
    userid, itemid, testid, tag = (
        sorted(list(set(data.userID))),
        sorted(list(set(data.assessmentItemID))),
        sorted(list(set(data.testId))),
        sorted(list(set(data.KnowleageTag)))
    )
    n_user, n_item , n_test, n_tag = len(userid), len(itemid), len(testid), len(tag)

    userid2index = {v: i for i, v in enumerate(userid)}
    itemid2index = {v: i + n_user for i, v in enumerate(itemid)}
    testid2index = {v:i+n_user+n_item for i ,v in enumerate(testid)}
    tagid2index = {v:i+n_user+n_item+n_test for i ,v in enumerate(tag)}
    id2index = dict(userid2index, **itemid2index, **testid2index, **tagid2index)
    return id2index


def process_data(data: pd.DataFrame, id2index: dict, device: str) -> dict:
    edge_user_item, edge_user_test, edge_user_know ,label = [], [], [], []
    for user, item, acode, test, know in zip(data.userID, data.assessmentItemID, data.answerCode, data.testId, data.KnowleageTag):
        uid, iid, tid, kid = id2index[user], id2index[item], id2index[test], id2index[know]
        edge_user_item.append([uid,iid])
        edge_user_test.append([uid,tid])
        edge_user_know.append([uid,kid])
        label.append(acode)

    edge_user_item = torch.LongTensor(edge_user_item).T
    edge_user_test = torch.LongTensor(edge_user_test).T
    edge_user_know = torch.LongTensor(edge_user_know).T
    label = torch.LongTensor(label)
    return dict(edge_user_item=edge_user_item.to(device),
                edge_user_test=edge_user_test.to(device),
                edge_user_know=edge_user_know.to(device),
                label=label.to(device),
                )

def train_to_tval_split(train_data, rule = "last_one"):
    df = train_data.sort_values(by=["userID", "Timestamp"], axis=0)
    if rule == "last_one":
        val_data = df.groupby('userID').apply(lambda group: group.iloc[-1]).reset_index(drop=True)
        train_data = df[~df.index.isin(val_data.index)]
    else:#percentage for user
        user_counts = df['userID'].value_counts()
        val_counts = (user_counts * 0.1).astype(int)
        val_data_indices = df.groupby('userID').apply(lambda group: group.sample(n=val_counts[group.name])).index.levels[1]
        val_data = df.loc[val_data_indices].reset_index(drop=True)
        train_data = df[~df.index.isin(val_data_indices)].reset_index(drop=True)
    return train_data, val_data


def print_data_stat(data: pd.DataFrame, name: str) -> None:
    userid, itemid = list(set(data.userID)), list(set(data.assessmentItemID))
    n_user, n_item = len(userid), len(itemid)

    logger.info(f"{name} Dataset Info")
    logger.info(f" * Num. Users    : {n_user}")
    logger.info(f" * Max. UserID   : {max(userid)}")
    logger.info(f" * Num. Items    : {n_item}")
    logger.info(f" * Num. Records  : {len(data)}")
