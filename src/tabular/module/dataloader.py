import os
import random
import time
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from model_configs.default_config import FEATS, cat_cols


class Preprocess:
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None
        self.le = LabelEncoder()

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    # def split_data(self,
    #                data: np.ndarray,
    #                ratio: float = 0.7,
    #                shuffle: bool = True,
    #                seed: int = 0) -> Tuple[np.ndarray]:
    #     """
    #     split data into two parts with a given ratio.
    #     """
    #     if shuffle:
    #         random.seed(seed)  # fix to default seed 0
    #         random.shuffle(data)

    #     size = int(len(data) * ratio)
    #     data_1 = data[:size]
    #     data_2 = data[size:]
    #     return data_1, data_2
    
    def split_data(self,
                    df: np.ndarray,
                    ratio: float = 0.7,
                    shuffle: bool = True,
                    seed: int = 0,
                    ):
        
        if shuffle:
            random.seed(seed)  # fix to default seed 0
            users = list(zip(df['userID'].value_counts().index, df['userID'].value_counts()))
            random.shuffle(users)

        max_train_data_len = ratio*len(df)
        sum_of_train_data = 0
        user_ids =[]

        for user_id, count in users:
            sum_of_train_data += count
            if max_train_data_len < sum_of_train_data:
                break
            user_ids.append(user_id)

        train = df[df['userID'].isin(user_ids)]
        test = df[df['userID'].isin(user_ids) == False]

        #test데이터셋은 각 유저의 마지막 interaction만 추출
        test = test[test['userID'] != test['userID'].shift(-1)]
        
        return train, test

    def __feature_engineering(self, df: pd.DataFrame, is_train: bool, encoding=False) -> pd.DataFrame:
            # 데이터 타입 변경
        dtype = {
            'userID' : 'category',
            'assessmentItemID' : 'category',
            'testId' : 'category',
            'KnowledgeTag' : 'category',
            'testTag' : 'category'
        }
        df.astype(dtype)
        # 날짜시간 데이터로 변환
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        if is_train == False:
            # LEAVE LAST INTERACTION ONLY
            # df = df[df['userID'] != df['userID'].shift(-1)]
            # DROP ANSWERCODE
            df = df.drop(['answerCode'], axis=1)
            return df[FEATS]
            
        return df
    
    def label_encoding(self, df: pd.DataFrame, is_train: bool):
        
        # Label Encoding catagorical data 
        if is_train == True:
            for col in cat_cols:
                df[col] = self.le.fit_transform(df[col].values)
                
            cat_idxs = [ i for i, f in enumerate(FEATS) if f in cat_cols]
            cat_dims = [ len(df[f].unique()) for f in FEATS if f in cat_cols]
            return df, cat_idxs, cat_dims
        else:
            for col in cat_cols:
                df[col] = self.le.transform(df[col].values)
            return df


    def load_data_from_file(self, file_name: str, is_train: bool = True) -> np.ndarray:
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path)  # , nrows=100000)
        df = self.__feature_engineering(df, is_train)
        
        # df = self.__preprocessing(df, is_train)
        return df

    def load_train_data(self, file_name: str) -> None:
        self.train_data = self.load_data_from_file(file_name)

    def load_test_data(self, file_name: str) -> None:
        self.test_data = self.load_data_from_file(file_name, is_train=False)
        
        
def xy_data_split(df):
    y_data = df['answerCode']
    X_data = df.drop(['answerCode'], axis=1)
    return X_data[FEATS], y_data