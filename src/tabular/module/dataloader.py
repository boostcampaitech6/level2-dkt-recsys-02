import os
import random
import time
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

class Preprocess:
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None

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
                    seed: int = 0):
        
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

    def __feature_engineering(self, df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        #유저별 시퀀스를 고려하기 위해 아래와 같이 정렬
        df.sort_values(by=['userID','Timestamp'], inplace=True)

        #유저들의 문제 풀이수, 정답 수, 정답률을 시간순으로 누적해서 계산
        df['user_correct_answer'] = df.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift(1))
        df['user_total_answer'] = df.groupby('userID')['answerCode'].cumcount()
        df['user_acc'] = df['user_correct_answer']/df['user_total_answer']

        # testId와 KnowledgeTag의 전체 정답률은 한번에 계산
        # 아래 데이터는 제출용 데이터셋에 대해서도 재사용
        correct_t = df.groupby(['testId'])['answerCode'].agg(['mean', 'sum'])
        correct_t.columns = ["test_mean", 'test_sum']
        correct_k = df.groupby(['KnowledgeTag'])['answerCode'].agg(['mean', 'sum'])
        correct_k.columns = ["tag_mean", 'tag_sum']

        df = pd.merge(df, correct_t, on=['testId'], how="left")
        df = pd.merge(df, correct_k, on=['KnowledgeTag'], how="left")
        
        if is_train == False:
            # LEAVE LAST INTERACTION ONLY
            df = df[df['userID'] != df['userID'].shift(-1)]
            # DROP ANSWERCODE
            df = df.drop(['answerCode'], axis=1)
            
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
    return X_data, y_data