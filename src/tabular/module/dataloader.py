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
        # 유저별로 정렬
        df.sort_values(by=['userID', 'Timestamp'], inplace=True)
        
        # 데이터 타입 변경
        dtype = {
            'userID': 'int16',
            'answerCode': 'int8',
            'KnowledgeTag': 'int16'
        }
        df = df.astype(dtype)
        
        # 'Timestamp' 열을 날짜/시간 형식으로 파싱
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S')

        # testTag 추가
        df['testTag'] = df['testId'].apply(lambda x: x[2]).astype('int16')

        # 유저별로 정답 누적 횟수 계산, 결측치 0
        df['user_correct_answer'] = df.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift(1))
        df['user_correct_answer'].fillna(0, inplace=True)
        
        # 유저별로 제출 누적 횟수 계산
        df['user_total_answer'] = df.groupby('userID')['answerCode'].cumcount() 
        
        # 유저별로 누적 정답률 계산, 결측치 0.75
        df['user_acc'] = df['user_correct_answer'] / df['user_total_answer']
        df['user_acc'].fillna(0.75, inplace=True)

        # userID별 정답률 추가
        df['user_sum'] = df.groupby('userID')['answerCode'].transform('sum')
        df['user_mean'] = df.groupby('userID')['answerCode'].transform('mean')
        
        # assessmentItemID별 정답률 추가
        df['assessment_sum'] = df.groupby('assessmentItemID')['answerCode'].transform('sum')
        df['assessment_mean'] = df.groupby('assessmentItemID')['answerCode'].transform('mean')
        
        # testId별 정답률 추가
        df['test_sum'] = df.groupby('testId')['answerCode'].transform('sum')
        df['test_mean'] = df.groupby('testId')['answerCode'].transform('mean')
        
        # KnowledgeTag별 정답률 추가
        df['knowledgeTag_sum'] = df.groupby('KnowledgeTag')['answerCode'].transform('sum')
        df['knowledgeTag_mean'] = df.groupby('KnowledgeTag')['answerCode'].transform('mean')
        
        # testTag별 정답률 추가
        df['testTag_sum'] = df.groupby('testTag')['answerCode'].transform('sum')
        df['testTag_mean'] = df.groupby('testTag')['answerCode'].transform('mean')

        # 상대적 정답률
        df['relative_answer_assessment'] = df['answerCode'] - df.groupby('assessmentItemID')['answerCode'].transform('mean')
        
        # 유저별 상대적 정답률 평균 - 학습 수준 레벨
        df['relative_answer_mean'] = df.groupby('userID')['relative_answer_assessment'].transform('mean')

        # 유저가 문항을 푼 시간
        df['time_to_solve'] = df.groupby(['userID', 'testId'])['Timestamp'].diff().dt.total_seconds().shift(-1)
        
        # 결측치 이전 행의 값으로 채움
        df['time_to_solve'].fillna(method='ffill', inplace=True)

        # 유저별 문항 시간 평균
        #df['time_to_solve_mean'] = df.groupby('userID')['time_to_solve'].transform('mean')
        df['time_to_solve_mean'] = df.groupby(['userID', 'testId'])['time_to_solve'].transform('mean')

        # clip(0, 255)는 메모리를 위해 uint8 데이터 타입을 쓰기 위함
        df['prior_assessment_frequency'] = df.groupby(['userID', 'assessmentItemID']).cumcount().clip(0, 255)

        # 각 태그별로 이전에 몇번 풀었는지
        df['prior_KnowledgeTag_frequency'] = df.groupby(['userID', 'KnowledgeTag']).cumcount()
        
        # 시험지 태그별 학년별 몇번 풀었는지
        df['prior_testTag_frequency'] = df.groupby(['userID', 'testTag']).cumcount()
        
        if is_train == False:
            # LEAVE LAST INTERACTION ONLY
            df = df[df['userID'] != df['userID'].shift(-1)]
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