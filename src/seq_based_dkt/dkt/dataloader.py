import os
import random
import time
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler



class Preprocess:
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def split_data(self,
                   data: np.ndarray,
                   ratio: float = 0.7,
                   shuffle: bool = True,
                   seed: int = 0) -> Tuple[np.ndarray]:
        """
        split data into two parts with a given ratio.
        """
        if shuffle:
            random.seed(seed)  # fix to default seed 0
            random.shuffle(data)

        size = int(len(data) * ratio)
        data_1 = data[:size]
        data_2 = data[size:]
        return data_1, data_2

    def __save_labels(self, encoder: LabelEncoder, name: str) -> None:
        le_path = os.path.join(self.args.asset_dir, name + "_classes.npy")
        np.save(le_path, encoder.classes_)   # .classes__: 변환된 값에 대한 원본값

    def __preprocessing(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        cate_cols = ["assessmentItemID", "testId", "KnowledgeTag"]

        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)

        for col in cate_cols:
            # +) 회귀 모델에서 LabelEncoder가 성능이 떨어질 수 있음
            le = LabelEncoder()
            if is_train:
                # For UNKNOWN class 
                # (∵train에는 없지만 valid || test에 존재할 수 있는 새로운 범주값을 처리하기 위함) 
                a = df[col].unique().tolist() + ["unknown"]
                le.fit(a)   # 각 범주형 값에 대해 고유한 정수값 할당하는 규칙 학습
                self.__save_labels(le, col)
            else:
                label_path = os.path.join(self.args.asset_dir, col + "_classes.npy")
                le.classes_ = np.load(label_path)

                # col에 대해 각 값이 le.classes_에 속하는지 확인 후 속하지 않으면 unknown으로 처리
                # 훈련 데이터에 존재하지 않는 새로운 범주값이 valid || test data에 등장하는 경우 "unknown"으로 처리
                df[col] = df[col].apply(
                    lambda x: x if str(x) in le.classes_ else "unknown"
                )

            # 모든 컬럼이 범주형이라고 가정
            # str변환: 입력 데이터 표준 형식 표준화 || "unknown" 같은 추가적인 문자열 값 처리 목적
            df[col] = df[col].astype(str)
            test = le.transform(df[col])   # transform 메서드를 사용해 컬럼 값을 수치형으로 변환
            df[col] = test   # 변환된 수치형 데이터로 df[col]

        # 필요시 다른 scale 적용
        # scale_col=["userID", "assessmentItemID", "testId", "answerCode", "KnowledgeTag","Dffclt", "Dscrmn", "Gussng"]
        # scaler=StandardScaler()
        # df[scale_col]=scaler.fit_transform(df[scale_col])

        return df

    def __feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO: Fill in if needed
        # 유저별로 정렬
        df.sort_values(by=['userID', 'Timestamp'], inplace=True)
        
        # # 데이터 타입 변경
        # # ksj) 추가된 feature에 대해서도 타입 변경? 
        # # 사용할 feature에 대해서만 FE 진행하는 추가 로직 필요
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

        # # userID별 정답률 추가
        # df['user_sum'] = df.groupby('userID')['answerCode'].transform('sum')
        df['user_mean'] = df.groupby('userID')['answerCode'].transform('mean')
        
        # assessmentItemID별 정답률 추가
        # df['assessment_sum'] = df.groupby('assessmentItemID')['answerCode'].transform('sum')
        # df['assessment_mean'] = df.groupby('assessmentItemID')['answerCode'].transform('mean')
        
        # testId별 정답률 추가
        # df['test_sum'] = df.groupby('testId')['answerCode'].transform('sum')
        # df['test_mean'] = df.groupby('testId')['answerCode'].transform('mean')
        
        # KnowledgeTag별 정답률 추가
        # df['knowledgeTag_sum'] = df.groupby('KnowledgeTag')['answerCode'].transform('sum')
        # df['knowledgeTag_mean'] = df.groupby('KnowledgeTag')['answerCode'].transform('mean')
        
        # testTag별 정답률 추가
        # df['testTag_sum'] = df.groupby('testTag')['answerCode'].transform('sum')
        # df['testTag_mean'] = df.groupby('testTag')['answerCode'].transform('mean')

        # 상대적 정답률
        df['relative_answer_assessment'] = df['answerCode'] - df.groupby('assessmentItemID')['answerCode'].transform('mean')
        
        # 유저별 상대적 정답률 평균 - 학습 수준 레벨
        df['relative_answer_mean'] = df.groupby('userID')['relative_answer_assessment'].transform('mean')

        # 유저가 문항을 푼 시간
        df['time_to_solve'] = df.groupby(['userID', 'testId'])['Timestamp'].diff().dt.total_seconds().shift(-1)
        
        # 결측치 이전 행의 값으로 채움
        df['time_to_solve'].ffill()

        # 유저별 문항 시간 평균
        #df['time_to_solve_mean'] = df.groupby('userID')['time_to_solve'].transform('mean')
        df['time_to_solve_mean'] = df.groupby(['userID', 'testId'])['time_to_solve'].transform('mean')

        # clip(0, 255)는 메모리를 위해 uint8 데이터 타입을 쓰기 위함
        # df['prior_assessment_frequency'] = df.groupby(['userID', 'assessmentItemID']).cumcount().clip(0, 255)

        # 각 태그별로 이전에 몇번 풀었는지
        # df['prior_KnowledgeTag_frequency'] = df.groupby(['userID', 'KnowledgeTag']).cumcount()
        
        # 시험지 태그별 학년별 몇번 풀었는지
        df['prior_testTag_frequency'] = df.groupby(['userID', 'testTag']).cumcount()
        
        return df

   
    def load_data_from_file(self, file_name: str, is_train: bool = True) -> np.ndarray:
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path)  # , nrows=100000)
        df = self.__feature_engineering(df)
        df = self.__preprocessing(df, is_train)

        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할 때 사용

        self.args.n_questions = len(
            np.load(os.path.join(self.args.asset_dir, "assessmentItemID_classes.npy"))
        )
        self.args.n_tests = len(
            np.load(os.path.join(self.args.asset_dir, "testId_classes.npy"))
        )
        self.args.n_tags = len(
            np.load(os.path.join(self.args.asset_dir, "KnowledgeTag_classes.npy"))
        )

        df = df.sort_values(by=["userID", "Timestamp"], axis=0)
        columns = ["userID", "assessmentItemID", "testId", "answerCode", "KnowledgeTag",
                    "Dffclt", "Dscrmn", "Gussng",
                    'testTag',
                    'user_correct_answer',
                    'user_total_answer',
                    'user_acc',
                    'user_mean',
                    'relative_answer_mean',
                    'time_to_solve',
                    'time_to_solve_mean',
                    'prior_testTag_frequency']
        group = (
            df[columns]
            .groupby("userID")
            .apply(
                lambda r: (
                    r["testId"].values,
                    r["assessmentItemID"].values,
                    r["KnowledgeTag"].values,
                    r["answerCode"].values,
                    r["Dffclt"].values,
                    r["Dscrmn"].values,
                    r["Gussng"].values,
                    r["testTag"].values,
                    r["user_correct_answer"].values,
                    r["user_total_answer"].values,
                    r["user_acc"].values,
                    r["user_mean"].values,
                    r["relative_answer_mean"].values,
                    r["time_to_solve"].values,
                    r["time_to_solve_mean"].values,
                    r["prior_testTag_frequency"].values,
                )
            )
        )
        """
        Baseline) 최종 반환 형식(group.values):
        [
            ([testId 값들], [assessmentItemID 값들], [KnowledgeTag 값들], [answerCode 값들]),
            ([testId 값들], [assessmentItemID 값들], [KnowledgeTag 값들], [answerCode 값들]),
            ...
        ]
        """
        return group.values

    def load_train_data(self, file_name: str) -> None:
        self.train_data = self.load_data_from_file(file_name)

    def load_test_data(self, file_name: str) -> None:
        self.test_data = self.load_data_from_file(file_name, is_train=False)


class DKTDataset_PC(torch.utils.data.Dataset):
    def __init__(self, data: np.ndarray, args):
        self.data = data
        self.max_seq_len = args.max_seq_len

    def __getitem__(self, index: int) -> dict:
        row = self.data[index]
        
        # Load from data
        test, question, tag, correct, dffclt, dscrmn, gussng, testTag, user_correct_answer, user_total_answer, user_acc, user_mean, relative_answer_mean, time_to_solve, time_to_solve_mean, prior_testTag_frequency = row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12], row[13], row[14], row[15]

        # tensor = torch.tensor(data, dtype=torch.int)
        # tensor로 변환할 data를 주고 data type지정
        # 0값을 특별한 용도로 쓰려고 범주형 data 각 요소에 +1 (0 padding)
        # correct는 정답 여부를 나타내는 binary label 이니까 +1 X
        data = {
            "test": torch.tensor(test + 1, dtype=torch.int),
            "question": torch.tensor(question + 1, dtype=torch.int),
            "tag": torch.tensor(tag + 1, dtype=torch.int),
            "correct": torch.tensor(correct, dtype=torch.int),
            # 범주형 변수가 아니므로 +1 X
            # padding을 위해 0이 아닌 다른 값을 사용할 수 있음
            "dffclt": torch.tensor(dffclt, dtype=torch.float),
            "dscrmn": torch.tensor(dscrmn, dtype=torch.float),
            "gussng": torch.tensor(gussng, dtype=torch.float),
            "testTag": torch.tensor(testTag, dtype=torch.int),
            "user_correct_answer": torch.tensor(user_correct_answer, dtype=torch.float),
            "user_total_answer": torch.tensor(user_total_answer, dtype=torch.float),
            "user_acc": torch.tensor(user_acc, dtype=torch.float),
            "user_mean": torch.tensor(user_mean, dtype=torch.float),
            'relative_answer_mean': torch.tensor(relative_answer_mean, dtype=torch.float),  
            'time_to_solve': torch.tensor(time_to_solve, dtype=torch.float),
            'time_to_solve_mean': torch.tensor(time_to_solve_mean, dtype=torch.float),  
            'prior_testTag_frequency': torch.tensor(prior_testTag_frequency, dtype=torch.float)
        }
        # int형으로 바꿀 feature list
        seq_len = len(row[0])
        # past, current df 정의
        if seq_len > self.max_seq_len:
            past_df = {f"past_{k}": v[-self.max_seq_len-1:-1] for k, v in data.items()} 
            current_df = {f"current_{k}": v[-self.max_seq_len:] for k, v in data.items()}
        else:
            past_df = {f"past_{k}": v[:-1] for k, v in data.items()} 
            current_df = {f"current_{k}": v for k, v in data.items()} # 전체 데이터
        

        for df in [past_df, current_df]:
            if(df==past_df):
                seq_len = len(df['past_test'])
                if seq_len < self.max_seq_len:   # sequence data 길이가 max보다 작은 경우
                    for k, seq in data.items():
                        tmp = torch.zeros(self.max_seq_len)   # max 길이만큼 0 생성
                        tmp[self.max_seq_len-seq_len:] = df[k]   # sequence data 뒤쪽에 넣음
                        df[k] = tmp
                    mask = torch.zeros(self.max_seq_len, dtype=torch.int16)
                    mask[-seq_len:] = 1
                else:
                    mask = torch.ones(self.max_seq_len, dtype=torch.int16)
                df["past_mask"]=mask    # padding에 대한 mask 추가

                interaction = df["past_correct"] + 1
                interaction = interaction.roll(shifts=1)
                interaction_mask = df["past_mask"].roll(shifts=1)
                interaction_mask[0] = 0
                interaction = (interaction * interaction_mask).to(torch.int64)
                df["past_interaction"] = interaction
            elif(df==current_df):
                seq_len = len(df['current_test'])
                if seq_len < self.max_seq_len:   # sequence data 길이가 max보다 작은 경우
                    for k, seq in data.items():
                        tmp = torch.zeros(self.max_seq_len)   # max 길이만큼 0 생성
                        tmp[self.max_seq_len-seq_len:] = df[k]   # sequence data 뒤쪽에 넣음
                        df[k] = tmp
                    mask = torch.zeros(self.max_seq_len, dtype=torch.int16)
                    mask[-seq_len:] = 1
                else:
                    mask = torch.ones(self.max_seq_len, dtype=torch.int16)
                df["current_mask"]=mask    # padding에 대한 mask 추가

                interaction = df["current_correct"] + 1
                interaction = interaction.roll(shifts=1)
                interaction_mask = df["current_mask"].roll(shifts=1)
                interaction_mask[0] = 0
                interaction = (interaction * interaction_mask).to(torch.int64)
                df["current_interaction"] = interaction
        
        # 하나로 합침
        combined_df = {**past_df, **current_df}
        keys_to_convert = ["past_test", "past_question", "past_tag", "past_correct", "past_testTag",
                            "current_test", "current_question", "current_tag", "current_correct", "current_testTag"]
        combined_df = {k: v.int() if k in keys_to_convert else v for k, v in combined_df.items()}

        # data = {k: v.int() for k, v in data.items()}
        # keys_to_convert = ["test", "question", "tag", "correct", "testTag"]
        # data = {k: v.int() if k in keys_to_convert else v for k, v in data.items()}


        """
        .int()?
        torch.tensor 변환 과정 등 data type이 변할 수 있음
        .int()로 data type 일관성 보장 + Embedding layer input
        """
        """
        (past, current 각각 생성)
        최종 반환 data 형식:
        data = {
                "test": torch.tensor(test + 1, dtype=torch.int),
                "question": torch.tensor(question + 1, dtype=torch.int),
                "tag": torch.tensor(tag + 1, dtype=torch.int),
                "correct": torch.tensor(correct, dtype=torch.int),
                "dffclt": torch.tensor(dffclt, dtype=torch.float),
                "dscrmn": torch.tensor(dscrmn, dtype=torch.float),
                "gussng": torch.tensor(gussng, dtype=torch.float),
                "testTag": torch.tensor(testTag + 1, dtype=torch.int),
                "user_correct_answer": torch.tensor(user_correct_answer, dtype=torch.float),
                "user_total_answer": torch.tensor(user_total_answer, dtype=torch.float),
                "user_acc": torch.tensor(user_acc, dtype=torch.float),
                "user_mean": torch.tensor(user_mean, dtype=torch.float),
                'relative_answer_mean': torch.tensor(relative_answer_mean, dtype=torch.float),  
                'time_to_solve': torch.tensor(time_to_solve, dtype=torch.float),
                'time_to_solve_mean': torch.tensor(time_to_solve_mean, dtype=torch.float),  
                'prior_testTag_frequency': torch.tensor(prior_testTag_frequency, dtype=torch.float)
            }
        """
        return combined_df

    def __len__(self) -> int:
        return len(self.data)


def get_loaders(args, train: np.ndarray, valid: np.ndarray) -> Tuple[torch.utils.data.DataLoader]:
    pin_memory = False
    train_loader, valid_loader = None, None

    if train is not None:
        trainset = DKTDataset_PC(train, args)
        train_loader = torch.utils.data.DataLoader(
            trainset,
            num_workers=args.num_workers,
            shuffle=True,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
        )
    if valid is not None:
        valset = DKTDataset_PC(valid, args)
        valid_loader = torch.utils.data.DataLoader(
            valset,
            num_workers=args.num_workers,
            shuffle=False,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
        )

    return train_loader, valid_loader
