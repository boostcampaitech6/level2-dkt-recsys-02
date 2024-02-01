## Setup
```bash
cd /src/tabular/
conda init
(base) . ~/.bashrc
(base) conda create -n dkt_tabular python=3.10 -y
(base) conda activate dkt_tabular
(dkt) pip install -r requirements.txt
(dkt) python main.py
```

## Files
`src/tabular`
* `main.py`: 실행 코드로 WandB sweep을 활용한 HPO가 탑재되어있습니다. count를 조절해서 학습 횟수를 조절할 수 있다.
* `requirements.txt`: 모델 학습에 필요한 라이브러리들이 정리되어 있습니다.

`src/tabular/module`
* `args.py`: `argparse`를 통해 모델 저장 경로, 데이터 경로, 학습 모델명 등 학습 파라미터 이외에 활용되는 여러 argument들을 받아줍니다.
* `datloader.py`: dataloader를 불러옵니다.
* `metric.py`: metric 계산하는 함수를 포함합니다.
* `model.py`: 여러 모델 소스 코드를 포함합니다. `LightGBM`, `Xgboost`, `Catboost`, `Tabnet`을 가지고 있습니다.
* `trainer.py`: 훈련에 사용되는 함수들을 포함합니다.
* `utils.py`: 학습에 필요한 부수적인 함수들을 포함합니다.


`src/tabular/model_configs`
* `default_config.py`: 학습에 필요한 각종 파라미터들을 관리하는 파일로, 각 모델들 학습에 적용되는 하이퍼파라미터, 학습에 사용할 피처 선정, 카테고리형 데이터 지정을 해줄 수 있다. 

`src/tabular/sweep`
* 해당 폴더에서는 WandB sweep을 활용한 하이퍼파라미터 튜닝 파라미터를 관리하며, 각 모델별로 json파일을 생성해서 원하는 설정값에 맞게 작성후 `python main.py`를 실행하면 HPO가 진행된다.