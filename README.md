![header](https://capsule-render.vercel.app/api?type=waving&color=gradient&height=250&section=header&text=Level2-DeepKnowledgeTracing&desc=RecSys-02&fontSize=50&fontColor=FFFFFF&fontAlignY=40)
- [랩업 레포트](https://github.com/boostcampaitech6/level2-dkt-recsys-02/blob/main/docs/DKT_RecSys_02_WarpUP_report.pdf)
- [최종 발표 자료](https://github.com/boostcampaitech6/level2-dkt-recsys-02/blob/main/docs/Recsys02-level2-DKT.pptx.pdf)
# 프로젝트 개요
**유저의 학습 상태에 대해 새로운 문제를 풀 가능성을 예측하는 Task**<br>
## 배경
시험 성적은 우리가 얼마만큼 아는지 평가하는 한 방법이지만, 학생 개개인의 이해도를 기리키는 ‘지식 상태‘는 알 수 없다.<br>
이를 해결하고자 딥러닝 방법론인 ‘Deep Knowledge Tracing’이 등장하였고 맞춤화 교육을 위해 아주 중요한 역할을 한다.<br>
대회에서는 지식 상태를 예측하기보다는, 주어진 문제를 맞출지 틀릴지 예측한다.<br>
최종적으로 test_data 마지막 문제의 정답여부가 주어지지 않으며, 이를 예측해야 한다.<br>

##  학습목표
트리 모델을 최대한 사용하지 않도록 한다.<br>
학습 목적이므로 내부 구조를 바꿀 수 있는 Deep Learning Model을 최대한 사용한다.<br>
각자 모델별 end-to-end를 경험 후, 경험을 합친다.<br>
공유하고, 질문하고, 토론한다.<br>
모델 하나를 깊게 파고 이해한다. <br>
가망이 없다면 미련 없이 돌아선다.<br>
궁금하면 일단 시도 해본다.<br>
모든 판단은 근거에 기반한다.<br>

# 팀소개

네이버 부스트캠프 AI Tech 6기 Level 2 Recsys 2조 **R_AI_SE** 입니다.

<aside>
    
💡 **R_AI_SE의 의미**

Recsys AI Seeker(추천시스템 인공지능 탐구자)를 줄여서 R_AI_SE입니다~


## 우리 조의 장점
### 특징
- 24시간코딩 이후 4시간 취침의 열정맨덜
- 티키타카
- 꿀잼
- 슬랙 허들은 개미지옥이야..
- 10시 데일리 스크럼, 16시 피어세션, 23시45분 앙상블세션
</aside>

## 👋 R_AI_SE의 멤버를 소개합니다 👋

### 🦹‍팀원소개
| 팀원   | 역할 및 담당                      |
|--------|----------------------------------|
| [김수진](https://github.com/guridon) | Transformer, SASRec, BERT4Rec, SAINT(+), CustomModel 구현, 모델 성능 실험, DKT Baseline 반자동화 |
| [김창영](https://github.com/ChangZero) | EDA, 정형데이터셋 베이스라인, TabNet, Tree모델 모듈화, Out Of Fold, WandB sweep HPO |
| [박승아](https://github.com/SeungahP) | 데이터 EDA, Last Query Transformer RNN, SAINT(+) 모델, T-fixup 구현, 하이퍼파라미터 튜닝 |
| [전민서](https://github.com/Minseojeonn) | Transformer, Graph, LSTM, SASREC, Bert4REC, ML, OOF, Wandb, LastQuery |
| [한대희](https://github.com/DAEHEE97) | EDA, Feature Engineering, XGBoost, LGBM, CatBoost, GridSearchCV, AutoGluon |
| [한예본](https://github.com/Yebonn-Han) | IRT 기반 Feature Engineering, Transformer, Graph, LSTM, SASREC,ML |

### 👨‍👧‍👦 Team 협업
### 📝 Ground Rule
#### 팀 규칙
- 모더레이터 역할
  - 순서 : 매일 돌아 가면서
  - 역할 : 피어세션 시 소개하고 싶은 곡 선정
- 데일리 스크럼
    - 오늘 학습 계획 정하기
    - Github PR 올린 것 코드리뷰 진행
- 피어세션
    - 모더레이터가 가져 온 노래 나올 때 각자 스트레칭 하기
    - 강의에 나오는 논문 리뷰하기
    - 미션 파일 코드 분석 발표하기
    - Github PR 올린 것 코드리뷰 진행

#### 깃 사용 규칙
1. 커밋 메세지 컨벤션 유다시티 스타일
2. 이슈 기반 작업
3. 깃허브 칸반 보드를 활용한 일정 관리
4. 데일리 스크럼/피어세션때 PR코드 리뷰 후 병합

<br>

# 프로젝트 개발 환경 및 기술 스택
## ⚙️ 개발 환경
- OS: Linux-5.4.0-99-generic-x86_64-with-glibc2.31
- GPU: Tesla V100-SXM2-32GB * 6
- CPU cores: 8

## 🔧 기술 스택
![](https://img.shields.io/badge/Pytorch-EE4C2C?style=flat-square&logo=Pytorch&logoColor=white)
![](https://img.shields.io/badge/jupyter-F37626?style=flat-square&logo=Jupyter&logoColor=white)
![](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=black)
![](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=Pandas&logoColor=white)
![](https://img.shields.io/badge/Numpy-013243?style=flat-square&logo=Numpy&logoColor=white)


![footer](https://capsule-render.vercel.app/api?type=waving&color=gradient&height=200&section=footer&)
