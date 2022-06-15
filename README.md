# RecSys16(사분의 오) DKT 프로젝트 저장소
![image](./images/dkt.png)

## 프로젝트 소개
### 개요
프로젝트는 주어진 Iscream Edu Dataset의 유저별 마지막 문제의 정오 유무를 예측하는 것이다. Dataset은 Train set과 Test
set으로 구성되어 있으며, 약 226만 개의 데이터가 사용자를 기준으로 90/10의 비율로 나누어져 있다. Dataset의 메인 정
보로는 7,442 명의 사용자(userID)와 9,454개의 고유 문항(assessment ID)의 정오 유무(answerCode)가 있으며, feature
engineering에 활용될 수 있는 부가 정보로는 3가지로 timestamp(문제를 풀기 시작한 시각), knowledgeTag(고유 태그 번
호), TestID(시험지 번호)로 구성되어 있다
### 데이터 수집 및 생성
#### 수집
구글 설문을 이용해 188개의 이미지를 수집했고, 7개를 제외하고 학습에 52개는 학습 과정에서 활용하고 129개는 모델 성능 평가에 활용했다.
#### 생성
필수정보(한글 이름, 직책과 소속), 부가정보(전화번호, 팩스번호, 핸드폰번호, 이메일, 주소), 그 외의 정보(회사명, 영어이름, 웹 주소, 로고, 구분자)를 랜덤으로 생성 후, 제작한 형식에 맞춰 명함 이미지를 생성했다. 명함 이미지에 사용한 정보들은 BIO Tag를 함께 생성하여 AI 모델 학습에 사용했다.
<img src="md_res/1.png" width="500">
### 모델
Rule 기반 모델과 AI 모델을 각각 제작해 두 모델의 성능을 평가하고, 각각의 장단점을 확인하였다.
<img src="md_res/2.png" width="500">
#### Rule 기반 모델
OCR API 결과에서 텍스트들을 grouping하는 과정을 거친 후, 각 텍스트의 카테고리를 분류하는 모델을 개발했다. 카테고리 사이의 간격이 짧다면 다른 카테고리의 단어가 묶인 경우에는 묶인 단어의 일부가 한 카테고리에 해당되면 다른 부분을 다시 Rule 기반 모델로 검사하는 반복 작업을 만들어 해결하였다.
#### AI(KoBERT) 기반 모델
OCR API로 추출한 텍스트들을 한 줄로 직렬화하고, 각 텍스트에 대해서 미리 정의한 카테고리로 분류하는 문제로 접근할 수 있다고 생각해 개체명 인식(NER; Named Entity Recognition)으로 해결하기 위해 모델을 설계했다. 개체명 인식 문제로 해결하는 방식 중 보편적인 BIO Tagging 방식을 채택했고, 분류 태그는 총 9개 태그로 구분하도록 했다.

<img src="md_res/3.png" width="200">

### 성능 평가 및 비교
평가 지표는 OCR API의 오류를 감안해 CER 기준 0.2 내의 오차는 수용하는 보정된 F1 score를 사용했다.

<img src="md_res/4.png" width="200">

### 서비스 시연
<img src="md_res/5.png" width="500">

## 팀 역할
| [ ![구창회](https://avatars.githubusercontent.com/u/63918561?v=4) ](https://github.com/sonyak-ku) | [ ![김지원](https://avatars.githubusercontent.com/u/97625330?v=4) ](https://github.com/Jiwon1729) | [ ![전민규](https://avatars.githubusercontent.com/u/85151359?v=4) ](https://github.com/alsrb0607) | [ ![정준우](https://avatars.githubusercontent.com/u/39089969?v=4) ](https://github.com/ler0n) |
|:----------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------:|
|                             [ 구창회 ](https://github.com/sonyak-ku)                              |                             [ 김지원 ](https://github.com/Jiwon1729)                              |                              [ 전민규 ](https://github.com/alsrb0607)                             |                              [ 정준우 ](https://github.com/ler0n)                             |
|                             LGBM, LightGCN embedding LSTM-attn 연결                              |                        Ultragcn, Rank based Ensemble, 실험 결과 예측 code 작성                         |                               Tabnet, data augmentation, voting ensemble                              |                                LSTM-attn 개선, Feature Engineering 진행                                

## Repo 구조

```
ai_model # AI 모델 구현 관련 폴더
├─ dataset.py
├─ model.py
├─ main.py
├─ utils.py
└─ ner_utils.py
```

## 최종 순위 및 결과

|리더보드| auroc  |     순위     |
|:--------:|:------:|:----------:|
|public| 0.8186 |  **11등**   |
|private| 0.8214 | **최종 12등** |

![image](./images/private.png)
