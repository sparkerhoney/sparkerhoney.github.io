---
title:  "[FiFa_competition] 1화 데이터 불러오기 및 모델 만들기"
excerpt: "[FiFa_competition] 01"

categories:
  - Copetition
tags:
  - [Copetition]

use_math: true
toc: true
toc_sticky: true
 
date: 2023-02-19
last_modified_at: 2023-02-19

--- 
# Introduction
겨울방학이 거의 끝나갈 무렵인 지금, [dacon.io](https://dacon.io/)에서 상시로 열리고 있는 FIFA 선수 이적료 예측 경진대회에 참가하고자 합니다.<br>
FIFA 선수 이적료 예측 경진대회는 데이터 분석과 머신 러닝 기술을 활용하여 선수 이적료를 예측하는 대회로, 데이터 분석 및 머신 러닝에 관심이 있는 분들에게 좋은 기회가 될 것입니다.<br>
참가자들은 경진대회에서 제공되는 데이터를 기반으로, 이적료를 예측하는 머신 러닝 모델을 구현하고, 정확도를 높이는 것이 목표입니다. 경진대회에 참가하여 데이터 분석과 머신 러닝을 실전에서 적용해보는 좋은 기회가 될 것입니다.<br>

# Data Set
```python
import pandas as pd
import numpy as np
```
우선 위 코드는 데이터 분석 및 조작을 위한 라이브러리인 pandas와 수치 계산을 위한 라이브러리인 numpy를 불러오는 코드입니다.<br>
따라서, 데이터 분석 및 조작을 위해 pandas와 수치 계산을 위해 numpy를 불러온 것입니다.<br> 이후에는 pandas와 numpy에서 제공하는 함수를 사용하여 데이터를 다루게 됩니다.<br>
```python
train = pd.read_csv('FIFA_train.csv',encoding = 'utf-8') # train data
test = pd.read_csv("FIFA_test.csv",encoding = 'utf-8') # test data
```
위 코드로 대회에서 제공하고 있는 데이터셋을 pd.read_csv을 활용하여 불러오게 됩니다.<br>
```python
train.head()
```
*.head()* method를 통해 데이터를 확인해봅니다.<br>

![1111](https://user-images.githubusercontent.com/108461006/219954874-23d46f06-f385-4c70-9969-59b4fa6807fe.jpg)

우리는 다음과 같은 데이터를 확인 할 수 있게됩니다.<br>
## data columns
- 연속형
1. age : 나이<br>
2. stat_overall : 선수의 현재 능력치 입니다.<br>
3. stat_potential : 선수가 경험 및 노력을 통해 발전할 수 있는 정도입니다.<br>
4. value(예측값) : FIFA가 선정한 선수의 이적 시장 가격 (단위 : 유로) 입니다<br>
- 이산/범주형
1. continent : 선수들의 국적이 포함되어 있는 대륙입니다<br>
2. contract_until(전처리 필요) : 선수의 계약기간이 언제까지인지 나타내어 줍니다<br>
3. position : 선수가 선호하는 포지션입니다. ex) 공격수, 수비수 등<br>
4. prefer_foot : 선수가 선호하는 발입니다. ex) 오른발<br>
5. reputation : 선수가 유명한 정도입니다. ex) 높은 수치일 수록 유명한 선수<br>
6. stat_skill_moves : 선수의 개인기 능력치 입니다.<br>
- 정리에서 제외한 변수
1. id : 선수 고유의 아이디<br>
2. name : 이름<br>

train data의 column 이름을 확인했다면 test data의 column을 확인해보자 우리는 우리의 $y_{data}$인 선수가치('value')가 없는 data이길 기대합니다.

![스크린샷 2023-02-19 오후 11 40 22](https://user-images.githubusercontent.com/108461006/219955207-7569373e-adfb-4bf1-92d9-9a15a1e41b52.jpg)
```python
train.describe()
```
위의 코드는 데이터프레임의 통계 정보를 보여주는 함수입니다.<br> 데이터프레임에 포함된 숫자형 열(컬럼)들의 기술 통계를 보여주며, 각 열의 개수, 평균값, 표준편차, 최소값, 25/50/75% 분위수, 최대값 등을 보여줍니다.<br>

이 함수를 사용하면 데이터셋의 전반적인 분포와 범위를 파악할 수 있어 데이터를 탐색하는 데 도움이 됩니다.<br> 
예를 들어, describe()를 통해 각 열의 최대값이나 최소값이 이상치(outlier)인지 확인할 수 있고, 평균과 중앙값의 차이가 큰 열이 있다면 데이터의 분포가 왜도가 있음을(skewed) 있을 수 있음을 알 수 있습니다.<br>

![스크린샷 2023-02-19 오후 11 50 00](https://user-images.githubusercontent.com/108461006/219955688-a1b2134c-9302-4b02-9469-bd5804190dee.jpg)

위와 같은 통계적 수치로 우리는 추후에 예측이나 정확한 예측에 필요한 파생변수등을 도출해낼 수 있습니다.<br>

데이터 전처리 과정 중에서 가끔 빠지게 되는 아차싶은 오류가 있는데 그것은 데이터의 이상치나 NaN값(즉, missing value)을 확인하지 않고 지나가는 부분이 있습니다.<br>
따라서 우리는 지금부터 우리의 데이터가 가지고 있는 NaN값을 찾아나갈 함수를 써서 확인해 볼 것입니다.<br>

```python
train.isnull().sum()
```
위의 함수를 사용하게 된다면 출력값은 다음과 같습니다.<br>
<img src="https://user-images.githubusercontent.com/108461006/219956089-99899241-b9be-4a52-9376-c65e71877e80.jpg" width="200" height="200">

우리의 데이터에는 NaN값이 없다는 사실이 밝혀졌으니 좀 더 진도를 나갈 수 있게 됐습니다.<br>

그 다음으로는 데이터 시각화에 관련한 내용입니다.<br>

## Data Visualization
제가 데이터를 확인할 때 가장 먼저 해보는 작업인 상관관계 heatmap 분석입니다.<br>
```python
#heatmap으로 상관관계를 표시
import matplotlib.pyplot as plt
import seaborn as sb
plt.rcParams["figure.figsize"] = (5,5)
sb.heatmap(train.corr(),
           annot = True, #실제 값 화면에 나타내기
           cmap = 'Greens', #색상
           vmin = -1, vmax=1 , #컬러차트 영역 -1 ~ +1
          ) 
```
위의 코드를 입력했을 때 출력값은 다음과 같습니다.<br>

![스크린샷 2023-02-20 오전 12 04 37](https://user-images.githubusercontent.com/108461006/219956507-41c6971e-c41d-4618-be5a-e0f56332e7d7.jpg)

heatmap은 다음과 같이 출력되게 됩니다.<br>
이를 통해서 'reputation', 'stat_overall',	'stat_potentia' 변수들이 우리의 $y_{data}$인 value와 상당히 높은 상관관계를 가지는 것을 알 수 있게 됐습니다!<br>
상관관계를 보고 유의미한 변수를 찾는 것, insight를 찾아보는 것은 data science를 공부 할 때 상당히 중요한 작업입니다.<br>
다음으로는 처음 data의 column을 확인할 때 정리에서 제외한 변수 두개를 빼주는 함수를 작성해보겠습니다.<br>
```python
train = train.drop(['id', 'name'], axis=1)
test = test.drop(['id', 'name'], axis=1)
```
위의 코드를 통해서 우리는 'id', 'name'을 제거했습니다.<br>
그 다음 내용은 위에서 불러온 데이터들을 전처리 하는 내용입니다.<br>