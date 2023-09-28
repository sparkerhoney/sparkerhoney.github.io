---
title:  "Maximum Likelihood Estimation 2nd"
excerpt: "Maximum Likelihood Estimation"

categories:
  - Statistics
tags:
  - [Statistics]

use_math: true
toc: true
toc_sticky: true
 
date: 2023-03-14
last_modified_at: 2023-03-14
---
의사결정나무에서 분기를 위해 사용하는 손실 함수에 대해 설명드리겠습니다.

분류에서 자연적인 손실 함수는 0/1 손실입니다. 하지만 이것은 가장 좋은 분할을 찾는 데 사용하기에는 계산적으로 힘들 수 있습니다. 따라서 대안적인 손실 함수가 사용됩니다.

여러 가지 손실 함수 중 가장 많이 사용되는 것은 지니 손실 함수입니다. 지니 손실 함수는 분류 오류 확률을 최소화하는 대신, 불순도를 최소화하려고 시도합니다. 불순도란 노드 내의 데이터가 서로 다른 클래스에 속한 정도를 측정하는 척도입니다.

노드 m의 불순도를 나타내는 지니 불순도는 다음과 같이 정의됩니다.

$I(m)=\sum_{k=1}^{K} \hat{p}{m k}(1-\hat{p}{m k})$

여기서 $\hat{p}_{mk}$는 노드 m에서 클래스 k의 비율을 나타냅니다.

지니 손실 함수는 다음과 같이 정의됩니다.

$H(X_m)=-\sum_{k=1}^{K} \hat{p}{mk} \log_2(\hat{p}{mk})$

이 함수는 불순도가 낮을수록 작은 값을 갖습니다. 따라서 나무 분할에서는 지니 불순도가 감소하는 방향으로 분할을 진행합니다.
[*[출처] : FOUNDATIONS OF MACHINE LEARNING by Bloomberg ML EDU*](https://bloomberg.github.io/foml/#home).