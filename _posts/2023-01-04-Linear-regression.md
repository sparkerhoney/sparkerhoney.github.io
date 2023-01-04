---
title:  "[Data Mining] 선형 회귀 분석(linear regression)"
excerpt: "data mining 01 선형회귀분석(linear regression). "

categories:
  - dm
tags:
  - [dm, data science]

use_math: true
toc: true
toc_sticky: true
 
date: 2023-01-04
last_modified_at: 2023-01-04
---
# 회귀분석

**회귀분석**이란 연구대상이 되는 시스템에 존재하는 **변수들 사이에 함수적인 관계를 규명하기위해 수학적인 모형을 상정**하고, 이 모형을 수집된 자료로부터 추정하는 통계적인 기법이다.<br>
예를들어, 아파트 평수와 전기소모량의 관계, 아파트 평수에 따라서 전기 소모량을 예측하는 것 등이 있다.


![](https://user-images.githubusercontent.com/108461006/210494006-9fbce97c-1c1b-45e7-88df-1f66db2e2ab3.png){: width="400" height="400"}


위 회귀선(추세선)의 함수는 아래와 같다.

> **$\hat{y}=ax+b$**

이 때 회귀분석은 *ε(오차)의 제곱합*이 최소가 되도록 a와 b값을 추정하는 방법이다.

---
이 때 반응변수와 회귀변수로 변수들을 나눌 수 있는데 각각을 알아보자.
1. 반응변수(response variable) : 다른 변수의 영향을 받는 변수 [ 일반적으로 ' y ', 종속변수]
2. 회귀변수(regressor variable) : 다른 변수에 영향을 미치는 변수 [ 일반적으로 ' x ', 설명변수]
  
반응변수 y 가 회귀변수 x 에 영향을 받는다고 하자.<br>
대부분의 경우, 반응변수와 회귀변수 간에 이론적인 관계는 잘 알려져 있지 않으므로, 다음과 같이 다양한 식으로 관계를 근사화한다.
​
- $y=\beta_0+\beta_1x$
- $y=\beta_0+\beta_1x+\beta_2x^2$
- $y=\beta_0e^{\beta_1x}$
- $y=\beta_0+\beta_1x$

위와 같은 식으로 반응변수 y 를 실제로 관측해보면 제어할 수 없는 요인에 의해 $y=\hat{y}+\epsilon$ 
으로 관측된다. <br>
즉, ε 은 y 와 추정된 모형 ŷ 간의 차이를 나타내는 오차로서 평균은 0, 분산은 σ 2 인 확률 변수로 가정한다.