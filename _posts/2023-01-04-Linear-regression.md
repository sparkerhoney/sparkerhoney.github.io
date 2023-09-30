---
title: 선형 회귀 분석(linear regression)
layout: post
description: Lecture summary
use_math: true
post-image: https://github.com/sparkerhoney/sparkerhoney.github.io/blob/master/_images/statistics.jpg?raw=true
category: statistics
tags:
- Data Science
- Statistics
---

# 회귀분석

**회귀분석**이란 연구대상이 되는 시스템에 존재하는 **변수들 사이에 함수적인 관계를 규명하기위해 수학적인 모형을 상정**하고, 이 모형을 수집된 자료로부터 추정하는 통계적인 기법이다.<br>
예를들어, 아파트 평수와 전기소모량의 관계, 아파트 평수에 따라서 전기 소모량을 예측하는 것 등이 있다.<br>

---

![](https://user-images.githubusercontent.com/108461006/210494006-9fbce97c-1c1b-45e7-88df-1f66db2e2ab3.png){: width="400" height="400"}<br>

---

## 회귀선의 함수

위 회귀선(추세선)의 함수는 아래와 같다.<br>

$$\hat{y}=ax+b$$ <br>

이 때 회귀분석은 *$$ε$$(오차)의 제곱합*이 최소가 되도록 a와 b값을 추정하는 방법이다.<br>

---

## 회귀분석에서 반응변수와 회귀분석의 관계

이 때 반응변수와 회귀변수로 변수들을 나눌 수 있는데 각각을 알아보자.<br>

1. 반응변수(response variable) : 다른 변수의 영향을 받는 변수 일반적으로 ' $$y$$ ', 종속변수
2. 회귀변수(regressor variable) : 다른 변수에 영향을 미치는 변수 일반적으로 ' $$x$$ ', 설명변수

반응변수 $$y$$ 가 회귀변수 $$x$$ 에 영향을 받는다고 하자.<br>
대부분의 경우, 반응변수와 회귀변수 간에 이론적인 관계는 잘 알려져 있지 않으므로, 다음과 같이 다양한 식으로 관계를 근사화한다.

$$y=\beta_0+\beta_1x$$ <br>
$$y=\beta_0+\beta_1x+\beta_2x^2$$ <br>
$$y=\beta_0e^{\beta_1x}$$ <br>
$$y=\beta_0+\beta_1x$$ <br>

위와 같은 식으로 반응변수 $$y$$ 를 실제로 관측해보면 제어할 수 없는 요인에 의해 $$y=\hat{y}+\epsilon$$ 
으로 관측된다. <br>
즉, $$ε$$ 은 $$y$$ 와 추정된 모형 $$ŷ$$ 간의 차이를 나타내는 오차로서 평균은 0, 분산은 $$\sigma^2$$ 인 확률 변수로 가정한다.<br>
반응변수 $$y$$ 가 하나의 회귀변수 $$x$$ 에 대하여 다음과 같은 관계를 가질 때, 이를 **일차 단순 회귀 모형** 이라고 부른다.

$$y=\beta_0+\beta_1x+\epsilon$$ <br>

---

## 최소제곱법(Least Squares Method)

위의 모형을 추정하기 위해서 최소제곱법 ( Least Squares Method ) 을 이용해 β 를 추정해보자.<br>

-  미지의 모수$$\beta_0$$과 $$\beta_1$$은 오차의 제곱합이 최소가 되도록 하는 값으로 추정하는데, 이러한 방법을 **최소제곱법**이라 한다.
  
$$L=\displaystyle\sum_{j=1}^{n}\epsilon_j^2=\displaystyle\sum_{j=1}^{n}(y_j-\beta_0-\beta_1x_j)^2$$<br>
$$\frac{\partial{L}}{\partial{\beta_1}}=-2\displaystyle\sum_{j=1}^{n}x_j(y_j-\beta_0-\beta_1x_j)=0$$<br>
$$\frac{\partial{L}}{\partial{\beta_0}}=-2\displaystyle\sum_{j=1}^{n}(y_j-\beta_0-\beta_1x_j)=0$$<br>

- 최소제곱법에 의한 $$β_0$$ 와 $$β_1$$ 의 추정량 

$$\hat{\beta_1}=b_1=\frac{\displaystyle\sum_{j=1}^{n}(x_j-\bar{x})(y_j-\bar{y})}{\displaystyle\sum_{j=1}^{n}(x_j-\bar{x})^2}$$<br>
$$\hat{\beta_0}=b_0=\bar{y}-b_1\bar{x}$$

- 추정된 $$b_0$$ ( $$β_0$$ 를 추정한 값 ) 와 $$b_1$$ ( $$β_1$$ 을 추정한 값 ) 을 이용하여 반응변수 $$y$$ 에 대한 추정식 ( 회귀식 ) 을 다음과 같이 나타낼 수 있다.
  
$$\hat{y}=b_0+b_1x$$ <br>

- $$x = x_j$$ 일 때의 반응변수 $$y$$ 의 추정값은 다음과 같다.
  
$$\hat{y}=b_0+b_1x_j$$ <br>

---

## 모형의 타당성 검토

이제 **모형의 타당성을 검토**해본다. 이는 모형이 데이터를 얼마나 잘 설명하는 지를 검토하는 과정이다.<br>

- 첫째로 잔차 Plot에서 오차 가정의 타당성을 검토해본다.<br>
이때, **잔차**란 실제 데이터 Plot에서 실제 반응변수의 값과 추정된 반응변수 값의 차이를  나타낸다.

1. 오차에 대한 독립성 가정을 검토한다.<br>
![오차 독립성 가정](https://user-images.githubusercontent.com/108461006/210522637-e1b3ab59-9da9-42e3-bcf0-c4d22aa5c714.png)<br>

2. 히스토그램 또는 정규확률지 Plot에서 오차에 대한 정규성 가정을 검토한다.<br>
![오차의 정규성 가정](https://user-images.githubusercontent.com/108461006/210523281-192eb31f-985b-433c-bc35-16b50e20b1ed.png)<br>

3. 잔차와 예측데이터의 Plot에서 오차에 대한 등분산성( 분산은 일정해야함 )을 검토한다.<br>
![오차의 등분산성 검정](https://user-images.githubusercontent.com/108461006/210523502-378570b2-fb1f-4f67-8261-4ef837762c23.png)<br>

---

- 두번째로, 분산분석에 의해서 모델의 유의성을 검토해본다.<br>
총제곱합(Total Sum of Squares, **SSTO**)을 회귀식에 의해 설명되는 변동(Regression Sum of Squares, **SSR**)과 회귀식에 의해 설명되지 않는 잔차변동(Residual Sum of Squares, **SSE**)로 분해할 수 있다. <br>

$$SSTO = SSR + SSE$$ <br>

따라서, **$$SSTO = SSR + SSE$$** 로 설명되어진다.<br>

---

*출처 : 명지대 산업경영공학과 김도현 교수님의 강의*
