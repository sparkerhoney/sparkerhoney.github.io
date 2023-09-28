---
layout: post
title: 선형회귀분석(linear regression)
description: 선형회귀분석 1st
summary: 
tags: statistics
minute: 1
---

# 회귀분석

**회귀분석**이란 연구대상이 되는 시스템에 존재하는 **변수들 사이에 함수적인 관계를 규명하기위해 수학적인 모형을 상정**하고, 이 모형을 수집된 자료로부터 추정하는 통계적인 기법이다.<br>
예를들어, 아파트 평수와 전기소모량의 관계, 아파트 평수에 따라서 전기 소모량을 예측하는 것 등이 있다.


![](https://user-images.githubusercontent.com/108461006/210494006-9fbce97c-1c1b-45e7-88df-1f66db2e2ab3.png){: width="400" height="400"}


위 회귀선(추세선)의 함수는 아래와 같다.

> **$\hat{y}=ax+b$**

이 때 회귀분석은 *ε(오차)의 제곱합*이 최소가 되도록 a와 b값을 추정하는 방법이다.


## 회귀분석에서 반응변수와 회귀분석의 관계
<br>

이 때 반응변수와 회귀변수로 변수들을 나눌 수 있는데 각각을 알아보자.
1. 반응변수(response variable) : 다른 변수의 영향을 받는 변수 [ 일반적으로 ' $y$ ', 종속변수]
2. 회귀변수(regressor variable) : 다른 변수에 영향을 미치는 변수 [ 일반적으로 ' $x$ ', 설명변수]
  
반응변수 $y$ 가 회귀변수 $x$ 에 영향을 받는다고 하자.<br>
대부분의 경우, 반응변수와 회귀변수 간에 이론적인 관계는 잘 알려져 있지 않으므로, 다음과 같이 다양한 식으로 관계를 근사화한다.
​
- $y=\beta_0+\beta_1x$
- $y=\beta_0+\beta_1x+\beta_2x^2$
- $y=\beta_0e^{\beta_1x}$
- $y=\beta_0+\beta_1x$

위와 같은 식으로 반응변수 $y$ 를 실제로 관측해보면 제어할 수 없는 요인에 의해 $y=\hat{y}+\epsilon$ 
으로 관측된다. <br>
즉, $ε$ 은 $y$ 와 추정된 모형 $ŷ$ 간의 차이를 나타내는 오차로서 평균은 0, 분산은 $\sigma^2$ 인 확률 변수로 가정한다.<br>
반응변수 $y$ 가 하나의 회귀변수 $x$ 에 대하여 다음과 같은 관계를 가질 때, 이를 **일차 단순 회귀 모형** 이라고 부른다.
- $y=\beta_0+\beta_1x+\epsilon$

## 최소제곱법(Least Squares Method)
<br>

위의 모형을 추정하기 위해서 최소제곱법 ( Least Squares Method ) 을 이용해 β 를 추정해보자.<br>

-  미지의 모수$\beta_0$과 $\beta_1$은 오차의 제곱합이 최소가 되도록 하는 값으로 추정하는데, 이러한 방법을 **최소제곱법**이라 한다.
>$L=\displaystyle\sum_{j=1}^{n}\epsilon_j^2=\displaystyle\sum_{j=1}^{n}(y_j-\beta_0-\beta_1x_j)^2$<br>
>$\frac{\partial{L}}{\partial{\beta_1}}=-2\displaystyle\sum_{j=1}^{n}x_j(y_j-\beta_0-\beta_1x_j)=0$<br>
>$\frac{\partial{L}}{\partial{\beta_0}}=-2\displaystyle\sum_{j=1}^{n}(y_j-\beta_0-\beta_1x_j)=0$<br>

- 최소제곱법에 의한 $β_0$ 와 $β_1$ 의 추정량 
>$\hat{\beta_1}=b_1=\frac{\displaystyle\sum_{j=1}^{n}(x_j-\bar{x})(y_j-\bar{y})}{\displaystyle\sum_{j=1}^{n}(x_j-\bar{x})^2}$<br>
>$\hat{\beta_0}=b_0=\bar{y}-b_1\bar{x}$

- 추정된 $b_0$ ( $β_0$ 를 추정한 값 ) 와 $b_1$ ( $β_1$ 을 추정한 값 ) 을 이용하여 반응변수 $y$ 에 대한 추정식 ( 회귀식 ) 을 다음과 같이 나타낼 수 있다.
>$\hat{y}=b_0+b_1x$

- $x = x_j$ 일 때의 반응변수 $y$ 의 추정값은 다음과 같다.
>$\hat{y}=b_0+b_1x_j$


이후 결과로 나온 **추정된 회귀식**과 **분산분석표**로 output을 해석한다.
## 모형의 타당성 검토
이제 **모형의 타당성을 검토**해본다. 이는 모형이 데이터를 얼마나 잘 설명하는 지를 검토하는 과정이다.<br>
- 첫째로 잔차 Plot에서 오차 가정의 타당성을 검토해본다.<br>
이때, **잔차**란 실제 데이터 Plot에서 실제 반응변수의 값과 추정된 반응변수 값의 차이를  나타낸다.
> 1. 오차에 대한 독립성 가정을 검토한다.
![오차 독립성 가정](https://user-images.githubusercontent.com/108461006/210522637-e1b3ab59-9da9-42e3-bcf0-c4d22aa5c714.png)<br>
위 처럼,  잔차들 간에 함수성을 가진다는 것을 잔차가 독립성을 위배한 것이다. <br>
따라서 잔차들 간에는 아무런 패턴이 존재해서는 안된다는 이야기다.
> 2. 히스토그램 또는 정규확률지 Plot에서 오차에 대한 정규성 가정을 검토한다.
​![오차의 정규성 가정](https://user-images.githubusercontent.com/108461006/210523281-192eb31f-985b-433c-bc35-16b50e20b1ed.png)<br>
위와 같이, 잔차들 간에는 정규성이 존재해야한다.
> 3. 잔차와 예측데이터의 Plot에서 오차에 대한 등분산성( 분산은 일정해야함 )을 검토한다.
![오차의 등분산성 검정](https://user-images.githubusercontent.com/108461006/210523502-378570b2-fb1f-4f67-8261-4ef837762c23.png)<br>
위 처럼, 등분산성이 일정해야만 가정이 타당하다고 말할 수 있다. 
- 두번째로, 분산분석에 의해서 모델의 유의성을 검토해본다.<br>
총제곱합(Total Sum of Squares, **SSTO**)을 회귀식에 의해 설명되는 변동(Regression Sum of Squares, **SSR**)과 회귀식에 의해 설명되지 않는 잔차변동(Residual Sum of Squares, **SSE**)로 분해할 수 있다.
> $(y-\bar{y})=(\hat{y}-\bar{y})+(y-\hat{y})$<br>
> $(y-\bar{y})^2=(\hat{y}-\bar{y})^2+(y-\hat{y})^2$<br>
> $\sum(y-\bar{y})^2=\sum(\hat{y}-\bar{y})^2+\sum(y-\hat{y})^2$<br>

따라서, **$SSTO = SSR + SSE$** 로 설명되어진다.<br>
이때 분산분석표는 아래와 같이 나타낼 수 있다.<br>
|Source of Variation|Degree of Freedom(DF)|Sum of Squares(SS)|Mean Square(MS=Ss/DF)|
|------|---|---|---|
|Regression Error|$1$<br>$n-2$|$SSR$<br>$SSE$|$MSR=\frac{SSR}{1}$<br>$MSE=\frac{SSE}{n-2}$|
|Total|$n-1$|$SSTO$||

Degree of Freedom (DF) 는 자유도고 칭하는데 이는 변수 또는 데이터 한 개당 제곱합이 얼마나 해당되는지에 대한 관여하는 정도를 나타내는 지표이다. 

$MSE=$오차분산 $σ_2$ 의 추정치 $=s^2$ 이라고 한다.

결정계수 $R^2$ (coefficient of determination) 는 전체 변동 중 회귀식에 의해 설명되는 변동의 비율이다. 
<br>이는 모델이 데이터를 잘 설명했는지 설명을 못했는지 판단이 가능하게 한다. ( 1에 가까울 수록 잘 설명하고 있다고 할 수 있다. )

$R^2=\frac{SSR}{SSTO}=1-\frac{SSE}{SSTO}=\frac{SSR}{SSE}=\frac{MSR}{MSE}$

 *출처 : 명지대 산업경영공학과 김도현 교수님의 강의*


