---
title: 다중 회귀 분석(multi linear regression)
layout: post
description: Lecture summary
use_math: true
post-image: https://github.com/sparkerhoney/sparkerhoney.github.io/blob/master/_images/statistics.jpg?raw=true
category: statistics
tags:
- Data Science
- Statistics
---

# 다중회귀분석

다중회귀분석은 앞서 설명했던 [선형회귀분석](https://sparkerhoney.github.io/dm/Linear-regression/)의 포괄적인 내용이다.<br>

---

## Vector/Matrix 접근방법

### 일반적인 형태

다루고자 하는 회귀 모형은 소위 선형 회귀모형이라고 불리며, 그 일반적인 형태는 다음과 같다.<br>
$$y_j=\beta_0+\beta_1x_{1j}+\beta_2x_{2j}+\cdots+\beta_px_{pj}+\epsilon_j,\quad (j=1,2,\cdots,n)$$<br>
$$\epsilon_j \sim NID(0,\sigma^2)$$<br>

즉, 일반 식에서 $$x_{pj}$$는 반드시 원래의 변수만을 의미하는 것이 아니다.<br>

### Vector/Matrix 표현

이를 vector/matrix형태로 나타내면, 위에서 보인매트릭스 형태이고 그에 관련된 식은 <br>
$$y=X\beta+\epsilon$$으로 나타낼 수 있다.<br>

---

## 최소제곱법에 의한 추정

### 최소제곱법

- 미지의 회귀계수 $$vector\beta$$는 다음 오차제곱합<br>
$$L=\displaystyle\sum_{j=1}^{n}\epsilon^2=\displaystyle\sum_{j=1}^{n}\{y_j-(\beta_0+\beta_1x_1+\cdots+\beta_px_{pj})\}^2$$이 **최소가 되도록 결정**한다.<br>
- 최소제곱법에 의한 $$\beta$$의 추정량<br>
$$b=(X^\prime X)^{-1}X^\prime y=$$<br>
$$
\begin{pmatrix}
   b_0 \\
   b_1 \\
   \vdots \\
   b_p \\
\end{pmatrix}
$$<br>

---

## 모형의 검토

### 모형의 타당성을 검토

1. 오차($$\epsilon_j$$)에 대한 가정의 검토
   - 독립성($$\epsilon_j$$은 서로 독립) $\rightarrow$ 잔차와 관측순서의 Plot
   - 정규성($$\epsilon_j$$가 정규분포를 따름) $\rightarrow$ 히스토그램 또는 정규확률지 Plot
   - 등분산성($$\epsilon_j$$의 분산이 모두 동일) $\rightarrow$ 잔차와 $$\hat{y}$$의 Plot<br>

2. 회귀식의 유의성 검토
   - 최소제곱법에 의해 구한 회귀식이 사용해도 좋을 만큼 유의한 것인가를 검토한다. $\rightarrow$ 전체 회귀식에 대한 유의성검정(분산분석)<br>

### 분산분석

- 총제곱합의 분해
  - 단순회귀모형에서와 마찬가지로 총제곱합은 아래와 같이 분해된다.<br>
  - $$\displaystyle\sum_{j=1}^{n}(y_j-\bar{y})^2=\displaystyle\sum_{j=1}^{n}(\hat{y}-\bar{y})^2+\displaystyle\sum_{j=1}^{n}(y-\hat{y})^2$$<br>
  - $$SSTO=SSR+SSE$$<br>
  - 단, $$\hat{y_j}=b_0+b_1x_{1j}+\cdots+b_px_{pj}$$<br>

- 분산분석표
  - |Source of Variation|Degree of Freedom(DF)|Sum of Squares(SS)|Mean Square(MS=Ss/DF)|$$F_0$$|
    |------|---|---|---|---|
    |Regression Error|$$p$$<br>$$n-p-1$$|$$SSR$$<br>$$SSE$$|$$MSR=\frac{SSR}{1}$$<br>$$MSE=\frac{SSE}{n-2}$$|$$\frac{MSR}{MSE}$$|
    |Total|$$n-1$$|$$SSTO$$|||<br>

- $$R^2$$ (결정계수) = $$\frac{SSR}{SSTP}=1-\frac{SSE}{SSTO}$$ $\rightarrow$ 변수가 많을수록 $$R^2$$값이 증가하게된다.<br>
- $$Adjusted\ R^2=1-\frac{n-1}{n-p-1}(1-R^2)=1-\frac{SSE/(n-p-1)}{SSTO/(n-1)}$$<br>
- $$Adjusted\ R^2$$는 $$R^2$$이 커진다고 해도 $$R^2$$가 억지로 커지는 것을 방지하는 것이다.<br>
- $$s^2=$$오차분산$$\sigma$$의 추정치 $$=MSE=\frac{SSE}{(n-p-1)}$$<br>

### 가설검정

- 전체 회귀식에 대한 검정
  - 회귀계수에 대해 먼저 아래와 같은 가설을 세운다.<br>
  - $$
    \begin{cases}
      H_0: \beta=(\beta_1,\beta_2,\cdots,\beta_p)=0\\
      H_1: not\ H_0
    \end{cases}
    $$<br>
  - 검정통계량 $$F_0=MSR/MSE$$는 $$H_0$$가 사실이 아니라는 가정하에서 자유도 $$(p, n-p-1)$$인 $$F$$분포를 따르며, 위 가설검정의 기각역은 다음과 같다.<br>
  - $$F_0=\frac{MSR}{MSE}>F(p,n-p-1;1-\alpha)$$<br>
  - 단 $$\alpha$$는 유의수준이다.<br>

---

*출처 : 명지대 산업경영공학과 김도현 교수님의 강의*
