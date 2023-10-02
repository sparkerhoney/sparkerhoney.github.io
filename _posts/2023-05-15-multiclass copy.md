---
title: Multiclass
layout: post
description: Lecture summary
use_math: true
post-image: https://github.com/sparkerhoney/sparkerhoney.github.io/blob/master/_images/machine%20learning.png?raw=true
category: machine learning
tags:
- Data Science
- machine learning
---

# Multiclass
## Introduction
### One-vs-All / One-vs-Rest
우리는 지금까지 binary한 분류에 대해서 공부해왔는데 해당 게시물에서는 다중 분류에 대한 공부내용을 정리할 것 입니다.<br>

![image](https://github.com/sparkerhoney/sparkerhoney.github.io/assets/108461006/a404cb0a-4276-4a18-91f2-2c9e741dbfa3)<br>
해당 다중분류는 One-vs-All로서 각 class에 대해 하나의 이진분류기를 훈련시킵니다.<br>

각 이진 분류기는 해당 클래스와 나머지 클래스들을 구별할 수 있도록 훈련됩니다.<br> 예를 들어, i번째 분류기는 i번 클래스와 나머지 클래스들을 구분할 수 있는 능력을 갖도록 훈련됩니다.<br>

$h_1, ..., h_k: X → R$으로 표현되는 이진 분류기들을 가정해봅시다.<br> 이진 분류기는 각각 입력 $x$에 대해 ${-1, 1}$사이의 이진 분류 결과나 실수 범위의 점수를 출력할 수 있습니다.<br>

최종 예측은 다음과 같이 이루어집니다:<br>

$$h(x) = \text{{argmax}}_{i \in \{1, \ldots, k\}} h_i(x)$$<br>

이 때, 여러 개의 이진 분류기가 동일한 점수를 가질 수 있는데, 이 경우 임의로 하나를 선택하여 결정합니다.<br>

이 방법은 다중 클래스 문제를 다루는 간단하면서도 효과적인 방법 중 하나입니다.<br> 이진 분류기를 k개 훈련시키는 것이 비교적 쉽기 때문에 대규모의 다중 클래스 문제에서도 적용 가능합니다.<br>
---
### Linear Classifers: Binary and Multiclass
선형 이진 분류기는 입력 공간 $X$ (여기서 $X = ( \mathbb{R}^d )$)에서 출력 공간 $Y = {-1, 1}$로의 매핑을 수행합니다.<br>

선형 분류기의 점수 함수 (score function)는 다음과 같이 정의됩니다:<br>
```
f(x) = <w, x> = w^T x
```


여기서 $w$는 분류기의 가중치 벡터이고, $x$는 입력 벡터입니다.<br> 이 함수는 입력 $x$에 대한 점수를 계산합니다.<br>

최종 분류 예측은 $\text{sign}(f(x))$의 부호에 따라 이루어집니다:<br>

- $\text{sign}(f(x)) = +1$: 입력 $x$는 양성 클래스에 속한다는 것을 의미합니다.<br>
- $\text{sign}(f(x)) = -1$: 입력 $x$는 음성 클래스에 속한다는 것을 의미합니다.<br>

기하학적으로, $\text{sign}(f(x)) = +1$인 경우와 $\text{sign}(f(x)) = -1$인 경우는 다음과 같습니다:<br>

- $\text{sign}(f(x)) = +1$: 입력 $x$가 가중치 벡터 $w$와 양의 각도를 이루는 방향에 위치합니다.<br> 즉, 입력 $x$와 가중치 벡터 $w$ 사이의 내적이 양수입니다.<br>
- $\text{sign}(f(x)) = -1$: 입력 $x$가 가중치 벡터 $w$와 음의 각도를 이루는 방향에 위치합니다.<br> 즉, 입력 $x$와 가중치 벡터 $w$ 사이의 내적이 음수입니다.<br>

따라서, 선형 이진 분류기는 입력 \( x \)와 가중치 벡터 \( w \) 사이의 내적을 계산하고, 그 결과에 따라 양성 클래스 (+1) 또는 음성 클래스 (-1)로 분류합니다.<br>



[*출처 : FOUNDATIONS OF MACHINE LEARNING by Bloomberg ML EDU*](https://bloomberg.github.io/foml/#home).
