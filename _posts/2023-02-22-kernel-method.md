---
title:  "kernel method 01"
excerpt: "kernel method"

categories:
  - ml
tags:
  - [ml, data science]

use_math: true
toc: true
toc_sticky: true
 
date: 2023-02-23
last_modified_at: 2023-02-23
---
# Linear Models Need Big Feature Spaces
## Review
우리는 앞서 살펴봤듯이 **모델의 표현능력(expressivity)**이 커지기 위해서는 가설공간을 얼마나 키울 수 있느냐에 따라 달라지게 된다고 했습니다.<br>
이 때 큰 표현능력을 가진다면 모델이 더 **성능이** 좋아질 겁니다.<br>

성능이 좋기 위해서 큰 표현능력을 갖는 것이 목표이고, 큰 표현능력을 얻기 위해서는 풍부한 가설공간을 가져야한다고 했습니다.<br>
이 풍부한 가설공간을 가지기 위해서는 가성공간을 확장시켜야 하는데, 가설공간을 확장시키기 위해서는 feature mapping function$(φ)$에서 더 많은 feature를 추가해야합니다.<br>

더 많은 feature를 추가하게 되면 가설공간이 확장될 것이고 더 복잡한 모델을 학습시킬 수 있을 것입니다.<br>
### Linear Models Need Big Feature Spaces
선형 모델을 사용하여 풍부한 가설공간을 얻으려면 고차원의 feature space가 필요합니다.<br>

$x = (1,x_1,..., x_d ) ∈ R^{d+1} = X$

우리가 위와 같은 선형 모델로 시작한다고 가정하겠습니다.<br>
이 때 우리는 차수 $M$의 모든 다항식 $x^{p_1}_1···x^{p_d}_d, with\ p_1 +···+p_d = M$을 추가하고 싶습니다.<br>
그러면 최종적으로 몇 개의 feature가 생성이 될 까요?<br>

이 문제는 조합에서 flower shop problem으로 설명이 됩니다.<br>

> **flower shop problem** :
> 
> 꽃집 문제(Flower Shop Problem)는 조합론(Combinatorics)에서 자주 등장하는 문제 중 하나입니다.<br> 이 문제는 어떤 종류의 꽃이 주어졌을 때, 이 꽃들로 만들 수 있는 모든 다항식의 개수를 구하는 문제입니다.<br>
>
> 예를 들어, 빨간색, 파란색, 노란색 꽃이 있다고 가정해봅시다. 이 꽃들로 만들 수 있는 모든 두 항의 다항식을 생각해보겠습니다.<br> 이 경우 가능한 다항식은 다음과 같습니다.<br>
>
> 1. 빨간색 + 파란색
> 2. 빨간색 + 노란색
> 3. 파란색 + 노란색
> 4. 빨간색 x 파란색
> 5. 빨간색 x 노란색
> 6. 파란색 x 노란색
>
> 따라서, 이 경우 총 6개의 다항식을 만들 수 있습니다.<br>
꽃집 문제에서 꽃의 종류와 항의 개수가 더 많아지면, 다항식의 개수는 기하급수적으로 증가하게 됩니다.<br> 예를 들어, 빨간색, 파란색, 노란색, 초록색 꽃이 있고, 3항의 다항식을 만들 경우 총 20개의 다항식을 만들 수 있습니다.<br> 이러한 문제를 해결하기 위해 조합론적 공식인 "꽃집 문제"를 사용하여 다항식의 개수를 계산할 수 있습니다.<br>

이 flower shop problem에서의 식은 다음과 같습니다.<br>

$\begin{pmatrix} M+d-1 \\ M \end{pmatrix}$

예를 들어, $d = 40$이고 $M = 8$이면, 총 314457495개의 특징이 생성됩니다. 이는 굉장히 큰 행렬을 만들어내게 됩니다.<br>

### Big Feature Spaces












[*[출처] : FOUNDATIONS OF MACHINE LEARNING by Bloomberg ML EDU*](https://bloomberg.github.io/foml/#home).