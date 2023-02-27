---
title:  "Kernel Methods 02"
excerpt: "Kernel Methods"

categories:
  - ml
tags:
  - [ml, data science]

use_math: true
toc: true
toc_sticky: true
 
date: 2023-02-27
last_modified_at: 2023-02-27
---
# Kernel Methods
## Review
자, 그럼 지금까지의 내용을 정리해 봅시다.<br>

우선 **선형(linear) svm** 은 초평면(hyperlane)으로 결정경계(decision boundary)를 나누어서 $yf(x)$인 **마진(margin)** 을 **최대화 시키는 방향으로 최적화** 를 진행해 나갑니다.<br>
그에 따른 dual에서의 $\alpha^*, w^*, b^*$등을 구해 초평면을 구해 $y_{new}$의 class를 최종적으로 도출해내는 알고리즘이라 할 수 있습니다.<br>

그렇다면 **비선형(non-linear)의 svm**에서는 어떤 방식으로 우리 $y_{new}$의 class를 얻을 수 있을까요?<br>

전 시간에 다루었던 [**Feature Extraction**](https://sparkerhoney.github.io/ml/Feature-Extraction-01/) 으로 설명드릴 수 있을 것 같습니다.<br>
feature extraction은 **매핑함수(mapping function) $\psi$** 를 통해서 더 높은 차원의 공간에 매핑(mapping)이 될 수 있을 것입니다.<br>

그림으로 예시를 들어보겠습니다.<br>

![images_lolhi_post_ba67b779-4322-4535-916b-22d9cbac666b_image](https://user-images.githubusercontent.com/108461006/221499589-f5a2828f-bb25-4369-9ca1-f929d8bfed73.png)

위의 그림에서 왼쪽에 있는 그래프는 **선형적으로** 결정경계를 그릴 수 없는데 mapping function을 통해서 **고차원의 feature space** 에 mapping을 시키게 되면 (그림에서는 $dim 1 \rightarrow dim 2$) 결정경계를 그릴 수 있게 됩니다.<br>

하지만 이런 feature extraction에서도 문제는 있었습니다.<br>
지난 시간에도 다루었던 내용입니다.<br>

![image](https://user-images.githubusercontent.com/108461006/220607091-bb0017bb-48c6-4d5e-8c94-b0e64e2712c2.png)

feature extraction을 통해 더 많은 feature를 추가하게되면 우리는 가설공간이 확장됨을 알 수 있습니다.<br>
그렇게 되면 더욱 복잡한 모델을 학습시킬 수 있을 겁니다.<br>
이때 이 가설공간이 확장되면 더욱 더 **풍부한 모델의 표현능력** 을 가진다고 얘기 했었는데 이렇게 됐을 때 2가지의 문제점이 생겼습니다.<br>

1. 과적합
2. 메모리 및 계산 비용

과적합은 일전 강의에서 **regularization** 과 같은 방식으로 해결이 가능하다고 했지만,<br>
메모리 및 계산 비용을 줄일 수 있는 방식은 배우지 못했습니다.<br>

지금부터 비선형의 svm을 다룰 때 메모리나 계산비용을 줄일 수 있는 방법인 **kernel methods** 에 대한 이야기를 하겠습니다.

## Kernel Methods: Motivation
### Review: Linear SVM and Dual
 
계속해서 지난시간 부터 이야기 해 온 feature extraction에 대한 내용입니다.<br>

$\underset{w∈R^d,b∈R}{\min}\frac{1}{2}\begin{Vmatrix} w \end{Vmatrix}^2+\frac{c}{n}∑_{i=1}^n(1-y_i[w^Tψ(x_i)+b])_+$

위의 식은 비선형 svm을 풀어나갈 때 $x$를 **고차원의 feature space** 에 mapping해준 **primal** 식입니다.<br>
이 식은 **dual** 로 표현한다면 다음과 같이 표현할 수 있습니다.<br>

$\underset{\alpha}{\sup}⁡∑_{i=0}^nα_i-\frac{1}{2}∑_{i,j=1}^nα_i α_j y_i y_j \psi(x_i)^T \psi(x_j)$

$s.t. ∑_{i=1}^nα_i y_i =0$

$α_i∈[0,\frac{c}{n}],\ i=1,…,n$

위 식에서 주의 깊게 봐야하는 내용은 매핑함수간에 유사도를 구하는 $\psi(x_i)^T \psi(x_j)$해당 부분입니다.<br>
이는 내적으로 구성되어있으며 각각 매핑함수를 따로 구한 뒤에 내적해야하는 계산비용이 기하급수적으로 올라간다고 생각할 수 있습니다.<br>

따라서 우리는 kernel methods를 이용해서 이를 해결해보겠습니다.

### Some Methods Can Be “Kernelized”
input data가 내적으로 표현될 수 있다는 이야기는 "**kernelized**" 될 수 있다는 것을 나타냅니다.<br>
즉, input data를 feature space로 매핑하는 것이 아니라 input data 쌍들 간의 내적으로만 계산할 수 있는 경우를 이야기합니다.<br>

$k(x, x') = \langle \psi(x), \psi(x') \rangle$

이렇게 "**kernelized**"된 경우에 위와 같이 kernel 함수 $k(x, x')$로 대체 할 수 있습니다.<br>

이것은 feature space가 매우 큰 경우에 유용합니다.<br>
**명시적으로 계산하는 것보다 훨씬 더 빠르게 계산을 할 수 있기 때문** 입니다.<br>
이를 우리는 **Kernel Trick** 이라 부릅니다.<br>

### Kernel Evaluation Can Be Fast
커널 트릭을 사용한 svm은 고차원 공간에 있는 feature vector들의 내적을 계산해야합니다.<br>
이는 

[*[출처] : FOUNDATIONS OF MACHINE LEARNING by Bloomberg ML EDU*](https://bloomberg.github.io/foml/#home).