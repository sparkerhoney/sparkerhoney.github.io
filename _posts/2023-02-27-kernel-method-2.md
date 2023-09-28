---
layout: post
title: Kernel Methods 2nd
description: Kernel Methods 2nd
summary: 
tags: machine learning
minute: 1
---

# Kernel Methods
## Review
자,  그럼 지금까지의 내용을 정리해 봅시다.<br>

우선 **선형(linear) svm** 은 초평면(hyperlane)으로 결정경계(decision boundary)를 나누어서 $yf(x)$인 **마진(margin)** 을 **최대화 시키는 방향으로 최적화** 를 진행해 나갑니다.<br>
그에 따른 dual에서의 $\alpha, w, b$등을 구해 초평면을 구해 $y_{new}$의 class를 최종적으로 도출해내는 알고리즘이라 할 수 있습니다.<br>

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

$\underset{w∈R^d,b∈R}{\min}\frac{1}{2}\begin{Vmatrix} w \end{Vmatrix}^2 + \frac{c}{n}∑_{i=1}^n(1-y_i[w^Tψ(x_i)+b])$

위의 식은 비선형 svm을 풀어나갈 때 $x$를 **고차원의 feature space** 에 mapping해준 **primal** 식입니다.<br>
이 식은 **dual** 로 표현한다면 다음과 같이 표현할 수 있습니다.<br>

$\underset{\alpha}{\sup}⁡∑_{i=0}^nα_i-\frac{1}{2} ∑_{i,j=1}^nα_i α_j y_i y_j \psi(x_i)^T \psi(x_j)$

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

### Kernel Function Calculation
1. **explicit 계산** : input data를 mapping function에 대입하여 각각의 feature map을 계산하고, 이를 내적해서 kernel의 값을 구합니다.
2. **implicit 계산** : 바로 kernel function의 값을 내적으로 계산합니다.

이것은 feature space가 매우 큰 경우에 유용합니다.<br>
**명시적으로 계산하는 것보다 훨씬 더 빠르게 계산을 할 수 있기 때문** 입니다.<br>
이를 우리는 **Kernel Trick** 이라 부릅니다.<br>

### Kernels as Similarity Scores
커널 트릭을 사용한 svm은 고차원 공간에 있는 feature vector들의 내적을 계산해야합니다.<br>
커널 함수를 **유사도 점수(similarity score)** 로 생각하는 것은 유용한 접근 방법입니다.<br> 하지만 이는 엄밀한 수학적인 정의는 아닙니다.<br> 유사도 점수를 정의하는 다양한 방법이 있기 때문입니다.<br> 따라서 우리는 일부 **특징 공간(feature space)** 에서 **내적(inner product)** 과 대응되는 **Mercer 커널** 을 사용할 것입니다.<br>

최종적으로 정리를 해보자면 **kernel method** 를 사용하게 된다면 다음과 같은  **장점** 이 있습니다.<br>
1. **계산상의 이점** : 특징 공간의 차원이 샘플 수보다 큰 경우에 유용합니다.<br> 이 경우, 커널 트릭을 사용하여 특징 맵을 명시적으로 계산하는 것보다 더 효율적인 암묵적 계산 방법을 사용할 수 있습니다.<br>
- ex) **image에 대한 분류 문제** 를 계산할 때 하나의 이미지를 하나의 벡터로 나타내면 각 픽셀의 값이 하나의 원소가 되는 매우 큰 차원의 feature space를 가지게 됩니다.<br> 이 때 **kernel trick** 을 사용하게 된다면 feature map을 explicit하게 계산하지 않아도 됩니다.<br>

2. **무한 차원의 특징 공간** : 일부 커널 함수는 무한 차원의 특징 공간을 사용할 수 있습니다. <br>이러한 함수를 사용하면 비선형 문제를 해결하는 데 도움이 됩니다.<br>
- ex) 커널 기법을 사용하여 비선형 SVM을 수행하는 경우, 일부 커널 함수는 **무한 차원의 특징 공간** 을 사용하여 데이터를 변환합니다.<br> 예를 들어, **RBF(radial basis function) 커널** 은 데이터를 무한 차원의 공간으로 매핑하여 **데이터 포인트 간의 거리(유사도)를 계산** 합니다.<br> 이렇게 하면 비선형 경계를 구분하기 위해 더 복잡한 함수를 사용할 필요 없이 선형 SVM을 사용할 수 있습니다.<br>

3. **유사도 개념 사용** : 커널 기법을 사용하면 데이터 샘플 간의 유사도를 계산할 수 있습니다.<br> 이는 특징(feature) 자체가 아니라 유사도(similarity)를 기반으로 문제를 해결하는 것을 의미합니다.<br> 이는 분류, 클러스터링, 추천 등의 문제에서 유용합니다.<br>
- **추천 시스템에서 커널 기법을 사용하여 유사도를 계산** 할 수 있습니다.<br> 예를 들어, 사용자간의 유사도를 계산하기 위해 사용자 간의 상호작용 기록을 사용할 수 있습니다.<br> 이렇게 하면 유사한 사용자 간에는 추천할 항목이 유사할 것이므로, 더 나은 추천을 제공할 수 있습니다.<br> 이러한 방식은 클러스터링 분석에서도 사용될 수 있으며, **데이터 샘플 간의 유사도** 를 기반으로 비슷한 특징을 가진 데이터 그룹을 찾을 수 있습니다.<br>

이제 SVM에서의 예시를 보며 자세하게 설명드리겠습니다.

## Example : SVM
### SVM Dual
기존에 우리가 가지고 있던 training set $(x_1,y_1),...,(x_n,y_n)$에 대한 SVM dual 최적화 문제는 다음과 같습니다.<br>

$\underset{α}{\sup}⁡∑_{i=0}^nα_i-\frac{1}{2}∑_{i,j=1}^nα_i α_j y_i y_j x_i^T x_i$<br>
$s.t.\ ∑_{i=1}^nα_i y_i=0$<br>
$\quad\quad α_i∈[0,\frac{c}{n}],\ i=1,…,n$

이때, 우리는 비선형의 데이터들에 대해서 kernel method를 사용해 **고차원의 유사도**를 계산할 것입니다.<br>

그렇다면 어떤 **kernel function** 이 있는지 한 번 알아보겠습니다.<br>

### Linear Kernel
**선형 커널(linear kernel)** 은 SVM에서 가장 기본적으로 사용되는 커널 함수 중 하나입니다. <br>이 커널 함수는 **입력 공간(input space)** 과 **특징 공간(feature space)** 이 같습니다.<br>

입력 공간은 $d$차원의 실수 벡터 공간 $R^d$이며, 특징 공간도 $d$차원의 실수 벡터 공간 $R^d$입니다.<br> **즉, 입력 데이터 자체를 특징으로 사용합니다.** <br>

선형 커널은 입력 데이터의 **내적(inner product)** 을 이용하여 **유사도(similarity)** 를 계산합니다.<br> 특징 공간에서 두 데이터 $x$와 $x'$의 내적은 다음과 같이 계산됩니다.<br>

- 이 때 혼동이 될 수 있는 부분은 $x$와 $x'$의 내적도 유사도라는 점입니다.<br> 이유는 기존데이터(support vector)와 $x_{new}$ data의 유사도를 측정해서 **y의 class** 를 정하게 되는데, 이는 비선형적인 데이터들에 대해서는 **초평면** 을 구할 수 없습니다.<br> 따라서 우리는 **Kernel Function K()** 를 이용해서 input space를 고차원의 feature space로 mapping시켜 초평면을 구하려고 하는 목적이 있기 때문입니다.<br>

featur map이 $ ψ(x)=x$인 경우 커널함수는 다음과 같이 형성이 됩니다.<br>

$k(x, x') = ψ(x)^T ψ(x') = x^T x'$

여기서 $ψ(x)$는 입력 데이터 $x$를 특징 공간으로 변환하는 함수이며, $x^T x'$은 입력 데이터 $x$와 $x'$의 내적입니다.<br> 따라서 **선형 커널** 은 입력 데이터 $x$와 $x'$의 **내적** 으로 두 데이터 간의 **유사도를 측정** 합니다.

선형 커널은 입력 공간이 선형 분리 가능(linearly separable)한 경우에는 잘 작동합니다.<br> 그러나 입력 데이터가 비선형(nonlinear)인 경우에는 다른 커널 함수를 사용해야 합니다.
### The Kernel Matrix (or the Gram Matrix)

$K = (<x_i,x_j>)_{i,j}=\begin{bmatrix}<x_1, x_1> & <x_1, x_2> & \dots & <x_1, x_n> \\
<x_2, x_1> & <x_2, x_2> & \dots & <x_2, x_n> \\
\vdots & \vdots & \ddots & \vdots \\
<x_n, x_1> & <x_n, x_2> & \dots & <x_n, x_n> \\
\end{bmatrix}$

Kernel Matrix 또는 Gram Matrix는 데이터 포인트 간 유사도(similarity)를 측정하는 방법 중 하나입니다. <br>이 행렬은 데이터 포인트의 내적(dot product)으로 구성되며, 모든 데이터 포인트 쌍에 대한 내적 값을 담고 있습니다.<br>


[*[출처] : FOUNDATIONS OF MACHINE LEARNING by Bloomberg ML EDU*](https://bloomberg.github.io/foml/#home).
