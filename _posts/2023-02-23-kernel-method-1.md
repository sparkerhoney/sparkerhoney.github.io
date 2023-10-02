---
title: Kernel Methods 1st
layout: post
description: Lecture summary
use_math: true
post-image: https://github.com/sparkerhoney/sparkerhoney.github.io/blob/master/_images/machine%20learning.png?raw=true
category: machine learning
tags:
- Data Science
- machine learning
---

# Linear Models Need Big Feature Spaces
---

## Review
우리는 앞서 살펴봤듯이 **모델의 표현능력(expressivity)** 이 커지기 위해서는 가설공간을 얼마나 키울 수 있느냐에 따라 달라지게 된다고 했습니다.<br>
이 때 큰 표현능력을 가진다면 모델이 더 **성능이** 좋아질 겁니다.<br>
성능이 좋기 위해서 큰 표현능력을 갖는 것이 목표이고, 큰 표현능력을 얻기 위해서는 풍부한 가설공간을 가져야한다고 했습니다.<br>
이 풍부한 가설공간을 가지기 위해서는 가성공간을 확장시켜야 하는데, 가설공간을 확장시키기 위해서는 feature mapping function $$(φ)$$에서 더 많은 feature를 추가해야합니다.<br>
더 많은 feature를 추가하게 되면 가설공간이 확장될 것이고 더 복잡한 모델을 학습시킬 수 있을 것입니다.<br>

---

### Linear Models Need Big Feature Spaces
선형 모델을 사용하여 풍부한 가설공간을 얻으려면 고차원의 feature space가 필요합니다.<br>
$$x = (1,x_1,..., x_d ) ∈ R^{d+1} = X$$<br>
우리가 위와 같은 선형 모델로 시작한다고 가정하겠습니다.<br>
이 때 우리는 차수 $$M$$의 모든 다항식 $$x^{p_1}_1···x^{p_d}_d, with\ p_1 +···+p_d = M$$을 추가하고 싶습니다.<br>
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
$$\begin{pmatrix} M+d-1 \\ M \end{pmatrix}$$<br>
예를 들어, $$d = 40$$이고 $$M = 8$$이면, 총 314457495개의 특징이 생성됩니다. 이는 굉장히 큰 행렬을 만들어내게 됩니다.<br>

---

### Big Feature Spaces
매우 큰 특징 공간은 두 가지 문제가 있습니다.<br>

1. 과적합
2. 메모리 및 계산 비용

과적합 문제는 규제(regularization)를 사용하여 해결할 수 있습니다.<br> 
규제는 모델의 복잡성을 제한함으로써 일반화 성능을 향상시키는 방법입니다.<br>

그러나 메모리 및 계산 비용 문제는 규제만으로 해결하기 어렵습니다.<br> 
대부분의 컴퓨터는 큰 데이터셋과 매우 큰 특징 공간을 처리하기에는 제한이 있습니다.<br> 
이 문제를 해결하기 위해 "kernel methods"이라는 방법이 사용될 수 있습니다.<br> 
kernel methods은 featur vector를 고차원의 feature space으로 변환하는 대신 kernel function를 사용하여 내적을 수행합니다.<br>
이를 통해 메모리 및 계산 비용을 줄일 수 있습니다.<br> 
그러나 모든 문제에 대해 kernel methods가 효과적인 것은 아니며, kernel function을 선택하는 것이 중요합니다.<br>

---

## Kernel Methods : Motivation
---

### Review: Linear SVM and Dual
우선 SVM은 Support Vector Machine의 약자로, 분류 문제를 해결하는 머신 러닝 알고리즘 중 하나입니다. SVM은 주어진 데이터를 가장 잘 분류할 수 있는 초평면(hyperplane)을 찾는 것이 목표입니다.<br>

이때 SVM은 주어진 데이터를 고차원 공간으로 매핑하는 함수 $$\psi(x)$$를 사용합니다. 이 함수는 주어진 데이터를 새로운 공간으로 변환하여, 기존의 선형 분리가 불가능한 데이터를 선형 분리 가능한 데이터로 변환합니다.<br>

이 SVM에서 사용되는 분류 함수는 아래와 같은 형태를 가집니다.<br>

$$y(x) = sign(w^T\psi(x) + b)$$<br>

여기서 $$w$$와 $$b$$는 SVM의 핵심 파라미터입니다. $$w$$는 초평면의 법선 벡터(normal vector)를 나타내고, $$b$$는 분류 경계면을 조절하는 bias 값입니다.<br>

위에서 소개한 SVM 분류 함수를 최적화하는 문제를 풀기 위해서는 학습 데이터를 이용하여 $$w$$와 $$b$$를 찾아야 합니다. 이때 SVM에서는 학습 데이터를 이용하여 $$w$$와 $$b$$를 찾기 위해 다음과 같은 최적화 문제를 풀게 됩니다.<br>

$$\underset{w∈R^d,b∈R}{\min}\frac{1}{2}\begin{Vmatrix} w \end{Vmatrix}^2+\frac{c}{n}∑_{i=1}^n(1-y_i[w^Tψ(x_i)+b])_+$$<br>

위 식에서 $$c$$는 SVM에서 regularization parameter로 사용되는 하이퍼파라미터입니다. 이 값은 모델이 오버피팅(overfitting)되는 것을 방지하기 위해 사용됩니다.<br>
그런데 위 최적화 문제는 $$w$$와 $$b$$에 대한 2차 제한 조건이 있는 복잡한 문제입니다. 따라서 이 문제를 푸는 것이 어렵습니다.<br>

## Dual Problem in SVM
---

SVM에서는 이 문제를 더 쉽게 풀기 위해 다음과 같이 dual problem(쌍대 문제)를 정의하고, 이를 푸는 것을 권장합니다.<br>

$$
\underset{\alpha}{\sup}⁡∑_{i=0}^nα_i-\frac{1}{2}∑_{i,j=1}^nα_i α_j y_i y_j x_i^T x_i
$$<br>

with the constraints:<br>

$$
s.t. ∑_{i=1}^nα_i y_i =0
$$<br>

$$
α_i∈[0,\frac{c}{n}],\ i=1,…,n
$$<br>

이때 dual problem의 제약 조건에 의해 $$\alpha$$값은 0과 $$\frac{c}{n}$$ 사이에 위치하게 됩니다.<br>

이 dual problem을 풀게 되면, 최적화 문제를 풀 때 필요한 $$w$$와 $$b$$값을 쉽게 계산할 수 있습니다. 실제로, $$w$$는 다음과 같은 식으로 계산됩니다.<br>

$$
w = ∑_{i=1}^n \alpha_i y_i \psi(x_i)
$$<br>

이 식에서 $$\psi(x_i)$$는 $$x_i$$를 고차원 공간으로 매핑한 값입니다. 따라서, $$\psi(x_i)$$는 기존의 $$x_i$$와 함께 학습 데이터로부터 주어지는 값이므로, 실제로는 $$w$$값을 계산할 때 $$\psi(x_i)$$ 대신 $$x_i$$를 사용하면 됩니다.<br>

마지막으로, $$b$$값은 support vector(서포트 벡터)라고 불리는 일부 학습 데이터에 대해서만 구하면 됩니다. 이 support vector는 최적화 문제를 푸는 과정에서 $$0<\alpha_i<\frac{c}{n}$$인 데이터 포인트들을 의미합니다. $$b$$값은 이 support vector들을 이용하여 구할 수 있습니다.<br>

---

## SVM and its Functionality
---

SVM은 분류 문제를 해결하기 위한 머신 러닝 알고리즘 중 하나입니다. SVM은 주어진 데이터를 고차원 공간으로 매핑하는 함수 $$\psi(x)$$를 사용하여, 기존의 선형 분리가 불가능한 데이터를 선형 분리 가능한 데이터로 변환한 뒤, 최적의 분류 경계면을 찾아내는 방식으로 동작합니다.<br>

이때 SVM에서 사용되는 분류 함수는 주어진 데이터 $$x$$를 고차원 공간으로 매핑한 결과 벡터 $$\psi(x)$$와 가중치 벡터 $$w$$의 내적에 편향 값 $$b$$를 더한 값의 부호를 취한 것입니다. 이 분류 함수를 최적화하기 위해 SVM은 학습 데이터를 이용하여 가중치 벡터 $$w$$와 편향 값 $$b$$를 찾아내는 최적화 문제를 푸는데, 이때 regularization parameter인 $$c$$를 사용하여 모델이 오버피팅되는 것을 방지합니다.<br>

### Regularization Parameter $$c$$
---

regularization parameter $$c$$: SVM의 regularization parameter인 $$c$$는 모델이 학습 데이터에 과적합(overfitting)되는 것을 방지하는 역할을 합니다.<br>

SVM은 최적의 초평면(hyperplane)을 찾기 위해 데이터를 고차원 공간으로 매핑하고, 그 공간에서 최적의 분류 경계를 찾습니다. 이때 모델의 복잡도는 $$w$$값의 크기에 비례하게 됩니다. $$w$$의 크기가 작을수록 모델은 단순해지고, $$w$$의 크기가 클수록 모델은 복잡해집니다.<br>

그러나 SVM에서는 단순한 모델뿐만 아니라 복잡한 모델 모두를 고려하여 최적의 모델을 찾습니다. 이를 위해 SVM은 일종의 균형을 맞춰야 합니다. 즉, SVM은 학습 데이터를 분류하는 데 있어서 오류를 최소화하는 동시에, $$w$$의 크기를 최소화해야 합니다. 이때 regularization parameter $$c$$가 중요한 역할을 합니다.<br>

$$c$$값은 오류를 얼마나 허용할 것인지를 결정합니다. $$c$$값이 크면 모델은 오류를 허용하지 않으며, $$w$$의 크기를 작게 유지하도록 노력합니다. 이 경우 모델은 단순해지지만, 학습 데이터에 과적합될 가능성이 커집니다. 따라서 $$c$$값이 작으면 모델은 오류를 조금 허용하며, $$w$$의 크기를 상대적으로 크게 유지합니다. 이 경우 모델은 복잡해지지만, 학습 데이터에 과적합될 가능성이 줄어듭니다.<br>

따라서 $$c$$값은 모델의 복잡도와 오버피팅 사이의 균형을 조절하는 역할을 합니다. $$c$$값을 적절히 설정하면 SVM은 일반화 성능(generalization performance)이 우수한 모델을 찾을 수 있습니다.<br>

---

## SVM and High Dimensional Mapping
---

SVM은 고차원으로 데이터를 매핑하는 것을 통해 비선형적인 결정 경계를 학습할 수 있는데, 이는 다음과 같은 이유 때문입니다.<br>

우선, 선형적으로 분리 가능한 데이터는 SVM에서 잘 동작합니다. 그러나 현실의 데이터는 대개 선형적으로 분리되기 어렵고 비선형적인 패턴을 보이는 경우가 많습니다. 이 때 SVM은 커널 트릭(kernel trick)이라는 기법을 이용하여 데이터를 고차원 공간으로 매핑하고, 고차원에서는 선형적으로 분리 가능한 데이터를 다룰 수 있습니다.<br>

예를 들어, 2차원에서 원형으로 분포된 데이터를 SVM으로 분류하고자 할 때, 이를 선형적으로 분리할 수 있는 3차원 공간으로 매핑할 수 있습니다. 이때, 커널 함수(kernel function)를 사용하여 2차원 데이터를 3차원으로 변환한 후, SVM 알고리즘을 적용하여 분류 경계를 학습합니다. 이러한 커널 함수를 이용한 비선형 매핑은 일반적으로 고차원 공간으로 데이터를 옮길 필요 없이, 더 빠르고 효율적으로 SVM 알고리즘을 적용할 수 있습니다.<br>

따라서 SVM은 고차원 공간으로 데이터를 매핑함으로써 비선형적인 패턴을 학습하고, 이를 통해 선형적으로 분리되지 않는 복잡한 문제를 해결할 수 있습니다.<br>

이렇게 SVM은 학습 데이터를 고차원 공간으로 매핑하여 분류 문제를 해결합니다. 이때, feature extraction이란 고차원 공간으로 매핑하기 위해 사용되는 함수 $$\psi(x)$$를 말합니다. 이 함수는 일반적으로 사용자가 직접 정의하며, 주어진 데이터를 새로운 특징 공간으로 매핑하는 역할을 합니다.<br>

또한, kernel method란 SVM에서 feature extraction을 수행하는 함수 대신 kernel 함수를 사용하여 데이터를 고차원 공간으로 매핑하는 방식입니다. kernel 함수는 두 데이터 간의 유사도를 계산하여, 유사한 데이터를 고차원 공간에서 가깝게 위치시키는 역할을 합니다. 이를 통해, SVM은 복잡한 비선형 분류 문제를 해결할 수 있습니다. 대표적으로 사용되는 kernel 함수로는 RBF(Radial Basis Function)가 있습니다.<br>

---

[*출처 : FOUNDATIONS OF MACHINE LEARNING by Bloomberg ML EDU*](https://bloomberg.github.io/foml/#home).