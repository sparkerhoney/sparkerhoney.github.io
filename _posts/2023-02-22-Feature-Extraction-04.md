---
title: Feature Extraction 4th
layout: post
description: Lecture summary
use_math: true
post-image: https://github.com/sparkerhoney/sparkerhoney.github.io/blob/master/_images/machine%20learning.png?raw=true
category: machine learning
tags:
- Data Science
- machine learning
---

# Features
## Handling Nonlinearity with Linear Methods

### Interactions: The Issue
**특징 간 상호작용(interactions between features)**이 있을 경우, 두 개 이상의 특징이 함께 사용될 때 결과가 예상과 다를 수 있습니다.<br>
Interactions between features는 두 개 이상의 feature가 함께 사용될 때 예측값에 영향을 미치는 경우를 의미합니다.<br>
예를 들어, 성별과 키 두 가지 feature가 있다고 가정해보겠습니다.<br>
만약 성별에 따라 키에 대한 평균값이 다르다면, 성별과 키 두 가지 feature가 함께 사용될 때 예측값에 영향을 미치게 됩니다.<br>
이러한 경우, 두 feature를 모두 고려해야 더 정확한 예측값을 얻을 수 있습니다.<br>
하지만 때로는 두 feature가 함께 사용될 때 예측값에 영향을 미치지 않는 경우도 있습니다.<br>
예를 들어, 온도와 강수량이라는 두 feature가 있다고 가정해보겠습니다.<br>
온도가 높아지면서 강수량이 많아진다고 해도, 강수량이 높은 경우 온도가 높아질 수도 있고 낮아질 수도 있기 때문에 두 feature가 함께 사용될 때 예측값에 큰 영향을 미치지 않을 수 있습니다.<br>
Interactions between features는 feature engineering 과정에서 매우 중요합니다.<br>
모델이 두 feature 간의 interaction을 고려하지 않는다면, 모델이 예측할 수 있는 범위가 한정될 수 있습니다.<br>
따라서 feature engineering 과정에서는 모든 가능한 interaction을 고려하여 예측값에 미치는 영향을 파악하고, 이를 모델에 반영하여 보다 정확한 예측을 할 수 있도록 해야 합니다.<br>
교호작용은 하나의 feature가 반영하는 정보가 다른 feature들에 따라서 영향을 받는 경우를 의미합니다.<br>
예를 들어, 날씨와 교통량이라는 두 feature가 있을 때, 날씨가 좋을 때는 교통량이 많아지는 경향이 있고, 날씨가 나쁠 때는 교통량이 감소하는 경향이 있다면, 날씨와 교통량은 교호작용을 가지고 있다고 할 수 있습니다.<br>
교호작용은 모델의 성능을 높이기 위해서 중요한 정보이며, feature engineering 단계에서 교호작용을 고려하여 새로운 feature를 만들어내는 것이 일반적입니다.<br>
이 경우 모델은 이러한 상호작용을 적절하게 고려하지 않을 경우 예측 결과가 부정확해질 수 있습니다.<br>

Input: Patient information $x$<br>
Action: Health score $y \in R$ (높을수록 좋습니다.)<br>
Feature Map: $\phi(x) = [height(x),weight(x)]$<br>
Issue: height에 관련된 weight가 중요합니다.<br>

이러한 관계는 linear한 분류로는 해결할 수 없습니다.<br>
즉, 높은 height에 대한 높은 wright는 높은 건강 점수를 가져오지만, 낮은 hright에 대한 낮은 weight는 건강 점수를 낮추는 것이므로, 이러한 상호작용을 고려해야합니다.<br>
이러한 상호작용을 고려하기 위해서는 둘을 곱한 값을 새로운 특성으로 추가하거나, 다항 특성(polynomial feature)을 추가하는 등의 방법으로 feature map을 구성해야합니다.<br>
위와 같은 방법을 통해서 더 복잡한 모델을 만들어서 height와 weight 사이의 상호작용을 고려할 수 있으며 더 높은 성능을 당성할 수 있습니다.<br>

### Interactions: Approach 1
상호작용을 고려하는 예시로서 구글에서 검색을 한"키에 따른 이상체중"을 예시로 설명할 수 있습니다.<br>
J.D.로빈손의 "이상체중"공식은 다음과 같습니다(남성기준):<br> 

$ideal\ weight(kg) = 52+1.9[height(inch)−60]$ <br>

이 수식을 이용하여 각 환자의 이상 체중을 계산하고, 실제 체중과의 차이를 제곱한 값을 $f(x)$로 정의합니다.<br>
이 때, $f(x)$는 다음과 같은 복잡한 수식이 됩니다.<br>

$f(x) = (52+1.9[h(x) −60]−w(x))^2$ <br>

위의 수식은 환자의 실제 체중

$(w(x))$와 J.D.로빈슨의 이상 체중 공식을 이용하여 계산한 이상 체중$(52+1.9[h(x) −60])$과의 차이를 제곱한 값을 $f(x)$로 정의하는 수식입니다.<br>
이렇게 정의된 $f(x)$값은 환자의 실제 체중과 이상 체중의 차이를 반영하는 점수로 사용될 수 있습니다.<br>

제곱을 취함으로써 차이가 큰 값들이 더 큰 영향을 미치게 되어, 이상 체중에 더 가까운 환자는 높은 점수를, 실제 체중이 이상 체중보다 높은 환자는 낮은 점수를 받게 됩니다.<br> 이렇게 점수를 산출함으로써, 상호작용을 고려한 더 정확한 건강 점수 모델을 만들 수 있습니다.<br>

WolframAlpha라는 검색엔진을 통해 복잡한 수학식을 풀어본다면 다음과 같습니다.<br>

$f (x) = 3.61h(x)^2 −3.8h(x)w(x) −235.6h(x) +w(x)^2 +124w(x) +3844$

### Interactions: Approach 2
상호작용에서의 두번째 접근방법은 모든 2차항의 특징들을 포함하는 것 입니다.<br>

$\phi(x) = [1,h(x),w(x),h(x)^2,w(x)^2, h(x)w(x)(교차 항)]$<br>

이 때의 mapping function $\phi(x)$ 은 앞서 나온 비선형적 요소들과 같이 polynomial 형식입니다.<br>
앞서 설명한 접근법 1과 같은 방법은 구글 검색이나 WolframAlpha를 사용하여 이상적인 체중을 계산하고, 특정 상호 작용을 계산하기 위해 해당 체중과 키와의 제곱 차이를 계산했습니다.<br>
이 방법은 매우 유연합니다.<br>
각 원시 특징에 대한 제곱 및 교차 항을 포함하여 모든 2 차원 특징을 포함할 수 있으며, 따라서 **선형 모델에서도 복잡한 비선형 상호 작용을 모델링** 할 수 있습니다.<br>
이 방법을 사용하면 모델이 자체적으로 모든 상호 작용을 학습하므로 사용자는 특정 상호 작용을 찾을 필요가 없습니다.<br>
또한, 이 방법은 **외부 도구를 사용하지 않으므로** 이전 방법보다 더 간단하며 유연합니다.<br> 이것은 또한 개발자가 복잡한 함수 및 외부 도구를 사용하여 일부 상호 작용을 추가 할 때 발생할 수 있는 오류를 방지합니다.<br>

### Predicate Features and Interaction Terms
- 정의: predicate는 입력 $x$를 받아들여 참(True) 또는 거짓(False)으로 평가하는 함수입니다.

예를들어, $x$가 어떤 사람의 정보를 나타낸다고 하면, "$s(x) = 1$"은 그 사람이 자고 있는지 여부를 참(True) 또는 거짓(False)으로 평가하는 술어입니다.<br>
마찬가지로, "$d(x) = 1$"은 그 사람이 운전 중인지 여부를 참 또는 거짓으로 평가하는 술어입니다.<br>

이러한 술어를 가진 특징(feature)을 예로 들면, "$s(x)d(x)$"는 "그 사람이 자고 있고 운전 중인가?"라는 질문에 대한 답을 참 또는 거짓으로 평가하는 특징입니다.<br>
이러한 술어 간의 상호작용(interaction)을 표현하는 데 AND 논리 연산자를 사용합니다.<br>

```python
is_sleeping = True
is_driving = False
```

여기서 is_sleeping과 is_driving은 각각 불리언 값(True 또는 False)을 가지는 변수입니다.<br>
이 변수들은 위에서 설명한 predicate features를 표현하는 데 사용될 수 있습니다.

위의 예에서 "$s(x)d(x)$"는 두 술어가 동시에 참일 때만 참이 되므로,

 "그 사람이 자고 있고 운전 중인가?"라는 질문에 대한 답이 참(True)이 됩니다.<br>
이와 같이 술어와 AND 연산자를 이용하여 특징 간의 상호작용을 표현하는 것을 "predicate features and interaction terms"이라고 합니다.<br>

술어에서 or 논리 연산자는 사용할 수 있습니다. 예를 들어, 다음과 같은 술어는 "$x$가 5보다 작거나 $x$가 10보다 크면 True이고, 그렇지 않으면 False"라는 불리언 값을 반환합니다.

```python
def predicate(x):
    return x < 5 or x > 10
```
### Geometric Example: Two class problem, nonlinear boundary
[*Feature Extraction*](http://youtu.be/3liCbRZPrZA)

이 영상에서는 두 개의 클래스를 가진 이진 분류 문제를 다루고 있습니다.<br>
기존 데이터의 feature map이 (x1,x2)인 경우, 즉 2차원 평면 상에서 선형 분리가 불가능한 비선형 경계를 가진 경우를 가정합니다.<br> 
이 경우, 선형 모델을 사용하여 분류하려 하면 해결이 불가능하다는 것을 보여주고 있습니다.<br>
![image](https://user-images.githubusercontent.com/108461006/220605445-3ebec93e-cb2d-4565-a52b-0637728afba7.png)

그러나 적절한 비선형 feature map을 사용하면 ($φ(x) = x1, x2, x_1^2 + x_2^2$와 같은 형태) 더 높은 차원의 공간으로 매핑될 수 있으며, 이 경우 2차원에서는 비선형 경계가 있었지만 고차원에서는 선형 경계를 가질 수 있습니다.<br>
이렇게 선형 모델을 사용하여 고차원에서 분류를 수행할 수 있으며, 이를 통해 비선형 문제를 해결할 수 있습니다.<br>

### Expressivity of Hypothesis Space
- $φ : X → R^d: F ={f (x) = w^Tφ(x)}$<br>

가설 공간(hypothesis space)은 학습 알고리즘이 가능한 가설(모델)의 집합입니다.<br>
이 가설 공간을 얼마나 크게 만들 수 있느냐가 모델의 표현 능력(expressivity)을 결정합니다.<br>

![image](https://user-images.githubusercontent.com/108461006/220607091-bb0017bb-48c6-4d5e-8c94-b0e64e2712c2.png)

선형 가설 공간에서는 각 데이터 포인트를 특성 매핑 함수$(φ)$로 매핑한 $d$차원 벡터와 이 벡터와의 가중치$(w)$의 내적으로 표현됩니다.<br>
이 때, 가중치$(w)$는 학습 알고리즘이 학습하는 모델의 파라미터입니다.<br>

가설 공간을 확장하기 위해서는 특성 매핑 함수$(φ)$에서 더 많은 특성을 추가하면 됩니다. <br>
이렇게 확장된 가설 공간에서는 더 복잡한 가설(모델)을 학습할 수 있습니다.<br>


[*[출처] : FOUNDATIONS OF MACHINE LEARNING by Bloomberg ML EDU*](https://bloomberg.github.io/foml/#home).