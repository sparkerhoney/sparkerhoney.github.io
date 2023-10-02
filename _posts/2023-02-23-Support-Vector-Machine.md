---
title: Support Vector Machine 1st
layout: post
description: Lecture summary
use_math: true
post-image: https://github.com/sparkerhoney/sparkerhoney.github.io/blob/master/_images/machine%20learning.png?raw=true
category: machine learning
tags:
- Data Science
- machine learning
---

# Support Vector Machines
---

## The SVM as a Quadratic Program
---

### The Margin
---

정의: 예측 score $$\hat{y}$$과 실제 class $$y∈{-1,1}$$에 대한 마진은 $$y\hat{y}$$ 입니다.<br>
마진은 자주 $$yf(x)$$와 같이 나타냅니다. 이때 $$f(x)$$는 우리의 score function입니다.<br>
마진은 우리가 얼마나 정확한지를 보여주는 척도이고, 우리는 마진을 최대화하기를 원합니다.<br>
대부분 classification loss는 마진에만 의존합니다.<br>

### Hinge Loss
---

- SVM/Hinge loss: $$l_{Hinge}=max⁡{1-m,0}=(1-m)_+$$<br>
- 마진 $$m=yf(x)$$: “positive part”인 $$x_+=x_1(x≥0)$$<br>
	 
Hinge는 convex이고, 0-1loss의 상계,	$$m=1$$일 때 미분이 불가합니다.<br>
우리는 $$m<1$$일 때 “margin error”를 가집니다.<br>

## Support Vector Machine
---

- Hypothesis space: $$F={f(x)=w^T x+b|w∈R^d,b∈R}$$<br>

###	SVM Optimization Problem
---

SVM 예측 함수는 다음의 식을 해결하는 것입니다.<br>

$$\underset{w∈R^d,b∈R}{\min}\frac{1}{2}\begin{Vmatrix} w \end{Vmatrix}^2+\frac{c}{n}∑_{i=1}^n\max⁡(0,1-y_i [w^T x_i+b])$$<br>

이 식은 maximize가 존재하기에 미분이 불가능합니다.<br>
따라서 제약식이 포함된 상태로 변형을 시킬 건데 형태는 다음과 같습니다.<br>

$$minimize\ \ \frac{1}{2}\begin{Vmatrix} w \end{Vmatrix}^2+\frac{c}{n}∑_{i=1}^nξ_i$$<br>
$$subject\ to \   \ ξ_i≥(1-y_i [w^T x_i+b])\ for\ i=1,…,n$$<br>
$$\qquad\qquad\ \ \  ξ_i≥0\ for\ i=1,…,n$$<br>

또 $$ξ$$또한 max를 가지기에 두 가지 형태로 나눌 수 있습니다.

$$minimize\ \ \frac{1}{2}\begin{Vmatrix} w \end{Vmatrix}^2+\frac{c}{n}∑_{i=1}^nξ_i$$<br>
$$subject\ to \   \ ξ_i≥(1-y_i [w^T x_i+b])\ for\ i=1,…,n$$<br>
$$\qquad\qquad\ \ \  ξ_i≥0\ for\ i=1,…,n$$<br>

위 식은 objective function이 이차식이고 제약식이 모두 affine이기에 quadratic program이라고 할 수 있고 미분이 가능해집니다.<br>

## The SVM Dual Problem
---

### SNM Lagrange Multipliers
---

[*Lagrange Multipliers*](https://sparkerhoney.github.io/op/Lagrange-Multiplier-Method/)에 대한 내용은 해당 링크를 참고해 주시면 감사하겠습니다.<br>

$$minimize\ \ \frac{1}{2}\begin{Vmatrix} w \end{Vmatrix}^2+\frac{c}{n}∑_{i=1}^nξ_i$$<br>
$$subject\ to \   \ ξ_i≥(1-y_i [w^T x_i+b])\ for\ i=1,…,n$$<br>
$$\qquad\qquad\ \ \  ξ_i≥0\ for\ i=1,…,n$$<br>

위 식에서 제약식을 최종적으로 변환한 이유는 결국 라그랑주 승수를 곱하기 위함인데 이는 다음식과 같습니다.<br>

$$L(w,b,ξ,α,λ)=\frac{1}{2}\begin{Vmatrix} w \end{Vmatrix}^2+\frac{c}{n}∑_{i=1}^nξ_i +∑_{i=1}^nα_i (1-y_i [w^T x_i+b]-ξ_i)+∑_{i=1}^nλ_i (-ξ_i)$$<br>   
$$=\frac{1}{2}w^T w+∑_{i=1}^nξ_i (\frac{c}{n}-α_i-λ_i)+∑_{i=1}^nα_i (1-y_i [w^T x_i+b])$$<br>

Primal과 dual의 관계는 다음과 같습니다.

$$p^*=\underset{w,ξ,b}{\inf}\underset{α,λ≽0}{\sup}⁡L(w,b,ξ,α,λ) ≥\underset{α,λ≽0}{\sup}\underset{w,ξ,b}{\inf}L(w,b,ξ,α,λ)=d^*$$<br>

이 때 objective function이 convex, constraint가 아핀이고 마지막으로 input point의 $$x$$가 strictly feasible 하기 때문에 SVM을 Quadratic program으로 해결할 수 있게 됩니다.<br>
이 때 이 것을 slater 조건이라 설명하고 있으며 이 때 strong duality를 가진다고 말할 수 있습니다.<br>
*Feasible region : 해의 영역*<br>

### SVM Dual Function
---

라그랑주 dual은 라그랑주의 primal 변수에 대한 하한값입니다.<br>

$$g(α,λ)=\underset{w,ξ,b}{\inf}⁡L(w,b,ξ,α,λ)$$<br>
$$=\underset{w,ξ,b}{\inf}[\frac{1}{2}w^T w+∑_{i=1}^nξ_i (\frac{c}{n}-α_i-λ_i)+∑_{i=1}^nα_i (1-y_i [w^T x_i+b])]$$<br>

Strong duality를 만족하게 된다면 KKT condition을 만족하기 때문에 KKT condition중 stationary condition인 최적화하려는 미지수로 편미분을 해 0이되는 조건을 계산해줍니다.<br>

### SVM Dual Function: First Order Conditions
---

$$∂_w L=0⟺w-∑_{i=1}^nα_i y_i x_i=0⟺w=∑_{i=1}^nα_i y_i x_i$$<br>
$$∂_b L=0⟺-∑_{i=1}^nα_i y_i=0⟺0=∑_{i=1}^nα_i y_i$$<br>
$$∂_{ξ_i} L=0⟺\frac{c}{n}-α_i-λ_i=0⟺\frac{c}{n}=α_i+λ_i$$<br>

### The SVM Dual Problem
---

앞서 봤던 First order condition에 의해서 라그랑지안 primal 문제가 Dual 문제로 바뀌게 됩니다.<br>

$$α_i$$에 관한 문제로 단순해졌습니다.(primal solution에서의 $$w$$를 구할 때, $$w=∑_{i=1}^nα_i y_i x_i$$이기에)<br>

$$\underset{α}{\sup}⁡∑_{i=0}^nα_i-\frac{1}{2}∑_{i,j=1}^nα_i α_j y_i y_j x_i^T x_i$$<br>
$$s.t.\ ∑_{i=1}^nα_i y_i=0$$<br>
$$\quad\quad α_i∈[0,\frac{c}{n}],\ i=1,…,n$$<br>

## Insight From Complementary Slackness: Margin and Support Vectors
---

### The Margin and Some Terminology
---

$$f^* (x)=x^T w^*+b^*$$이 존재한다고 가정할 때 마진 $$yf^* (x)$$는 다음과 같습니다.

![image](https://user-images.githubusercontent.com/108461006/220888241-e5ae1c57-3a55-44a9-8568-82f1e21c58c2.png)<br>

###	Support Vectors and The Margin
---

“Slack Variable”이라고 부르는 $$ξ_i^*=\max ⁡(0,1-y_i f^* (x_i))$$는 $$(x_i,y_i)$$에서의 hinge loss입니다.<br>
$$ξ_i^*=0$$이라고 가정하면 그 때의 $$y_i f^* (x_i )≥1$$일 것입니다.<br>

즉, 분류가 올바르게 된 $$y_i f^* (x_i )≥1$$의 hinge loss 0이 될 것입니다.<br>

###	Complementary Slackness Conditions
---

우리의 primal constraint와 라그랑주 승수를 상기해보자면 다음과 같습니다.<br>
First order condition에서의 $$∇_{ξ_i} L=0$$은 $$λ_i^*=\frac{c}{n}-α_i^*$$를 도출 시킵니다.<br>

Strong duality(KKT condition)에 의해서, 우리는 complementary slackness를 가져야 합니다.

$$α_i^* (1-y_i f^* (x_i )-ξ_i^* )=0$$<br>
$$λ_i^* ξ_i^*=(\frac{c}{n}-α_i^* ) ξ_i^*=0$$<br>


### Consequences of Complementary Slackness Conditions
이렇게 된다면 마진 값에 의해서 결과 값을 도출해낼 수 있는데 과정은 다음과 같습니다.<br>
- 만약 $$y_i f^* (x_i)>1$$이면 그 때의 margin loss는 $$ξ_i^*=0$$이되고 위의 식에 대입해본다면 $$α_i^*=0$$이 됩니다.<br>
- 만약 $$y_i f^* (x_i )<1$$이면 그 때의 margin loss는 $$ξ_i^*>0$$이되고 위의 식에 대입해본다면 $$α_i^*=\frac{c}{n}$$이 됩니다.<br>
- 만약 $$α_i^*=0$$이면 그 때의 margin loss는 $$ξ_i^*=0$$이되고 이 말인 즉슨 loss가 없다는 것을 의미해 최종적으로 $$y_i f^* (x_i )≥1$$이라고 할 수 있습니다.<br>
- 만약 $$α_i^*∈(0,\frac{c}{n})$$이면 그 때의 margin loss는 $$ξ_i^*=0$$이되고 이 말인 즉슨 $$1-y_i f^* (x_i )=0$$이라고 할 수 있습니다.<br>

이 모든 것을 요약 정리해본다면 다음과 같습니다.<br>

1. $y_i f^* (x_i )<1⇒ α_i^*=\frac{c}{n}$
2. $y_i f^* (x_i )=1⇒ α_i^*∈[0,\frac{c}{n}]$
3. $y_i f^* (x_i )>1⇒ α_i^*=0$
	
---

# Support Vectors
만약 $$α^*$$가 dual problem의 해라고 하면, primal의 해는 다음과 같습니다.<br>
$$w^*=\sum_{i=1}^n α_i^* y_i x_i$$<br>
with  $$α_i^*∈[0,c/n]$$<br>
$$α_i^*>0$$에 대응되는 $$x_i^*$$를 우리는 support vector라고 부른다.<br>
Margin error가 거의 없거나 마진 위에 있는 경우 우리는 input에서 sparsity를 가진다고 한다.<br>

---

## Complementary Slackness To Get $$b^*$$
The Bias Term: b<br>
SVM primal에서 complementary slackness condition은 다음과 같다.<br>
$$α_i^* (1-y_i f^* (x_i )-ξ_i^* )=0$$		(1)<br>
$$λ_i^* ξ_i^*=(c/n-α_i^* ) ξ_i^*=0$$		(2)<br>
$$α_i^*∈(0,c/n)$$와 같은 i가 있다고 가정하자.<br>
(2)에 의해서 $$ξ_i^*=0$$을 만족한다<br>
(1)에 의해서 다음식을 만족한다.<br>
$$y_i [x_i^T w^*+b^* ]=1$$<br>
⇔$$[x_i^T w^*+b^* ]=y_i$$         (use $$y_i∈{-1,1}$$)<br>
⇔$$b^*=y_i-x_i^T w^*$$<br>
따라서 최적의 b는 마지막 식과 같다.<br>

---

## Subgradient Descent
svm에서 gradient descent를 사용하지 못하는 이유는 SVM의 목적은 마진을 최대화하는 것입니다.<br>
이때 마진은 결정 경계와 가장 가까운 데이터 포인트까지의 거리로 정의됩니다.<br>
예를 들어, 마진을 `M`이라고 하면, `M = y(w^T x + b)`로 표현될 수 있습니다.<br>

SVM을 훈련시키는 과정에서는 마진을 최대화하는 결정 경계를 찾는 최적화 문제를 푸는 것이 중요합니다.<br>
그러나 SVM에서는 비용 함수가 다른 대부분의 머신 러닝 알고리즘들과 달리 매끄럽지 않은 형태를 가지고 있습니다.<br>
SVM의 비용 함수는 `L = 1/2 ||w||^2 + C Σ ξ_i`와 같이 표현될 수 있습니다.<br>

SVM의 비용 함수는 최적화하려는 파라미터에 대해 불연속적인 구조를 가지고 있으며, 이는 그래디언트를 사용한 최적화 알고리즘(예: 경사 하강법)을 적용하기 어렵게 만듭니다.<br>
결국 SVM에서는 그래디언트를 사용한 최적화 알고리즘을 직접 적용하기보다는, 원래 최적화 문제를 대신할 수 있는 수학적 기법(쌍대 문제)을 사용하여 최적화 문제를 해결합니다.<br>
쌍대 문제는 `D(α) = Σ α_i - 1/2 Σ Σ α_i α_j y_i y_j x_i^T x_j`와 같이 표현될 수 있습니다.<br>

쌍대 문제는 훨씬 더 부드러운 비용 함수를 가지므로 그래디언트를 사용하여 최적화를 할 수 있습니다.<br>
이렇게 함으로써 SVM은 비용 함수를 최적화할 때 그래디언트를 사용하지 않고도 비용 함수의 최적 값을 찾아낼 수 있습니다.<br>

SVM의 dual 문제는 quadratic program(QP)입니다.<br>
이러한 문제는 subgradient를 사용하여 해결하는 것이 일반적으로 어렵습니다.<br>
하지만 SVM의 경우 dual 문제가 convex 문제이기 때문에 subgradient 대신에 좀 더 효율적인 알고리즘을 사용할 수 있습니다.<br>
그럼에도 불구하고 subgradient 방법으로 SVM 문제를 해결하는 것은 가능합니다.<br>
다만, subgradient 방법은 수렴이 느리며 최적의 해를 보장하지 않습니다.<br>

---

$$\underset{\alpha}{\sup}⁡\sum_{i=0}^nα_i-\frac{1}{2}∑_{i,j=1}^nα_i α_j y_i y_j x_i^T x_i$$<br>
$$s.t. ∑_{i=1}^nα_i y_i =0$$<br>
$$α_i∈[0,\frac{c}{n}],\ i=1,…,n$$<br>

---

Primal Feasibility: $$y_i(w^Tx_i+b) - 1 + \xi_i \geq 0,\ \forall i$$<br>
Dual Feasibility: $$\alpha_i \geq 0,\ \forall i$$<br>
Complementary Slackness: $$α_i(y_i(w^Tx_i+b)-1+\xi_i) = 0,\ \forall i$$<br>
Stationarity: $$\nabla_w L(w,b,\xi,\alpha) = 0$$ and $$\frac{\partial L(w,b,\xi,\alpha)}{\partial \xi_i} = 0,\ \forall i$$<br>

---

## Hinge Loss
Hinge Loss는 SVM에서 사용하는 손실 함수 중 하나로, 다음과 같이 정의됩니다.<br>
$$\ell(y, f(x)) = \max{0, 1 - yf(x)}$$<br>
여기서 $$y$$는 실제 클래스(1 또는 -1)를 나타내며, $$f(x)$$는 입력 데이터 $$x$$에 대한 모델의 예측값을 나타냅니다.<br> 
Hinge Loss는 SVM의 비용 함수와 밀접한 관련이 있습니다.<br> 
SVM의 비용 함수는 마진을 최대화하는 것을 목표로 하며, 마진은 결정 경계와 가장 가까운 데이터 포인트까지의 거리를 나타냅니다.<br>
따라서, Hinge Loss는 결정 경계와 데이터 포인트 사이의 거리를 측정하는 데 사용됩니다.<br>
Hinge Loss의 값은 0과 1 사이에 있으며, 값이 1에 가까울수록 모델의 예측이 정확하다는 것을 나타냅니다.<br>
반대로, 값이 0에 가까울수록 모델의 예측이 부정확하다는 것을 나타냅니다.<br>
따라서, Hinge Loss의 값이 작을수록 모델의 성능이 좋다고 할 수 있습니다.<br>
Hinge Loss의 subgradient는 입력 데이터 $$x$$와 실제 클래스 $$y$$에 의존하며, 모델이 실제 클래스와 다른 예측을 하게 되는 경우 오류를 줄이는 방향으로 가중치 벡터를 업데이트할 수 있도록 합니다.<br>

---

[*출처 : FOUNDATIONS OF MACHINE LEARNING by Bloomberg ML EDU*](https://bloomberg.github.io/foml/#home)

---
