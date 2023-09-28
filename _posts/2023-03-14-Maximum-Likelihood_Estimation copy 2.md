---
layout: post
title: Maximum Likelihood Estimation
description: Maximum Likelihood Estimation
summary: 
tags: statistics
minute: 1
---

# Maximum Likelihood Estimation
## Likelyhood of an Estimated Probability Distribution
최대우도추정(Maximum Likelihood Estimation, MLE)은 확률적 모델링에서 매개변수를 추정하는 방법 중 하나입니다. MLE는 어떤 데이터가 주어졌을 때, 이 데이터가 생성될 확률을 최대화하는 모수값을 찾는 것을 목표로 합니다.

MLE는 주어진 데이터의 분포를 가장 잘 설명하는 모수를 찾기 위한 방법으로, 데이터가 확률 분포에서 생성된다는 가정 하에서 모수를 추정합니다. 이때, 확률 분포의 모수를 추정하기 위해 모수에 대한 가능도 함수를 정의하고, 이 함수를 최대화하는 모수 값을 찾습니다.

예를 들어, 베르누이 분포에서 동전 던지기 데이터가 있을 때, 이 데이터가 생성될 확률을 모수인 $\theta$를 이용해 다음과 같이 나타낼 수 있습니다.

$$P(x|\theta) = \theta^{x}(1-\theta)^{(1-x)}$$

여기서 $x$는 동전의 앞면이 나올 확률을 나타내는 변수입니다. 이때, 주어진 데이터가 $D$일 때, 가능도 함수는 다음과 같이 정의됩니다.

$$L(\theta|D) = \prod_{i=1}^NP(x_i|\theta) = \prod_{i=1}^N\theta^{x_i}(1-\theta)^{(1-x_i)}$$

이제, 이 가능도 함수를 최대화하는 $\theta$ 값을 찾으면 됩니다. 일반적으로 로그 변환을 적용하여 계산합니다.

$$\log L(\theta|D) = \sum_{i=1}^N[x_i\log\theta + (1-x_i)\log(1-\theta)]$$

$\log L(\theta|D)$를 $\theta$에 대해 미분하고 0으로 놓으면 최대값을 갖는 $\theta$ 값을 구할 수 있습니다.

$$\frac{\partial}{\partial\theta}\log L(\theta|D) = \frac{\sum_{i=1}^Nx_i}{\theta} - \frac{\sum_{i=1}^N(1-x_i)}{1-\theta} = 0$$

따라서, $\theta$의 MLE는 다음과 같이 구할 수 있습니다.

$$\hat{\theta}{MLE} = \frac{\sum{i=1}^Nx_i}{N}$$

MLE는 모수 추정의 일반적인 방법 중 하나이며, 다양한 분야에서 활용됩니다.

### Estimating a Probability Distribution: Setting

설정: 확률 분포의 추정
$p(y)$가 $Y$ 상의 확률 분포를 나타내고, $p(y)$를 추정하려고 합니다. 이때, $p(y)$가 연속적인 공간 $Y$ 상의 확률 밀도 함수(probability density function) 또는 이산적인 공간 $Y$ 상의 확률 질량 함수(probability mass function)인 경우가 있습니다. 이러한 경우에 대해 대표적인 예시로는 다음과 같은 것이 있습니다.

$Y = R$: 연속적인 공간 상에서의 일반적인 확률 분포 추정
$Y = R^d$: 다변량 연속적인 공간 상에서의 확률 분포 추정
$Y = {-1,1}$: 이진 분류(binary classification) 문제에서의 추정
$Y = {0,1,2,...,K}$: 다중 분류(multiclass problem) 문제에서의 추정
$Y = {0,1,2,3,4,...}$: 무한한 카운트(unbounded counts) 값을 가지는 경우에서의 추정
이러한 경우에 대해, 확률 분포 $p(y)$의 추정을 위해 다양한 방법이 사용될 수 있습니다.

여기서 설명하는 것은 확률 변수 $Y$에 대해 정의된 알려지지 않은 확률 분포 $p(y)$를 추정하는 문제입니다. $Y$가 연속 또는 이산 확률 변수인지에 따라 $p(y)$는 다른 형태를 취할 수 있습니다.

연속 확률 변수의 경우, $p(y)$는 연속 공간 $\mathbb{R}$ 또는 고차원 공간 $\mathbb{R}^d$와 같은 연속 공간 위에 정의된 확률 밀도 함수입니다. 목표는 관측 데이터 포인트 집합에서 기본 밀도 함수를 추정하는 것입니다.

이산 확률 변수의 경우, $p(y)$는 이산 공간 위에 정의된 확률 질량 함수입니다. 예를 들어, 이항 분류 문제에 대한 이진 집합 $Y={-1,1}$, $K$개의 가능한 범주를 가진 다중 분류 문제에 대한 정수 집합 $Y={0,1,2,\dots,K}$, 계수 데이터에 대한 $Y={0,1,2,3,\dots}$과 같은 공간이 있습니다.

알려지지 않은 분포 $p(y)$를 추정하는 문제는 통계 및 기계 학습에서 기본적인 문제이며, 이를 해결하기 위한 다양한 방법이 있습니다. 이러한 방법은 일반적으로 관측된 데이터 포인트 집합을 사용하여 기본 분포의 추정치를 구성하는 것을 포함합니다. 분포를 추정하는 인기 있는 방법 중 하나는 최대 우도 추정입니다. 최대 우도 추정은 관측된 데이터의 확률을 최대화하는 분포를 찾는 것을 목표로합니다.

### Evaluating a Probability Distribution Estimate

확률 분포 추정에 대해 이야기하기 전에, 평가하는 방법에 대해 이야기해보겠습니다.
어떤 사람이 우리에게 확률 분포의 추정치 pˆ(y)를 제공했다고 가정해 봅시다.
이 추정치가 얼마나 좋은지 평가하는 방법은 무엇일까요?
우리는 pˆ(y)가 미래 데이터를 잘 설명하는 것이 중요합니다. 즉, pˆ(y)가 우리가 앞으로 관측할 데이터를 잘 대표하는 것이 필요합니다.

### Likelihood of a Predicted Distribution

우리가 가지고 있는 것은 $D = (y1,..., yn)$이라는 데이터셋입니다. 이 데이터셋은 모두 $i.i.d.$ 방식으로 참인 분포 $p(y)$에서 추출된 것입니다.

그러면 이 데이터셋 $D$가 주어졌을 때, $pˆ$가 이 데이터셋 $D$에 대한 가능도(likelihood)를 가지게 됩니다. 이는 다음과 같이 정의됩니다.

$pˆ(D) = Yn(i=1) pˆ(yi)$

만약 $pˆ$이 확률 질량 함수일 경우, likelihood는 확률과 동일합니다. 이는 $pˆ(D)$가 $pˆ$가 $D$에 대한 확률 질량 함수이기 때문입니다.

최대 우도 추정법에서는 로그 우도 함수 $\ell(\theta \mid \mathcal{D})$를 최대화하는 모수 $\theta$를 찾는 것이 목적입니다. 로그 우도 함수를 미분하면 다음과 같습니다.

$$\frac{\partial}{\partial \theta} \ell(\theta \mid \mathcal{D}) = \sum_{i=1}^{n} \frac{\partial}{\partial \theta} \log p(x_i \mid \theta)$$

여기서 $x_1, x_2, \ldots, x_n$은 주어진 데이터 셋이고, $p(x_i \mid \theta)$는 모수 $\theta$ 하에서 데이터 $x_i$가 발생할 확률입니다.

이제 $\frac{\partial}{\partial \theta} \ell(\theta \mid \mathcal{D})$을 0으로 놓고 $\theta$에 대해 풀어서 최대 우도 추정값을 구합니다. 즉,

$$\frac{\partial}{\partial \theta} \ell(\theta \mid \mathcal{D}) = 0$$

을 만족하는 $\theta$를 찾습니다. 이때, $\frac{\partial}{\partial \theta} \ell(\theta \mid \mathcal{D})$이 0이라는 것은 $\ell(\theta \mid \mathcal{D})$이 극대값 혹은 극소값을 가진다는 것을 의미합니다. 최대 우도 추정에서는 $\ell(\theta \mid \mathcal{D})$가 최대가 되는 $\theta$를 찾는 것이 목적이므로, 이때 $\frac{\partial}{\partial \theta} \ell(\theta \mid \mathcal{D})$가 0이 되는 $\theta$가 최대 우도 추정값이 됩니다.

이러한 방식으로 최대 우도 추정값을 찾는 것은, 우도 함수를 미분하여 최대값을 찾는 방식으로써, 우도 함수가 모수 공간에서 볼록(convex)한 형태를 가지고 있을 때 항상 가능합니다. 이를 위해서는 우도 함수가 특정한 조건을 만족해야 하며, 이러한 조건을 만족하는 경우에는 편미분하여 최대값을 구할 수 있습니다.

## Parametric Families of Distributions
### Parametric Models

포아송 회귀 분석은 종속 변수가 이산형이며, 예측 변수가 하나 이상인 경우 사용되는 통계학적 모델링 방법입니다. 주로 카운트 데이터를 예측하는 데에 사용됩니다.

예를 들어, 고객이 한 주문에서 구매하는 제품의 수, 광고 클릭 수, 사고 발생 건수 등과 같은 이산적인 값들을 예측하는 데에 사용됩니다.

모델의 기본 가정은 종속 변수가 포아송 분포를 따른다는 것입니다. 이때 포아송 분포의 평균은 예측 변수들의 선형 결합으로 나타내어집니다. 이를 수식으로 나타내면 다음과 같습니다.

$$\log(\lambda_i) = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \cdots + \beta_p x_{ip}$$

여기서 $\lambda_i$는 i번째 사건의 기대값(평균)이며, $x_{ij}$는 i번째 사건의 j번째 예측 변수 값입니다. $\beta_0, \beta_1, \cdots, \beta_p$는 모델의 계수로, 이 값들이 모델 학습 과정에서 추정됩니다.

이 모델을 학습하기 위해서는 최대 우도 추정법(Maximum Likelihood Estimation, MLE)을 사용합니다. 즉, 주어진 데이터에 대해 가능도 함수를 최대화하는 계수 값을 찾아내는 것입니다. 가능도 함수는 주어진 데이터와 모델의 예측값 간의 차이를 나타내는 손실 함수(loss function)로, 로그-우도 함수(log-likelihood function)로 정의됩니다.

최종적으로 구한 모델은 예측 변수들과 종속 변수 사이의 관계를 설명할 수 있으며, 이를 이용하여 새로운 예측 변수 값에 대한 종속 변수 값을 예측할 수 있습니다.

조건부 가우시안 회귀모델에서는 입력 변수 $\boldsymbol{x}$와 출력 변수 $y$ 사이의 관계를 알아내기 위한 모델입니다. 이 모델에서는 선형 회귀모델을 기반으로 하며, 입력 변수와 출력 변수 사이의 관계가 선형적이라고 가정합니다. 즉, 출력 변수 $y$는 다음과 같이 선형 회귀식으로 표현됩니다.

$$y = \boldsymbol{w}^T\boldsymbol{x} + \epsilon$$

여기서 $\boldsymbol{w}$는 가중치(weight) 벡터이고, $\epsilon$은 오차항(error term)입니다. $\epsilon$은 정규분포를 따르며, 평균이 0이고 분산이 $\beta^{-1}$인 정규분포를 따릅니다. 이때, $\beta$는 모델의 정확도를 나타내는 매개변수입니다.

따라서, 모델에서는 $\beta$를 이용하여 오차항의 분산 대신 오차항의 정확도를 조절합니다. 이때, 분산과 정확도는 서로 역수 관계에 있으므로, 모델의 정확도가 높을수록 오차항의 분산은 낮아집니다.
 
따라서, 조건부 가우시안 회귀모델에서는 다음과 같은 조건부 확률 분포를 사용하여 출력 변수 $y$의 예측값을 구합니다.

$$p(y|\boldsymbol{x}, \boldsymbol{w}, \beta) = \mathcal{N}(y|\boldsymbol{w}^T\boldsymbol{x}, \beta^{-1})$$

이 확률 분포는 입력 변수 $\boldsymbol{x}$가 주어졌을 때 출력 변수 $y$의 분포를 나타냅니다. 이때, $\mathcal{N}$은 정규분포를 나타내며, $y$의 평균은 $\boldsymbol{w}^T\boldsymbol{x}$이고 분산은 $\beta^{-1}$입니다.

따라서, 조건부 가우시안 회귀모델에서는 입력 변수 $\boldsymbol{x}$와 가중치 벡터 $\boldsymbol{w}$, 정확도를 나타내는 매개변수 $\beta$를 이용하여 출력 변수 $y$의 예측값을 계산합니다.

[*[출처] : FOUNDATIONS OF MACHINE LEARNING by Bloomberg ML EDU*](https://bloomberg.github.io/foml/#home).
