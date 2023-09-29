---
title: MixMatch-A holistic approach to semi-supervised learning에 대하여
layout: post
description: paper review
use_math: true
post-image: https://user-images.githubusercontent.com/79881119/235352836-99c2a2ef-1197-4bb2-a730-1e48fcf60ded.jpg
category: paper review
tags:
- SSL
- AI
- Deep learning
---

준지도학습이란 흔히 일부 레이블이 존재하는 데이터에 대해 학습한 모델을 토대로 레이블이 없는 데이터셋에 대해서 학습을 진행하는 것을 의미한다. 즉, unsupervised learning과 동일하게 representation mapping을 어떠한 방식으로 해결할 것인지에 대한 분석이 중요하며, 본 게시글에서는 SSL(Semi-supervised learning)에서 가장 성능을 끌어올렸던 유명한 논문 중 하나인 MixMatch에 대해 리뷰하도록 하겠다. [논문링크](https://arxiv.org/abs/1905.02249)

# Deep learning의 구성 요소
뜬금없이 딥러닝의 구성 요소를 언급하고자 한 이유는 바로 그 안에 supervised, semi-supervised 등 딥러닝 연구 분야에 대한 갈림길이 내재되어있기 때문이다. 딥러닝이 성공한 주요 이유는 머신 러닝의 한 기법 중 하나인 Neural Network를 보다 효율적인 형태로 구성하고, 이를 gradient based 방식으로 최적화하기 위해 SGD, Adam 등 다양한 optimization 방법이 제안되었다. 또한 ReLU function이나 residual learning과 같이 현재에도 SOTA로 쓰일 정도로 좋은 모델링 방법이 많이 연구되었고, 여전히 transformer 구조 및 multimodal 등 다양한 형태의 연구가 진행되는 중이다.
이러한 서사를 막론하고, 결국 우리가 목표로 잡는 학습을 cost function, objective function이라 부르고, 이를 최적화하기 위해서는 충분히 많은 데이터가 필요하다. 예를 들어 이미지 분류 작업같은 경우, 단순히 이미지를 n개의 class로 구분하는 지표화 작업이 필요하지만, segmentation과 같은 경우 이미지 각 픽셀에 대한 지표화가 필수적이고, 이런 경우 time consuming 문제가 있다. 다른 문제로는 만약 의학적인 지식이 필요한 상황이라면(의료 CT, MRI 이미지를 통한 진단 딥러닝 알고리즘을 설계하려고 한다면), 레이블링에는 충분히 많은 사전 지식(도메인 지식)이 필요하기도 하다. 또한 만약 어떠한 영상이나 이미지를 보고 등장하는 사람이나 사물에 대한 private information이 지표화에 필요할 경우, 사생활 문제가 야기될 수 있다.
결국 길게 말하고자 한 것은 방대한 데이터셋과 augmentation 방법으로 딥러닝 네트워크 파라미터로 하여금 보다 일반화에 가까운 representation을 학습한 것이 딥러닝이 좋은 성능을 낼 수 있는 키포인트인데, 이러한 데이터셋을 얻기 위한 과정이 순탄치가 않다는 것.

# SSL : Semi-Supervised Learning
위에서도 사용했지만, 준지도 학습에 대한 단어는 SSL로 통일하도록 하겠다. 준지도 학습에서 지표화된 데이터셋의 필요를 줄이기 위해서 접근한 연구 방식은 모델로 하여금 unseen data에 대해 unlabeled dataset이 보다 일반화에 도움을 줄 수 있게끔 해주기 위한  "loss term"을 잘 설정하는 데에 있었다.
예를 들어 라벨링이 되어있지 않은 그림에 대해서 우리는 고양이 사진을 보고 당연히 "고양이"라고 오차 없이 예측할 수 있지만, 네트워크는 모든 분류 작업을 확률 기반(softmax)으로 계산하기 때문에 이러한 형태의 확신을 가지기 힘들다. 이는 이후 SSL을 MixMatch라는 알고리즘으로 접근한 해당 논문의 내용에서 더 자세히 다루도록 하고, loss term을 바꾸는 연구의 세 가지 큰 방식을 구분하도록 하겠다.

## Entropy minimization
첫 번째는 Enropy minimization이다. 일반적으로 지표화가 진행되지 않은 데이터를 모델이 예측하면 확률값으로 매핑이 된다.
<p align="center">
  <img src="https://user-images.githubusercontent.com/79881119/209059028-f13d36bd-c752-4c19-9cc8-b016185d4f75.jpg"/>
</p> 
만약 지표화가 된 상황에서의 데이터라면 "개, 고양이, 원숭이"의 3가지 클래스를 구분하는 작업에 있어서 1-hot encoding(이를 hard label이라고도 부른다)을 수행한다. One-hot encoding이란 정답인 확률이 1이고 나머지가 0인 상황이다. 따라서 위의 그림을 그대로 지표화하게 되면 (0, 1, 0)이 되는 것이다.
<p align="center">
  <img src="https://user-images.githubusercontent.com/79881119/209059035-69102d07-26c6-48fb-9b8d-2b15aa0ac992.png"/>
</p>
그러나 이를 네트워크에 통과시킨 결과는 다르다. 매우 잘 학습한 모델이 세 클래스에 대해 결과를 아무리 잘 예측하더라도 (0, 1, 0)이 되기 힘들다.
이는 cross entropy loss의 특성상 softmax를 포함하는 데에서 그 한계를 찾을 수도 있는데, 점수표가 어떤 방식으로 설정되든 이에 대한 CE loss를 계산하기 전 softmax 연산을 통해 확률값으로 매핑한다.
<p align="center">
  <img src="https://user-images.githubusercontent.com/79881119/209059037-8d487115-45d6-46e6-88d7-920feb69f796.png"/>
</p>
그러나 softmax 함수의 경우 0과 1을 점근선으로 가지기 때문에, 실제로는 점수표 상의 그 어떠한 value도 softmax 상에서 0과 1의 절대적인 값을 가질 수 없고, 결론적으로는 잘 학습된 모델의 경우에도 '고양이일 확률이 가장 높다' 정도의 예측이 최선인 것이다.
결국 네트워크를 통과한 각 클래스에 대한 예측값은 확신이 없다고 볼 수 있는데, 이는 지표화되지 않은 데이터셋에 대해 학습을 하게 될 경우 문제가 생긴다. 그렇기 때문에 entropy minimization을 통해 애매한 확률값들을 확실한 값으로 바꿔준다. 이에 대한 내용은 이후 모든 loss term에 대해 설명한 후에 종합적인 분석이 필요한 부분이 있어 뒤로 넘기도록 하겠다.

## Consistency regularization
두 번째는 Consistency regularization이다. 이는 생각보다 간단한 개념인데,
<p align="center">
  <img src="https://user-images.githubusercontent.com/79881119/209059039-fb2d2135-7f02-46e5-b713-c73cb6a67e41.png"/>
</p>
예를 들어 위와 같이 고양이 사진을 90도 회전시킨 augmented sample을 unlabeled dataset으로 사용한다고 생각해보자. 서로 다르게 augmented(노이즈 추가, 컬러 변경, 회전 등)된 두 이미지는 사실 서로 같은 probability distribution을 가져야 한다. 즉 지표화되지 않은 샘플에 대해 예측이 들쭉날쭉하게 변하지 않게 하는 것이 정규화 방식이다.

## Generic regularization
Model의 overfitting을 방지하는 정규화 방식으로, 이후 설명할 MixMatch에서의 MixUp과 관련이 있다. 예를 들어 두 이미지에 대해 지표화가 되어있다고 가정하자. 강아지의 경우 (1, 0, 0)의 라벨을 갖게 되고, 고양이의 경우 (0, 1, 0)의 라벨을 가지게 된다. 이 두 샘플에 대한 convex sample은

\[
    \theta \cdot cat + (1-\theta) \cdot dog,~(0<\theta<1)
\]

라고 볼 수 있다. 여기서 convex sample이란 두 샘플을 convex set의 한 지점으로 보고, 그 사이를 보간하는 모든 sample이 포함되는 convex set의 정의를 그대로 따른다고 볼 수 있다. 이를 실제로 시각화하면,
<p align="center">
  <img src="https://user-images.githubusercontent.com/79881119/209059042-6f90bb89-7f19-4a0f-b7cc-a8756fd5c746.png"/>
</p>
이렇게 개냥이가 각각 50%씩 첨가된 샘플이 생성된다. 해당 샘플에 대한 라벨은 마찬가지로 convex sample과 같은 공식에 따라

\[
    \theta \cdot (0,\ 1,\ 0) + (1-\theta) \cdot (1,\ 0,\ 0)\ = (0.5,\ 0.5,\ 0)
\]

이러한 방식을 MixUp이라 부른다.
앞서 여러 가지의 정규화 방식을 소개하였고, 이제 본격적으로 MixMatch에서 어떠한 방식을 통해 위와 같은 여러 알고리즘을 통합하여 준지도학습을 진행할 수 있었는지 천천히 소개하도록 하겠다.

# Related works
준지도 학습의 경우 관련 내용이 좀 있는데, MixMatch 논문에서는 전혀 언급하지 않는 분야도 있다. 이 중에 가장 유명한 transductive model, graph-based model 그리고 generative model에 대해 간단하게 소개하도록 하겠다.
<p align="center">
  <img src="https://user-images.githubusercontent.com/79881119/209059045-b04b1a3a-5b87-42c1-8b93-0d4ca790cc43.png"/>
</p>
Transductive learning은 Inductive learning과 다르게, 각 노드(데이터셋)와 엣지(라벨)에 대해 일부 노드에 대한 엣지 정보만 가지고 나머지 노드에 엣지를 부여하는 작업이다. 따라서 그래프 개념으로 해석한 SSL 그 자체로 보면 된다. 그래프 based model도 비슷한 형태로 생각해주면 된다. 물론 위와는 다르게 노드와 엣지의 느낌이 약간 다른데, Graph-based modeling에서 각 노드를 데이터셋으로 보는 방식은 transductive learning에서 해석하는 것과 같지만, 엣지는 유사성을 나타낸다.
<p align="center">
  <img src="https://user-images.githubusercontent.com/79881119/209059047-34b7ea16-352b-4d71-84f2-f5109687188d.png"/>
</p>
간단하게 MNIST 데이터셋으로 기준을 보인다면, 같은 숫자일수록 그래프 상에서 엣지(선으로 표현된 부분)가 강하게 나타날 것이고 이는 곧 유사한 클래스의 데이터일수록 높은 유사도(그래프 상에서는 거리가 가깝다고 역으로 이해할 수 있다)를 보인다고 생각할 수 있다. 에너지 based로 생각하는 것, Hessian과 관련된 수식 증명의 경우 나중에 기회가 된다면 따로 다룰 것이고 오늘 언급할 페이퍼는 해당 내용을 신경쓰지 않기 때문에 넘어가도록 하겠다.
<p align="center">
  <img src="https://user-images.githubusercontent.com/79881119/209059049-7a484a0f-631e-47a8-89a2-629b361e2fe9.png"/>
</p>
Generative modeling 방식은 말 그대로 생성 모델링을 통해 heuristic한 준지도 학습을 진행하게 된다. 이를 테면 노이즈를 제거하는 방식이 될 수도 있고, 이미지에 color를 입히는 작업이 될 수도 있으며 perturbation(빈 부분, 손상된 부분)을 복원하거나 서로 다른 채널을 예측하는 형태로 진행된다.

---

# Build up for MixMatch
Mixmatch를 언급하기 전에 관련된 준지도학습 관련 내용을 간단하게 언급했다. 위의 내용은 사실상 related works라고 보기는 힘들되, semi-supervised learning을 풀어가려는 다양한 방법론으로 제시가 되고 있다.
그렇다면 MixMatch에서 아이디어로 삼게 된 여러 알고리즘에 대한 기본 내용을 보다 자세히 언급하도록 하겠다.
그 중 가장 첫 번째는 Consistency regularization으로, Augmentation이 서로 다르게 적용되었다고 하더라도 같은 라벨을 예측해야한다는 것을 네트워크 학습에 이용하게 된다.
따라서 stochastic한 함수 Augment(x)가 존재하고, 만약 같은 input image X에 대해 랜덤한 augmentation을 적용하면, 이에 대한 parameterized 모델의 예측은

\[
    p_{model}(y \vert \text{Augment}(x); \theta), p_{model}(y \vert \text{Augment}; \theta)    
\]

와 같이 두 개로 나온다. 여기서 주의할 점은 Augment() 함수 자체가 stochastic하다고 했으므로, 두 개의 term은 서로 다른 예측값을 가진다(같은 value가 아님). 따라서 모델은 다음과 같은 loss term을 최소화하는 방향으로 학습된다.

\[
    \parallel p_{model}(y \vert \text{Augment}(x); \theta) - p_{model}(y \vert \text{Augment}; \theta) \parallel_2^2
\]

Mean Teacher 방식에서는 두 개의 term을 서로 다른 모델링을 통해 해결하는데, 바로 아래와 같은 그림을 보면 student model의 경우에는 똑같은 방식으로 최적화가 진행되지만, teacher model은 student model의 parameter를 exponential moving average 방식으로 가져와 사용한다. Exponential moving average를 잘 모른다면 그냥 단순히,

\[
    w_{k+1}^{teacher} = \beta w_k^{teacher} + (1-\beta)w_k^{student}    
\]

<p align="center">
  <img src="https://user-images.githubusercontent.com/79881119/209059052-e06308cd-62b4-4969-8a05-1be0d97cb338.png"/>
</p>
처럼 기존 weight에 student weight를 업데이트하는 방식을 사용한다고 생각하면 된다. 자세한 내용은 [여기](https://arxiv.org/abs/1703.01780)를 참고. 그리고 VAT라는 방식(Virtual Adversarial Training)에서는 Adversarial sampling 방법 중에 maximally changes output class distribution을 이용한 perturbation 방식(모델을 가장 혼란스럽게 만드는 augmentation이라고 보면 된다)을 사용, hard sampling을 통해 같은 방식으로 최적화를 한다. MixMatch에서는 단순한 data augmentation 방식으로 볼 수 있는 random horizontal flips and crops를 사용한다.

두 번째는 엔트로피 최소화이다. 사실 그냥 Entropy minimization을 단순 번역한 것. 곰곰히 생각해보면 정보 이론에서 엔트로피가 어떤 식으로 정의되는지 혹시 기억할지 모르겠다. 만약 랜덤 변수 space X에서 각 랜덤 변수가 추출될 확률을 $P = (p_1, p_2, p_3, ...)$ 등으로 정의한다면 해당 space에서의 엔트로피는

\[
    \begin{aligned}
        \text{for }X =& (x_1,~x_2,~\cdots,~x_N)\text{ where each random variable }x_i(i=1,~\cdots,~N)\text{ has a probability }P = (p_1,~\cdots,~p_N), \newline
        H(X) =& -\sum_{i=1}^N p_i \log(p_i)
    \end{aligned}    
\]

라 할 수 있다. 물론 지금 이 상황에서는 이산 확률에 대한 가정이지만, 결론적으로 말하자면 얼마나 분포가 고르냐/고르지 않냐의 문제로 귀결된다.

앞서 설명했던 바와 같이 지표화가 진행되지 않은 샘플에 대한 예측은 실제 one-hot encoding 방식에 비해 라벨링 자체의 엔트로피가 높게 생성된다. 또한 앞서 언급한 여러 data augmentation을 거친 샘플들에 대한 예측 결과는 더욱 entropy를 증가시키는 요인이 될 것이다.

이렇게 라벨 대신에 사용할 모델의 예측 확률값들을 Pseudo-Label이라 부르기로 했고, 우리는 이러한 유사 라벨들을 실제 학습에 활용하기 위해 다음과 같은 전략을 세운다.

1. $K$개의 augmentation을 같은 데이터 $X$에 취한다.

2. 각각의 augmented data $K$개를 모델에 통과시킨 예측 확률 map에 평균을 취하고, average label을 생성한다.

3. Average label에 앞서 언급한 entropy minimization을 수행한다.

Sharp label을 만들기 위한 entropy minimization은 temperature hyperparameter $T$에 의해 결정된다. 일반적인 probability에,

\[
    Sharpen(p,~T)_i = \frac{p_i^{1/T}}{\sum^L_1 p_j^{1/T}}
\]

이와 같이 적용한 새로운 확률 맵을 이용하는 것이다. 이를 실제로 시각화하여 보면 다음과 같다.
<p align="center">
  <img src="https://user-images.githubusercontent.com/79881119/209059056-4f683a5d-ce95-4a72-8248-e3a1dd1abb4c.png"/>
</p>
$T = 1$이면 원래의 확률 맵과 동일하다. 위와 같이 균등균등하게 설정한 확률 맵에서는 유사한 확률값(0.15, 0.13, 0.12)가 실제 모델 학습에서 dense region problem을 일으킬 수 있다. 이게 무슨 소리냐면,
<p align="center">
  <img src="https://user-images.githubusercontent.com/79881119/209059057-a0ece861-efba-405b-b135-deb59f99f388.jpeg"/>
</p>
위와 같은 그림에서, 진하게 표시된 십자가 모양과 삼각형 모양이 라벨링 된 데이터고 이에 대해 학습을 진행한 후에 unlabeld 샘플(파란색/주황색 점들)에 대해 decision boundary를 고려하는 상황이라면, 실선으로 나와있는 경계선보다 점선으로 나와있는 경계선이 분포 상으로 덜 밀집된 부분을 지나가기 때문에 적절한 경계선으로 보인다. 이렇듯 경계선이 밀도가 높은 지점을 지나게 되면, 해당 경계선 위치에 있는 샘플의 경우 보다 확률이 애매하게 매핑되기 때문에 이를 방지하기 위한 minimization 방법을 고안하게 된 것이다.
<p align="center">
  <img src="https://user-images.githubusercontent.com/79881119/209059060-042b9c8c-5546-443b-bd44-6b8050fba5ca.gif"/>
</p>
실제로 $T$값을 점차 감소시키면서 위의 식을 적용해보는 모습이다. $T$가 $0$에 가까워질수록 분포는 one-hot encoding에 가까워지고 $T \rightarrow 0$가 되면 one-hot encoding에 수렴한다.
<figure class="half">
    <img src="https://user-images.githubusercontent.com/79881119/209059063-258cba67-de1e-4de7-ada3-b6d32971acdb.png" width="600" />
    <img src="https://user-images.githubusercontent.com/79881119/209059066-0e54d4e6-842d-4fc8-9ed7-3545f8d9a574.png" width="600" />
</figure>

# MixMatch algorithm
MixMatch 알고리즘이 사용하는 loss objective는 크게 두 가지로 구분된다. 준지도 학습을 구성하는 labeled dataset과 unlabeled dataset 각각에 대해 적용되는 loss(CE loss for labeled sample, consistency loss for unlabeled sample)이 서로 다르기 때문이다.

뒤이어 알고리즘 전반에 대해 디테일하게 설명하기 전에, $X', U'$는 각각 labeled dataset으로부터의 augmented dataset 그리고 unlabeled dataset으로부터의 augmented dataset을 의미한다.

\[
    X',~U' = MixMatch(X, U, T, K, \alpha)    
\]

$X, U$는 augmentation이 진행되기 전 각 dataset을 의미하고 $T$는 entropy minimization에 사용되는 temperature, $K$는 unlabeled dataset에 적용될 augmentation 개수, $\alpha$는 MixUp에 사용될 convex coefficient에 해당된다.

\[
    L_X = \frac{1}{\vert X' \vert} \sum_{x, p \in X'} H(p, p_{model}(y \vert x; \theta))   
\]

당연하게도 라벨이 존재하는 데이터에 대해서는 원래의 label에 대한 cross entropy loss를 적용하게 되고,

\[
    L_U = \frac{1}{L\vert U' \vert} \sum_{u, q \in U'} \parallel q-p_{model}(y \vert u; \theta) \parallel_2^2    
\]

라벨이 존재하지 않는 데이터세 대해서는 pseudo label $q$를 적용한 consistency regularization loss를 사용한다. 이 두 개를 잘 섞어서 사용한다고 생각하면 된다. 사실 수식만 봐서는 아직 잘 이해가 안될 분들을 위해 직접 알고리즘 코드 한 줄 한 줄 설명해드리도록 하겠다.

<p align="center">
  <img src="https://user-images.githubusercontent.com/79881119/209059068-c301914f-e460-4e6f-accd-0c745e74107f.png"/>
</p>
1~6번째 줄을 먼저 보도록 하자. 입력으로는 같은 배치 크기의 labeled dataset과 unlabeled dataset을 사용하고, labeled dataset $x$에 대해서는 stochastic augmentation을 한 개 적용하고, unlabeled dataset $u$에 대해서는 stochastic augmentation을 $K$개 적용한다.

<p align="center">
  <img src="https://user-images.githubusercontent.com/79881119/209059073-bbebd988-7b15-4cef-9606-556bb05ac16c.png"/>
</p>
요 부분에서 pseudo label이 결정되는데, $K$개의 augmented된 unlabeled sample인 $\hat{u}$ 애들을 가지고 각각 모델의 예측값을 뽑아낸 뒤, 이를 $K$로 나누어 평균 예측값을 구하게 된다. 논문에서도 설명하겠지만 Pseudo-labeling 과정에 대해서는 최적화를 먹이지 않는다고 한다. 즉 오로지 현재 모델의 예측값을 기준으로 삼는다는 것. 그런 뒤 temperature hyperparameter $T$에 대해 sharpening을 진행하면 pseudo label $q_b$를 생성할 수 있게 된다.

<p align="center">
  <img src="https://user-images.githubusercontent.com/79881119/209059075-530284c1-90de-4a4e-ab0f-9139e508f01f.png"/>
</p>

10~12번째 줄은 label data(augmentation 이후) + unlabeled data(augmentation 이후)를 서로 합친 뒤에 셔플링하는 과정이다. 섞게 되면 총 $B + B \times K$개의 샘플이 무작위로 나열되고, 이를 하나의 queue 혹은 dequeue 자료형으로 생각한다.

<p align="center">
  <img src="https://user-images.githubusercontent.com/79881119/209059078-b9d40da6-ec11-4bab-904f-dad2d09ae580.png"/>
</p>

무작위로 나열된 샘플 배치 내에서 X'(labeled dataset)과 MixUp을 진행하고, 나머지 샘플들을 이용하여 U'(unlabeled dataset)과의 MixUp을 진행한다. 이렇게 진행된 MixUp은 lambda 값에 따라 labeled data 혹은 unlabeled data와의 중요도를 결정하는데,
만약 단순히 샘플링한 W를 MixUp에 사용하면, 구체적으로 labeled dataset과의 MixUp, 혹은 unlabeled dataset과의 MixUp에 대한 중요도가 사라지게 된다. 따라서

\[
    \begin{aligned}
        \lambda =& Beta(\alpha,~\alpha) \newline
        \lambda' =& \max (\lambda,~1-\lambda) \newline
        x' =& \lambda'x_1 + (1-\lambda')x_2 \newline
        p' =& \lambda'p_1 + (1-\lambda')p_2
    \end{aligned}
\]

이와 같이 샘플을 MixUp하게 되면 Vanila MixUp(lambda를 서로 같은 값으로 둠)에서 무시했던 batch order를 유지하면서 MixUp sample를 생성할 수 있게 된다.

<p align="center">
  <img src="https://user-images.githubusercontent.com/79881119/209059081-b3812560-90f9-4fed-954b-52d6519ecc9c.png"/>
</p>

실제로 CIFAR-10, SVHN에 대해 250~4000 label를 가지고 SSL을 진행한 MixMatch 방식과 오차율을 비교하게 된다. Supervised method는 당연히 다른 방법들에 비해 좋은 것이 맞고, 검은색(제안된 방법)이 적은 라벨을 가지고도 representation을 효과적으로 학습할 수 있음을 보인다.

<p align="center">
  <img src="https://user-images.githubusercontent.com/79881119/209059086-cc8045b7-cc5a-4a0c-844e-f43dd1dfeba7.png"/>
</p>

주요 contribution이라 함은 이런 저런 loss term과 관련된 SSL 방식을 최적화하는 알고리즘을 효율적으로 잘 설계했다는 점이 될 수 있겠다.