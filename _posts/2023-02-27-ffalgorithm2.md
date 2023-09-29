---
title: 딥러닝의 체계를 바꾼다! The Forward-Forward Algorithm 논문 리뷰 (2)
layout: post
description: Forward-forward algorithm
use_math: true
post-image: https://user-images.githubusercontent.com/79881119/222030729-fc5d7460-8d26-4d6a-a1a8-fbc71131c9f1.gif
category: paper review
tags:
- Deep learning
- FF algorithm
- Methodology
- Generative model
- Restricted Boltzmann Machine
---
# 들어가며...
이번 포스트는 바로 이전에 작성된 글인 <U>FF(Forward-forward algorithm)</U> 소개글([참고 링크](https://junia3.github.io/blog/ffalgorithm))에 이어 Hinton이라는 저자가 FF 알고리즘의 학습을 contrastive learning과 연관지어 설명한 부분을 자세히 다뤄보고자 작성하게 되었다.

# Hinton의 RBM(Restricted Boltzmann Machine)에 대하여
머신러닝을 공부했던 사람이라면 모를 수 없는 Andrew Ng이라는 사람이 Hinton과 [인터뷰](https://www.youtube.com/watch?v=-eyhCTvrEtE&list=PLfsVAYSMwsksjfpy8P2t_I52mugGeA5gR&ab_channel=PreserveKnowledge)하면서 인생에 있어 가장 큰 업적을 고르라했을 때, Hinton은 [restricted boltzmann machine](https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine)을 손꼽았다.
RBM은 backpropagation과 더불어 딥러닝 연구의 학습 방법론 중 하나인데, 놀랍게도 GAN, Diffusion model처럼 RBM 또한 generative(생성) 모델의 한 축이라고 볼 수 있다.
사실 Hinton은 RBM을 처음으로 제시하지는 않았고 그의 [저서 중 하나](https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf)에서 학습법에 대해 논하였으며, 다양한 application으로 개념을 올린 사람이라고 할 수 있다.

### Generative model
생성 모델에 대해서 간단하게 설명하자면 기존의 understanding based neural network 구조인 ANN, DNN 그리고 CNN같은 disciminative model과는 다르게 '확률 밀도 함수(<U>Probablistic Density Function</U>)'를 모델링하고자 하는 것이다.
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/222032590-7803a29d-b91c-47af-8799-547927b6ef3d.png" width="800">
</p>
뒤에서 보다 자세히 설명하겠지만 RBM을 이용하는 sampling 방식은 deep neural network의 초기화와 관련이 있으며, <U>특정 distribution을 가지는</U> 데이터가 가장 효과적으로 생성될 수 있는 <U>latent factor</U>를 찾는 과정이 된다.

확률 밀도 함수에 대해 알고자 하는 것은 복잡한 task이기 때문에, 이를 위해서 다음과 같이 예시를 들어보도록 하자. 예를 들어 '<U>사람의 얼굴</U>'을 그럴듯하게 생성하고자 한다고 생각해보자. 얼굴을 구성하기 위해서는 여러 요소들이 필요한데, 이들을 각각 'feature'로 생각하고, feature에는 <U>여러 가지 가능한 변수 상태</U>(state)가 존재한다. 뒤에서 energy based approach로 식을 전개하는 과정에서 <U>변수의 상태</U>가 곧 해당 <U>변수가 차지하는 확률 공간에서의 입지</U>(probability)를 의미하기 때문에 잘 기억해두면 좋다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/222034574-6220e53c-6b1d-4fe6-aed6-f2be571f7082.png" width="800">
</p>

즉 어떠한 이미지를 구성하는 각 요소들을 가능한 state의 집합으로 생각하고, 집합의 각 요소들이 가지는 확률 분포를 모델링할 수 있다는 것이다. 꼭 얼굴 이미지가 아니더라도 <U>특정 데이터셋에 대해</U> 이러한 방식으로 <U>세부 요소들</U>을 machine이 모델링할 수 있다면, 각 요소들의 조합을 통해 샘플을 그럴듯하게 만들 수 있다는 것이다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/222035138-704f11a3-cd5e-4b5b-a37e-6cf596b8e29f.png" width="300">
</p>

예컨데 StyleGAN의 경우에는 Synthesizer가 $\mathcal{W}$ space의 latent vector를 style vector로 affine transform하여 coarse detail부터 fine detail까지 그려내는 모습을 볼 수 있는데, 이러한 GAN 구조도 결국 <U>각 hidden layer</U>에서 구현할 수 있는 <U>얼굴 특징에 대한</U> 확률 밀도 함수를 학습한 것과 같다.

### Boltzmann machine
앞서 설명한 내용을 기반으로 기계학습에 pdf를 피팅하기 위해서는 각 state에 대해 <U>computational 구조를 모델링</U>해야하는데, 바로 여기서 가져올 수 있는 모델링이 <U>볼츠만 머신</U>이다. 볼츠만 머신은 생각보다 굉장히 간단한 개념이다. Visible(시각적으로 보이는 부분) part에서 샘플링을 하는 방법이 복잡하기 때문에, hidden(은닉층 부분) part에서 <U>implicit한 요소들에 대한 state를 정의</U>하고자 하는 것이다. 즉 <U>'눈에 보이진 않지만 존재하는 무언가'</U>를 내재하는 그래프 구조를 만들게 된다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/222036056-4895ec47-6132-4ed0-b504-369b568cbb85.png" width="300">
</p>

위의 그래프 구조에서 각 노드(노란색/초록색 부분)이 각 state라고 생각하면 된다. <U>노란색</U>을 우리에게는 좀 더 <U>가까운 level에서의 feature</U>(얼굴 이미지에서의 눈, 코, 입), <U>초록색</U>을 우리가 알 수는 없지만, 각 feature node들의 connection에 의한 activation으로 표현되는 은닉 요소들이라고 보면 된다. 하지만 위의 구조를 토대로 계산하게 되면 <U>은닉 요소(hidden state)</U>와 <U>가시 요소(visible state)</U> 간의 관계 이외에도 같은 레이어에서의 요소 간의 관계 또한 정의해야한다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/222037181-ef2359c5-cc9b-4522-b732-cf8895907538.png" width="800">
</p>

그렇기 때문에 위와 같이 <U>같은 레이어 상에서의 관계를 없앤</U> 모델링을 통해 constraints를 주어, 확률 연산이 보다 간단할 수 있게 구성한 것이 바로 <U>'Restricted'</U> 볼츠만 머신이다. 기본 볼츠만 머신은 각 레이어의 학습을 조건부로 표현할 수 없기 때문에 $p(h,~v)$의 joint를 연산해야하지만, 제한된 볼츠만 머신은 각 레이어의 학습을 조건부인 $p(h \vert v),~p(v \vert h)$로서 정의가 가능하다.

단순히 각 레이어에서의 의존성을 없애게 되면 굉장히 흥미로운 일이 일어나는데, 볼츠만 머신이 우리가 흔히 알고 있는 <U>feed forward neural network 구조처럼</U> 변하게 된다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/222039046-50d50184-5bbd-4e8f-9723-150158598807.png" width="400">
</p>

이렇게 단순화된 RBM을 학습하는 방법은 Hinton이 제시했는데, 먼저 간단하게 설명하자면 forward propagation을 통해 visible state로부터 hidden state를 결정하게 되고, 다시 hidden state로부터 visible state 상태를 결정하는 loop를 구성하게 된다.

### RBM의 수학적 모델
RBM은 확률 분포를 학습하기 위한 state의 모델링 방식으로, layer 의존성만 남겨 결국 <U>neural network와 같은 학습이 가능</U>한 형태라고 앞서 설명했다. RBM이 실제로 확률 분포를 학습하는 과정에서의 <U>goodness</U>(잘 학습했음)을 <U>수학적으로 표현하는 과정</U>과 그 <U>구조</U>에 대해서 살펴보도록 하자.

RBM의 구조는 <U>visible unit</U>들로 구성된 <U>visible layer</U>, 그리고 각 layer를 연결하는 weight matrix로 구성된다(노드를 <U>연결하는 선</U>이 각각 weight가 있다고 생각하면 된다).

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/222041275-a983201c-8f6f-43af-9311-b6c90ab25756.png" width="800">
</p>

수학적 모델을 정의하는 과정에서 <U>사용될 notation과 구조</U>는 위와 같다. State인 $v, h, b, c$는 모두 벡터가 되며, 이들 간의 관계를 정의하는 $W$는 matrix가 된다. 예를 들어 visible layer의 unit의 갯수가 $d$이고 hidden layer의 unit의 갯수가 $n$이라면,

\[
    \begin{aligned}
        v,~b \in & \mathbb{R}^{d \times 1} \newline
        h,~c \in & \mathbb{R}^{n \times 1} \newline
        W \in & \mathbb{R}^{n \times d}
    \end{aligned}    
\]

이처럼 표현할 수 있다.

### Energy based learning
RBM의 학습에 대한 goodness 또한 cost function으로 정의할 수 있는데, 이때 'Energy'라는 개념을 사용하게 된다. 앞서 각 노드가 의미하는 것은 확률 밀도 함수의 한 축을 담당하는 state라고 했는데, 이때 <U>특정 상태</U> $x$에 대한 에너지 $E(x)$를 곧 해당 상태(state)에 대응하는 값으로 생각해볼 수 있다. 물리학 개념을 생각해보면 에너지가 <U>낮을수록 안정적</U>이기 때문에, 모든 가능한 state $x$에 대한 에너지 분포를 해당 state가 존재할 확률로 normalizing할 수 있게 된다. 바로 다음과 같이 말이다.

\[
    \begin{aligned}
        p(x) =& \frac{\exp(-E(x))}{Z}    \newline
        \text{where } Z =& \sum_x \exp(-E(x))
    \end{aligned}
\]

바로 여기서 왜 해당 모델링이 '볼츠만 머신'이라 불리는지 이해할 수 있다. 통계 역학이나 수학에서 <U>Boltzmann distribution</U>은 시스템이 특정 state의 에너지와 온도의 함수로 <U>존재할 확률을 제공</U>하는 probability distribution function이다. 실제로 볼츠만 머신이 볼츠만 분포와 같이 temperature에 대한 정의를 주지는 않지만, 각 노드가 가지는 position을 해당 상태의 에너지로 정의하고, 이를 정규화한 <U>볼츠만 분포 형태로 근사하고 싶은 것</U>이다.

RBM에서는 visible unit인 $v$와 hidden unit인 $h$의 각 unit state에 따라 에너지를 결정할 수 있는데, <U>hidden unit</U>에 대한 energy는 <U>관찰할 수 없기 때문</U>에 <U>visible unit</U>에 대한 <U>확률 분포를 결정</U>할 수 있다.

\[
    \begin{aligned}
        p(v) =& \frac{\exp(-E(v,~h))}{Z}    \newline
        \text{where } Z =& \sum_v \sum_h \exp(-E(v,~h))
    \end{aligned}
\]

그리고 다시 hidden unit에 의해 복잡해진 energy function을 free energy $F(\cdot)$를 통해 다음과 같이 단순화할 수 있다.

\[
    \begin{aligned}
        p(v) =& \frac{\exp(-F(v))}{Z^\prime}    \newline
        \text{where } F(v) =& -\log \sum_h \exp(-E(v,~h)) \newline
        \text{and } Z^\prime =& \sum_v \exp(-F(v))
    \end{aligned}
\]

Free energy를 해석하면 <U>'모든 hidden state'</U>에 대한 unnormalized negative log likelihood의 총합이고, 이렇게 정의를 할 수 있는 이유는 RBM에서 <U>hidden state 서로에 대한 의존성을 배제</U>했기 때문이다. RBM에서 Energy는 다음과 같이 정의된다.

\[
    E(v,~h) = -b^\top v - c^\top h - h^\top Wv    
\]

Energy 식은 총 세 부분으로 구성된다. Visible layer의 state vector에 대한 biased term($-b^\top v$) 그리고 hidden layer의 state vector에 대한 biased term($-c^\top h$), 마지막으로 두 state간의 weight 관계($-h^\top Wv$)이다. Bias term인 $b$ 그리고 $c$는 <U>각 layer 전반의 특성을 반영</U>한 값이 되고, weight는 bias term이 보지 못하는 <U>레이어 사이의 관계를 반영</U>한 값이 된다고 생각하면 된다.

---

# Bernoulli RBM

앞서 설명했던 <U>얼굴 생성과 관련된 state</U>는 각각 여러 상태를 가질 수 있지만, Bernoulli RBM의 경우에는 visible/hidden unit 각각 $0$ 혹은 $1$의 상태만 가지는 경우를 다루게 된다. 앞서 살펴본 free energy $F(\cdot)$로 전개된 식을 살펴보면,

\[
    \begin{aligned}
        F(v) =& -\log \sum_h \exp (-(-b^\top v - c^\top h - h^\top Wv)) \newline
        =& -\log \sum_h \exp(b^\top v + c^\top h + h^\top Wv) \newline
        =& -\log \sum_h \exp(b^\top v) \exp(c^\top h + h^\top Wc) \newline
        =& -\log \left( \exp(b^\top v) \sum_h \exp(c^\top h + h^\top Wc) \right) \newline
        =& -b^\top v -\log \sum_h \exp(c^\top h + h^\top Wc)
    \end{aligned}
\]

위와 같이 정리할 수 있다. 그리고 $h$가 곧 $0$ 혹은 $1$ 이므로(Bernoulli)

\[
    \begin{aligned}
        -b^\top v& - \sum_{i=1}^n \log (\exp (0) + \exp (c_i + W_i v)) \newline
        =& -b^\top v - \sum_i \log (1 + \exp(c_i + W_i v))
    \end{aligned}    
\]

위와 같이 표현할 수 있다. RBM은 input에 대해 의존하는 neural network 구조와는 다르게 visible layer를 이용하여 hidden layer의 state를 생성할 수도 있지만, <U>반대로 hidden layer를 이용하여</U> 다시 <U>visible layer를 생성</U>할 수 있다. Visible layer가 주어졌을 때의 조건부 확률을 energy based로 전개하면 다음과 같다.

\[
    \begin{aligned}
        p(h \vert v) =& \frac{p(h,v)}{p(v)} = \frac{\exp (-E(h, v))/Z}{\sum_h p(h, v)} \newline
        =& \frac{\exp(-E(h, v))/Z}{\sum_h \exp(-E(h, v))/Z} \newline
        =& \frac{e^{b^\top v} e^{c^\top h + h^\top Wv}}{\sum_h e^{b^\top v} e^{c^\top h + h^\top Wv}} \newline
        =& \frac{e^{c^\top h + h^\top Wv}}{\sum_h e^{c^\top h + h^\top Wv}}
    \end{aligned}    
\]

총 $n$개의 hidden unit이 있을때, 이 중에서 하나의 hidden unit이 $1$인 값을 가질 확률은 sigmoid 함수인 $\sigma(\cdot)$으로 표현할 수 있다.

\[
    \begin{aligned}
    p(h_i = 1 \vert v) =& \frac{e^{c_i + W_i v}}{\sum_h e^{c_i h_i + h_i W_i v}} \newline
    =& \frac{e^{c_i + W_i v}}{e^0 + e^{c_i + W_i v}} \newline
    =& \frac{e^{c_i + W_i v}}{1 + e^{c_i + W_i v}} = \frac{1}{1 + \frac{1}{e^{c_i + W_i v}}} \newline
    =& \sigma (c_i + W_i v)
    \end{aligned}    
\]

반대 방향에 대해서도 <U>같은 공식을 적용</U>할 수 있고, 이때 바뀌는 것은 bias와 weight에 곱해지는 input이기 때문에

\[
    p(v_j = 1 \vert h) = \sigma(b_j + W_j^\top h)    
\]

와 같다. RBM 모델의 수학적 모델링은 위의 공식대로 sigmoid 함수를 따르는 조건부 확률이 되며, RBM이 학습하고자 하는 것은 <U>데이터의 확률 분포</U>이다.
만약 RBM의 hidden layer가 적절한 property에 대한 distribution $p(h)$를 제대로 학습했다면, sampling을 통해 획득할 수 있는 $p(v \vert h)$가 원래 데이터인 $p(v)$와 같아야 한다. 마치 Variational autoencoder랑 비슷하긴한데 조금 다른 점은 VAE 구조는 encoding이 목적이고 RBM은 <U>확률 밀도 함수 자체를 레이어에 피팅</U>하고자 하는 것이다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/222154860-16a4c59c-576d-4477-8e7d-fab9c3700f98.png" width="800">
</p>

이제 실제로 왜 $p(v)$에 대한 negative log likelihood를 최적화하는 것이 <U>contrasitve learning과 관련될 수 있는지</U> 수식으로 증명할 수 있는 이론적 배경이 완성되었다. FF 설명하고자 여기까지 왔다... 

---

# Parametric learning in RBM
RBM의 파라미터를 $\theta$라고 해보자. 여기서 parameter란 weight와 bias가 될 수 있다. 앞서 전개했던 식들을 토대로 negative log likelihood를 구하면 다음과 같다.

\[
    \begin{aligned}
    -\frac{\partial}{\partial \theta} \log p(v) =& -\frac{\partial}{\partial \theta} \log \left( \frac{\exp (-F(v))}{Z} \right)   \newline
    =& -\frac{\partial}{\partial \theta} \left( \log \exp(-F(v)) -\log (Z) \right) \newline
    =& -\frac{\partial}{\partial \theta}(-F(v) - \log (Z)) \newline
    =& \frac{\partial}{\partial \theta} F(v) + \frac{\partial}{\partial \theta} \log (Z) \newline
    =& \frac{\partial}{\partial \theta} F(v) + \frac{\partial}{\partial \theta} \log \left( \sum_{\tilde{v}} \exp (-F(\tilde{v})) \right)
    \end{aligned} 
\]

$\tilde{v}$는 RBM에 의해 생성되어 visible unit에 정의된 state vector를 의미한다. 즉, <U>은닉층으로부터 생성된 샘플</U>을 의미한다.

\[
    \begin{aligned}
        =& \frac{\partial}{\partial \theta} F(v) + \frac{\sum_v \exp (-F(\tilde{v})) \frac{\partial}{\partial \theta} (-F(\tilde{v}))}{\sum_{\tilde{v}} \exp(-F(\tilde{v}))} \newline
        =& \frac{\partial}{\partial \theta} F(v) - \sum_{\tilde{v}} \frac{\exp (-F(\tilde{v}))}{Z} \cdot \frac{\partial}{\partial \theta} (F(\tilde{v})) \newline
        =& \frac{\partial}{\partial \theta} F(v) - \sum_{\tilde{v}} p(\tilde{v}) \frac{\partial}{\partial \theta} F(\tilde{v})
    \end{aligned}    
\]

식을 정리하고 났더니 <U>생성된 샘플</U> $p(\tilde{v})$에 의한 <U>자유 에너지 변화율 평균</U>이 되었다. 확률 기댓값으로 정의되는 구조이기 때문에

\[
    = \frac{\partial}{\partial \theta} F(v) - E_\tilde{v} \left( \frac{\partial F(\tilde{v})}{\partial \theta} \right)    
\]

이처럼 정의할 수 있다. 물론 <U>당연한 이야기지만</U> feasible visible sample $\tilde{v}$에 대한 기댓값 연산이 불가능하므로 <U>샘플의 평균을 통해 근사하는 학습</U>이 진행된다.

\[
    \approx \frac{\partial}{\partial \theta} F(v) - \frac{1}{\vert \mathcal{N} \vert} \sum_{\tilde{v} \in \mathcal{N}} \frac{\partial F(\tilde{v})}{\partial \theta}
\]

앞서 말했던 바와 같이 볼츠만 머신은 각 레이어가 의미하는 것이 특정 representation의 확률 분포가 되고, 따라서 위의 식은 <U>서로 다른 두 분포</U>(하나는 원래 데이터, 다른 하나는 모델링된 데이터)의 간격을 줄이는 것과 같다. 바로 익숙한 KL divergence로 생각해볼 수 있으며, 이는 <U>energy based approach</U>에서 <U>에너지의 변화가 곧 확률 분포의 변화</U>이기 때문이다.
샘플링할 수 있는 $\tilde{v}$의 개수에 따라 loss term은 달라지지만, Hinton 교수는 한 번의 샘플링으로 gradient descent를 사용하더라도 RBM 학습이 가능하다고 밝혔고, loss 식은 <U>다음과 같이 단순화하여 표현</U>할 수 있다.

\[
    \text{loss} = F(v) - F(\tilde{v})
\]

바로 이러한 맥락에서 유도된 <U>볼츠만 머신에서의 contrastive divergence</U>란 real data와 네트워크가 만들어낸 가상의 data 사이의 간격을 줄이는 학습을 의미한다.

\[
    \frac{\partial KL(P_\text{data} \vert\vert P_\text{model})}{\partial w_{ij}} = \left< s_i s_j \right>_\text{data} - \left< s_i s_j \right> \text{model}
\]

---

# Relationship with RBM and FF algorithm 

식에서의 brackets는 <U>레이어 사이의 state</U>가 fluctuation(weight에 따른 변화)하는 것을 표현한다. 앞에서 유도한 식을 일반적으로 표현한 것과 같다. 결국 RBM이 학습하는 것이 네트워크 전체에 대해 error를 propagation하는 구조가 아니라 두 개의 레이어 사이에 <U>real sample과 modeled sample을 유사하게</U> 만드는 것이다. 볼츠만 머신에 대한 아이디어는 저자가 간단하게 다음과 같이 정리해주었다.

1. Learn by minimizing the free energy $F(\cdot) on real data and maximizing the free energy on negative data generated by network$
2. Use the Hopfield energy as the energy function and use repeated stochastic updates to sample global configurations from the Boltzmann distribution defined by the energy function

저자가 추가로 FF 알고리즘에 대해 언급하는 것은 <U>wake</U>는 일종의 <U>bottom-up</U> 구조로, real data를 통해 hidden state에 대한 representation을 축적하고, <U>sleep</U>은 <U>top-down</U> 구조로, 학습된 hidden state로 sampling하는 것과 같다. 그렇기 때문에 RBM 학습 방법이 이전 글에서 살펴보았던 구조랑 동일한 해석으로 이어진다는 것. Hinton 씨는 아마도 <U>볼츠만 머신</U>이 간단한 iterative 구조를 가짐으로써 <U>복잡한 task에 적용되지 못하고</U> backpropagation 알고리즘에 비해 제대로 연구가 진행되지 못한 점을 아쉽게 생각한 듯하다.

# Relationship with GAN

그리고 <U>backpropagation</U> 방법을 통한 generative model 중 유명한 녀석인 <U>GAN</U> 역시도 이와 비슷하다. FF 알고리즘이 각 레이어마다 greedy algorithm을 통해 iterative한 최적화를 진행한다면, GAN은 disciminator가 <U>네트워크로 생성된 데이터</U>가 positive sample인지 negative sample인지 구분하게 된다. 다만 GAN과는 다른 점은 <U>probability 학습에 사용된 layer</U>가 그대로 goodness 판별에 사용되므로, backpropagation이 불필요하다는 것이다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/222173754-41942750-2d49-4204-b183-f881b74e5303.png" width="800">
</p>

---

# Relationship with contrastive learning
단순히 RBM에서의 contrastive(KL divergence)를 최적화하는 것이 아니라, [SimCLR](https://arxiv.org/abs/2002.05709)와 같은 <U>self-supervised contrastive learning</U> 방법에서 사용하는 학습법과의 관련성에 대한 부분이다.
해당 방법들에서 사용하는 방법은 동일한 이미지에서 <U>다른 crop된 image patch</U>에 대한 representation을 구한 뒤, <U>동일한 이미지</U>에서 나온 image patch라면 서로 <U>similarity가 높도록</U>, <U>다른 이미지</U>에서 나온 image patch라면 서로 <U>similarity가 낮도록</U> 하는 objective function을 적용하거나,
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209150800-c9a469b8-da9a-47c8-9068-1f8fd5657ea4.gif" width="600"/>
</p>
동일한 이미지에 <U>서로 다른 augmentation을 적용</U>하여 positive pair를 만들고, 나머지 이미지에 대한 augmentation pair들을 모두 negative로 간주하여 contrastive learning을 진행하는 방법도 있다. 

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209150790-bc14237d-2686-43e9-8059-8220bb554dd1.gif" width="600"/>
</p>

자세한 내용은 본인이 작성한 글들 중에서 [여러 딥러닝 학습법](https://junia3.github.io/blog/transfer) 게시글을 참고하면 좋을 것 같다. 아무튼 이렇게 positive/negative pair에 대한 similarity를 구하는 것을 <U>agreement</U>라는 용어로 통칭하도록 하겠다. <U>Image crop</U>에 대한 agreement을 생각했을때, 만약 두 crops가 완전히 일치한다면 agreement를 학습하는 의미가 없게 된다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/222297983-d7833018-c2db-4764-877f-404550e840ff.png" width="600"/>
</p>

그리고 실제 인간의 신경망 구조는 두 개의 <U>서로 다른 representation</U>이 <U>완전히 동일한 neural weight 세팅</U>에서 추출되었다는 보장을 할 수 없다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/222299625-745b5469-9fcc-4214-ae32-9e119927f66b.png" width="600"/>
</p>

위의 그림을 보면 딥러닝에서 사용하는 contrastive learning은 <U>각 배치 단위로 학습</U>이 진행되기 때문에 update 시기와 정확하게 동기화가 가능하지만, 실제 인간이 정보를 처리하는 과정에서는 <U>이러한 형태</U>의 contrastive learning이 <U>불가능하다는 것</U>을 알 수 있다(조금만 weight가 달라져도, 이에 따른 noise와 disagreement를 배제할 수 없어지기 때문).

FF에서는 agreement를 다르게 측정하는 것을 알 수 있고, 이러한 세팅이 실제 신경망에서와 비슷하다고 한다. 만약 <U>source</U>가 <U>neuron을 activate</U>한다면(positive sample) 높은 goodness를 보일 것이고, 아니라면(positive sample) 낮은 activation 값을 가질 것이다. 단순히 각 레이어의 output을 통해 measuring하는 방식은 <U>각 샘플에 대한 동기화 없이</U>도 <U>독립적인 연산이 가능</U>하다는 장점이 있다. SimCLR와 같은 방법의 문제점은 학습 효율 및 성능을 위해 <U>batch size를 키우고</U>, 각각 representation을 구해야하는 등 <U>연산량이 급증</U>한다는 문제가 있다. 그러나 사용되는 <U>objective function</U>은 유사도에 대한 정보를 통해 weight를 업데이트하는 과정이 전부이기 때문에(representation의 일부만 constraints로 학습) <U>학습 효율을 충분히 활용하지 못한다</U>는 분석이 있다.

---

# Problem with stacked contrastive learning
<U>Supervision이 없는</U> 상황에서 여러 층의 <U>representation layer</U>를 학습하는 방법은 하나의 layer가 특정 data에 대한 representation을 학습하게끔 한 다음, 학습된 <U>hidden layer의 activation</U>을 다시 input으로 하여 다음 레이어를 학습하는 것이다. <U>RBM</U>이나 <U>stacked autoencoder</U>에서 representation을 학습할 때 이러한 방법을 사용한다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/222305800-96f8e08c-774b-41e9-b8f5-0545835eae32.png" width="600"/>
</p>

이러한 학습법은 그러나 큰 문제가 있다. 예를 들어, 임의의 noise images를 임의의 weight matrix로 mapping하는 상황을 생각해보자. 구한 activity는 연산 결과에 <U>input</U>에 대한 <U>weight matrix</U>와의 <U>correlation</U>이 될텐데, 이는 실제 데이터와는 아무런 상관이 없을 것이다. 위의 그림에서 보는 것과 같이 $h \rightarrow h$로의 activity 연산은 <U>external world</U>(visible layer)에는 아무런 영향을 끼치지 못한다는 것을 알 수 있다.

이러한 문제점을 해결하기 위해서 볼츠만 머신의 학습 알고리즘에서 Hinton은 두 개의 서로 다른 external boundary condition을 둔다. 바로 이 방법이 positive와 negative data를 contrasting하는 것이며, 다층 레이어 학습시 생기는 문제점을 <U>해결할 수 있는 방법으로 제시</U>되었다고 한다.

---

# Learning fast and slow
이 부분이 사실상 굉장히 흥미롭게 다가왔던 파트이다. 바로 각 레이어에서의 goodness에 의한 parameter update는 <U>layer normalized된 output</U>에 아무런 <U>영향을 미치지 않는다는 것</U>이다. 만약 이게 가능하다면 backpropagation처럼 레이어 간 학습 과정에 동기화 필요성이 사라지기 때문에 <U>특정 레이어에서의 activation 연산이 느려지더라도</U> 병렬 처리가 가능하다는 놀라운 발전이 가능하다. <U>End-to-end 학습</U>이지만, 그렇다해서 각 요소가 다른 요소의 학습을 <U>기다릴 필요성은 없어진다</U>는 관점이다.

만약 layer가 fully connected 되어있다고 생각하면, <U>weight update</U>($\Delta w_j$)는 input $x$에 대해 다음과 같은 값을 가지게 된다.

\[
    \Delta w_j = 2 \epsilon \frac{\partial \log (p)}{\partial \sum_j y_j^2} y_j x   
\]

$y_j$는 layer normalization 이전의 ReLU 노드의 activation 그 자체를 의미하고, $w_j$는 weight에서 neuron $j$에 대한 벡터만을 의미한다. 따라서 $y_j = \text{ReLU}(\left< w_j,  x\right>)$라고 생각할 수 있다. $\epsilon$은 learning rate라고 생각하면 되고, neuron $j$에 대한 activity는 위의 <U>weight update에 의해</U> $\Delta w_j x$ 만큼 변하게 된다.

그렇다면 결국 output의 변화가 index $j$에 의존하는 것은 곧 $y_j$에 의존하는 것과 같다. 이를 제외하고는 다른 요소들은 index에 의존하지 않기 때문에 weight의 변화는 activity vector의 크기를 변화시킬 수 있어도 <U>방향은 변화시키지 못한다</U>.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/222310424-a075dc2d-be6e-4f69-a4b4-c165a95833b8.png" width="600"/>
</p>

즉, 앞서 weight parameter가 update가 되는 상황이 <U>layer normalize 결과</U>를 변화시키지 않기 때문에 한 번에 <U>여러 레이어를 업데이트할 수 있는 장점</U>이 있다. 따라서 input $x$를 기준으로 모든 레이어가 desired goodness를 얻기 위한 weight update가 <U>동시에 발생</U>할 수 있다는 것이다(one step으로 업데이트 가능). 모든 input vector와 layer normalized hidden vector의 길이가 $1$이라고 가정한다면, learning rate는 layer $L$에서의 squared activity 합인 $S_L$, 얻고자 하는 goodness인 $S^\*$에 대해 다음과 같이 설정할 수 있다.

\[
    \epsilon = \sqrt{\frac{S^\*}{S_L}} -1    
\]

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/222312743-770466eb-7a88-46da-8d0e-55a36ff0e266.png" width="600"/>
</p>

하지만 실제 저자는 아직 이론상으로만 밝힌 사실이고, <U>mini-batch 단위</U>로 학습을 진행하는 구조에서는 이를 <U>사용할 수 없다</U>고 한다.

추가적으로 FF의 경우에는 이런 방법도 사용할 수 있다. Backpropagation 알고리즘과는 다르게 FF는 <U>학습 구조 내부</U>에 <U>black box</U>가 있더라도 상관없이 학습이 가능하다는 것을 언급했는데, 여기서의 black box는 한쪽 layer의 output을 stochastic(구조를 모르기 때문)하게 변형시켜 다음 layer의 input으로 넣는 경우를 생각해볼 수 있다. 만약 black box를 몇 개의 hidden layer로 이루어진 neural network로 구성해보는 경우에, 이 black box가 학습하는 속도가 상대적으로 outer loop보다 느려질 경우 <U>black box</U>가 <U>stationary</U>(확률 분포가 <U>시간에 따라 변하지 않고 일정</U>한 process)라 가정할 수 있게 된다. 분포가 일정한 시스템이 중간에 있게 되면 나머지 activity들이 <U>새로운 데이터에 적응할 수 있는 시간</U>을 벌어주게 되고, 이러한 black box의 slow learning을 활용하면 보다 긴 timescale에 대해 <U>시스템을 성장시킬 수 있는 배경</U>이 된다. 에컨데 slow reinfocement learning 과정에서 black box의 input으로 약간의 noise를 더한 perturbed input을 통해 positive phase에서의 cost function 변화를 볼 수 있게 되는데(원래 과정이었다면 <U>학습 속도가 동일하기 때문에</U> noise에 의한 효과를 확인하지 못함), 이로써 black box 내부의 뉴런들의 activity에 의한 cost function의 도함수를 예측할 수 있게 되는 것이다. 이 부분을 보면서 score estimation 중 <U>denoising score matching</U>이 생각났다.

---

# Analog computation
Activity vector에 weight를 곱하는 연산을 energy efficient한 방법으로 구현하는 것은 activity를 voltage로, weight는 conductance로 보는 것이다. 단위 시간당 곱해진 value는 곧 charge(전류)가 된다. Conductance는 저항의 역수로, 각 유닛에서의 <U>value</U>(voltage)에 대해 다음 레이어로 얼만큼 값을 넘겨줄지 결정하는 요소가 된다.

\[
    G = 1/R,~I = GV
\]

이러한 구조가 디지털 환경에서 높은 power로 구동되는 트랜지스터를 사용하여 $O(n^2)$ 만큼의 single bit operation을 진행하는 것(multiplication을 구현하기 위해)보다 훨씬 효율적인 것을 알 수 있다. 하지만 backpropagation 과정을 구현하기엔 아날로그 회로가 부적합하기 때문에 보통은 <U>A-to-D converter</U>를 사용하는 방법을 고안했다.
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/222327263-af864684-e211-4461-8a8c-5b530fad70f9.png" width="600"/>
</p>
그런데 만약 forward-forward 알고리즘을 사용한다면, 굳이 <U>backpropagation</U>을 진행할 <U>필요가 없어지게</U> 된다.

---

# Mortal computation
Mortal computation을 직역하면 말 그대로 <U>'유한한 삶을 사는 연산'</U>을 의미한다. 대부분의 디지털 컴퓨터의 경우 해결해야할 특정 task가 있다면, 이를 해결하기 위해서는 단지 <U>컴퓨터가 할 일</U>을 잘 <U>정리해서 넘겨주는 과정</U>이 필요하다. 결국 실생활의 문제에 generalize하기 위해서는 computation에 활용될 수 있는 instruction을 제대로 짜는 것이 중요하다는 관점이었다. 물론 지금은 상황이 많이 바뀌었기 때문에 <U>더이상 프로그래머들이 고통받으며</U> 알고리즘을 짜지 않고, <U>딥러닝으로 해결하고자 하는 시도가 증가</U>하였다.
그와 동시에 연구 단체에서는 <U>딥러닝</U>의 구체적 작동방식과 <U>하드웨어의 관계</U>에 대해 제대로 알아보기도 전에, 단순히 여러 hardware에서 돌릴 수 있는 software framework를 만들기 시작했으며, 이러한 방향은 프로그램과 하드웨어가 서로 독립적으로, 즉 하드웨어가 사라지더라도 <U>knowledge는 남는</U>, <U>immortal knowledge</U>를 낳게 되었다.

Software를 hardware와 <U>분리시킨 것</U>은 많은 장점이 있었다. 분리된 소프트웨어는 <U>하드웨어에 대한 제한이 없기 때문에</U> electrical engineering에 대한 고려 없이 연구가 진행될 수 있기 때문이다. 단순히 프로그램을 작성하고, 이를 수많은 컴퓨팅 환경에 복사하기만 하면 된다. 대규모의 데이터셋에 대한 도함수를 계산하고, 최적의 모델을 찾는 과정은 점차 <U>병렬화되기 시작</U>했고 그만큼 <U>가속화</U>되었다.

하지만 만약, 지금 소프트웨어가 가지고 있는 <U>불멸성</U>(immortality)을 희생해서라도 연산에 필요한 에너지를 절약할 수 있다고 생각해보자. 기존의 알고리즘이 허용했던 한정된 activation과 네트워크 구조에서 벗어나 <U>훨씬 다양한 variation</U>이 각 hardward 플랫폼에 적용 가능하게 되고, 동일한 task를 수행하는 과정에도 모두 <U>서로 다른 parameter set</U>이 최적의 결과를 만들 것이다. 이런 parameter는 각 하드웨어에 따라 다르기 때문에 <U>대체 불가능</U>하며, <U>mortal</U>하다는 특징을 가질 것이다.

같은 parameter를 <U>다른 하드웨어에 적용</U>한다는 것이 어쩌면 합리적이지 못하지만(fully-optimized된 결과인지 <U>확신할 수 없기 때문</U>), 하나의 HW에서 다른 HW로 knowledge transfer를 하는 <U>더 biological한 방법이 존재</U>한다. 이를테면 이미지를 보고 <U>물체의 종류를 구분</U>하는 task가 있다고 하면, 실제로 관심이 있는 것은 하드웨어에서 모델링된 function이 pixel value를 각 class label로 매핑하는 것이지, parameter value 자체가 아니다. 그리고 이러한 representation function은 <U>distillation 방법</U>으로 다른 하드웨어로 transfer가 가능하다.

Distillation이란 단순히 학습의 <U>teacher 역할</U>을 하는 네트워크가 내놓는 정답을 맞추는 것이 아니라, 오답에 대한 예측도 동일하게 따라가고자 하는 것이다. 이러한 방법이 <U>generalization 성능을 높이고</U> overfitting을 방지하는 효과가 있다고 알려져있다. Distillation은 teacher network가 modality에 대해 <U>internal representation</U>을 <U>풍부하게 담는 output</U>을 내보내는 경우에 가장 효과적이다. 마치 언어가 가지는 기능과 같은데, 무언가를 묘사하는 문장이 있다고 생각해볼때, 언어가 단지 상징적 의미를 가진 정보 전달의 단위라고 보는 것보다는, 해당 문장을 말해주는 사람이 이해해서 설명해주는 것을 <U>듣는 사람으로 하여금 비슷한 이해력을 가질 수 있게</U> 하는 경우로 생각해볼 수 있다. 말이 좀 어려웠는데 좀 더 쉽게 설명하자면, 어떠한 상황을 설명하는 문장이 단순히 <U>상징적인 상황을 축약하는 것</U>이 아니라 일타 강사가 <U>잘 설명해주는 내용</U>이라고 생각하면 된다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/222336865-2ba8da9b-9fa0-4529-ab21-8ed3dbc10dce.png" width="600"/>
</p>

그림에서 보이는 뇌의 분홍색 부분이 transfer 하고자 하는 representation이고, 잘 전달하기 위한 output(scene을 가장 <U>잘 설명하는</U> description)으로 지식을 넘기는 과정이 된다. 만약 언어가 각 나라의 culture를 잘 대표할 수 있는 형태로 발전이 되었다면, 여러 나라의 culture를 <U>제대로 이해하는 딥러닝 모델</U>은 언어를 기반으로 해야할 것이다.

앞서 distillation에 대해서 언급한 이유는 output에 의한 완전한 knowledge transfer에 <U>한계가 분명히 존재한다는 점</U>이다. 저자가 언급하는 것은 만약 파라미터 수를 늘리면서 energy를 절약하고 싶다면, mortal computation(하드웨어에 따라 파라미터를 설계하는 것)이 유일한 방법이라고 하며, 이러한 연구를 진행할 수 있는 candidate로 <U>forward-forward algorithm</U>을 제안하였다. 그럼에도 불구하고 만약 hardware에서 <U>동일한 소프트웨어를 학습하기 위해서</U>, weight를 공유하는 방식으로써 학습의 bandwidth를 올릴 수 있고, knowledge를 공유하는 것이 distillation보다 나은 방법이라고 제시한다.

---

# 결론
저자는 몇가지 FF를 통해 해볼 수 있는 연구들을 제시하고 마무리한다. 아무래도 FF라는 알고리즘은 backpropagation이 불가능한 아날로그 컴퓨팅 환경에 시사하는 바가 굉장히 큰 점과, 아직 디지털 환경에서 효과적으로 FF 알고리즘을 통해 대용량의 네트워크를 학습시키는 방법이 발견되지 않았다는 점에서 발전 가능성이 굉장히 큰 연구라고 생각된다.
딥러닝 연구를 시작했을 당시에는 이미 초창기 논문들을 읽고 이해하는데 시간을 모두 썼었고, 거기에서 insight를 얻어서 새로운 연구를 고안하는 것이 이미 늦은 단계였지만 최근 들어 이렇게 <U>preliminary 연구</U>를 딥러닝 분야에서 본 것이 굉장히 감동적인 일이라고 생각된다.

결국 컨퍼런스는 수치 싸움이고, 기존 SOTA를 이길 수 없다면 새로운 task를 정의하여 성능의 feasibility를 보이는 것이 최선이라고 생각했었는데 이런 새로운 방향성을 제시한 연구를 보니 딥러닝을 공부하던 초창기로 돌아가는 기분이었고, 논문 내용보다 더 배운 것이 많은 시간이었다.