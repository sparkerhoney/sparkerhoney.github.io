---
title: Contrastive test time adaptation 논문 리뷰
layout: post
description: Test time domain adaptation
use_math: true
post-image: https://user-images.githubusercontent.com/79881119/227496551-8a7c2609-6841-4801-810b-da6a9fbfd0a5.gif
category: paper review
tags:
- Deep learning
- DA
- Test time adaptation
- Contrastive learning
---

# 딥러닝 : Representation을 효율적으로 학습하는 법

제목에서도 볼 수 있듯이 <U>Test-time adaptation</U>이라는 task가 해당 paper의 main이다. 바로 본론으로 들어가기 전에 잠시 언급하고 싶은 내용은, 딥러닝은 네트워크가 특정 데이터셋의 representation을 잘 학습하도록 하고, 학습된 representation을 feature map으로 사용하여 목적이 되는 task를 해결하고자 하는 알고리즘이다. 그러다보니 자연스럽게 딥러닝이라는 방법론은 **데이터에 의존할 수 밖에** 없으며, 한정된 자원 내에서 문제 해결을 위해 <U>output을 generalization</U>할 수 있도록 빅데이터나 정규화 방법을 사용하곤 한다. 고로 딥러닝이란 **closed-set**(유한한 데이터셋 범위)를 다루게 된다. AI라는 것이 엄청 대단한 것처럼 느껴지지만 사실상 그렇진 않은 것이다. 사람으로 치면 경험이라고 볼 수 있는 데이터셋 없이는 아무것도 모르는 무지의 상태와 같고, 컴퓨팅 환경에서 뇌의 역할을 하는 parameter가 한정된 수를 가질 수 밖에 없기 때문에 무한정 넓은 범위의 distribution을 학습할 수 없다. 결국 끊임없이 <U>효율과 성능 사이 저울질</U>하는 과정이 되는 것이다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/227495210-3dc15e45-ce2c-4826-8238-5ec3c53edf42.png" width="600">
</p>

---

# Contrastive TTA

이번에 다룰 내용은 contrastive learning 방법을 이용한 <U>효율적인 representation 학습</U>과 관련된 TTA로 이어진다. Domain adaptation에서 source dataset에 대한 접근 없이 target dataset을 기준으로 하는 <U>메트릭 성능</U>을 높이는 것이 곧 test-time adaptation이다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/227495143-ba65a665-6b94-42a1-bb44-9f7b26cdee1b.png" width="500">
</p>

**Domain adaptation**의 주목적은 위에 나와있는 그림에서 볼 수 있듯이 source domain에 최적화된 classifier(꼭 분류기가 아니라, task에 맞는 형태의 head라고 이해하면 좋다)가 target domain에는 제대로 적용되지 않는 것을 볼 수 있다. 이러한 문제를 <U>domain shift</U>라고 부른다. 만약 source dataset이 있다면, target domain에 대한 supervision 없이도 feature(푸른색, 붉은색으로 표현된 각각의 분포) 간의 분포 거리를 줄임으로써 source domain의 hypothesis를 target에도 적용할 수 있게 된다. **Hypothesis**란 ‘가설’이라는 의미인데, 위의 그림과 같은 도메인 어뎁테이션에서는 classifier가 <U>‘~에 대해서도 그럴 것이다’</U>라는 **기준선**이 되고, 이 기준선에 맞지 않는 특징자들을 alignment(정렬)하는 것이 의미론적으로 통한다고 이해하면 된다. 하지만 source dataset을 참고할 수 없게 된다면 이런 방식의 정렬은 <U>불가능</U>하게 된다.

Source에 대한 hypothesis를 그대로 유지하면서 target domain의 특징들을 활용할 수 있는 여러 연구들이 소개되었다. 대체로 TENT, SHOT와 같은 논문에서 <U>entropy minimization</U>(모델의 확신도를 올려줌) 방법이나 <U>pseudo-labeling</U>(source classifier의 예측을 임시방편으로 라벨로 사용함) 방법이 제시되었지만, 근본적으로 두 방법들은 <U>source/target 분포를 정렬</U>하는 것에 목적함수가 되지 못한다는 문제가 있다.

해당 논문에서는 **AdaContrast**라는 contrastive learning 방법을 SSL(Self-supervised learning) 연구로부터 착안하여 TTA에 적용하려는 시도를 한다. Contrastive learning은 supervision 없이 유의미한 representation을 학습하기 위한 학습법이므로 entropy minimization이나 pseudo labeling과 같은 수동적/간접적인 방법에 비해 feature alignment에 직접적인 목적을 가질 수 있다는 점이 기존 연구와 차이가 될 수 있다. 이를 논문에서는 <U>‘Model calibration’</U>이라 표현한다.

---

# Limitations and Suggestion

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/227495158-6b432885-97c3-4964-a165-d270073bc00d.png" width="600">
</p>

결국 논문이 제시하고자 한 문제점은 기존 test-time(source dataset을 사용하지 못하는 상황)에서의 <U>approach에 내재하는 한계들</U>이다. 기존 방식은 source domain classifier를 가지고 target domain classifier를 생성하는 것이 쉽지 않은 일이고, 이에  image/feature generation이나 class prototype, entropy minimization 그리고 self-training/pseudo-labeling과 같은 방법들로 접근하였다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/227495162-0891dfc7-e09b-464c-b2d6-05f2b2e4c4e3.png" width="500">
</p>

직접 target에 대한 class conditional sample을 생성하는 방법의 경우([논문 링크](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Model_Adaptation_Unsupervised_Domain_Adaptation_Without_Source_Data_CVPR_2020_paper.pdf)) 위의 그림에서 볼 수 있듯이 adaptation 단계에서 image/feature generation을 하기 위해 <U>computation capacity</U>가 커진다는 단점이 있다. 

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/227495163-813203ef-19a0-4190-810d-09eb02e1f041.png" width="600">
</p>

또한 <U>entropy minimization</U>으로 접근하는 방법의 경우에는 분포 간 차이가 크게 되면 위와 같이 classifier를 기준으로 멀어지게끔 학습하는 것이 오히려 <U>error accumulation</U>을 일으키게 된다. 이러한 문제를 target dataset에 대해서 model calibration(모델의 정렬)이 맞지 않는다고 표현한다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/227495170-9d2925f3-cd82-4601-bd38-b32d89d71796.png" width="700">
</p>

또한 대부분의 연구에서 <U>pseudo-labeling</U>을 사용하게 되는데, 실제로 생성된 pseudo-label의 퀄리티를 보장할 수 없기 때문에 보다 noise를 줄이는 방법을 고안하기 시작하였다. 대표적인 방법으로는 <U>mean-teacher</U> 방식을 사용하거나, <U>prototype</U>을 사용하여 앙상블하는 정규화 장치를 사용하기도 한다. Test-time training이라는 일종의 <U>meta-learning</U> 방식은 pre-training 단계에서 supervised learning과 self-supervised learning을 동시에 최적화함으로써, auxiliary task에 적합한 네트워크의 representation을 학습하는 것이 주된 목적이다. 해당 방법론에서는 SSL 방식으로 rotation prediction을 사용하였고(의도적으로 각도 회전 후, 이를 예측하는 task), 이러한 방식들의 문제는 pre-training 과정을 off-the-shelf(source dataset에 대해 학습된 네트워크를 그대로 사용)할 수 없다는 것이다. 그리고 가장 큰 문제점은 SSL 방식 자체가 학습하고자 하는 supervision이 downstream task와 직접적인 연관이 없기 때문에(from 각도 예측 문제 to Object detection 문제) 모든 네트워크/학습 데이터에 대해 같은 방법을 적용했을 때 좋은 성능을 보일 것이라고 판단하기도 어렵다.

물론 논문에서 제시한 <U>contrastive learning</U>을 test-time training에 적용할 수도 있지만, 이런 식으로 학습 방법을 바꾸는 것이 adaptation 단계에서의 성능을 높인다는 것이 보장되지 않는 것은 마찬가지이다. 따라서 AdaContrast를 제안한 contrastive TTA 논문은 직접 test-time adaptation 전략으로서 contrastive learning을 제시하게 되고, 이를 기반으로 pseudo-labeling과의 <U>동시 최적화를 목적</U>으로 한다. Target representation이 contrastive learning을 통해 잘 학습된다면 decision boundary(pseudo label)의 퀄리티가 올라갈 것이고, 좋아진 퀄리티를 가지는 prior는 다시 contrastive learning의 성능에 영향을 미치는 구조가 된다. 

---

# Method

### Source model training

해당 논문이 해결하고자 하는 domain adaptation task는 가장 간단하게 image classification 중 <U>closed-set test time adaptation</U>에 해당된다. Source model가 image/label pair 인 $(x\_s^i, y\_s^i)\_{i=1}^{N_s} \in \mathcal{D\_s}$ 에 대해 학습된 다고 생각했을 때, 논문에서의 test time adaptation은 학습된 pre-trained network와 target dataset의 image $(x\_t^i, \cdot)^{N\_t}\_{i=1} \in \mathcal{D}\_t$ 를 사용해서 실제 label인 $(y_t^i)_{i=1}^{N_t} \in \mathcal{Y}_t$를 잘 예측하는 네트워크를 만들고 싶은 것이다. Test time adaptation에서는 domain adaptation 단계에서 source dataset를 참고할 수도 없으며 label은 evaluation 목적으로만 사용된다. 또한 closed-set이란 source와 target dataset이 서로 <U>같은 label space를 공유</U>하는 것이다. 예를 들어 source dataset의 class가 (cat, dog, car, truck, bird)로 구성되어있다면 target dataset의 class 또한 (cat, dog, car, truck, bird)로 구성되어있는 경우에 해당된다. 이를 label spcae(집합)으로 표현하자면,

\[
\mathcal{Y}_s = \mathcal{Y}_t = \mathcal{Y}
\]

위와 같이 나타낼 수 있다. 논문에서 source model을 얻는 과정으로는 일반적인 <U>feature extracter/classifier</U> 구조를 가지는 neural network에 대해서,

\[
\begin{cases}
\text{feature extractor }f_s(\cdot), & \mathcal{X}_s \rightarrow \text{R}^D \newline
\text{classifier }h_s(\cdot), & \text{R}^D \rightarrow \text{R}^C
\end{cases}
\]

기존 fully test time adaptation approach 방법론 중 entropy minimization과 pseudo labeling을 적용했던 [SHOT 논문](https://arxiv.org/abs/2002.08546)과 동일하게 label-smoothing을 거친 이후에 cross entropy loss를 최적화하는 식으로 source training을 진행하였다.

\[
L_s^{CE} =-\sum_{c=1}^C \tilde{y}_s^c \log (p_s^c)
\]

위의 식은 soft label에 대한 cross entropy를 나타낸 것인데, $p_s^c$는 $c$번째 element에 대한 model의 output인 $a$에 다음과 같이 softmax operation을 적용한 확률값을 의미한다.

\[
p_s^c = \sigma_c (a) = \frac{\exp(a_c)}{\sum_{k=1}^C \exp{a_k}}
\]

실제로 ground truth로 사용되는 $\tilde{y}_s^c$는 원래의 one-hot encoding 대신 soft label을 사용하는데, 다음과 같은 식을 통해 smoothing을 거친다.

\[
\tilde{y}_s^c = (1-\alpha)y_s^c + \alpha/C
\]

즉 원래 정답인 class에 대해 $1$(모든 확률)이 mapping된 상태를 기준으로 $\alpha$만큼의 확률을 균등하게 나머지 class에 분배하는 것이다. 예컨데 CIFAR-10 dataset에 대해 고양이에 대한 class 라벨을 다음과 같이 바꾼다고 할 수 있다. 여기서 $\alpha = 0.1$를 사용했다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/227495174-e97f5f80-595c-4c68-bf45-648c79583d20.png" width="400">
    <img src="https://user-images.githubusercontent.com/79881119/227495176-30c4918e-be70-4fc1-a9d6-71bb86c35fd6.png" width="400">
</p>

### Pseudo labeling

Adaptation 과정에 pseudo label인 $(\tilde{y}^i)^{N_t}_{i=1}$을 <U>unlabeled target dataset</U>에 대해 생성하고  이를 사용하여 source model을 target domain의 dataset에 대해 bootstrapping을 진행하는 과정을 거친다. Pseudo label을 정제하고 예측하는 과정을 epoch 단위로 하지 않고 batch 단위로 하여, 단일 epoch 학습 내에서 최대한 최근의 refinement가 반영될 수 있게 한 것이 특징이다. Refinement는 target feature space로 정의되는 memory queue $Q_w$에서 <U>nearest-neighbor soft voting</U>을 통해 생성한다. 뒤에서 전체 framework에 대한 그림을 통해 보다 구체화되겠지만 간단하게 설명하자면 input 이미지로 target domain의 샘플 $x_t$이 들어가게 되면 weak augmentation(약한 변형) t$_w$이 랜덤하게 적용된($t_w$는 augmentation 집합인 $T_w$에서 무작위로 선택) augmented sample $t_w(x_t)$를 feature extractor에 통과시킨 embedding $w = f_t(t_w(x_t))$를 구하게 된다. 바로 이 weak augmented sample의 embedding vector가 nearest neighbor를 찾는 query(기준점)가 되며, 앞서 설명했던 바와 같이 queue에서 nearest neighbor searching 이후 $\arg\max$ 연산을 통해 pseudo label $\hat{y}$를 구하게 된다.

### Memory queue

Memory Queue에 저장되어야 하는 정보는 <U>embedding 간의 유사도</U>를 구해야하기 때문에 이전 학습 과정에서 지속적으로 업데이트되어 들어오게 되는 weakly augmented sample의 embedding vector $w$와 해당 embedding vector가 classifier에 통과하여 얻은 category별 output에 대한 확률 분포 $\sigma(h_t(w))$가 된다.  Update 방식은 <U>MoCo랑 유사</U>하다고 보면 된다. MoCo에서도 encoder 안정성을 위해 momentum encoder를 적용하였는데, 같은 방법을 사용하여 update하게 된다. 예컨데 원래의 source encoder를 그대로 가져와 초기화시킨 모델 구조 $g_t^\prime (\cdot) = f_t^\prime (t_w(x_t))$가 있을때, 해당 인코더 전체의 parameter인 $\theta_t^\prime$는 gradient descent 방법으로 최적화하지 않고 최적화 중인 encoder의 weight를 <U>exponentially weighted하여 가져오는 방법</U>을 사용한다(momentum learning).

\[
\theta_t^\prime \leftarrow m\theta_t^\prime + (1-m) \theta_t
\]

실질적으로 <U>momentum queue</U>에 저장되는 feature는 이렇게 update된 encoder를 사용한다고 생각하면 된다.

### Nearest-neighbor soft voting

위에서 언급한  memory queue를 업데이트하는 것은 지속적으로 target domain에 가까운 feature를 통해 pseudo label의 <U>bootstrapping</U>을 위한 작업이었고, 실질적으로 구성된 momory queue에서 target sample의 representation embedding vector와 가장 가까운(nearset neighbor)의 output probability를 통해 <U>soft voting</U>을 진행하고, 이를 <U>pseudo-label</U>로 간주한다. 만약 weakly augmented sampe인 $t_w(x_t)$에 대해 $N$개의 nearest neighbor를 soft voting에 사용한다고 하면,

\[
\hat{p}^(i, c) = \frac{1}{N} \sum_{j \in \mathcal{I}_i} p^{\prime(j, c)}
\]

$N$개의 nearest neighbor에 대한 index 집합 $\mathcal{I}_i$ 전체의 probability의 평균을 구하는 것과 같다. 이렇게 평균을 구한 뒤, <U>가장 높은 확률</U>을 보이는 category를 기준으로 <U>pseudo label</U>을 정하게 된다.

\[
\hat{y}^i = \arg \max_c \hat{p}^{(i, c)}
\]

### Jointly self-supervised contrastive learning

위의 pseudo-labeling과 동시에 test time adaptation 과정에 contrastive learning을 적용한다. Contrastive learning은 instance에 따라 discrimination을 하는 방법(같은 이미지에 대해 서로 다른 view의 샘플을 가깝게 샘플링하고, 서로 다른 이미지에 대한 샘플을 멀게 샘플링)을 적용하였다. Image view에 대한 샘플은 augmentation을 통해 획득되고, 이전에 pseudo-labeling을 진행하기 위해 적용했던 weak augmentation superset $T_w$ 대신, <U>strong augmentation superset</U> $T_s$에서 랜덤하게 두 augmented $t_s,~t_s^\prime$을 적용한 샘플 $t_s(x_t),~t_s^\prime(x_t)$을 positive sample로 간주하게 된다.

### Overall framework

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/227495180-38aa0d48-673b-4db4-a405-5a01969ba433.png" width="1000">
</p>

전체 구조를 나타낸 이미지는 위와 같다.  미처 설명하지 않은 부분에 대해 마저 정리하면 다음과 같다.

Stronly augmented samples $t_s(x_t)$, $t_s^\prime(x_t)$을 encoding한 feature embedding vector를 각각 query, key인 $q = f_t(t_s(x_t))$, $k = f_t^\prime(t_s^\prime (x_t))$으로 간주하게 된다. Memory queue는 한정된 길이 $P$만큼 feature를 저장하게 되고 $(k^j)^P_{j = 1}$은 새로 들어오는 $k$로 업데이트 된다. 기존 InfoNCE loss는 positive sample이 되는 query($q$), key($k$) 간의 <U>positive matching</U>과 더불어 이를 제외한 query 내부의 나머지 key와의 <U>negative matching loss</U>로 최적화가 진행되지만 AdaContrast 논문의 경우에는 queue에 key를 저장하면과 동시에 예측된 pseudo label $(\hat{y}^j)^P_{j=1}$을 함께 저장하여 만약 같은 class의 negative pair가 있다면, 이는 <U>최적화 loss term에서 제외</U>시키는 방법을 사용한다.

\[
\begin{aligned}
L_T^{ctr} =& L_{\text{InfoNCE}} = -\log \frac{\exp q \cdot k_{+}/\tau}{\sum_{j \in \mathcal{N}_q} q \cdot k_j / \tau} \newline
\mathcal{N}_q =& \{j \vert 1 \le j \le P,~j \in \mathbb{Z},~\hat{y} \neq \hat{y}^j\}~\cup~ \{0\}
\end{aligned}
\]

또한 기존의 SSL 방식들을 적용한 논문의 방향과는 다르게(pre-training stage를 통해 transferrable feature를 학습하는 것) contrastive learning을 test-time adaptation phase와 함께 적용하는 것이 특징이라고 할 수 있다.

### Regularization

앞서 설명했던 것과 같이 weakly augmented target sample $t_w(x_t)$를 사용하여 nearest neighbor 방법을 통해 pseudo label $\hat{y}$을 예측하게 된다. 이렇듯 entropy thresholding을 통해 특정 category의 probability를 최대화하는 방법이 아닌 nearest neighbor의 soft voting을 사용하는 것이 일종의 regularization이고, 실제로 strongly augmented image에 supervision으로 사용될 때 <U>정규화 효과가 내재</U>된다고 볼 수 있다.

\[
L_i^{CE} = -\mathbb{E}\_{x\_t \in \mathcal{X}\_t} \sum_{c=1}^C \hat{y}^c \log p\_q^c
\]

위의 loss function은 실제로 target sample에 pseudo supervision을 주는 과정이고, $p_q$는 strongly augmented query image $t_s(x_t)$에 대한 network의 예측 확률인 $\sigma(h(f(t_s(x_t))))$을 의미한다.

추가적으로 사용된 loss로는 diversity regularization이 있는데, 이는 각 category에 대한 prototype에 대해 <U>trivial solution을 가지지 않을 수</U> 있게 방지하는 역할을 한다.

\[
\begin{aligned}
L_t^{div} =& \mathbb{E}\_{x\_t \in \mathcal{X}\_t} \sum\_{c=1}^C \bar{p}\_q^c \log \bar{p}\_q^c \newline
\bar{p}\_q =& \mathbb{E}\_{x\_t \in \mathcal{X}\_t} \sigma (g\_t (t\_s(x\_t)))
\end{aligned}
\]

이렇게 앞서 소개한 모든 loss function에 대해 최적화를 진행하면 Contrastive TTA에서 안정적인 domain adaptation을 위해 제시한 방법들을 모두 적용할 수 있게 된다.

\[
L_t = \gamma_1 L_t^{ce} + \gamma_2 L_t^{ctr} + \gamma_3 L_t^{div}
\]

보통 위와 같이 loss function이 많아지게 되면 hyperparameter 튜닝을 통해 각 loss의 중요도가 성능에 미치는 영향이 중요해지는데, 저자들은 $\gamma_1 = \gamma_2 = \gamma_3 = 1.0$로 설정하고 본인들이 제시한 방법이 <U>hyperparameter</U> 변화에 <U>robust한 성능</U>을 보인다고 했다.

---

# Experiments

검증을 위한 Dataset으로는 VisDA-C 그리고 DomainNet-126을 사용하였다. 원래 DomainNet 데이터셋은 noisy label을 포함하고 있었기 때문에, Real, Sketch, Clipart 그리고 Painting에 대해 126개의 class를 가지는 일부 subset을 사용하였고, 이를 DomainNet-126이라고 언급하였다. 모델 구조로는 ResNet-18/50/101 모델을 각각의 실험에서 backbone으로 사용하였다. 

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/227495183-3acc4366-54dd-4053-bee5-5741eb5e3a1a.png" width="700">
</p>

위의 그림은 선행 논문들 중 [SHOT](https://arxiv.org/abs/2002.08546)에서 제시한 네트워크 구조이며, 이 논문에서도 마찬가지의 방법을 적용하여 backbone 이후 $256$ 차원의 fully-connected layer와 batch normalization을 사용하는 구조를 그대로 사용하였고, classifier 뒤쪽에 weight normalization을 진행하는 구조 또한 그대로 사용하였다.  기존 [MoCo](https://arxiv.org/pdf/1911.05722.pdf)에서 [MoCo-v2](https://arxiv.org/pdf/2003.04297.pdf)로 넘어가는 과정에서 SimCLR의 projection head 구조를 사용하게 되는데, 이 논문에서는 굳이 해당 head 없이 contrastive learning을 진행했음에도 성능 하락 없이 잘 진행되었다고 한다. Baseline 비교는 기존 UDA 으로 유명한 방식들과 진행하였고, 이에 추가로 지속적으로 연구된 TTA 방식들도 비교 baseline에 추가하였다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/227495186-5a1a36f6-04b7-4370-873e-b3d4fc111db0.png" width="850">
</p>

VisDA-C train dataset에서 validation으로 넘어가는 과정에서의 성능 평가는 위와 같다. VisDA0C dataset은 아래와 같이 구성된다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/227495189-fb93a03b-47df-49b7-ac92-ba7a53fbd248.png" width="600">
</p>

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/227495195-6340a53f-0b52-4eb1-9fa4-f43bc4ed6b93.png" width="700">
</p>

이번에는 DomainNet-126에 적용한 결과를 보여주고, 총 $7$가지의 domain shift에 대해 실험 진행 및 정확도를 계산하였다. 그런데 논문에서 조금 흠이라고 한다면 epoch를 $15$나 돌려서 얻어낸 결과라는데, 본인 생각에는 contrastive learning이 유의미한 representation을 안정적으로 얻기 위해서는 충분한 학습이 필요하기 때문이고, 이로 인해 <U>bottleneck이 생기지 않았을까</U>, 추측해본다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/227495201-e3dcb6ac-7bfc-42c5-a7ff-98de3afc3abb.png" width="600">
</p>

대표적인 <U>entropy minimization</U> 방법론을 활용한 TTA 논문인 SHOT과 Confidence/Accuracy에 대한 그래프는 위와 같다. SHOT의 경우 entropy minimization을 통해 전반적인 샘플의 accuracy를 향상시키는데 성공하였으나, 실제로 confidence가 낮은 샘플들(엔트로피가 높은 샘플들)이 accuracy 경향성을 그대로 따르지는 않는다는 것이다. 이 그래프를 사용하여 model calibration(정렬)에 대한 해석을 하였는데, $y = x$ 그래프를 보다 잘 따르는 본인들의 방법이 실제로 network의 confidence를 accuracy에 잘 반영하는 형태를 보여주기 때문에 <U>calibration에 효과적</U>이라고 언급하였다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/227495203-59c882a9-047a-4573-a74c-beac1b6dcee0.png" width="550">
</p>

추가로 <U>queue size</U>(메모리 크기)와 soft voting을 진행할 때 참고하는 <U>nearest neighbor의 개수</U>에 대한 ablation도 진행하였다. Queue size는 많을 수록 좋아지는 경향은 있지만, 특정 갯수 이후로는 성능 수렴이 시작되는 것을 볼 수 있고 neighbor은 K-NN 알고리즘에서와 같이 너무 적게 참고하게 되면 **overfitting**, 너무 많이 참고하게되면 **underfitting** 때문에 성능이 떨어지는 것을 확인할 수 있다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/227495206-ce1f3c17-67ec-4723-a01b-b1e19ab1d2ed.png" width="550">
</p>

본인들의 contribution 중 하나인 hyperparameter에 대해 robustness를 보여주는 과정은 조금 실망스러웠는데, lr scale을 단순히 $10$배 증가시키는 것이 정말 유의미한 효과가 있는지, optimizer의 안정성 때문인지 그 효과를 명확하게는 보여주지 못한다고 보였고, 특히 방법론에서 loss function을 총 3개 제시하였고 이를 단순히 동일하게 더한 값을 최적화하였는데, 이에 대한 <U>weight value ablation</U> 진행이 없었다는 점이다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/227495207-325faf51-bcdc-48a9-ba02-a409d1c2bfdc.png" width="600">
</p>

---

# Conclusion

본인이 의문이 남았던 점은 VisDA-C에 대해서는 training epoch보다 test adaptation epoch을 더 많이 사용했다($10 < 15$). 결국 test time adaptation을 하고자 하는 이유는 사전 학습된 representation에 target domain의 unsupervised setting만으로도 빠른 alignment를 진행하는 방법을 찾는 것인데, 안정적인 성능을 위해 epoch를 더 많이 투자해야만 한다면 이는 효율적인 방법으로 제시될 수 없다는 것의 반증이 아닐까 생각했다. 추가로 논문에서 본인들이 많은 loss(classification, contrastive learning, diversity regularization 등등)을 적용하면서 weight에 대한 고려를 하지 않은 이유는 hyperparameter에 대한 robustness 때문이었다고 했는데, 실제로 뒤에 experiment에서 보여준 결과에서는 단순히 loss function을 제외했을때 성능이 하락하는 모습을 보여준 것이다.