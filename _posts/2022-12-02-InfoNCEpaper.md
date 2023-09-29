---
title: Simple explanation of NCE(Noise Contrastive Estimation) and InfoNCE
layout: post
description: paper review
use_math: true
post-image: https://user-images.githubusercontent.com/79881119/220251868-6804a167-f742-41f7-ae6e-03fbb25d9dac.gif
category: paper review
tags:
- Contrastive learning
- AI
- Deep learning
---
해당 내용은 Representation Learning with Contrastive Predictive Coding에서 소개된 InfoNCE를 기준으로 그 loss term의 시작에 있는 InfoNCE에 대해 간단한 설명을 하고자 작성하게 되었다. [논문링크](https://arxiv.org/abs/1807.03748)   

InfoNCE는 contrastive learning의 기본에 있는 연구가 되며, 흔히 multimodal(멀티모달)이라 불리는 AI의 새로운 지평을 열기 위해 보다 다양한 task에서도 공통적으로 학습 가능한 형태의 representation space를 찾기 위한 방법 중 하나라 볼 수 있다. 그렇다면 구체적으로 InfoNCE를 알아보기 전, Noise Contrastive Estimation에 대해서 간단하게 소개해보도록 하겠다. NCE가 궁금하지 않다면 그냥 넘겨서 InfoNCE 부분만 읽어도 된다.

# NCE: A gentle introduction to NCE
다음 링크의 글을 참고하였다. [링크](https://www.kdnuggets.com/2019/07/introduction-noise-contrastive-estimation.html)

## Background
이 개념을 설명하기 위해서는 먼저 문제를 상정해야하는데, 이는 NLP task로 예를 드는 것이 좋다. 가장 유명한 접근 방식으로는 Word2Vec이 있는데, 다음과 같은 process를 따라서 문제를 이해해보도록 하자.

---

"빠른 주황색 여우가 점프를 한다." 라는 문장이 있다.

Sliding window 방식으로 문장의 각 단어들에 대해 (문맥, 타겟)의 pair를 생성한다. 여기서 타겟은 sliding window가 포함하는 영역에서의 중간 단어를 의미하고, 문맥은 그 주변 단어를 의미한다. 간단한 문제 제시를 위해 여기서는 문맥을 neighboring 1 word라고 가정한다.

- ((빠른, 여우가), 주황색),

- ((주황색, 점프를), 여우가),

- ((여우가, 한다), 점프를)

각 문맥에 대한 word를 vector로 바꾸는데, 이러한 방식은 lookup table를 사용하는 방법이 될 수 있다. 자세한 방법론에 대한 부분은 tensorflow가 제시하는 튜토리얼을 참고하면 좋다. 임베딩은 보통 벡터 형태로 바꾸는 것을 의미하는데, 이렇게 되면 전체 context를 대표하는 context embedding은 각 context word의 임베딩의 평균으로 생각할 수 있다.

이렇게 임베딩의 평균을 사용한 context vector를 MLP(Fully connected NN layers)에 넣게 되고, softmax를 토대로 target word에 대한 확률 매핑을 추출하게 된다. 즉 output이 target 단어들의 후보군에 대한 확률 map이라 생각하면 된다.

맞는 단어에 대한 one-hot encoding에 대해 CrossEntropyLoss 최적화를 진행하면 된다.

---

<p align="center">
  <img src="https://user-images.githubusercontent.com/79881119/209057172-5ca85d2a-98ee-4e32-973b-4b154a7130bc.png"/>
</p> 
이러한 방식을 통해, sentence 내에서 특정 단어들의 추출 빈도나 통계를 학습하게 된다. 또한 lookup table의 경우 학습이 가능하므로, 최종적으로 학습이 마무리된 경우 embedding space 상에서 엇비슷한 단어들은 비슷한 위치에 있게 되고, 서로 다른 단어들은 동떨어져서 존재하게 될 것이다.   
그러나 위에서 4번째 process를 잘 보다보면, 네트워크는 dense layer 구조를 weight matrix of (임베딩의 차원 수, 단어 수)로 가지게 된다. 즉, 우리가 각 vocabulary 단어에 대한 예측을 위해서는, 먼저 각 단어에 대한 layer output을 구하고,

​\[
    \text{for } i : \text{ index of word, }z_i = Wx_i    
\]

이에 softmax transformation을 통해 확률 값으로 매핑한다.

\[
    p(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{\vert V \vert} e^{z_j}}​
\]

​​​물론 이와 같이 계산하기 위해서는 기존의 vocabulary size가 지정되어 있어야 한다. 그리고 이를 기반으로 cross entropy loss를 계산하는 것이다.

\[
    L = -\sum_j^{\vert V \vert} y_j \log (p_j) = -\log (p_{\text{target}})    
\]

여기서 중요한 점은 loss function이 모든 prediction에 대해서 더해지는 것처럼 보이지만 사실은 0이 아닌 term은 결국 실제 라벨에 해당되는 $y_i$​​ 가 1인 지점만 생각하게 되는  것이다. 즉, actual target word에 대한 확률값만 고려하게 된다는 의미. 그런데 $p(z_i)$를 보면 모든 위치의 단어에 대한 probability는 항상 같은 분모를 가지게 된다. 이 분모의 term을 보다보면,

\[
    \sum_{j=1}^{\vert V \vert} e^{z_j}
\]

인데, 바로 이 denominator 덕분에 굳이 매번 parameter가 모든 training example에 대해 non-zero gradient term을 가지게 되는 것이다. Noisy한 학습이 진행된다고 생각해주면 될 것 같다.

## Negative sampling
이러한 문제를 해결하기 위해 한가지 고려할 수 있는 것은 잘못된 vocabulary(negative term)을 모두 더하는 것이 아니라 이 중에 일부를 선택하는 것이다. 이렇게 선택된 일부의 non-target word를 negative samples라 부를 것이다. 이 글 전체에서 쓰이기도 하고, 실제로 infoNCE를 언급할 때 사용할 용어 전반은 앞에서 정의한 단어 그대로를 사용할 것이다.   
앞서 언급한 모든 process를 동일하게 진행하되, negative sampling만 추가된 것이다. 즉 이를 다시 풀어쓰게 되면,

---

"빠른 주황색 여우가 점프를 한다." 라는 문장이 있다.

Sliding window 방식으로 문장의 각 단어들에 대해 (문맥, 타겟)의 pair를 생성한다. 여기서 타겟은 sliding window가 포함하는 영역에서의 중간 단어를 의미하고, 문맥은 그 주변 단어를 의미한다. 간단한 문제 제시를 위해 여기서는 문맥을 neighboring 1 word라고 가정한다.

- ((빠른, 여우가), 주황색),

- ((주황색, 점프를), 여우가),

- ((여우가, 한다), 점프를)

각 문맥에 대한 word를 vector로 바꾸는데, 이러한 방식은 lookup table를 사용하는 방법이 될 수 있다. 자세한 방법론에 대한 부분은 tensorflow가 제시하는 튜토리얼을 참고하면 좋다. 임베딩은 보통 벡터 형태로 바꾸는 것을 의미하는데, 이렇게 되면 전체 context를 대표하는 context embedding은 각 context word의 임베딩의 평균으로 생각할 수 있다.

이렇게 임베딩의 평균을 사용한 context vector를 MLP(Fully connected NN layers)에 넣게 되고, softmax를 토대로 target word에 대한 확률 매핑을 추출하게 된다. 이 때, 전체 단어에 대한 확률 매핑이 아닌, 일부 추출된 negative sample과 positive sample을 entire sample space라 가정하고 계산한다.

맞는 단어에 대한 one-hot encoding에 대해 CrossEntropyLoss 최적화를 진행하면 된다.

---

물론 이렇게 일부 샘플들을 추출할 경우 매번 denominator가 달라지게 되므로 normalize가 제대로 진행되지 않을 수도 있다는 문제가 있지만, 이는 수많은 학습을 통해 approximation이 가능하다고 보고, 이러한 학습법에서의 가장 주요한 점은 gradient update의 수를 줄일 수 있다는 것이다. 즉,

\[
    \vert Embedding \vert \times \vert V \vert \rightarrow \vert Embedding \vert \times \vert N+1 \vert    
\]

negative samples N에 대해서 위와 같이 전체 space(vocabulary samles)이 아닌 일부 영역에 대해서만 최적화가 진행된다.
이는 상당히 많은 vocabulary가 존재하는 NLP task에서 그럴듯하게 들리는게, 애초에 문맥상 **"얼룩말"**이 전혀 쓰이지 않는 상황에서까지 해당 embedding을 최적화하는 것은 학습에도 악영향을 미치기도 하며, 추가적인 메모리 손실을 불러온다.

바로 Noise Contrastive Estimation(NCE)는 이러한 negative sampling을 일부 이론을 추가하여 구현한 형태가 되겠다.

## Learning by comparison

Negative Sampling에서는, true target을 1, 그리고 random samples의 target을 0s로 지표화한다. 이러한 지표화는 네트워크로 하여금 자연스럽게 어떤 샘플이 real이고 어떤 샘플이 noise인지 구분하게끔 한다. NCE는 바로 logistic regression 모델링을 토대로 이러한 문제에 답을 하게끔 해준다. Logistic regression modeling은 input이 다른 클래스가 아니라 해당 클래스에서 왔을 log-odds 비율과 같다.

\[
    logit = \log (\frac{p_1}{p_2}) = \log (\frac{p_1}{1-p_1})   
\]
여기서 우리는 log-odds가 true word distribution P에서 왔을 확률과 noise distribution Q에서 왔을 확률의 비를 정할 수 있다.
\[
    logit = \log (\frac{P}{Q}) = \log (P)- \log (Q)   
\]
즉, negative sampling인 noise에 대해 real distribution을 상대적으로 logit 학습을 통해 접근하기 때문에 noise contrastive estimation이라는 용어로 표현 가능한 것이다.   
우리는 실제 distribution인 P는 intractable하지만, noise distribution Q는 어떤 식으로든 정의할 수 있다. 예컨데 만약 all vocabulary를 동일 확률로 샘플링하거나, training data에서 word의 출현 빈도를 고려하는 방식으로 샘플링할 수도 있다. 이런 저런 방법을 떠나 여기서 중요하게 적용할 점은 우리에게 있어서 $\log (Q)$ 계산에 대한 명확한 방향을 제시해준 셈이다.   
다시 한 번 앞서 소개한 word2vec network를 살펴보고 Q가 어떤 방식으로 사용될 수 있는지 살펴보도록 하자.

<p align="center">
  <img src="https://user-images.githubusercontent.com/79881119/209057172-5ca85d2a-98ee-4e32-973b-4b154a7130bc.png"/>
</p> 
우리는 context vector를 네트워크의 입력으로 사용할 것이다. 그러나 vocabulary 모든 단어에 대한 output 계산이 아니라, 우리가 미리 정의한 **distribution 'Q'**로부터 무작위 샘플링 된 단어(random samples)에 대해 계산할 것이다. 그렇게 되면 network의 출력값에 대한 계산은 target 단어와 noise distribution으로부터 샘플링된 $N$개의 단어에 대해 진행되는 것이다. 즉, network evaluation은 $N+1$(random samples + target)에 대해 진행된다고 생각하면 된다.   
우리가 negative sampling을 진행하게 될 noise distribution을 정의하였고, 해당 분포로부터 추출된 noise를 사용하기 때문에 Q에 따라 analytically하게 각 단어의 확률을 계산할 수 있다.   
예컨데 만약 "고양이"이 10%의 확률로 샘플링이 되고, "네코"가 90%의 확률로 샘플링이 된다면 "고양이"가 샘플로 추출되었을 때의 Q는 0.1인 것이다. 결국, Q는 각 단어 샘플에 대한 추출 빈도라고 인식하면 된다. 그리고 앞선 수식에서의 probability $P$는 네트워크를 통과한 형태의 prediction이다. 학습은 일반적인 logistic regression task와 같이 target word는 1로, negative samples은0s로 되게끔 할 것이다.   

---

# InfoNCE
앞서 조금 길게 NCE에 대한 내용을 짚고 넘어왔다. 앞서 noise constrastive estimation(NCE)에 대해 길게 설명을 했던 이유는 이 논문에서 제시하는 loss의 형태와 intuition이 그로부터 나왔기 때문이다.   
AI, 딥러닝이라 불리는 분야가 성공하는데 있어서는 gradient based optimization을 기준으로 SGD, RMSprop, Adam 등 다양한 알고리즘을 통해 훌륭한 최적화 알고리즘이 나왔으며, labeled dataset을 통한 supervised learning이 주된 역할을 담당했다고 볼 수 있다. 그러나 여전히 딥러닝은 다양한 모달리티(modality)에 적용되기 힘들다는 점에서 한계가 있다. 여기서 modality는 무언가를 나타내는 형식/형태로 해석하면 될 것 같다. 즉 데이터셋의 유형이라고 생각해보자.   
사람의 목소리를 구분하는 task에서(화자 인식) 학습된 데이터셋은 음악 장르를 구분하거나 번역하는 task에 적용되기 힘들다. 비슷한 형태의 representation이 학습되었다는 가정에 대해서는 transfer learning이나 domain adaptation과 같은 방법이 있지만, vision/audio 및 vision/text와 같이 전혀 무관한 modality에서는 이러한 가정이 전혀 성립될 수 없다는 것이다.   
이러한 측면이 가지는 또다른 문제점은 unsupervised learning에서도 나타나는데, 결국 high-level representation을 나타내는데 유의미한 representation을 별다른 supervision 없이 학습할 수 있을지에 대한 궁금증이 생긴다.   
비지도 학습에서는 별다른 지표화 진행 없이 supervision을 가해야하기 때문에 흔히 predictive coding이라는 기법이 사용되는데, 이는 causal한 형태의 데이터를 가정하고 빈 부분이나 미래의 값을 예측하는 형태로 학습을 하게 된다. 이 논문에서는 바로 이런 predictive coding과 NCE를 수식화한 형태의 objective function을 모티브로 삼는다.   
해당 논문에서 제시하는 방법은 다음과 같이 세 가지로 나눌 수 있다.   
- High dimensional data를 보다 compact한 latent embedding space로 mapping, 이에 따라 conditional prediction이 모델로 하여금 더 간단하게 수행될 수 있게 하는 것이 목적이다.
- 이러한 latent condition을 통해 prediction을 함에 있어서 powerful autoregressive(AR) 모델을 사용한다.
- NCE(Noise Contrastive Estimation)을 loss function에 활용함으로써 모델 전반이 end-to-end로 학습되게 한다.
   
길고 길었던 소개글이고, 이제 본격적으로 Contrastive predicting coding에 대해 살펴보도록 하자.

## Main intuition, motivation
이 논문에서 기억해야할 것은 등장하는 수식보다는, 바로 어떻게 최적화를 고안했는지에 대한 intuition이다. 공유되는 정보를 high dimensional signal이라고 했을 때, high dimensional signal의 서로 다른 부분 정보들의 인코딩 방식이 된다. 즉 modality의 자체(low level information, raw dataset)에 대한 representation을 학습하는 것이 아닌, 각 도메인 latent space에서의 relation을 모델로 하여금 implicit하게 학습하고 싶다는 것이 주목적이 되겠다.   
그렇기에 새로운 형태의 loss를 고안해야 했고, 그 방법으로 NCE가 제안이 된 것. 그리고 보통 prediction을 할 때(regressive model과 같은 구조에서), 보다 먼 미래의 값을 예측할 때 global information을 보아야하는 경우가 있는데, 이를 slow feature라고 부른다. Slow feature란 오디오 데이터에서는 억양이나 분위기가 될 수 있고 이미지에서는 object 자체가 될 수도 있으며 텍스트에서는 줄거리나 큰 맥락에서의 주제 등과 같이 보다 데이터 전반에 걸친 inference가 필요한 경우를 의미한다.   

고차원 데이터셋에서의 어려운 점을 꼽으라면, 단연 MSE나 Cross Entropy Loss가 크게 효과적이지 않다는 점을 들 수 있다. 예컨데 이미지 생성 모델을 기준으로 1024 x 1024 샘플을 생성할 때, per pixel loss를 사용하게 되면 penalty가 커지는 문제 때문에 예측 형태가 edge 부분을 명확하게 그리지 못하고 뭉개지는 현상(blurry)이 생긴다. Cross Enropy Loss 또한 번역 모델에서 예측 단어의 수가 증가할수록, 모든 class에 대해 gradient를 먹여줘야한다는 문제랑, 수많은 probability label이 sparse하게 분포된다는 점에서 학습 성능이 저하된다는 문제가 생긴다.   

또다른 문제점으로는 powerful conditional generative model이 필요하다는 점이고, 이는 모델의 학습에 있어서 학습하고자 하는 encoder($E$)와 함께 GPU 상에서 학습이 되어야하기 때문에(비록 parameter update는 되지 않더라도) 그만큼의 하드웨어를 차지한다는 것이 문제가 되며, 또한 generation time에 따라 학습 bottleneck이 생긴다는 단점이 있다. Conditional generative model의 또다른 문제로는 context $c$에 대한 ignorance인데, 이는 고차원 데이터셋인 이미지를 생성함에 있어 class label과 같은 high-level latent 변수는 적은 수의 정보를 담고 있기 때문이다. 즉, decoder를 설계함에 있어 conditional한 확률 분포 $p(x \vert c)$를 직접적으로 모델링하는 것은 사실은 $x$와 $c$의 공유된 정보를 통해 데이터를 생성하는 것이 아니라 단순히 $c$를 무시한 채 생성을 하도록 학습이 진행되었을 가능성이 있다는 것이다.   

따라서 해당 논문에서는 future information을 예측할 때 target x(미래의 값)와 context c(현재의 값)을 compact한 distribution으로 학습하여, 서로 정보 공유가 되게끔 non linear mapping을 고안하였다. 이는 Mutual information식을 통해,
\[
    I(x; c) = \sum_{x,~c} p(x, c) \log \frac{p(x \vert c)}{p(x)}    
\]

위와 같이 표현할 수 있다. Mutual information(MI)는 인자가 되는 두 분포$(x, c)$의 joint distribution인 $p(x, c)$가 단순곱 $p(x)p(c)$와 얼마나 비슷한지 측정하는 척도로 쓰이며, 아래와 같이 유도된다.

\[
    I(x, c) = \sum_c \sum_{x \in X} p(x,c) \log \frac{p(x, c)}{p(x)p(c)}    
\]

위의 식은 KL divergence 식과 똑같이 유도되고, 이를 bayes rule을 통해 간단히 표현하면,

\[
    I(x, c) = \sum_c \sum_{x \in X} p(x,c) \log \frac{p(x, c)}{p(x)p(c)} = \sum_{x, c} p(x, c) \log \frac{p(x \vert c) p(c)}{p(x) p(c)} = \sum_{x, c} p(x, c) \log \frac{p(x \vert c)}{p(x)}  
\]

위와 같이 표현 가능하다. $x$와 $c$의 dependent한 척도를 나타내는데, 만약 $x$가 $c$에 대해 independent하다면 위의 값은 작아지고, 반대로 dependent할수록 그 값이 커지게 된다. KL divergence와는 다르게 위의 식은 commutable하기 때문에 대칭성이 성립한다.

## Contrastive predictive coding
<p align="center">
  <img src="https://user-images.githubusercontent.com/79881119/209057175-14b2eb0c-a332-4d8d-b2a2-9e1e0d195812.png"/>
</p> 
위의 그림은 CPC(Contrastive Predictive Coding) model의 구조를 보여준다. 먼저, non-linear encoder로 표현된 g_enc가 input sequence인 x_t를 latent representation의 sequence로 바꾼다.

\[
    g_{enc}(x_0, x_1, \cdots, x_t) = (z_0, z_1, \cdots, z_t)    
\]

여기서 autoregressive model $g_{ar}$은 causal한(현재 시점으로 과거의) input들을 활용하여 context latent representation을 생성한다.

\[
    g_{ar}(z_{\leq t}) = c_t    
\]

여기까지는 흔한 AR 구조의 흐름과 사실상 동일하다. 그러나 위에서 언급했던 것과 같이 future observation을 직접적으로 예측하는 것이 아니라, future prediction과 context vector 사이의 mutual information에 비례하는 density ratio를 모델링한다.

\[
    f_k(x_{t+k,~c_t}) \propto \frac{p(x_{t+k} \vert c_t)}{p(x_{t+k})}    
\]

위에서 설명한 식에서 $\log$ 내부에 들어가는 값이 결국 $x$, $c$의 joint distribution에 대한 $x$, $c$각각의 분포의 곱과의 차이를 보여주는 비율이 되겠다. 그렇기 때문에 위의 값에 비례하는 형태로 모델을 구성하게 되면 자동적으로 future prediction이 유의미하게 context를 보도록 강조할 수 있다는 것!(density ratio를 최적화함으로써, 모델한테 "이걸 보고 배우면 돼" 식으로 과외 선생님처럼 가르쳐줄 수 있다는 것이다)   

density ratio $f$는 정규화되지 않은 값으로, 다음과 같이 $\log$ bilinear model이 될 수도 있고 non linear network 및 RNN 구조로 대체될 수 있다.

\[
    f_k(x_{t+k}, c_t) = exp(z_{t_k}^T, W_kc_t)    
\]

이러한 density ratio를 활용, encoder와 함께 다음 $z$를 예측함으로써 model이 고차원의 데이터인 $x$를 예측해야하는 문제를 해결해줄 수 있다. Encoding된 latent space 상에서의 $z$를 예측하는 방식은 직접적으로 $p(x)$나 $p(x \vert c)$를 건들 수 없지만, 우리가 가정할 수 있는 분포를 가지고 sampling을 진행하는 Noise contrastive estimation과 importance sampling이 사용될 수 있는 것이다. 드디어 우리가 고생고생해서 이해한 NCE를 써먹을 타이밍이 왔다.

## InfoNCE, Mutual information estimation
위의 구조를 잘 보면 결국 우리가 훈련해야 하는 것은 encoder, autoregressive model에 대한 두 개의 모델이다. 두 모델은 모두 NCE를 기반한 loss function에 의해 최적화가 될 것이고, 이를 InfoNCE라 명명하였다.   
앞서 봤던 NCE와 동일한 방법으로, $N$개의 무작위 샘플을 가정할 건데, 이 중에 1개는 posivie sample이 될 것이고 $N-1$개는 proposal distribution을 통해 추출된 negative sample이 될 것이다.

\[
    X = (x_1, x_2, \cdots, x_N)    
\]
\[
    \begin{cases}
        positive~x, & x \sim p(x_{t+k} \vert c_t) \newline
        negative, & x \sim p(x_{t+k})
    \end{cases}   
\]
그래서 최적화할 loss function은 결국,

\[
    \mathcal{L} = -E_X (\log \frac{f_k(x_{t_k}, c_t)}{\sum_{x_j \in X} f_k(x_j,c_t)}) 
\]

요렇게 된다. 근데 사실 이 objective function만 보면, 단순히 categorical cross entropy를 표현한 것처럼 보여 크게 다를 것이 없어 보이는데, 대체 왜 이게 density estimation에 근접한 형태로 학습이 되는지 혼란스러울 수 있다. 왜냐면 내가 그랬는데 똑똑이인 분들은 이걸 보고 바로 이해할 수도 있겠지만 암튼 결국 우리가 얻고자 하는 최종 optimal probability를 생각해볼 수 있는데, 이는 $p(d = i \vert X, c_t)$라고 할 수 있겠다.   
위에서 말했던 것과 같이 모델이 예측하는 것은 특정 샘플이 context를 보고 나온 녀석인지 아니면 context를 보지 않고 나온 녀석인지에 대한 비율이고, 이 값이 고대로 Mutual information의 logarithm을 통해 수치화된다고 했었는데, 결국 우리는 optimal probability를 주어진 샘플 X와 context vector $c$에 대해서 다음 식으로 나타내볼 수 있다.   
\[
    p(d = i \vert X, c_t) = \frac{p(x_i \vert c_t) \prod_{l \ne i} p(x_l)}{\sum_{j=1}^N p(x_j \vert c_t) \prod_{l \ne j} p(x_l)} = \frac{\frac{p(x_i \vert c_t)}{p(x_i)}}{\sum_{j=1}^N \frac{p(x_j \vert c_t)}{p(x_j)}}
\]
따라서 앞서 본 식에서 네트워크 f가 예측하는 값이 probability ratio에 비례함을 알 수 있으며, 이는 negative sample의 개수와는 무관한 것을 확인할 수 있다. Training에는 단순히 loss만 사용되지만, mutual information은 다음과 같은 lower bound를 통해 확인할 수 있다.

\[
    I(x_{t+k}, c_t) \geq \log(N) - L_N    
\]

이는 $N$이 커질수록 tight해진다. 여기서 tight의 의미는 lower bound가 실제 mutual information의 infimum에 가까워진다는 것이다. 즉 실제 값에 근사한다는 의미. 사실 더 쓰자기엔 Appendix가 있는데 이걸 여기서 다 풀어쓰기에는 무리가 있어 여기서 마무리하도록 하겠다.

# Experiments
단연 이 모델의 가장 큰 장점이라고 한다면 encoder를 arbitrary한 구조를 가져와 적용 가능하다는 점과 representation learning에 대한 loss의 기준을 마련했다는 점이 될 수 있다. 그렇기에 해당 논문에서도 여러 modality에 대한 실험을 함께 진행했다.

## Audio dataset
<p align="center">
  <img src="https://user-images.githubusercontent.com/79881119/209057177-0574457c-a2bc-4272-979e-d6e811a04ada.png"/>
</p> 
Librispeech는 총 251명의 발화자에 대한 음성 파일로 구성되고, 각각에 대한 text script가 주어진다. 이러한 text script는 실제 phone sequence와는 alignment가 되어있지 않기 때문에 추가적인 annotation이 필요하다. 필자는 이걸 모 랩실에서 구현할 때 노답 알고리즘을 파이썬으로 구성해서 만들었었는데 [Kaldi Toolkit](https://kaldi-asr.org/)이란게 있다고 한다. 이런 좋은게 있으면 미리 알려주지.. 흔히 CV에서 opencv-python 모듈을 많이 사용하는데, 이건 그거의 오디오 버전이라고 보면 될 것 같다. 좋은 사실 하나 알아갑니다...

<p align="center">
  <img src="https://user-images.githubusercontent.com/79881119/209057181-2a6a7337-5fc4-454a-af9a-173a7846960e.png" width="400"/>
  <img src="https://user-images.githubusercontent.com/79881119/209057183-c781b0d7-318c-42a9-b4f4-ed2510522eb3.png" width="400"/>
</p> 

아니 세상에 심지어 이 논문에서 분리하고 alignment한 데이터셋을 구글 드라이브로 친절하게 공유도 해주셨다. 본인 speech transformer 학습시킬 때 이런 게 있는 줄 진작 알았더라면 이거 다운받아서 할 걸 그랬다. 후회가 막심하구만..   
암튼 이렇게 구성한 데이터셋으로 할 수 있는 것은 음성을 듣고 발화자를 분류하는 speaker classification, 각 시점에서의 phone(음절)을 분류해내는 phone classification이 있다. 두 task 모두 CPC 방식이 supervised 방식에 대해 좋은 성능으로 보답하는 걸 볼 수 있다. 우측 표는 Phone classification task에 대해 CPC 모델에 대한 ablation을 이것저것 보여준다.

<p align="center">
  <img src="https://user-images.githubusercontent.com/79881119/209057187-038f0dc9-cdbe-42df-a45c-0743bcf343fd.png" width="400"/>
</p> 

이건 단순히 loss 그래프처럼 보이는데 그게 아니고 현재 시점을 기준으로 latent step을 몇 번 거치냐에 따른 phone prediction 평균 정확도를 보여준다. Phone 개수는 41 possible classes가 있으니, 아무런 예측을 하지 못하는 확률값 기준은 대강 0.025로 생각하면 될 듯하다. 한 step의 latent은 10ms를 차지하기 때문에, 약 20 step을 기준으로 해당 범위까지 예측이 이어질 수 있음을 그래프로 보여준 것 같다.

# Vision
<p align="center">
  <img src="https://user-images.githubusercontent.com/79881119/209057189-814e8e88-7ae3-483c-8bec-9d7eecfc09e2.png" width="400"/>
</p> 
Vision은 좀 특이하게 ResNetv2 101 backbone을 이미지용 인코더로 사용하되, 원래 구조에서 batch normalization을 제외하고 사용하였다. Batch normalization은 classification과 같이 정해진 확률 범위 내에서 결과값을 뽑는데는 학습이 안정적이고 좋지만, representation을 학습해야 하는 generation이 연관된 학습 과정에서는 주로 쓰이지 않는 편이 낫다는 해석이 있다. 이건 뭐 믿거나 말거나인데 실제로 batch normalization이 들어가면 feature map 정규화가 진행되면서 보다 distribution collapse가 발생하기도 한다. Unsupervised learning이 끝난 뒤의 linear layer는 ImageNet labels를 추정하기 위해 따로 학습을 진행한다.

학습 과정은 다음과 같다.

- $256 \times 256$ 이미지로부터 $7 \times 7$ grid of $64 \times 64$ crops를 뽑아낸다. 어? 그럼 $64 \times 7 = 448$ 이어서 개수가 안맞는데요?? 싶은 사람들을 위해 말씀드리자면 각 patch는 32픽셀 만큼 오버랩되어서 추출한다. 즉 $256+(32 \times 6)/64 = 4+3 = 7$ 이 되는 것
- 각 crop은 인코딩되고 나서, 각 채널 별로 mean pooling하면 1024 벡터가 나오게 된다. crop이 총 $7 \times 7$ 만큼 있었으니까 결국 output은 $7\times 7 \times 1024$ 짜리 텐서가 나오게 된다.
- PixelCNN 형태의 AR 모델(궁금하면 찾아보면 되는데 간단하게 설명하자면 이전의 픽셀들을 보고 다음 픽셀을 예측하는 형태의 generative model) 해서 다음 픽셀들을 예측하는 형태로 unsupervised learning을 진행한다. 이건 위의 그림을 보면 이해가능

- Linear classifier는 앞서 훈련해놓은 CPC feature map을 통해 학습하는데, CPC 학습에는 Adam optimizer를 사용하고 Linear classifier는 SGD를 활용했다고 한다.

<p align="center">
  <img src="https://user-images.githubusercontent.com/79881119/209057191-a0a8b0cd-d182-4f25-9d5a-2a8c697b0275.png" width="400"/>
  <img src="https://user-images.githubusercontent.com/79881119/209057192-6fec3629-5d57-44c0-8409-00a9e026bbdb.png" width="400"/>
</p> 

결과는 매우 좋았던 것으로 보인다. 참고로 여기서의 Top-1 ACC, Top-5 ACC는 classification task와는 좀 다르게 앞서 언급한 procedure 처럼 unsupervised하게 학습된 feature map을 기준으로 성능을 평가한 것이다.

## Natural Language
이번엔 또 NLP다. Mobality 하나하나에 대한 실험 과정을 설명할라다보니 내가 인간 멀티모달이 되어가는 것 같긴한데, 요즘같은 AI 블루오션 시대에는 이렇게 여러 분야 찍먹하면서 살아남아야지 어쩌겠나 싶다. 암튼,,   

먼저 모델을 BookCorpus dataset에 대해 학습시킨다. NLP task 자체가 autoregressive하므로 학습 과정에 대해서는 앞서 word2vec과 관련된 NCE를 참고하면 될 듯하다. 학습 자체에서 학습되지 못한 단어들을 evaluation하기 위해서, word2vec과 모델이 학습한 embedding 사이에 linear mapping이 추가된다.   

저자들이 사용한 분류 task에 대한 데이터셋은 MR(영화 리뷰 감정), CR(고객 상품 리뷰), 주관적/객관적 평가, 의견(MPQA) 그리고 질문 타입(TREC) 분류 등등을 사용하였다.   
<p align="center">
  <img src="https://user-images.githubusercontent.com/79881119/209057193-fe7bad9c-fa88-4b46-9170-5376e0c60084.png" width="400"/>
</p> 
학습에 있어서 transfer learning setup은 skip-though vectors 논문에서 사용한 방식과 동일하게 했으며, 해당 논문을 기준으로 비교 대상을 선정한 것을 확인할 수 있다. 암튼 어느 정도 잘 된다는 것을 결과로써 보여준 모습.

## Reinforcement learning
하다하다 이제 RL(강화학습)까지 흘러들어왔다. 이쯤이면 저자들이 변태라는게 학계의 정설은 아니고 Deepmind 최고   
<p align="center">
  <img src="https://user-images.githubusercontent.com/79881119/209057194-b5e93178-d0c9-4178-8275-e2a0885a8bef.png" width="400"/>
</p> 
참고로 RL은 딥러닝과 그 결이 조금 다르기 때문에 objective를 다르게 줘야 한다. A2C agent를 base model로 사용하고 CPC를 auxiliary loss로 주었다. 자세한 내용은 사실 나도 잘 몰라서 넘어가도록 하겠다. 빨간색이 잘 나온걸 보면 효과적이라고 결론을 낸 것 같다.   

# Appendix

\[
    \begin{aligned}
        \mathcal{L}\_N^\text{opt} =& -\mathbb{E}\_X \log \left( \frac{\frac{p(x_{t+k} \vert c_t)}{p(x_{t_k})}}{\frac{p(x_{t+k} \vert c_t)}{p(x_{t+k})} + \sum\_{x_j \in  X_\text{neg}}\frac{p(x_j \vert c_t)}{p(x_j)}} \right) \newline
        =& \mathbb{E}\_X \log \left( 1+\frac{p(x_{t+k})}{p(x_{t+k} \vert c_t)} \sum\_{x_j \in X_\text{neg}} \frac{p(x_j \vert c_t)}{p(x_j)} \right) \newline
        \approx& \mathbb{E}\_X \log \left( 1+\frac{p(x_{t+k})}{p(x_{t+k} \vert c_t)} (N-1) \mathbb{E}\_{x_j} \left( \frac{p(x_j \vert c_t)}{p(x_j)} \right) \right) \newline
        =& \mathbb{E}\_X \log \left( 1+\frac{p(x_{t+k})}{p(x_{t+k} \vert c_t)} (N-1) \right) \newline
        \geq& \mathbb{E}\_X \log \left( \frac{p(x_{t+k})}{p(x_{t+k} \vert c_t)} N \right) \newline
        =& -I(x_{t+k}, c_t)+\log(N)
    \end{aligned}
\]

Loss function의 lower bound 유도 과정

\[
  \begin{aligned}
    \mathbb{E}\_X \left( \log \frac{f(x, c)}{\sum\_{x_j \in X} f(x_j,c)} \right) =& \mathbb{E}\_{(x, c)} \left( F(x, c) \right) - \mathbb{E}\_{(x, c)} \left( \log \sum\_{x_j \in X} e^{F(x_j, c)} \right) \newline
    =& \mathbb{E}\_{(x, c)} \left( F(x, c) \right) - \mathbb{E}\_{(x, c)} \left( \log \left( e^{F(x, c)} + \sum\_{x_j \in X_\text{neg}} e^{F(x_j, c)} \right) \right) \newline
    \leq& \mathbb{E}\_{(x, c)} \left( F(x, c) \right) - \mathbb{E}\_c \left( \log \sum\_{x_j \in X_\text{neg}} e^{F(x_j,c)} \right) \newline
    =& \mathbb{E}\_{(x, c)} \left( F(x, c) \right) - \mathbb{E}\_c \left( \log \frac{1}{N-1} \sum\_{x_j \in X_\text{neg}} e^{F(x_j, c)} + \log (N-1) \right)
  \end{aligned}
\]

Mutual information neural estimation과의 연관성을 보여줌. 결국  mutual information에서 intuition을 어떠한 방식으로 얻으셨는지 수식을 통해 보여주는 것 같다.