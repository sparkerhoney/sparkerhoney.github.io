---
title: Transformer와 Multimodal에 대하여
layout: post
description: Attention mechanism
use_math: true
post-image: https://user-images.githubusercontent.com/79881119/235353492-2ca621c9-8091-4c8a-b852-39195e45928b.gif
category: paper review
tags:
- Transformer
- Attention
- AI
- Deep learning
---

# Convolutional neural network의 발전
기존의 딥러닝에서 사용되던 대부분의 네트워크 구조는 <U>Multilayer perceptron</U>(MLP) 혹은 <U>Convolutional neural network</U>(CNN)이 주를 이루고 있었다. 단순히 modality를 $1 \times N$ 차원 벡터로 늘려서 연산하는 MLP와는 다르게 Convolutional neural network 구조는 image와 같은 modality에서 성능을 입증받았고, 다양한 연구들이 그 뒤를 이었다. Convolutional neural network가 MLP에 비해 가지는 장점은 많았다. 우선 첫번째로 MLP보다 동일한 hidden layer 개수를 가지는 deep neural network를 구성함에 있어 적은 parameter를 학습시켜도 된다는 점과, 학습 가능한 parameter 수는 더 적으면서도 학습 가능한 representation이 <U>modality에 대해</U> 일반화가 잘된다는 점이다.   
이러한 퍼포먼스 덕분에 NLP(Natural language processing)이나 Audio 관련 딥러닝에서도 CNN 구조를 통해 연구를 시작했으며, 이는 'Inductive bias' 덕분이라고 언급할 수 있다. <U>Inductive bias</U>란, 학습하는 모델이 관측하지 못할 모든 modality에 대해서 추정하기 위해 학습이나 추론 과정에서 주어질 수 있는 모든 '가정'의 집합이다. 이게 무슨 의미냐면 예를 들어 딥러닝 네트워크가 '왼쪽을 보는'라는 고양이에 대한 데이터셋이 없이 고양이에 대한 classification을 학습했음에도, 추론 과정에서 '왼쪽을 보는' 고양이 이미지가 주어졌을 때 해당 객체가 고양이라는 사실을 인지하도록 하기 위한 일종의 constraint라고 보면 된다. 물론 이 예시는 사실 약간 부적절한 내용이지만 **inductive bias**라는 용어가 transformer라는 모델에 대한 개념을 파악하기에 필수적이기 때문에 이를 먼저 간단하게 이해하고 넘어가고 싶었다.
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209152403-2d55f1f9-fb7b-4440-8cd4-e97db1f267a0.png" width="600"/>
</p>
위의 그림을 보면 간단하게 inductive bias에 대해서 설명할 수 있다. Convolution 연산이 진행되는 과정을 보면 필터가 특정 영역(ex. $3 \times 3$)을 기준으로 필터의 paramter와 feature 값을 모두 multiplication한 뒤 더하는 구조가 된다. 따라서 다음과 같은 두 가지의 가정을 해볼 수 있다.

- 이미지에 속한 object의 경우, 해당 object를 나타내는 픽셀의 상대적 위치는 보장된다(localization).
- 이미지에 속한 object가 움직일 경우, 해당 object에 대한 feature output 또한 이미지 상에서 움직인다(translation equivariance).

첫번째 조건의 경우엔 큰 문제가 없으나, 두번째 조건은 큰 문제가 생긴다. 왜냐하면 translation equivariance는 MLP에서도 처리하지 못했던 문제였고, 만약 같은 object의 <U>위치만 달라짐</U>으로써 얻어지는 <U>feature map의 형태에 변화</U>가 생긴다면, 같은 object에 대해 동일한 예측을 취할 수 있다는 게 보장되지 않기 때문이다. 물론 MLP와는 다르게 CNN에서는 물체의 형태가 달라지지는 않고, 단순히 그 상대적인 위치는 유지된 채로 feature map 상에서의 전반적인 localization만 바뀐다는 것이다.   
그러나 CNN의 구조 상에서 max-pooling과 같이 같은 kernel 내의 feature value를 단일한 값으로 축약하는 filtering module, 그리고 softmax를 통한 확률값 계산을 통해 localization이 보장된 feature map에 대해 동일한 예측값을 낼 수 있다는 사실이 확인되었다. 즉 <U>feature extraction</U> 부분은 translation equivariance와 localization을 기반으로 object에 대해 유의미한 feature 형태를 유지하는 과정이 되며, <U>classifier</U> 부분은 translation invariance를 통해 이렇게 추출된 feature map을 기반으로 일관성 있는 prediction을 하게된다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209152405-a2d45124-f1fc-447e-a1e0-9c37e14250fb.png" width="600"/>
</p>

바로 이러한 장점 덕분에 convolutional neural network는 더 적은 parameter 수를 가지고도 같은 class에 대한 feature representation 학습에 유리했고, ImageNet에서의 첫 성공 이후 수많은 연구가 진행되었던 것이다. 무엇보다 sequential data를 처리해야하는 NLP, Audio, Video 등등 temporal information을 추출하는데에도 CNN 구조를 많이 사용하기 시작했다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209152408-f033371c-33f4-4970-aa42-6faddd9f7386.png" width="600"/>
</p>

그러면서 자연스럽게 sequential data를 처리하기 위한 <U>RNN</U>(Recurrent neural network)이 등장하게 되었으며, LSTM, GRU와 같은 long-term memory 모듈을 활용하여 input과 output에 대한 global contextual meaning을 학습하려는 노력이 시작되었다.

---

# 기계 번역에서의 RNN과 한계점
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209152411-83f8a105-9fad-4f44-adbb-ef3790ea903d.png" width="600"/>
</p>
예를 들어 '<U>나는 고양이 이름을 에폭이라 짓기로 결정하였다</U>'라는 한국어 문장을 '<U>I decided to name my cat Epoch</U>'라는 영어 문장으로 번역하는 것을 머신러닝/딥러닝으로 해결하려는 기계 번역 task가 있다고 생각해보자. 단순히 이런 task를 RNN의 관점에서 접근하게 되면,   
<p align="center">
    한국어 문장 $\rightarrow$ $E_{\theta}$ $\rightarrow$ 임베딩 $\rightarrow$ $D_{\phi}$ $\rightarrow$ 영어 문장
</p>

위와 같이 표현할 수 있다. 중간에 있는 $E_\theta, D_\phi$는 각각 인코더와 디코더를 의미한다. 이러한 <U>Encoder-Decoder 구조</U>를 가지는 RNN 형태로 대표적인 것이 sequence-to-sequence 모델이다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209152412-1c51e386-83ea-4cd7-b202-b7776a1910a7.png" width="700"/>
</p>

첫번째 가장 큰 문제점은 연산 속도가 된다. 딥러닝이 레이어를 깊게 가져가면서도 연산 속도를 빠르게 할 수 있었던 것은 텐서 연산에 대해 GPU를 통한 병렬 처리가 가능했다는 점이다. 같은 level에서의 feature map value는 동일하기 때문에 convolution 연산을 <U>굳이 순차적으로 진행하지 않고도</U> 모든 연산을 **동시에** 할 수 있기 때문에 구조적 이점을 가져갈 수 있었다. 그러나 RNN의 경우는 그럴 수 없었다. 구조를 보면 알 수 있지만 LSTM(hidden layer로 사용된다고 보면 된다)의 각 계산을 하기 위해서는 이전 LSTM의 결과가 필요하다. 병렬 처리를 통해 LSTM 하나의 연산을 빠르게 한다고 하더라도 문장의 길이 $N$에 대해서는 여전히 <U>bottleneck</U>이 걸리고 있는 것이다. 두번째 문제점은 연산 과정에서 사용되는 context vector의 크기가 고정적이라는 것이다. 이 또한 문장의 길이가 길어질수록 큰 문제가 생기는데, RNN 구조에서 번역 과정에서 참고할 수 있는 feature embedding은 오직 encoder의 final hidden layer의 output이다. 따라서 복잡한 문장이나 번역이 까다로운 경우 문장의 길이가 길어질수록 encoder에서의 문장을 제대로 참고하지 못하는 문제가 발생하였다. 바로 이러한 문제로부터 등장한 것이 attention 메커니즘이고, transformer의 근간이 되는 기술이기도 하다.

---

# Attention mechanism
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209152414-898f80a8-522c-435b-b490-2dbb1a6db297.png" width="500"/>
</p>

앞서 말했듯이 RNN에는 크게 두 가지의 문제점이 있다. 첫번째는 long sentence에 대해 연산 속도가 너무 느리다는 점이고, 두번째는 문장이 길어지면 길어질수록 context vector의 <U>고정적인 길이</U>에 대해 performance 제약을 받는다는 점이다. 이러한 문제를 해결하기 위해서는 RNN을 구성하는 <U>모든 LSTM의 output에 대해</U> reasoning이 필요하게 되었고, 여기서 'attention'이라는 모듈이 제시가 되었다. Attention module은 input에 대해(single input이 될 수도, multiple input이 될 수도 있음) 서로 얼마나 연관도를 가지는지를 weight로 표현한 tensor를 추출한다. 예를 들어 다음과 같은 문장이 제시가 되었다고 생각해보자.

\[
    \text{I decided to name my cat Epoch}
\]

이를 단어 단위로 tokenize한 뒤에,

\[
    \text{(I, decided, to, name, my, cat, Epoch)}
\]

각 token을 모두 embedding 조건에 따라서 value로 mapping한 뒤, 단어 '<U>name</U>'에 대해서 같은 문장에 있는 단어들과의 유사성을 구해보려고 한다. 우선 알 수 있는 사실은 'name'과 가장 관련이 있는 단어는 '<U>Epoch</U>'이며, 그 다음으로 중요한 단어는 '<U>cat</U>'이 될 수 있을 것이다.

\[
    \text{(I, decided, to, name, my, cat, Epoch)} \rightarrow (0, 0, 0, 0.3, 0, 0.1, 0.6)
\]

극단적인 경우로 가정했지만, 아무튼 이처럼 input의 각 embedding 사이의 유사성을 weight로 매핑하고, 이렇게 추출된 <U>weight의 softmax 확률값</U>을 **attention value**로 삼게 되는 것이다. 이렇게 attention을 사용하게 되면 LSTM와 같은 long term 모듈에 의지하지 않고도 문장 전체에 대한 global correlation 혹은 long range interaction을 획득할 수 있다.   
그러나 여전히 이러한 sequence to sequence를 사용해서도 해결할 수 없는 문제가 있는데, 그것은 바로 decoder에 사용될 feature vector를 추출하기 위한 RNN의 '<U>순차적 연산</U>'을 가속화할 방법이다. 여전히 학습 성능과 추론 시간 사이에 해결할 수 없는 trade-off가 남아있게 된다. 이제 드디어 설명하려는 것이 이러한 bottleneck이 해결될 수 없는 RNN 구조에 <U>의존하지 않고</U> 유의미한 기계 번역을 할 수 있다는 내용의 새로운 패러다임을 제시한 <U>transformer</U>가 되겠다.

---

# Attention is all you need!
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209152419-c632ff15-9079-4f22-9a6d-9c3c2709d73d.gif" width="700"/>
</p>

~~뉴진스 최고~~ [Transformer](https://arxiv.org/abs/1706.03762)가 바로 이 제목과 함께 <U>거대한 어그로</U>를 끌며 나타났다. 해당 논문의 main idea는, 굳이 RNN과 같은 convolution 구조를 사용해서 global context를 뽑지 않더라도 어차피 attention을 쓰면 즉각적으로 global reasoning이 가능하고, 이러한 attention 연산을 여러번 진행하는 deep neural network 구조가 오히려 기계 번역과 같은 task에 더 적합하지 않겠냐는 것이다. 실제로 기존 RNN에 비해서 연산 속도를 줄이면서도 그 성능을 입증받았고, 이러한 <U>구조적 변화</U>는 NLP 및 여러 연구에 변화를 일으키는 초석이 되었다.
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209152421-69e23550-3f6e-4360-93f2-d6153e4c49f8.png" width="500"/>
</p>
지금부터 transformer 구조를 구성하는 중요 요소 중 하나인 self-attention, multi-head attention 그리고 positional encoding과 masked attention에 대해서 알아보도록 하자.

## Self Attention
RNN에서는 encoder 구조를 통해 feature vector를 추출하였다. 이 feature vector는 모든 input token에 대한 context를 담고 있게끔 학습된다는 것이 RNN 구조에서의 assumption이었고, 여기에 추가적으로 attention value를 통해 이전의 RNN feature vector까지 고려했던게 기존 방식이다. <U>Attention mechanism</U>만을 사용하는 **transformer**는 이러한 의존성을 없애고, 단순히 attention을 같은 문장에 여러번 적용함으로써 이러한 contextual embedding을 추출한다. Attention에서 사용되는 용어에 대해서 먼저 짚고 넘어가도록 하자.

- Query($Q$) : '질문' 혹은 '물음'이라는 의미를 가지고 있다. 문장 내의 각 토큰이 다른 토큰과 어떤 연관을 가지고 있을지 추론해가는 과정에 있어서 '각 토큰'을 의미한다. 예를 들어 위의 예시에서 봤던 것처럼 단어 '<U>name</U>'에 대해서 같은 문장(I decided to name my cat Epoch)에 있는 단어들과의 유사성을 구해보려고 할 때의 'name'과 같다.

- Key($K$) : Query와 pair로 작용하여, query가 특정 key에 대해 value를 물었을 때의 이를 줄 수 있는 역할을 한다. 예를 들어 위의 예시에서 봤던 것처럼 단어 '<U>name</U>'에 대해서 같은 문장(I decided to name my cat Epoch)에 있는 단어들과의 유사성을 구해보려고 할 때 문장 내에 있는 단어들(I, decided, cat 등등)과 같다.

- Value($V$) : Key에 대응되는 value를 의미한다. Query, key, value에 대한 개념은 단일로 이해하는 것보다는 셋의 <U>연관성</U>을 생각해보는 것이 훨씬 간단하다.

입력으로 주어지는 문장 $S$가 있고 이를 embedding 함수 $\mathcal{E}$를 통해 embedding tensor $X$로 만들었다고 해보자. 이 embedding tensor에 대해 query, key, value는 각각의 weight에 따라 liear projection 된다.

\[
    \begin{aligned}
        Q =& W_1X \newline
        K =& W_2X \newline
        V =& W_3X
    \end{aligned}    
\]

예를 들어 문장을 총 $n$개의 토큰으로 나눈 뒤, 각 토큰을 $e$의 dimension을 가지는 텐서로 치환했다고 하자. $Q,~K,~V$의 dimension $d$에 대해서,

\[
    \begin{aligned}
        X \in & \mathbb{R}^{n \times e} \newline
        K \in & \mathbb{R}^{n \times d},~W_1 \in \mathbb{R}^{e \times d} \newline
        Q \in & \mathbb{R}^{n \times d},~W_2 \in \mathbb{R}^{e \times d} \newline
        V \in & \mathbb{R}^{n \times d},~W_3 \in \mathbb{R}^{e \times d}
    \end{aligned}    
\]

위와 같이 나타낼 수 있다. Attention value를 구하는 여러 방법들 중 transformer에서 사용된 scaled dot product 과정을 소개하면 다음과 같다.   
   

가장 먼저, 서로 다른 input embedding에 대한 score를 계산한다. $S = Q \cdot K^\top$. $Q$의 각 row vector는 각각의 token embedding에 대한 값이고, $K^\top$의 각 column vector는 각각의 token embedding에 대한 key가 된다. 이를 내적하게 되면 $Q,~K$의 row vectors $r^q_i,~r^k_i \in \mathbb{R}^d$ $(i = 1,~2,~\cdots,~n)$ 에 대해 다음과 같이 표현할 수 있다.

\[
    QK^\top = \begin{bmatrix}
        r^q_1 \newline
        r^q_2 \newline
        \vdots \newline
        r^q_n
    \end{bmatrix}
    \cdot \begin{bmatrix}
        {r^k_1}^\top & {r^k_2}^\top & \cdots & {r^k_n}^\top
    \end{bmatrix}
\]
이렇게 계산된 값은 각 row vector끼리의 내적으로 구성되며, 이는 row vector의 dimension $d$ 값이 커질수록 커지는 구조가 된다.

\[
    S = QK^\top = \begin{bmatrix}
        {r^k_1}^\top r^q_1 & {r^k_2}^\top r^q_1 & \cdots & {r^k_n}^\top r^q_1 \newline
        \vdots & \vdots & \ddots & \vdots \newline
        {r^k_1}^\top r^q_n & {r^k_2}^\top r^q_n & \cdots & {r^k_n}^\top r^q_n
    \end{bmatrix}  
\]

따라서 안정적인 학습을 위해(gradient를 맞춰주기 위해) 위의 score를 dimension의 square root value로 나눠준다.

\[
    S_n = \frac{QK^\top}{\sqrt{d}} = \begin{bmatrix}
        ({r^k_1}^\top r^q_1)/\sqrt{d} & ({r^k_2}^\top r^q_1)/\sqrt{d} & \cdots & ({r^k_n}^\top r^q_1)/\sqrt{d} \newline
        \vdots & \vdots & \ddots & \vdots \newline
        ({r^k_1}^\top r^q_n)/\sqrt{d} & ({r^k_2}^\top r^q_n)/\sqrt{d} & \cdots & ({r^k_n}^\top r^q_n)/\sqrt{d}
    \end{bmatrix}  
\]

구한 score를 확률값으로 바꿔주기 위해 softmax를 취한다. Softmax function은 exponential 함수를 통해 $0 \sim 1$ 사이의 값으로 normalization 해주고, 가장 중요한 점은 특정 dimension으로의 합이 $1$이 될 수 있도록 해준다.

\[
    \begin{aligned}
        \text{Let }z_{ij} =& ({r^k\_j}^\top r^q_i)/\sqrt{d}, \newline
        softmax(z_{ij}) =& \frac{e^{z_{ij}}}{\sum\_{i=1}^n e^{z_{ij}}} \newline
        P =& softmax(S_n)
    \end{aligned}    
\]

이제 여기에 마지막으로 key에 대응되는 value를 곱해주면, 우리가 얻고자 하는 weighted value matrix를 얻을 수 있다.

\[
    Z = V\cdot softmax(\frac{Q \cdot K^\top}{\sqrt{d}})    
\]

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209152427-711293fb-c92f-4efc-b50c-8208073833cb.png" width="200"/>
</p>

## Masked attention in decoder
Attention 연산은 모두 위에서 설명한 것과 거의 동일하다. 다만 decoder에서 encoder와의 attention을 할 때는 encoder에서의 결과를 토대로 key, value를 상정하고 decoder의 output을 query로 삼는다. 그러나 학습 시에 <U>RNN과는 다르게</U> input으로 tokenized sentence 전체를 넣어주다 보니 causality 문제가 생긴다. 이 문제는 다음과 같다.   
Decoder에서 input 문장에 대해 번역된 결과를 내보낼 때는 recurrent 구조를 가진다. 이는 **RNN based sequence to sequence model**과 **transformer** 모두 동일하다. 그렇기 때문에 학습 과정에서는 '번역 결과'를 알고 있어도 이를 사용하면 안된다.
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209152430-6f3ce74c-b547-4c7f-89b8-6fac89547819.png" width="800"/>
</p>
이 그림을 보도록 하자. 그림에서의 task는 'I want to buy a car'이라는 문장을 다른 언어로 번역하는 과정을 나타낸 것이다. Decoder에서 가장 먼저 <U>BOS</U>(Begin of Sequence를 의미하는 토큰)와 encoder에서의 output을 기반으로 첫 단어('<U>Ich</U>')를 예측한다. 그 다음 단어는 예측된 '<U>Ich</U>'와 encoder에서의 output을 기반으로 두번째 단어('<U>will</U>')을 예측한다. 그 다음 단어는 예측된 '<U>Ich, will</U>'과 encoder에서의 output을 기반으로 세번째 단어('<U>ein</U>')을 예측한다. 이런 식으로 진행된다.   
논문에 나와있지는 않지만 여기서 중요한 실제 학습될 시에는 모델이 예측한 단어가 아닌 실제 ground truth를 사용한다는 것이고, 이를 자연어 연구에서는 **teacher forcing**이라고 부른다. 예컨데 번역 결과를 이미 알고 있다면 $n$번째 단어를 예측할 때 굳이 $n-1$번째까지 예측한 단어를 쓸 필요가 없고 이미 알고있는 $n-1$개의 GT를 사용할 수 있고, 이러한 방식이 encoder 단에서의 attention 학습을 보다 안정적으로 만들어줄 수 있다. 초반에 모델이 수렴되지 않은 상태에서는 decoder가 생뚱맞은 단어를 내뱉게될 것이고, 이를 그대로 decoder 예측의 attention query로 사용하게 되면 학습이 불안정해지는 문제가 발생하기 때문이다.

물론 현재 예측될 단어가 다음에 예측될 단어를 참고하지 못한다는 사실은 'inference'(테스트)에서는 항상 성립한다. 그러나 학습 시에는 이미 번역된 결과를 모두 알고 있고, 이를 supervision으로 삼아서 네트워크를 학습하기 때문에 <U>학습 과정에서의 decoder는 뒷 단어들을 참고할 수 없게</U> 해야한다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209152431-a618a50d-b838-47fb-8e90-8333dd5075df.svg" width="800"/>
</p>

연산 과정을 간단하게 표현하면 위와 같다. 맨 윗줄부터가 decoder의 input token에 대한 attention을 나타내고, mask에서 색칠된 부분이 value가 $1$, 그렇지 않은 부분이 value가 $0$으로 $n$번째의 embedding은 $n$보다 작거나 같은 attention weight만 참고할 수 있다.

## Multi-head attention
머신러닝 기법 중 '앙상블'이란 개념이나, 딥러닝 convolutional neural network에서 사용하는 kernel의 channel 수는 모두 비슷한 기능을 가진다. 그것은 바로 같은 feature map에 대해 <U>여러 가지 representation을 학습할 수 있다</U>는 것이다. CNN에서의 구조는 이미 이를 보장할 수 있는 network **width**라는 특성을 가지지만, transformer의 경우에는 attention layer가 여러 채널을 가질 수 없다는 문제가 생긴다. 이를 해결하기 위해 등장한 개념이 바로 <U>Multi-head attention</U>이다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209152436-7bedef8c-8afe-4456-90a0-956fde4836a9.png" width="300"/>
</p>

개념은 상당히 간단하게도, attention을 할 수 있는 linear layer를 head의 개수 $h$만큼 늘려서 사용하겠다는 것이다. Attention weight를 연산할 수 있는 head를 여러 개 사용하여 계산한 뒤, 결과값을 head의 축으로 concatenate한 뒤 Linear 연산을 통해 원래의 dimension으로 맞춰준다. 예컨데 위에서 사용한 notation 그대로를 사용해보면, head index $h$에 대한 query, key, value linear operator $W_h^Q,~W_h^K,~W_h^V$와 multi-head linear operator $W_o \in \mathbb{R}^{hd \times d}$에 대해서,

\[
    \begin{aligned}
        Q_h =& W_h^QX,~K_h = W_h^KX,~V_h = W_h^VX, \newline
        Z_h =& V_h\cdot softmax(\frac{Q_h \cdot K_h^\top}{\sqrt{d}}) \in \mathbb{R}^{n \times d}, \newline
        Z_{concat} =& \text{Concat}(Z_1;~Z_2;~\cdots;~Z_h) \in \mathbb{R}^{n \times hd}, \newline
        Z_{output} =& W_o \cdot Z_{concat} \in \mathbb{R}^{n \times d}
    \end{aligned}
\]

다음과 같이 계산되는 구조다.

## Positional encoding
RNN 구조에서 없었던 내용 중에 causality는 여전히 존재한다. 그런데 그것보다 더 문제인 것은 각 임베딩 사이의 거리도 어느 정도 context를 판단할 때 고려가 되는 사항인데, 각 token을 sequential하게 연산하는 RNN은 이를 걱정하지 않아도 되지만 transformer의 경우에는 해당 constraint를 줄 수 있는 방법이 필요하다. 예를 들면 다음과 같은 문장이 있다고 생각해보자,

\[
    \text{I have a dog named Adam and my friend has a cat named Epoch}    
\]
한글로 번역하면, 나한테는 <U>'아담'이라는 이름을 가진 강아지</U>가 있으며, 내 친구는 <U>'에폭'이라는 이름을 가진 고양이</U>가 있다는 뜻이다. 여기서 중요한 점은 강아지를 가지고 있는 주체와 고양이를 가지고 있는 주체에 대한 문장에서의 거리다. 만약 특정 이름과 동물이 연관된다는 특성에 대해서 token의 위치와 무관하게 학습이 진행되다보면, 다음과 같은 참사가 발생한다.   
   
'I have **a dog** named Adam and my friend has a cat named **Epoch**'   
'I have a dog named **Adam** and my friend has **a cat** named Epoch'   
   
여기서 <U>굵게 표시된 부분끼리 만약 attention value가 높게 책정이 되면</U>, 내가 키우는 강아지와 친구가 키우는 고양이의 이름이 뒤바뀌는 일이 생기는 것이다. 혹은 내 친구가 강아지를 키우고 내가 고양이를 키우는 식으로 번역이 될 수도 있다. 따라서 CNN에서 object가 서로 비슷한 픽셀 위치에 위치한다거나, translation에 불변한다는 특징과 같이 학습하는 과정에서의 inductive bias를 어느 정도 주기 위해서 token의 위치에 따른 임베딩을 추가해주는 방법을 고안하였다.
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209152440-74866c8f-538e-46de-86a9-6e3de9f32142.png" width="400"/>
    <img src="https://user-images.githubusercontent.com/79881119/209152441-05269b19-26ac-48e1-8312-1979c03876d9.png" width="400"/>
</p>
Positional encoding으로 사용된 함수는 흔히 알고 있는 sinusoidal 함수를 사용한다. 

\[  
    \begin{aligned}
        PE_{(pos,~2i)} =& \sin (pos/10000^{2i/d_{model}}) \newline
        PE_{(pos,~2i+1)} =& \cos (pos/10000^{2i/d_{model}})
    \end{aligned}
\]

여기서 $i$는 embedding dimension을 따라가는 축이고, $pos$가 각 token의 위치를 나타내는 position이다. 이를 실제로 시각화한 그림이 좌측에 있는 얼룩덜룩한 형태가 된다. 식을 보면 알 수 있듯이 $d_{model}$은 앞서 언급했던 embedding의 차원 수 $d$와 같으며, $i$가 <U>증가할수록</U> sinusoidal 함수의 <U>주파수가 점차 감소</U>하는 모양새가 된다. 주기함수를 보면 알 수 있듯이 y축을 따라가면서 함숫값을 보게 되면 **같은 함숫값을 가지는 index**가 reference 위치와는 관계없이 동일한 것을 알 수 있다. 즉 <U>짙은 파란색이랑 또다른 짙은 파란색 사이의 y축 거리</U>(pos 차이)나, <U>짙은 빨강이랑 또다른 짙은 빨강 사이의 y축 거리(pos 차이)</U>가 일정하게 나타난다는 것이다. 그리고 또한 위의 함수는 각 embedding position마다 서로 다른 encoding을 더해줄 수 있다. 함숫값은 주기 함수에 의해 반복이 되지만, 그 어떤 encoded embedding도(x축에 평행한 value들을 의미) 서로 <U>일치하지 않고 unique</U>하다. 즉, 문장이 얼마나 길어지든 상관없이 각 token에 유일한 값으로 매핑할 수 있기 때문에, <U>문장의 길이와 무관하게 모든 token을 구분할 수 있다</U>.
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209152442-d12a5bfc-aaf1-442e-9733-ea9911e1b1b1.png" width="400"/>
</p>
이처럼 구해준 각 token 별 encoding을 dimension에 맞게 생성해서 모든 embedding에 더해주면 된다.

## Transformer 학습

학습은 간단하게 supervision이 있기 때문에 cross-entropy loss에 대해 최적화가 가능하다. Output으로 나오는 softmax prediction에 대해 가장 최댓값을 가지는 index의 단어를 매핑하고, 제대로 된 단어의 확률값이 1에 가까워질 수 있게 학습된다.

## 그래서 요즘은?
Transformer는 단연코 neural network 중 최근에 가장 활발히 연구된다고 해도 과언이 아닐 정도로 다양한 분야에 접목하기가 좋다. 2017년에 기계 번역 task에 대해서 처음으로 transformer가 제시된 이후, BERT, GPT-3 등등 NLP 관련 모델들이 많이 등장했으며, 특히나 vision 분야에서도 vision transformer가 등장하면서 <U>multimodal approach</U>가 가능해졌다. 모든 modality에 대해 일관적인 구조로 encoding이 가능하다면 이를 토대로 여러 modality를 함께 supervision으로 사용할 수 있다는 관점이다.
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209152438-03afe1d5-bd0f-4aa5-8ae5-1f7be17c522f.png" width="600"/>
</p>

---

# Vision transformer

앞서 기계 번역에서 <U>transformer</U>의 구조 및 학습 방법에 대해 알아보았다. 따로 여기서는 첨부하지는 않았지만 실제로 transformer를 사용했을 때의 성능은 굉장히 좋았고, 이후 다양한 tansformer based approach를 통해 많은 <U>NLP 관련 기술들</U>이 발전할 수 있었다. 이에 비전에서도 NLP에서 사용된 transformer 구조를 사용할 수 없을지에 대한 연구가 'Attention is all you need' 논문 이후로 활발하게 진행되기 시작했다.   
하지만 명확하게 <U>tokenize되기 쉬운</U> natural language나 audio(단어나 음절 단위로 끊는 경우)와는 다르게 image의 경우에는 보다 연속적인 신호로 구성되어 있어서 정확히 어떠한 기준으로 tokenize를 해야할 지 애매하고, 이를 임베딩하는 방법 또한 일반적인 word2vec과 같은 방법을 사용할 수 없다는 점이 문제가 되었다.
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209152451-54daf42a-40fa-49fc-91cf-e6aaf2570e52.png" width="400"/>
</p>
이러한 과정에서 처음으로 vision transformer의 성능을 보여준 논문이 [ViT: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)라는 논문이었고, 해당 논문을 간단하게 살펴보면 다음과 같다. 논문에서 접근한 방식은 상당히 간단하다.
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209162457-43a1b681-e2d6-4e20-9e85-b8e964efebf5.gif" width="600"/>
</p>
이미지를 총 $n$개의 patch로 나눈다. 논문 제목에서 확인할 수 있겠지만, 이미지 크기가 $256 \times 256$ 이라면 이를 균등하게 $16 \times 16$의 patch로 분리한다. 위의 그림에서는 이를 좀 더 간단하게 표현하기 위해 이미지의 $H,~W$를 각각 3등분하여 9개의 샘플을 만들어내는 것을 볼 수 있다. 분리한 patch를 각각 flatten해서 1차원의 텐서로 만든 뒤 linear projection을 통해 embedding space로 보내고, 이 patch sequence 앞쪽에 <U>cls token</U>(이미지의 클래스를 구분할 때, global inference를 하기 위해 기준점이 되는 부분이라고 기억해두자)을 추가해준다. 앞서 transformer에서 했던 것과 같이 마찬가지로 여기서도 positional embedding을 더해주는데, 이미지의 경우 natural language embedding에서 사용했던 것과 같은 sinusoidal embedding의 효과가 나타나지 않기 때문에 여기선 그대신 <U>learnable positional embedding</U>을 사용하게 된다. 모든 attention 연산이 끝난 뒤 encoder output에서 <U>cls token</U>을 받아오고, 여러 번의 attention 연산이 진행되면서 이 cls_token에는 16개의 patch embedding에 대한 <U>global reference</U>가 완료된 상황이다. 이 cls token은 $B \times D$ 크기를 가지며, 이를 바로 class 예측에 사용하기보다는 attention 연산이 진행되면서 '<U>다른 패치들의 정보를 요약</U>'한다는 느낌으로 접근한다. 따라서 class embedding을 추출한 다음에는 다음과 같은 연산이 진행된다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209152455-2fb10959-e2cf-48cb-89f6-7237f5a803b7.png" width="600"/>
</p>

16개의 patch가 있었기 때문에 transformer model의 $Q,~K,~V$ dimension $d$에 대해서 output은 cls token과 함께 $(16+1) \times d$ 크기의 tensor로 추출된다. 여기에 MLP head로 class 개수만큼의 output을 내보내는 linear 연산($d \rightarrow C$ where $C$ is the number of classes)을 진행한 뒤, layer normalization 결과를 <U>class 예측에 대한 score map으로 사용</U>한다. 즉, 이를 logit으로 삼아 softmax 연산을 하게 되면 비로소 cls token의 output에 대한 class 예측이 진행되는 것이다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209152460-d5a17d63-28b3-4860-a9fa-c56412abd7af.png" width="800"/>
</p>

그러나 vision transformer의 경우 데이터셋이 충분하지 않으면 BiT(ResNet 기반 transformer)와 같은 CNN 기반 모델에 비해 성능을 기대하기 힘들었는데, 이 이유는 바로 앞서 설명했던 inductive bias의 부재와 관련된다. CNN based network는 적은 데이터셋으로도 기본적으로 가지고 있는 inductive bias를 통해 빠른 일반화 및 최적화가 가능하지만, 패치 단위로 나누어 각각에 대한 attention을 연산하는 구조인 <U>vision transformer는 이런 의존성이 전혀 없기 때문</U>이다.   
그렇기 때문에 가지고 있는 단점이자 장점은, CNN의 경우 inductive bias 때문에 정해진 연산을 통해 추출할 수 있는 representation map 혹은 feature map이 어느 정도 학습이 진행되고 나면 수렴이 발생하지만, vision transformer는 데이터셋을 대량으로 학습시키면 학습시킬수록 attention 성능을 높일 수 있고, 이는 특정 구조에 따라 수렴하는 형태가 아니라 끊임없이 발전할 수 있다고 해석하였다.   
그렇기 때문에 Vision transformer를 <U>대량의 데이터셋</U> JFT dataset($3 \times 10^8$ samples)로 사전 학습한 뒤, 원하는 **downstream task**에 맞게 fine-tuning을 하면 더 좋은 성능을 보일 수 있다고 한다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209152462-5da11123-b4d4-46f5-b56e-d1c4106d0732.png" width="800"/>
</p>

그리고 <U>convolution</U> 연산에 비해 <U>attention</U> 연산이 가지는 parameter의 개수나 네트워크의 전반적인 규모가 CNN보다 <U>훨씬 크다</U>는 점이 문제가 될 수 있다. VGG-16의 경우 경량화가 많이 부족한 모델임에도 불구하고 175.12M의 parameter 개수를 가지기 때문에 ViT-Large 혹은 ViT-Huge에 비해 훨씬 가벼운 것을 확인할 수 있다. 심지어 ResNet의 경우에 가장 레이어가 깊은 네트워크 중 하나인 <U>ResNet-152</U>의 경우에도 paramter 개수가 <U>60.34M</U>에 그치는 것을 보면, 확실히 transformer 네트워크의 규모가 지나치게 크게 구성된 것을 볼 수 있다. 

---

# 왜 Vision transformer가 CNN을 이길 수 있었을까?
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209152466-28ea9cdf-ab8d-400c-8c10-d6e17abde10c.png" width="800"/>
</p>
가장 간단하게 풀자면 단순히 self-attention 이라는 연산의 mechanism이, 초기의 layer부터 입력된 input 전체에 대한 inference가 가능하기 때문이다. 예를 들어 CNN의 경우에는, $3 \times 3$의 커널 크기를 통해 $32 \times 32$ 크기의 receptive field를 가지기 위해서는

\[
    3+2 \times (n-1) \ge 32,~n > 14    
\]

14보다 많은 개수의 layer가 필요하다. 이 과정에서의 연산량을 줄이기 위해 추가적으로 max-pooling과 같은 filtering이 들어가면, 이미지의 특정 부분에서는 <U>global한 fine detail을 알아채기도 전</U>에 **image classification**을 해야 하는 문제가 발생하는 것이다. 이를 다르게 말하면 ViT는 모든 레이어에서 균일한 형태의 representation map을 획득할 수 있고, self-attention 연산을 통해 보다 빠르게 global information을 축약해낼 수 있다. 이렇게 빠르게 요약한 global 정보를 여러 레이어를 통해 각 패치마다 공유하면서, 잘 학습된 representation을 전달할 수 있게 되는 것이다. 그러나 어김없이 <U>dataset의 개수가 대량으로 필요하다는 조건</U>(약 10억 개의 샘플)과 그만큼 <U>parameter 수도 많이 필요</U>하다는 것이 제약 조건이 될 수 있다. 추가적으로는 high-resolution image의 경우 attention 연산이 $Q,~K,~V$에 대한 attention score를 연산하는 과정이 되기 때문에 multi-head self attention의 computation이 $(H \times W)^2$에 비례해서 증가하는 가장 큰 문제가 발생한다. 이제부터 그 각각의 문제점을 해결하기 위한 연구들을 소개해보겠다.

---

# Dataset을 효율적으로 사용할 수 있는 방법

가장 먼저, 데이터셋을 많이 사용하지 않고 학습시킬 수 있는 방향에 대해서 연구한 논문인 [DeiT: Training data-efficient image transformers & distillation through attention](https://arxiv.org/pdf/2012.12877.pdf)에 대해서 살펴보도록 하자.
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209152469-4fde8fda-f626-4d1c-999a-6d4d75435c68.png" width="400"/>
</p>
DeiT는 ViT와 동일한 transformer 구조에 대한 학습을 진행하되, JFT와 같은 대용량 데이터셋에 대한 pre-training을 하지 않고도 ImageNet dataset에 대해서만 학습하는 방향을 제시했다. 이전에 작성했던 글들 중에서 Knowledge distillation에 대해 간단하게 소개했던 글이 있는데([참고](https://junia3.github.io/blog/transfer)), 바로 distillation을 통해서 transformer가 지지부진할 때 도움을 많이 줄 수 있는 <U>teacher</U> 역할을 <U>convolutional neural network</U>가 해줄 수 있다는 것이다. Convolutional neural network 구조에 의존하지 않더라도, 단순히 distillation loss를 줄 수 있는 token 하나만 추가하면 된다.
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209152470-4bcf4a9a-85f6-4145-a535-20449cd5d23e.png" width="400"/>
</p>
그런데 여기서 두 가지의 선택권이 있다. 바로 teacher network의 prediction에 대해 soft-label distillation을 loss로 줄 것인지, 아니면 teacher network의 prediction의 최댓값을 one-hot encoding으로 한 hard-label distillation을 loss로 줄 것인지이다.

\[
    \mathcal{L}\_{\text{global}} = (1-\lambda) \mathcal{L}\_{\text{CE}}(\psi(Z\_s), y) + \lambda \tau^2 \text{KL}(\psi(Z_s/\tau),~\psi(Z_t/\tau))    
\]

이런 식으로 temperature value $\tau$와 그에 따라 $lambda$ 값을 적절히 조절하여 weighted soft-distillation loss를 더해주는 방법이 있는 한편,

\[
    \mathcal{L}\_{\text{global}}^\text{hard Distill} = \frac{1}{2} \mathcal{L}\_{\text{CE}}(\psi(Z\_s), y) + \frac{1}{2} \mathcal{L}\_{\text{CE}}(\psi(Z_s),~y_t)   
\]

위와 같이 hard target에 대해서 예측을 하는 hard distillation이 있다. 여기서 표현된 $y_t$는 $\arg \max_c Z_t(c)$로 표현한다. 즉 teacher prediction의 prediction 최댓값을 1로, 나머지를 0으로 하는 one-hot encoding 형태로 주어진다. DeiT 논문에서는 아래 방법을 소개하면서 해당 loss term을 최적화하였다. 그러나 이 네트워크의 경우 한계점이 상당히 명확하게 존재하는데, 바로 <U>CNN의 성능에 대해 upper bound가 생긴다</U>는 점이다. **ViT**가 제시되고 나서, <U>JFT와 같은 대량의 데이터셋</U>을 활용하여 학습할 수 있다면 <U>CNN보다 성능이 좋아질 수 있다는 것</U>이 transformer based vision approach의 분석이었는데, 데이터를 효율적으로 쓰려고 하면서 CNN을 teacher network로 사용하다보니 다시 원점으로 돌아오게된 것이다.

---

# High resolution image에 대한 효율적인 연산
다음으로 언급할 내용으로는 앞서 Vision transformer에서 해결하지 못했던 <U>high resolution image</U>에 대한 연산이다. Vision transformer 연산 과정을 보게 되면, image를 동일한 크기의 patch로 분리하고, 이를 embedding으로 linear projection한 뒤에 일련의 attention 연산을 진행하게 된다. 그러므로 만약 image의 resolution이 2배가 되면, linear projection은 그 값의 제곱인 4배로 늘어나게 되고, attention 연산에 필수적인 $Q,~K,~V$를 통한 attention weight 연산은 그 값의 다시 제곱인 16배로 증가하는 것이다.
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209162456-586c1b3e-7855-426e-a332-73e7542e2863.gif" width="400"/>
</p>
그리고 무엇보다 연산 때문에 token의 개수가 한정적이라면, CNN에서 했던 것과 같이 다양한 scale의 feature를 뽑기가 힘들 것이다. Global information을 빠른 시간에 축적하는 건 장점이 될 수 있는데, 결국 그렇게 축적된 데이터는 하나의 scale에만 국한된다는 점. 즉 vision transformer를 더 발전시켜서 segmentation이나 image restoration과 같은 <U>high-level vision task</U>에도 적용할 수 있어야하는데, 지금의 구조로는 가능성이 보이지 않는다. 바로 여기서 나온 것이 지금 설명할 [Swin-Transformer](https://arxiv.org/pdf/2103.14030.pdf)를 통한 hierarchical feature extraction이다.
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209152473-0853ced7-3895-4745-93cd-810b097bf40f.png" width="700"/>
</p>
만약 high-resolution인 이미지에 대해서 <U>정해진 개수</U>의 patch를 뽑는 것이 아니라 <U>단계적으로 patch에 대한 연산</U> 이후 이를 merge하는 과정을 추가하면, '초반에 fine detail을 잡는 과정과 후반에 coarse feature를 잡아내는 CNN과 유사하게 동작하지 않을까?'에 대한 내용을 다룬다.
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209152478-69203891-82b7-4c8b-86e9-8ca45c1610a7.png" width="800"/>
</p>
네트워크 구조는 위와 같다. ViT에서 크게 달라진 점을 찾아보면 patch partition 부분에 대해 가장 처음에는 $\frac{HW}{4^2}$ 만큼의 patch를(여기서는 갯수를 의미한다), 그리고 점차 patch 크기를 키워서 마지막에는 $\frac{HW}{32^2}$ 크기의 patch를 사용한다. 크기를 키우는 과정은 인접한 패치끼리 붙이는 과정이다.

1. 가장 처음 패치는 patch partition을 통해 $4 \times 4 \times 3$의 크기를 가진다. 참고로 위의 그림에서 보이는 값에 대해서 헷갈릴 수도 있기 때문에 언급하자면, $\frac{H}{4} \times \frac{W}{4}$는 patch 하나의 크기가 되고, 각 패치 내부에는 총 $4 \times 4$ 단위로 RGB 값을 가진 픽셀들이 들어있다고 생각하면 된다.

2. Stage 1을 통해 Linear embedding으로 patch의 채널 수를 $4\times4\times3$에서 $4\times4\times C/16$배로 증가시킨다. 각 패치에 대한 임베딩을 뽑는 stage라고 생각하면 된다. 실제 논문에서는 C = 192를 사용했으니, 원래는 RGB 채널 3개로 시작했지만 $192/16 = 12$로 4배 증가시킨 것과 같다.

3. Stage 2 부터는 우리가 알고있던 attention이 진행된다. 다만, 여기서 patch-merging이 일어난다. Patch merging이 일어나게 되면 인접한 4개의 patch가 합쳐지면서 큰 patch가 되고, 그와 동시에 연산량이 줄기 때문에 그만큼을 channel로 증가시켜준다. 즉, $\frac{H}{4} \times \frac{W}{4}$ 크기를 가지고 있던 패치 4개를 붙이면 각 패치의 크기는 $\frac{H}{8} \times \frac{W}{8}$이 되고, 패치 내부의 구획 개수는 일정하기 때문에(이 부분이 제일 중요하다!) **merge 레이어를 지날수록 Attention 연산량이 줄어드는 것**이다. 왜냐? 패치 내부에 attention을 하게 될 구획 개수는 일정한데, 전체 패치 개수는 merge하면서 점차 줄어들기 때문. 그렇기 때문에 channel 수를 2배 증가시켜준다.

4. 참고로 Attention은 각 윈도우 내부에서만 진행된다. 여기서 말하는 윈도우란 곧 패치를 의미하며, 사실상 윈도우라는 개념과 패치라는 개념을 구분해서 써야하지만 이 논문에서는 혼용해서 쓴 것으로 보인다. 아마도 대부분의 리뷰어들이 여기서 혼란을 겪었을 듯 싶다. 각 윈도우 내에서는 구획 개수가 일정하기 때문에(3번에서 언급한 '패치 내부의 구획 개수는 일정하기 때문에'라는 문장과 같은 의미이다! 혼란스럽지 않기 위해 한번 더 언급)

5. 이러한 과정을 attention block을 거치면서 진행한다. 여기에 추가적으로 W-MSA와 SW-MSA라는 개념은 뒤에서 설명하도록 하겠다.

암튼 이렇게 저렇게 패치 사이즈를 조절해가면서 attention을 한다는 과정을 길게 설명했다. 사실 이 부분은 본인이 직접 깨닫기 전까지는 워딩만 보고서는 절대 이해할 수 없다. 왜냐하면 '<U>patch</U>'라는 워딩과 '<U>window</U>'라는 워딩이 paper에서 너무 겹치기 때문이다. 진짜 엄밀하게 따지면 완전 다르진 않고 어느 정도는 겹치는 개념이긴 하지만, shifted-window라는 method를 위해서 이런 식으로 독자(?)들을 혼란스럽게 하다니 조금은 괘씸하다. 아무튼 이제 W-MSA와 SW-MSA에 대해서 보면,   

W-MSA는 윈도우 내부의 패치들에 대한 attention이다. 하나의 윈도우 크기에 아마 논문에서 구현하기로는 $7 \times 7$의 embedding이 포함될 것이고, 이렇게 하나의 윈도우 내에 포함된 애들끼리의 연산을 의미한다. 그러나 이렇게만 연산을 하게 되면 <U>서로 다른 윈도우에 속한 임베딩은 다른 임베딩의 정보를 얻을 수 없는 경우</U>가 생긴다. 다음과 같은 그림으로 예시를 그려보았다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209152481-5933823c-3a1e-494b-9059-f66622435286.png" width="800"/>
</p>
분명 파란색으로 표시된 패치와 붉은색으로 표시된 패치는 위치상 서로 붙어있지만, 가장 마지막 attention layer 전까지는 <U>서로 attention 연산이 불가능</U>하다. 이러한 분단국가 문제는 locality가 중요하게 적용되는 image에 대해서 치명적으로 적용할 수 있을 뿐만 아니라, ViT의 장점 중 하나인 global information을 빠르게 취득하는 것이 불가능해진다. 따라서 다음과 같은 전략을 사용한다.
윈도우 내의 attention을 하는 것은 위의 그림에서 그려진 대로 계산한다. 그러나 shifted window attention은 다음과 같이 window를 이동시키는 전략을 사용한다. 이해가 쉽도록 이미지 전체를 하나의 window로 생각하고, 그 내부를 $4 \times 4$ 만큼의 패치로 구분해보았다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209152489-66f1f113-14bc-4870-a7b2-268ea4441a18.png" width="800"/>
</p>
다음과 같이 윈도우를 일부 이동시키면, 움직인 만큼 원래의 윈도우에 포함되던 부분이 빠져나오게 된다. 이를 채워주는 방법으로는 간단하게 '<U>잘라서 붙이기</U>' 방법을 사용한다. 좀 더 명확하게 말하자면, cyclic shift와 같다. 화학에서 분자 구조(체심 입방 구조 뭐 이런거..)에서의 개수를 구하는 과정에서 물질 전체가 반복되는 구조라고 가정하면 쉽게 풀 수 있었던 방법을 기억하면 된다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209152493-93751ee4-db63-4ef8-8009-1b555bd27a11.png" width="800"/>
</p>

이렇게 window를 옮긴 뒤에 attention을 진행하게 되었을 경우에는 다음과 같이 <U>다른 window의 패치와도</U> attention이 연산될 수 있음을 보여준다.
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209152498-b0ec4bc1-d4c5-4974-acc9-ee8c482c4cf6.png" width="800"/>
</p>

그리고 단순히 이전에 사용되던 scaled dot product self-attention 식이 아닌, 이번에는 relative position을 반영한 bias를 score에 더해주게 된다.

\[
    \text{Attention}(Q,~k,~V) = softmax(QK^\top / \sqrt{d}+B)V    
\]

이는 이전의 positional embedding이 절대적인 좌표(sinusoidal embedding)을 더해주었던 접근과는 다르게 각 패치의 위치를 반영하는 attention embedding 방법이다. 이를 간단하게 설명하자면 만약 현재 query로 삼고있는 패치가 좌측 상단이고, attention을 구할 패치가 우측 하단이라면 $x, y$ 모두 증가하는 방향으로 bias를 더해주고, 만약 그 반대 방향이라면 $x, y$ 모두 감소하는 방향으로 bias를 더해주는 것이다. 

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209152501-db6b06b0-11df-4d00-a5ad-61d3b9966b4e.png" width="400"/>
    <img src="https://user-images.githubusercontent.com/79881119/209152505-289cca9a-ae69-414b-872b-cd81661513d8.png" width="400"/>
</p>

위의 그림을 예시로 삼아보겠다, 만약 Attention을 진행할 하나의 윈도우 내에 총 9개의 patch가 있다면(1~9로 숫자가 붙은 영역들), 이 patch에 대해서 <U>row index의 차이</U>를 나타내는 $x$-axis matrix, 그리고 <U>column index의 차이</U>를 나타내는 $y$-axis matrix를 표현할 수 있다. 

그런 뒤, 여기에 윈도우의 크기를 반영하는 다음과 같은 연산을 거치게 된다.

```
# 각 matrix에 (window 크기-1) 만큼 더해주기
y_axis_matrix += window_size - 1
x_axis_matrix += window_size - 1

# 2M-1로 scaling한 뒤 더해주기
x_axis_matrix *= (2 * window_size -1)
relative_position_M = x_axis_matrix + y_axis_matrix
```
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209152507-ff3cc005-39bb-4e7a-a54c-dee94c41934d.png" width="400"/>
</p>

이렇게 scaling을 해주게 되면 원래 $-M-1 \sim M-1$ 범위를 가지고 있던 relative matrix들이 $0 \sim 2M-1$로 scaling된 뒤, 서로 곱해져서 $0 \sim (2M-1)^2$ 까지의 범위로 표현될 수 있는 것이다.

---

# CNN과 ViT을 어떻게 하면 잘 조합할 수 있을까?
앞서 살펴본 Shifted window 방법을 사용한 <U>Swin-T</U>의 경우, 훨씬 효율적인 연산이 가능하다는 점과 hierarchical 구조로 인해 CNN이 가지던 다양한 scale에 대한 flexibility를 가질 수 있으며, 그러면서도 연산 속도는 image size에 대해 quadratic하지 않고 <U>Linear</U>하게 유지할 수 있다는 점이 큰 장점이 되었다. 또한 이러한 scale flexibility로 인해 classification 이외에도 다양한 vision task, 예를 들어 detection이나 segmentation 등에도 일반적으로 적용될 수 있는 구조를 제시했다는 점이 되겠다.   
그러나 이러한 Swin-T의 단점은 결국 작은 데이터셋으로 scratch부터 학습하기 힘들다는 점이고, 이런 문제는 ViT에서 제시된 이후로 여전히 해결되지 못했다. 앞서 소개했던 Data effficient transformer의 경우 CNN을 teacher model로 사용하여 knowledge distillation을 진행했지만, 결국 CNN의 성능에 따라 좌우되는 문제가 발생했었다. 그렇다면 과연 CNN이랑 ViT의 장점을 함께 활용하여, inductive bias를 통한 representation의 효율적 학습(<U>data efficiency</U>)과 <U>global information</U> 활용을 함께 할 수 있는 방법은 없을까?   
바로 이러한 질문으로부터 등장했던 연구가 [Convolutional vision Transformer(CvT)](https://arxiv.org/pdf/2103.15808.pdf)이다. CNN의 여러 장점들을 ViT 구조에 결합하고자 했던 것이다. 예를 들어 CNN은 object에 대해 shift, scale 그리고 어느 정도 용인 가능한 distortion이 일어나더라도 이를 <U>invariance하게 처리</U>할 수 있다. 그에 반하여 ViT는 dynamic attention, global information 그리고 보다 나은 generalization이 가능하다.   
CvT에서 바꾼 두 가지의 주된 구조적 형태는 바로 다음과 같다.

1. Transformer의 계층적 구조에 대해, convolutional token embedding이라는 새로운 임베딩 방식을 소개한다.
2. Convolutional transformer block은 연산될 convolutional projection을 생성해낸다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209152513-48020390-58f3-4788-8e29-1a6ecab74d94.png" width="600"/>
</p>

결론부터 보자면, 구조를 그대로 사용하면서 더 적은 데이터셋을 활용했던 DeiT는 parameter 수를 효과적으로 줄이거나 성능을 CNN teacher network 이상으로 끌어올리지 못했으나, CvT는 parameter 수를 보다 효과적으로 활용하면서도 성능을 ViT 이상으로 높일 수 있음을 보여주었다. 네트워크의 전반적인 구조를 보면,

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209152516-f6734f30-56b8-45dd-8bd9-d22545e46aec.png" width="900"/>
</p>

결국 하고자 하는 것은 input image 혹은 중간의 feature map에 대해서 <U>convolution</U> 연산으로 <U>token embedding</U>을 진행하고, 이 token에 대한 <U>attention</U>을 수행한다. 그런 뒤 output으로 나오게 되는 <U>lower resolution</U>의 feature map에 대해 다시 convolution 연산을 진행, <U>token에 대한 attention을 수행</U>하는 과정이 반복된다.

$l-1$번째 layer에서의 output $x_{l-1}$에 대해 $l$번째 convolution 연산의 output으로 얻을 수 있는 new token map을 $f(x_{l-1})$라고 하자. 여기서의 $f$는 필터링 함수를 의미한다. 실제 official 코드를 참고해보면 convolutional embedding 부분이 다음과 같이 되어있는데,

```python
class ConvEmbed(nn.Module):
    """ Image to Conv Embedding
    """

    def __init__(self,
                 patch_size=7,
                 in_chans=3,
                 embed_dim=64,
                 stride=4,
                 padding=2,
                 norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding
        )
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.proj(x)

        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        if self.norm:
            x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        return x

```
default setting을 참고하게 되면 ```patch_size = 7```, ```stride = 4```, ```padding = 2```라고 되어있기 때문에 input image의 spatial resolution $H_i \times W_i$에 대해서 output embedding의 resolution $H_o \times W_o는$

\[  
    \begin{aligned}
        H_o = \left( \frac{H_i + 2 \cdot 2 - 7}{4} + 1 \right) \newline
        W_o = \left( \frac{W_i + 2 \cdot 2 - 7}{4} + 1 \right) 
    \end{aligned}
\]

위와 같다. 이렇게 차원을 축소하면서 embedding을 추출한 뒤, layernorm을 적용하고 난 결과를 추출한다. 이제 이렇게 추출된 <U>token</U>에 대해서 <U>attention 연산</U>을 진행하게 되는데, 기존 방식의 attention과는 다르게 Convolutional projection이 이전의 projection과 다른 점은 다음과 같다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209152520-24aa2a13-3d4d-4560-9a55-eafc4bb0f32d.png" width="900"/>
</p>

가장 왼쪽에 보이는 (a)는 ViT에서 token을 Query, Key 그리고 Value에 mapping할 때 사용했던 linear projection 방식이다. 단순히 weight $W^Q,~W^K,~W^V$를 곱해줌으로써 만들어낼 수 있다. 이와는 다르게 convolutional projection은(중간에 보이는 그림 (b)) token을 reshape 및 padding을 통해 convolution 연산이 가능한 window 구조를 만들어주고, 여기에 convolutional projection을 수행하여 query, key 그리고 value를 만들어낸다. 연산 과정에 대한 official 코드 중 일부를 가져와서 간단하게 설명하면 다음과 같다.

```python
def forward_conv(self, x, h, w):
    # Class 토큰을 따로 분리하는 과정/ Convolution 연산은 오직 이미지에 대한 토큰에만 적용하겠다는 의미
    if self.with_cls_token:
        cls_token, x = torch.split(x, [1, h*w], 1)

    # einops의 rearrange를 통해 HW * C로 들어온 input을 C * H * W로 펴주게 된다
    x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

    # conv_proj_ 는 모두 embedding dimension을 channel input 그리고 output으로 하는 convolution 연산
    # 참고로 convolution은 (kernel=3, stride=1, padding=1)로 사용함으로써 spatial dimension을 그대로 유지한다
    if self.conv_proj_q is not None:
        q = self.conv_proj_q(x)
    else:
        q = rearrange(x, 'b c h w -> b (h w) c')

    if self.conv_proj_k is not None:
        k = self.conv_proj_k(x)
    else:
        k = rearrange(x, 'b c h w -> b (h w) c')

    if self.conv_proj_v is not None:
        v = self.conv_proj_v(x)
    else:
        v = rearrange(x, 'b c h w -> b (h w) c')

    # 모든 연산이 끝나면 다시 rearrange를 통해 원래와 같이 쭉 펴주게 된다.
    if self.with_cls_token:
        q = torch.cat((cls_token, q), dim=1)
        k = torch.cat((cls_token, k), dim=1)
        v = torch.cat((cls_token, v), dim=1)

    # 그리고 아까 떼어냈던 class 토큰을 다시 붙여주게 되면, attention 연산에 필요한 query, key, value를 얻을 수 있다
    return q, k, v
```
참고로 위에서 사용되는 convolutional block은 depthwise + pointwise convolution을 사용한다. 

```python
def _build_projection(self,
                    dim_in,
                    dim_out,
                    kernel_size,
                    padding,
                    stride,
                    method):
    if method == 'dw_bn':
        proj = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(
                dim_in,
                dim_in,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=False,
                groups=dim_in
            )),
            ('bn', nn.BatchNorm2d(dim_in)),
            ('rearrage', Rearrange('b c h w -> b (h w) c')),
        ]))
    elif method == 'avg':
        proj = nn.Sequential(OrderedDict([
            ('avg', nn.AvgPool2d(
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                ceil_mode=True
            )),
            ('rearrage', Rearrange('b c h w -> b (h w) c')),
        ]))
    elif method == 'linear':
        proj = None
    else:
        raise ValueError('Unknown method ({})'.format(method))

    return proj
```
내부 함수인 projection 생성 함수를 보게 되면 알 수 있다. 실제로 논문에서 설명하기로는

- Depth-wise convolution 2d
- BatchNorm2d
- Point-wise convolution 2d

라고 되어있지만 실제로 구현되어있는 걸 보니까 두번째 단계까지는 맞고, pointwise convolution 대신 <U>linear projection을 통해</U> 각 channel 간의 correlation을 준 것 같다.   
결론적으로 Transformer based model에 비해 CvT는 더 적은 parameter 수와 FLOPS를 가지고도 더 높은 accuracy를 획득하였다. Attention 연산 시에 MLP에 의존하지 않다보니, 같은 projection을 내보내는데 더 적은 수의 parameter를 요구하기 때문이다. 그러면서도 CNN based model과는 다르게 ViT based network의 성능과 같이 높은 수치를 보여주었다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209162407-d9b17c8b-faf9-4458-987b-ee98bb0abab4.png" width="800"/>
</p>

---

# Multimodal using transformer

지금까지 생각보다 많은 논문들을 리뷰했다. 가장 처음에는 <U>CNN, RNN</U>부터 시작해서 <U>Sequence to sequence</U>. 그리고 이어지는 <U>attention mechanism</U>과 이를 기반으로한 <u>attention only network(transformer)</U>의 발전. 그리고 이러한 NLP에서의 성공이 vision task로 이어질 수 있었던 <U>ViT</U>의 제안 방식과 더불어 여러 한계점을 극복하기 위한 방법들(<U>DeiT, Swin-T, CvT</U>)까지 모두 살펴보았다. 그렇다면 **multimodal**의 지평을 열 수 있게 도와준 transformer가 정확히 어떤 측면에서 다양한 연구에 활용될 수 있는지 짤막하게 소개하며 이번 글을 마무리해볼까 한다.   

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209162417-87272d02-5e7d-49fd-bec4-55df5e350680.png" width="300"/>
</p>

<U>Modality</U>란 내포할 수 있는 양상이 너무 막연하기에 다양한 설명이 될 수 있다. 하지만 지금 살펴보고자 하는 '딥러닝'의 측면에서는 다음과 같이 정의할 수 있다. '<U>Modality</U>'는 vision, audio 그리고 language와 같이 특정 sensor나 관측 방법을 통해 취득할 수 있는 개별적인 <U>communication channel</U>이다. 용어가 많이 생소하겠지만 예를 들어 카메라나 LiDAR(센서) 등등 어떠한 형식의 정보만 수집할 수 있다면 이를 modality라고 표현 가능하다. Thermal 센서를 통해 취득한 열화상 이미지도 또다른 modality고, CT나 MRI 기계를 통해 취득한 의학 영상 이미지 또한 또다른 modality 중 하나가 된다.   
따라서 <U>multimodality</U>, 혹은 <U>멀티 모달</U>이라고 불리는 딥러닝의 task는 vision, text, sound, data 등등 서로 다른 취득 방식으로 획득한 데이터셋을 유의미하게 함께 활용하여 representation learning을 하고자 하는 목적에 있다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209162421-7299f3f4-06d3-4ad1-9fbb-d16ff22339ba.png" width="700"/>
</p>

간혹 multimodal과 cross-modal 사이에 워딩이 겹치는 문제가 있는데, 각각의 차이점은 다음과 같다. 멀티모달은 딥러닝에서 새롭게 제시된 알고리즘으로, 여러 가지의 modality를 <U>함께 활용</U>하여 학습을 하는 것이다. 예를 들어, 사람은 시각과 청각을 모두 활용해서 사람이나 특정 물체를 판별하는데, 바로 이러한 능력을 computer에 대해서 적용하고자 하는 것이다. 그와는 다르게 cross-modal은 multimodal deep learning으로 접근을 하되, 하나의 <U>modality의 정보</U>가 <U>다른 modality</U>의 <U>성능을 높이는데</U> 사용되는 것이다. 만약 고양이의 이미지를 보았다면, 고양이 울음소리를 들음으로써 '아 이 사진은 고양이겠구나'라고 판단할 수 있는 것이다.   
AI system 중 다양한 modality에 대해 함께 작동할 수 있는 모델을 multimodal이라고 부르며, cross-modal은 서로 다른 task를 활용함으로써 중간에 있는 지식을 활용하는 것이다. **여러 스타트업**이 모여있는 <U>공동 사무실에서</U> 함께 일하는 과정에서 서로 시너지 효과를 내서 모두의 사업이 성공하는 것이 **cross-modal**의 예시가 될 수 있고, 하나의 스타트업 내 <U>여러 부서</U>가 함께 co-work를 해서 각자 맡은 일을 열심히 해서 회사를 키우는게 **multi-modal**의 예시가 될 수 있겠다.   
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209162424-488ac1d5-e7fb-4b7e-aa55-91343d969211.png" width="600"/>
    <img src="https://user-images.githubusercontent.com/79881119/209162425-9ed2e379-a796-43de-b55c-6940358e2a56.png" width="600"/>
</p>
대표적인 text와 vision을 함께 활용하는 multimodal 방식은 위에서 보는 바와 같이 video captioning과 같은 캡셔닝 기술과, video question answering과 같은 reasoning 기술이다. 이외에도 비디오나 미디어의 특정 부분을 text description을 통해 찾아내는 retrieval task도 있으며, text를 적으면 video나 image를 만들어내는 기술이나 audio를 통해 video를 만드는 기술 등등 다양하게 적용될 수 있다.   
바로 이러한 측면에서 'Transformer'는 multimodal에 접근하기 가장 좋은 네트워크 구조다. 

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209162442-15f32068-0ad1-455b-b1ea-70bcaf09d8e8.png" width="700"/>
</p>
Transformer는 단순하게도 모든 형태의 input을 tokenize할 방법만 찾으면, 이를 대표할 만한 representation space로 embedding한 뒤 attention 학습을 진행하면 된다. Embedding 과정이 복잡하지도 않으며, 가장 중요한 점은 '다양한 데이터 형태'에 적용이 된다는 것이다. 그리고 Multimodal transformer에서는 <U>fusion</U> 및 <U>alignment</U>와 같은 cross-modality interation이 attention을 통해서 자동적으로 발생한다. 혹시라도 궁금한 사람은 [survey paper](https://arxiv.org/pdf/2206.06488.pdf)를 보면 잘 정리되어 있어서 좋은 것 같다.
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209162448-0652bcf8-93e9-4427-aa81-96085c12f147.png" width="900"/>
</p>
보라색과 초록색이 서로 다른 modality embedding이라고 생각하고 각각을 살펴보면 위와 같다. 처음부터 아예 <U>더해서 attention을 진행하는 과정</U>도 있고(a), 더하지 않고 <U>차원 단위로 붙여서</U> 연산하는 과정도 있다(b). 앞선 방법들과는 다르게 transformer layer를 <u>다르게</U> 사용한 뒤, 이후 각각의 output에 대해 <U>하나의 transformer layer로 합치는</U> 과정도 있으며(c), 오히려 처음에는 하나의 layer로 학습한 뒤에 <U>여러 layer로 분리하는 과정</U>도 있다(d). 또한 서로 다른 layer로 학습하되, 각각의 query가 <U>교차되면서</U> 상대방 layer의 attention을 학습하는 방법도 있고(e), 이러한 cross-attention을 진행한 뒤에 결과를 concatenate하는 방식도 존재한다(f). 이 여러 가지 방법들은 모두 conceptually 그럴듯하게 보이며, 각 modality의 연관성이나 문제 해결 방법에 따라 더 분화할 수 있는 구조를 가진다.   
사실 어떤 task가 어떤 방법을 쓰는지에 대해서 구체적으로 알 필요가 있는 것은 아니다. 단지 이 그림에서 시사하는 바는 '<U>Transformer 구조는 여러 modality를 함께 학습하는 과정에서 취할 수 있는 전략이 매우 다양하다</U>'라는 것이다.

---

# CLIP: Learning Transferable Visual Models From Natural Language Supervision

이러한 multimodal 관점에서 등장한 가장 유명하고, 또 많이 인용되고 있는 논문인 CLIP에 대해서 설명하도록 하겠다.
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209162453-316f70d1-ae33-4bda-9b11-2cd983ad6d6b.png" width="900"/>
</p>
학습법은 간단하게도, 특정 이미지가 있다면 이를 설명하는 text prompt가 각각 주어지고, 이를 transformer encoder를 통한 embedding으로 각각 바꾼다. image embedding과 text embedding 사이에 positive pair는 가깝게(diagonal 부분), negative pair는 서로 멀게(나머지 부분) 학습하게 되면, 최종적으로는 image에 대한 classification을 text-driven으로 학습이 가능하다는 것이다. 첫번째 contrastive pre-training 부분 다음을 보면 label text로부터 dataset classifier를 만드는게 나오는데, <U>학습 시</U>에 <U>text description</U>을 사용했기 때문에 classification 시에도 <U>비슷한 형태의 description</U>을 주기 위해 '이것은 ○○○의 사진입니다'의 형태로 넣어주게 된다.   
수많은 이미지에 대한 설명과 함께 학습하다보면, 학습 시에 사용되지 않은 classification 이미지에 대해서도 좋은 성능을 보여줄 수 있다는 것이 바로 이 논문이었고, 이후 다양한 형태로 활용되며 현재 multimodal 시장에서 가장 핫한 baseline이라고 볼 수 있다.
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209162454-22fc6fb0-c39a-457e-b9a5-c25945eabd37.png" width="400"/>
</p>
녹색으로 표시된 부분이 zero-shot clip이 fully-supervised ResNet보다 더 좋은 성능을 보인 데이터셋이다. 모든 데이터셋에 대해 supervision을 가지고 학습한 ResNet보다 전혀 training sample에 대해 접근하지 못했음에도 높은 정확도를 보이는 것은 정말 혁신적이지 않을 수 없다. [CLIP 논문](https://arxiv.org/pdf/2103.00020.pdf)은 사실 실험적인 부분에서 자세하게 보고 넘어갈 부분이 정말 많아서 이후에 따로 다른 게시글에 논문 리뷰로 다룰 예정이다.
