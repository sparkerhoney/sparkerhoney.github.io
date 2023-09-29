---
title: 딥러닝의 체계를 바꾼다! The Forward-Forward Algorithm 논문 리뷰 (1)
layout: post
description: Forward-forward algorithm
use_math: true
post-image: https://user-images.githubusercontent.com/79881119/221447149-7ed12c39-a466-4f34-82d8-cecd13e304ef.gif
category: paper review
tags:
- Deep learning
- FF algorithm
- Methodology
---

# 들어가며...

제목이 너무 어그로성이 짙었는데, 논문에서는 backpropagation을 <U>완전히 대체하는 알고리즘을 소개한 것은 아니고</U> 딥러닝의 새로운 연구 방향을 잡아준 것과 같다.

이 논문에서는 neural network를 학습하는 <U>기존 방법들</U>로부터 벗어나 새로운 학습법을 소개한다. 새롭게 제시된 방법인 <U>FF(forward-forward)</U> 알고리즘은 뒤에서 보다 디테일하게 언급되겠지만 Supervised learning과 unsupervised learning의 몇몇 간단한 task에 잘 적용되는 것을 볼 수 있고, 저자는 이를 통해 FF 알고리즘이 기존의 foward/backward 알고리즘과 더불어 더 많은 연구가 진행될 수 있을 것이라고 전망한다. 아마 딥러닝을 하던 사람들은 가장 기초부터 배울 때 backpropagation이라는 개념을 필수로 배울 수 밖에 없으며, 본인이 블로그에 작성한 글 중 신경망 학습을 위해 제시된 backpropagation이라는 개념을 perceptron의 역사와 함께 소개하는 내용이 있었다([참고 링크](https://junia3.github.io/blog/cs231n04)).   
기존 backpropagation 방법은 forward pass를 통해 <U>오차를 계산한 뒤</U>(supervision이 있다고 가정하면) backward pass 시 chain rule을 통해 각 parameter를 learning rate에 따라 업데이트했다면, forward forwad algorithm(FF)은 한 번의 <U>positive pass(real data에 대한 stage)</U>와 한 번의 <U>negative pass(네트워크 자체에서 생성되는 data에 대한 stage)</U>로 구성된다.   

---

# 논문에서 제시한 backpropagation의 근본적인 문제점
사실상 딥러닝은 큰 갯수의 parameter를 가진 model을 stochastic gradient 방법을 통해 대량의 데이터셋에 fitting하는 과정이었다. 그리고 gradient는 <U>backpropagation</U>을 통해 연산하게 되었다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/220493453-18b6ebe8-cc5e-4316-932b-3ae8006f52db.png" width="400">
</p>

인간의 뇌가 작동하는 구조를 모방한 것이 신경망이었다. 시냅스가 서로 연결되어있으며 이전 신호의 magnitude에 따라 activation이 일어나는 방식으로 구성한 <U>다층 신경망 구조</U>는 실생활의 이런 저런 문제들을 해결할 수 있는 힘이 있었다. 이러한 발전이 있기 위해서는(다층 신경망을 학습시키기 위해서는) <U>backpropagation 알고리즘</U>이 필수적이었으며, 여기서 발생하는 의문점은 다음과 같았다.

- 인간의 뇌가 실제로 학습할 때 backpropagation 방법을 사용하는가?
- 만약 인간의 뇌에서도 학습할 때 backpropagation과 같은 방법을 사용하지 않는다면, 시냅스 간의 연결 사이에 가중치를 조절하기 위한 메커니즘이 따로 존재하는가?

## 피아제의 인지발달이론

잠시 논문을 소개하기 전에 심리학 이론에 대한 설명을 하고 넘어가도록 하겠다. 직접적으로 이 논문과 관련이 있을지는 모르겠지만, 논문을 읽으며 근본적으로 backpropagation에 의문을 가진 과정 자체가 인간이 어떤 <U>정보를 학습하는 메커니즘과의 차이</U> 때문이라고 생각했다. 
실제로 우리 뇌의 cortex(피질)를 생각해보면 backpropagation은 뉴런으로 구현할 수 있음이 증명되었지만 우리가 실생활에서 학습하는 방식과는 차이가 있다. 예를 들어 <U>어떤 아이</U>가 난생 처음으로 강아지를 본다고 생각해보자([참고 링크](https://www.simplypsychology.org/what-is-accommodation-and-assimilation.html)). Piaget(피아제)의 <U>인지발달이론</U>을 인용하자면, 인간에게 있어서 accommodation과 assimilation 과정이 반복되면서 인지 적응이 진행된다고 설명한다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/220494883-88e9a0a7-8bbf-4baf-8784-cc07bac1da9f.png" width="400">
</p>

아이의 보호자는 아이에게 강아지의 생김새에 대한 묘사를 해주거나, 그림책을 기반으로 '강아지'라는 존재에 대한 특징을 입력받는다.
아이는 기존에 강아지에 대한 어떠한 지식도 없었기 때문에 '강아지'라는 존재는 <U>본인의 인식 체계 속</U>에서 dissimilar한 존재다(낯선 존재). 따라서 강아지에 대한 <U>정보가 주어진 순간</U>에는 해당 지식에 대한 <U>불안정한 체계</U>가 잡히게 되고, 아이는 강아지에 대한 새로운 특징이나 정보를 입력할 때마다 '강아지'에 대한 인식 체계를 확립하며 이를 안정화하는 단계에 이른다.
이런 상황에서 길을 걷다가 실제로 <U>강아지</U>를 만났을 경우를 생각해보자. 아이는 본인이 안정화시킨(확립한) 강아지에 대한 특징을 토대로 목격한 대상을 강아지라고 판단하게 된다. 그런데 갑자기 강아지가 <U>예상치 못한 행동</U>을 하는 경우를 생각해보자. 강아지가 <U>'짖고', '물고', '핥고'</U>하는 특징들은 아이가 기존에 경험해보지 못했기 때문에 본인이 확립한 '강아지'라는 <U>인식 체계에 disequillibrium을 주는 특징</U>들이다. 이러한 혼란스러운 상황에서 아이는 부모나 정보를 제공해줄 수 있는 사람을 통해 '강아지가 맞다'라는 확답을 듣게 되면, disequillibrium 상태에 있었던 <U>지식 체계가 강화</U>된다(reinforcement). 이를 동화(assimilatioon) 과정이라고 부른다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/220498798-01bddac3-658f-49cc-99c0-453f3da78280.png" width="400">
</p>

Accomodation은 조금 다르게, 처음 보는 존재를 분류할 때 본인이 인식하고 있는 특징과 <U>다른 점들을 통해 새로운 정보 체계를 확립</U>하는 과정이다. 예를 들어 길을 가다가 고양이를 본 경우를 생각해보자. 고양이는 강아지와 다르게 '야옹'하는 소리를 내고, '나무에 올라가거나' 등등 여러 다른 특징들을 보여준다. 기존에 강아지가 본인이 알고 있는 특징들과 다른 모습들을 보여준 경우에도 아이는 같은 <U>disequillibrium 과정</U>을 거쳤고, 이런 상황에서 정답을 제공해줄 수 있는 사람을 통해 <U>정보 체계를 강화</U>했던 것과 비슷한 방식으로 아이는 정보를 제공해줄 수 있는 사람에게 강아지가 맞는지 묻게 된다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/220499468-238695a2-b377-40e4-8998-5a2eb773d618.png" width="400">
    <img src="https://user-images.githubusercontent.com/79881119/220499618-eb68d8de-a747-4e63-ba1a-d8d83a788940.png" width="400">
</p>

그러나 이번엔 강아지가 아니라 본인이 기존에 알지 못했던 정보인 '고양이'라는 대답을 듣게 되고, 이를 통해 본인이 <U>기존에 알고 있는 강아지에 대한 특징</U>에 새로운 존재인 <U>고양이의 특징</U>을 접목시켜 새로운 정보에 대해 적응하는 과정을 겪는다. 이러한 과정을 <U>accomodation</U>(적응)이라고 부른다.

## 그래서 인간의 학습 과정은?
피아제의 인지 발달 이론에 대해 굳이 짚고 넘어 온 이유는 인간은 본인이 <U>알지 못했던 사실</U>이나 <U>새로운 사실</U>을 받아들이는 과정에서 본인이 가진 인식 체계(일종의 뉴런 weight)가 도출한 잘못된 결과에 대해 오차를 계산한 뒤 이를 <U>다시 적용시키는 과정</U>이 <U>explicit하게 존재하지 않는다</U>는 사실이다. 

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/220500694-89a1e6d3-ad86-4420-9744-62ff8c24eac4.png" width="400">
</p>
시각 정보를 처리하는 visual cortex가 연결된 구조는 top-down 형태로, 시각 정보를 받아들이는 <U>가장 바깥쪽 부분</U>부터 차례로 정보를 처리하게끔 되어있다. 만약 backpropagation이 진행되는 구조는 이와는 반대로 <U>가장 안쪽 cortex부터 망막까지 이어지는 시신경 세포들까지</U>의 bottom-up mirror 구조를 가져야하는데, 실제로는 그렇게 되지 않는다는 것이다. 오히려 우리가 보는 시각 정보는 연속적인 프레임을 가진 일종의 동영상이며, <U>잘못된 판단에 대한 ground truth가 주어졌을 경우</U>(강아지라고 했는데 사실은 고양이였을 경우) 이전에 관찰한 시각 정보에 대한 <U>nerve signal</U> 오차를 계산해서 역방향으로 정보를 학습하는 것이 아니라, 우리가 <U>지금 보고 있는 이미지에 대해</U> 정보 체계를 수정하게 된다. 즉 backpropagation 구조라기 보다는 시각 정보를 통해 신경 activity가 발생하는 내부에서 <U>하나의 루프를 생성</U>하고, 이 과정으로 <U>정보 체계를 바꿔가는 것</U>으로 볼 수 있다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/220513070-eb6e8bb0-f122-4fe9-9da6-8dac3584a4d9.png" width="800">
</p>

그리고 만약 <U>우리가 학습하는 정보에 대해서도</U> backpropagation이 진행된다면 형광등이 빠른 속도로 점멸하는 것처럼 우리의 <U>인식 체계에도 주기적인 time-out</U>이 필요하다. 딥러닝에서 하는 일종의 <U>online-learning</U>과 비슷한데, 우리가 일상생활을 유지하면서 그와 동시에 backpropagation이 가능하기 위해서는 뇌의 각 단계에서의 sensory processing 결과를 저장할 pipeline이 필요하고, 이를 오차에 맞춰 수정한 뒤 원래의 인식 체계에 적용할 수 있어야 한다. 하지만 <U>pipeline의 뒤쪽에 있는 정보</U>가 backpropagation을 통해 <U>earlier stage</U>(보다 input에 가까운 위치)에 영향을 끼치지 위해서는 <U>실시간으로 인식을 진행하는</U> 우리의 학습 과정과는 차이가 있어야 한다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/220514216-e547f003-4f09-4cf9-867a-3857f6f95a41.png" width="800">
</p>

Backpropagation은 또한 forward pass 과정에서의 <U>모든 연산 결과</U>를 알아야한다는 것이다. Chain-rule에 의해 각 노드에서의 <U>local gradient를 계산</U>하기 위해서는 노드에서의 input을 알아야하고, 이는 곧 이전 노드들의 output을 모두 알아야 가능하기 때문이다. 그렇기에 <U>forward pass 과정이 black box</U>라 가정하면(어떤 연산이 진행되는지 전혀 모른다고 생각하면), 미분 가능한 모델이 확립된 상황이 아니라면 <U>backpropagation이 진행될 수 없는 것</U>을 알 수 있다. 이를 바꿔 설명하자면 만약 인간의 인식 체계가 backpropagation을 적용하기 위해서는 시신경을 포함하여 판단을 내리는 모든 구조에 대해 <U>differentiable closed form</U>으로 알고 있다는 전제가 필요하다. 이러한 문제들을 forward-forward algorithm에서는 고려할 필요가 없다.

또다른 방법으로는 강화학습을 생각해볼 수 있다. Forward process에 대한 정보가 부재할 경우에는 단순히 neural activity에 대한 weight의 일부에 random한 변화를 가해주고, 변화에 따라 바뀌는 <U>결과값에 대한 보상</U>을 해주면 된다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/220514711-1707850b-19dc-45ac-b9dc-55275351196a.png" width="400">
</p>

하지만 특정 parameter의 변화가 <U>다른 parameter의 변화에 종속</U>할 수 있는 기존의 backpropagation과는 달리, 강화학습의 경우에 <U>variance(경우의 수)가 너무 크기 때문</U>에 각 parameter의 변화가 output에 미치는 영향을 제대로 확인할 수가 없다. 이러한 문제를 학습 과정에서 생기는 noise라 하는데, 이를 완화하기 위해서는 변화가 가해지는 parameter의 개수에 반비례하게 learning rate를 구성하는 방법이 있다. 결국 <U>parameter의 개수가 증가할수록</U> 학습 속도는 이에 반비례해서 <U>계속 감소</U>하게 되며, 대용량의 네트워크를 학습시킬 수 있는 backpropagation 알고리즘에 비해 <U>학습 속도 측면</U>에서 불리하게 작용한다.

이 논문에서는 <U>ReLU나 softmax</U>와 같이 closed form으로 구할 수 있는 non-linearity를 포함하지 않는 네트워크도 학습할 수 있는 forward-forward algorithm(FF)을 제안한다. FF의 가장 큰 장점은 backpropagation 방법에서는 forward pass에 대한 <U>레이어 연산이 불명확한 경우</U>에는 <U>학습이 불가능</U>하다는 점을 해결할 수 있다는 사실이다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/220517775-fe18eb62-c188-4ba5-99cb-6478931496e6.png" width="1000">
</p>

또한 <U>연속된 데이터</U>가 주어졌을 때 다음과 같이 neural network의 output에 대한 <U>error를 통해 parameter를 업데이트</U>하는 과정에서 pipelining을 멈출 필요가 없다는 점도 장점이 될 수 있다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/220519115-745feb79-b4f3-4631-82ac-fbcdc1d680a7.png" width="400">
</p>

하지만 논문에서 밝히는 것은 backpropagation보다는 forward-forward algorithm이 <U>속도가 더 느리고</U> 실험한 몇몇의 toy problem 이외에는 아직 일반화가 힘들다는 문제가 있기 때문에 FF 알고리즘이 backpropagation을 완전히 대체하기는 힘들다고 밝힌다. 그렇기 때문에 여전히 대용량의 데이터셋을 기반으로 하는 딥러닝은 <U>backpropagation을 계속 사용할 것</U>이라고 한다.

---

# Forward forward algorithm이란?

FF는 <U>볼츠만 머신</U>이나 <U>noise contrastive estimation(NCE)</U>에서 말하는 greedy multi-layer learning procedure라고 볼 수 있다. Greedy 알고리즘이라 불리는 이유는 다음과 같다. 어떠한 <U>문제(task)</U>를 해결하려면 그에 맞는 <U>해결책(solution)</U>이 필요하고 이를 우리는 <U>알고리즘</U>이라고 부른다. 실생활에서 우리가 접하는 것과 같이 <U>복잡한 문제를 컴퓨팅 환경에서 해결</U>하는 상황에서 단번에 최적의 해를 구할 수 없는 것이 일반적이다. 따라서 복잡한 문제들을 여러 sub-task로 분류하여 해결해가는 형태의 <U>dynamic programming</U>을 활용하기도 하지만, 복잡한 문제를 타개할 마땅한 sub-task 조차도 정의하기 어려운 상황이 있다면 그럴 땐 <U>직면한 상황을 해결하면서</U> 최종 task의 solution에 근접해가는데, 이를 <U>greedy algorithm</U>이라 부른다.
볼츠만 머신이나 NCE 그리고 신경망 구조도 결국 layer-wise greedy algorithm이라고 할 수 있다. 예컨데 학습 과정에서 <U>서로 다른 위치</U>의 레이어는 <U>input 정보</U>가 다르기 때문에(distribution) 이들을 통해 얻을 수 있는 representation(feature) 또한 달라지게 되고, 결국 각 layer는 <U>각자가 직면한 상황</U>에서 <U>독립적인 solution을 구하는</U> task로 귀결된다. Greedy MLP algorithm은 각 레이어가 학습되는 단계에서는 각자가 내릴 수 있는 최선의 해결책을 내놓아야 한다는 점이고, 이는 모든 레이어가 <U>독립적으로 학습된다는 assumption</U>이 포함된다.

FF algorithm이 제시하는 방법은 다음과 같다. <U>서로 반대되는 objective</U>를 가지는 두 data가 <U>각각 forward pass</U>되면서 backpropagation을 대체한다는 것이다. 이러한 두 forward pass를 각각 'positive pass' 그리고 'negative pass'라고 한다. Positive pass는 <U>real data</U>에 적용되고, 각 <U>hidden layer의 weight</U>로 하여금 <U>'goodness'</U>를 증가시키게끔 작동한다. 그와는 반대로 negative pass는 <U>negative data</U>에 적용되며 각 hidden layer에 대해 <U>'goodness'</U>를 감소시키는 방향으로 작용한다. 각각의 pass에 대한 goodness는 서로 같은 식이지만 부호가 반대라고 생각하면 되는데, positive pass의 경우에는 goodness가 neural activities의 squared sum이 되고, negative pass의 경우에는 goodness가 neural activities의 negative squared sum이 된다. 이 논문에서 제시한 goodness란 greedy 알고리즘에서 '최적의 선택'을 의미하며, 굳이 이 논문에서 주장한 <U>squared sum</U>이 아니라 다른 형태가 될 수 있다고 말한다. 본인이 생각하기에는 이 부분이 아마도 앞으로 FF algorithm을 활용한 다양한 딥러닝 연구의 기준이 되지 않을까 생각해본다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/220526904-ba8ddae0-6cc4-466a-bda7-535ccd91b2d8.png" width="700">
</p>

위의 그림을 토대로 보면, 각 레이어는 each pass에 따라 real data(positive input)에 대해서 negative input을 기준으로 특정 threshold 만큼 높이는 것이 목적이 된다. 기존의 backpropagation은 <U>output node에서만</U> objective function을 가졌다면, FF algorithm에서는 <U>각 레이어마다</U> objective function을 가진다고 생각할 수 있다. 각 레이어에서의 objective function은 <U>positive input</U>과 <U>negative input</U>을 이진 분류하는 classifier와 같기 때문에 특정 레이어의 노드 index $j$에 대해 positive sample일 확률을 logistic $\sigma$를 통해 다음과 같이 표현할 수 있다.

\[
    p(\text{positive}) = \sigma \left( \sum\_j y_j^2 - \theta \right)    
\]

식에서의 $y_j$는 <U>layer normalization 이전의 activity</U>에 해당되고, $\theta$는 threshold다. 식에서 확인할 수 있는 것은 저자가 목적함수로 삼은 goodness가 <U>데이터의 이진 분류</U>를 위한 logistic의 input으로 사용되기 때문에 likelihood(squared sum)를 최대화하는 방향으로 <U>positive pass</U>의 goodness를, negative likelihood(negative squared sum)를 최대화하는 방향으로 <U>negative pass</U>의 goodness를 설정했다고 해석할 수 있다.

---

# Learning MLP with greedy algorithm
위의 방법을 그대로 생각해보면 <U>단일 hidden layer</U>에 대해서는 각 데이터에 대한 goodness를 구하는 방식이 명확하고, 학습할 때의 objective 또한 마찬가지인 것을 볼 수 있다. 하지만 만약 <U>첫번째 layer의 output</U>이 그대로 <U>두번째 layer의 input으로</U> 사용된다면 이미 output의 squared sum을 통해 positive/negative 구분이 가능하게끔 학습되었기 때문에 첫번째 layer의 output vector의 크기를 비교하는 것만으로도 두번째 layer에서는 goodness를 판별할 수 있게 된다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/220530960-9729f287-48b7-46a3-a55f-a74838df93b7.png" width="700">
</p>

쉽게 말하자면 <U>볼츠만 머신과 같은 greedy 알고리즘</U>에서 추구하고자 하는 것은 각 레이어마다 <U>input이 다르기 때문에</U> 그에 맞게 <U>서로 다른 representation</U>을 학습하게 되는 것인데, 이미 이전 레이어에서 학습한 결과만 있다면 goodness 판별이 어렵지 않기 때문에 이후 레이어는 새로운 feature(representation)을 <U>학습할 필요가 없게 된다</U>(일종의 identity mapping이라고 생각하면 될 것 같다).

이러한 <U>feature collapse</U> 문제를 막기 위해서 FF는 hidden layer output으로 나오는 feature vector의 길이를 다음 layer의 input으로 넣기 전에 normalize하게 된다. 그렇게 되면 길이에 대한 정보를 통해 <U>상대적인 길이만 유지</U>한 채로 input으로 들어가게 된다. 다르게 표현하자면 activity vector는 크기와 방향을 가지는데, <U>크기 정보를 필터링</U>하고 <U>방향에 대한 정보</U>만 다음 layer로 보낸다고 생각하면 된다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/222123454-0e336269-f646-450b-8695-c6e9fbc5ab5e.png" width="600">
</p>

\[
    \begin{aligned}
        \hat{a^l} =& \frac{a^l}{\vert \vert a^l \vert \vert_2} \newline
        a^l =& (a_1^l,~a_2^l,~a_3^l,~a_4^l,~a_5^l)
    \end{aligned}
\]

FF algorithm에서의 layer normalization은 activity에 대해 <U>layer mean을 빼주는 과정 없이</U> activity vector의 <U>길이로 나눠주는 작업</U>을 진행했다고 한다.

---

# FF experiments
앞서 말했던 바와 같이 이 논문의 주된 목적은 <U>FF algorithm의 feasibility를 확인</U>하는 것이기 때문에 상대적으로 적은 parameter 수를 가지는 작은 neural network에 대해 적용한 실험이 대부분이고, 저자는 <U>이후의 연구들을 통해</U> FF를 <U>large neural network에 적용</U>하는 것은 future work로 남겼다.

## Backpropagation baseline for MNIST
이 논문에서 대부분의 실험은 <U>MNIST dataset</U>(숫자 손글씨)을 기반으로 한다. MNIST의 원래 구성은 $60,000$개의 training images와 $10,000$개의 test images 인데, 이 중에서 $50,000$의 training image와 $10,000$개의 validation image로 분리하여 최적의 hyper-parameter를 찾는 과정을 거친다. MNIST는 딥러닝에서 원래 사용되던 backpropagation 알고리즘을 적용한 연구에서 많이 활용되던 데이터셋이기 때문에, <U>저자가 제시한 새로운 알고리즘의 feasibility를 확인</U>하기 좋은 구조가 된다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/220541733-4f57acde-0604-4d97-9d23-bea840547492.png" width="600">
</p>

CNN(Convolutional neural network)와 같은 구조는 MNIST에 대해 대략 $0.6\%$의 오차를 보인다. 보통 CNN과 같은 구조는 permutation-invariant하지 않다고 하는데, permutation invariant란 <U>순열의 변화</U>가 <U>output에 영향을 미치지 않는 경우</U>를 의미한다. 즉 ReLU를 activation function으로 사용하는 MLP 구조는 permutation invariant 구조인데, 이 경우 대략 $1.4\%$의 test error를 보이고 dropout이나 label smoothing 같은 regularizer를 사용할 경우 $1.1\%$까지 성능이 올라간다. 이에 추가로 이미지의 확률 분포를 모델링하는 unsupervised learning 방법을 추가하면 더 성능이 올라가지만, 요약하자면 정규화를 고려하지 않는다면 CNN은 $0.6\%$, MLP는 $1.4\%$의 test error를 보인다.

## Unsupervised learning in FF
FF algorithm을 길게 설명했는데, 여기서 두 가지의 의문이 나오게 된다. 첫번째는 <U>negative data를 학습하는 과정</U>이 dataset의 <U>multi-layer representation</U> 학습에 어떤 방식으로 효과를 주는가이고, 두번째는 <U>negative data를 어떻게 만들어내는가</U>이다. 저자는 첫번째 질문에 대한 대답을 하기 위해 <U>hand-crafted negative data</U>를 만들어내게 되었다.

<U>Contrastive learning</U>을 supervised learning에서 활용할 때 주로 적용하는 방식은 input vector를 label에 대한 정보 없이 <U>representation vector로 바꾸는 작업</U>을 진행한 뒤에, 이를 label에 대한 softmax probability를 구하는 <U>linear transformation</U>으로 학습하는 구조를 활용한다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/220544597-6a6533da-3184-4d29-86bd-654c87e2d0a5.png" width="600">
</p>

Linear transformation을 학습하는 과정은 supervision을 가지지만, <U>hidden layer 없이</U> 학습되기 때문에 <U>backpropagation</U>이 따로 필요하지 않다.
FF는 바로 이러한 관점에서 <U>positive example</U>과 <U>corrupted example</U>을 활용한 representation learning을 한다고 볼 수 있다. Dataset을 corrupt하는 방법은 data augmentation과 마찬가지로 여러 가지가 있을 수 있다.

이러한 여러 augmentation 중에서 FF 알고리즘의 효과적인 학습을 위해서는 다음과 같은 조건을 충족해야한다고 한다. FF가 image에서의 object shape와 같은 long-range correlation(이미지 전반을 보게끔)을 가질 수 있게 하는 방법은 <U>negative dataset</U>이 real dataset과 <U>long range correlation</U>은 **다르게**, <U>short range correlation</U>은 **유사하게** 구성하는 것이다. 이렇게 데이터셋을 구성하게 되면 네트워크의 각 레이어는 short range correlation로는 fake/real을 <U>구분할 수 없기 때문에</U> longer range correlation에 집중하는 경향성이 생긴다. 가장 간단한 방법은 ones/zeros로 구성된 마스크를 넓은 영역으로 구성하는 것이다. 그런 뒤 서로 다른 real image를 mask와 reversed mask를 적용하여 합한 데이터셋을 구성하게 되면, <U>적은 영역에 대해서는 real dataset과 구분할 수 없는</U> negative sample을 구성할 수 있게 된다.

넓은 영역의 mask를 만드는 방법은 다음과 같은데, 먼저 random한 bit image를 기준으로 가로/세로 모두에 $(1/4,~1/2,~1/4)$의 값으로 블러링하는 과정을 계속 반복한다. 실제로 negative sample을 만드는 과정이 궁금해서 논문에서 제시한 방법을 코드로 옮겨보았다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/220559410-cc6e67cf-7e3e-41b9-b6fe-d12eb2da9dda.png" width="600">
</p>

## Hand-crafted negative sample

#### MNIST dataset sample 가져오기

```python
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import numpy as np

train_dataset = MNIST('./data/', train=True,download=True)
sample1, _ = train_dataset[5]
sample2, _ = train_dataset[20]
sample1_array = np.array(sample1)
sample2_array = np.array(sample2)
```

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/220557403-e28b43bf-8617-4af5-b747-d7944a567025.png" width="400">
</p>

#### $28 \times 28$ 크기의 random bit image 생성하기

```python
random_bit_image = np.random.randint(2, size=(28, 28))
```

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/220557637-bf65cdf3-6e43-4139-9aae-98f691a215ae.png" width="200">
</p>

#### $3 \times 3$ 크기의 blur kernel 생성하기

```python
blur_kernel = np.array([[0, 1/4, 0],[1/4, 1/2, 1/4],[0, 1/4, 0]])
```

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/220558021-d792c287-77ad-4419-ad88-d26e659ac99e.png" width="200">
</p>

#### Numpy 배열에 대한 2D convolution 함수
```python
# Define convolution operation for 2D numpy matrix
def conv2d(image, kernel):
    output_height, output_width = image.shape
    output = np.zeros((output_height, output_width))

    # zero padding
    image = np.pad(image, ((1, 1), (1, 1)), 'constant', constant_values=0)
    
    # calculate 2d convolution
    for h in range(output_height):
        if (h + 3) <= image.shape[0]:
            for w in range(output_width):
                if (w + 3) <= image.shape[1]:
                    output[h][w] = np.sum(
                        image[h : h + 3, w : w + 3] * kernel
                    ).astype(np.float32)
    return output
```

#### Iteration 돌리면서 blur kernel 적용하기

```python
iteration = 5
blurred = random_bit_image[:, :]

for i in range(iteration):
  blurred = conv2d(blurred, blur_kernel)

output = (blurred-np.min(blurred))/(np.max(blurred)-np.min(blurred))
```

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/220558492-f8ce61ec-1d0b-4a25-a876-3af816573cf4.png" width="200">
</p>

#### Thresholding 적용하여 mask 생성하기
```python
def thresholding(image, thresh=0.5):
    return np.where(image > thresh, 1, 0)

mask = thresholding(output)
```

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/220558727-932c0804-839f-4f7a-9764-f26bc1571b6d.png" width="200">
</p>

#### Sample 섞기
```python
mixed = sample1 * mask + sample2 * np.where(mask==0, 1, 0)
```

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/220558989-74c7a012-ad90-4a75-8c5c-c920d4082f32.png" width="200">
</p>

이게 맞나...? <U>논문에서 주어진 그림처럼</U> 깔끔한 mask가 잘 안나온다.

아무튼 negative sample을 잘 만들어주었고, 이를 토대로 $4$개의 layer를 가지는 MLP를 학습한 결과 test error rate로 $1.37\%$를 얻을 수 있다고 한다. Label을 prediction할 때 <U>마지막 세 layer의 normalized activity vector를 사용</U>했으며, 첫번째 hidden layer의 output을 사용했을 경우에는 performance가 악화되었다고 언급한다. 물론 fully-connected layer 말고도 local receptive field를 가지는 구조를 활용할 수도 있는데, 이때는 보다 성능이 올라서 $1.1\%$의 오차율을 보였다고 한다. 참고로 <U>CNN이랑 조금 다른 점</U>은 원래 CNN에서는 weight parameter가 sharing되는데, FF 알고리즘에서는 필터의 weight가 공유되지 않는다는 것이다. 저자들이 밝힌 local receptive field 구조는 다음과 같다.

- The first hidden layer used a $4 \times 4$ grid of locations with a stride of $6$, a receptive field of $10 \times 10$ pixels and $128$ channels at each location. The second hidden layer used a $3 \times 3$ grid with $220$ channels at each grid point. The receptive field was all the channels in a square of $4$ adjacent grid points in the layer below. The third hidden layer used a $2 \times 2$ grid with $512$ channels and, again, the receptive field was all the channels in a square of $4$ adjacent grid points in the layer below. This architecture has approximately 2000 hidden units per layer.

---

# Supervised learning in FF
앞서 진행한 학습은 label 없이 representation을 학습하는 방법에 대한 문제였다(Like contrastive learning method). Unsupervised learning 방법은 네트워크 크기가 크고, 학습 가능한 feature가 여러 downstream task에 활용될 수 있을 때 유용하다는 특징이 있지만 굳이 그러지 않고 작은 네트워크를 <U>원하는 task</U>에 대해 <U>fine-tuning</U> 혹은 단순 <U>fitting</U> 시켜서 사용하고 싶을 수도 있다.

FF에서는 이를 해결하는 방법이 input에 label을 추가하는 것인데, 예를 들면 <U>text 학습 시에 앞단에 prompt를 붙여주는 것</U>처럼 이미지에 <U>label에 대한 정보를 함께 주는</U> 방식이다. 앞서 unsupervised learning에서 했던 것과는 다르게 이번에는 <U>label 정보가 맞다면</U>(correct label) positive data이고 <U>다르다면</U>(incorrect label) negative data가 된다. 이번에는 <U>positive</U>와 <U>negative</U> 간의 차이가 오직 label이기 때문에, FF 과정에서 negative label이 있다면 해당 상황에서 <U>모든 feature를 무시하게끔</U> 학습해야한다. 왜냐하면 잘못된 라벨과 이미지를 매칭하는 상황 자체가 잘못된 correlation이 형성되는 과정이기 때문이다.

MNIST 이미지는 black border를 가지고 있기 때문에 CNN을 적용하기 적절한 데이터 구조가 된다. 첫번째 $10$개의 픽셀을 $N$ 만큼의 <U>label representation 중 하나로 치환</U>하게 되면 첫 hidden layer가 학습하는 형태를 간단하게 시각화할 수 있다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/220803989-e3b018e6-b76c-4c74-938e-d0e36ec19d85.png" width="900">
</p>

앞서 <U>unsupervised에서 사용했던 MLP</U> 구조(4개의 ReLU를 포함한 hidden layers)를 학습시켰을 때 $1.36\%$의 test error가 나왔다고 한다. Backpropagation은 FF보다 약 $1/3$만큼의 epoch에도 비슷한 결과가 나오는 것으로 봤을 때 아직 이 논문에서 제시하는 FF 알고리즘이 완벽하지는 않다는 것을 보여주는 것 같다. <U>수렴 속도를 늘리기 위해서</U> Learning rate를 높이고 더 적은 epoch에 대해 학습하게 되면 오히려 test error 성능이 $1.46\%$로 하락했다고 한다. FF 방법을 통해 학습하고 난 후에는 test digit에 neutral label($10$개의 픽셀이 모두 $0.1$ 값을 가짐)이 추가된 input을 분류할 수 있게 된다. 이런 방법을 사용하게 되면 사실 <U>들어가는 input을 제외하고</U>는 unsupervised learning과 동일한데, 첫번째 hidden layer의 activation을 제외하고 <U>나머지 activation을 모두 사용</U>하여 학습된 softmax를 통해 classification을 진행한다. 그러나 굳이 이런 방법을 사용할 필요가 없고, 단순히 특정 라벨과 test image를 통과시킨 후, <U>activation 결과로 축적된 goodness</U>가 가장 높은 label을 prediction하는 경우를 생각해볼 수 있다. 학습 과정에서 neutral label을 hard negative label로 사용할 때는 <U>학습이 더 어려워졌다</U>고 한다.

## Supervised learning MNIST sample

#### MNIST sample 가져오기

```python
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import numpy as np

train_dataset = MNIST('./data/', train=True,download=True)
sample1, label1 = train_dataset[5]
sample2, label2 = train_dataset[20]
sample1_array = np.array(sample1)
sample2_array = np.array(sample2)
```

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/220806706-912847a9-ee8d-401e-b736-9f3b60ade2b8.png" width="400">
</p>

#### MNIST에 Label 씌우기

```python
def label_on_mnist(image, label):
    overlay = image[:, :]
    overlay[0, :10] = 0
    overlay[0, label] = np.max(image)
    return overlay
```
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/220807486-75c16ef7-ece3-41b5-9c2a-f8a4c60e2706.png" width="600">
</p>

#### Positive sample 만들기

```python
overlay_pos1 = label_on_mnist(sample1_array, label1)
overlay_pos2 = label_on_mnist(sample2_array, label2)
```

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/220806997-706465ff-ba3d-4c2e-8960-0052922621c0.png" width="400">
</p>

#### Negative sample 만들기

```python
overlay_neg1 = label_on_mnist(sample1_array, np.random.choice([num for num in range(10) if num != label1]))
overlay_neg2 = label_on_mnist(sample2_array, np.random.choice([num for num in range(10) if num != label2]))
```

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/220808333-bb077ab7-9eae-41db-b552-c17731750bfe.png" width="400">
</p>

---

# FF를 사용하여 perception 모델링하기
지금까지 제시된 방법은 모두 <U>단일 layer</U>를 각각 <U>greedy algorithm</U>을 기반으로 따로 학습하는 과정이었다. 즉, 이후의 레이어가 학습하는 것이 기존의 <U>backpropagation과는 다르게</U> 이전 레이어의 학습에 아무런 영향을 미치지 않고 <U>독립적으로 학습된다</U>는 뜻이다.   
이는 backpropagation에 대해 가지는 <U>FF 알고리즘의 명백한 한계점</U>이다. 따라서 computation에서 학습되는 과정보다는 인간이 실제로 시각 정보를 받아들이는 과정을 생각해보았다. 예를 들어, 단순히 이미지를 학습에 사용하고 끝나는 것이 아니라, 정적 이미지를 좀 지루한 비디오로 간주하고, 기존의 신경망 구조를 <U>다층 recurrent neural network</U>로 보자는 것이다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/220810512-6c43d83a-b28e-4751-a096-559b28204b4d.png" width="700">
</p>

그런데 여기서 의문점이 생긴다. 앞서 설명했던 FF는 forward pass를 진행하는 과정에서 positive와 negative data를 각각 processing하는 과정을 거쳤었는데, 결국 단일 이미지를 연속된 input으로 간주하게 되면 <U>negative sample은 학습에 활용할 수 없는 구조</U>가 되는 것이 아닌가라는 것이다. 

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/220811042-0f013fa1-2e35-458a-b80c-2433209320bf.png" width="700">
</p>

저자는 다음과 같이 밝힌다. FF가 forwarding을 진행하는 것은 기존에 제시했던 positive data와 negative data에 대해 모두 해당되는 것은 맞지만, recurrent 구조에서는 <U>각 layer의 output이 activity에 관여하는 과정</U>만 달라지는 것이다.

예컨데 $t$ 시점에서의 이전 레이어의 output과 다음 레이어의 output(인접한 레이어를 의미)가 $t+1$ 시점에서의 <U>기준 레이어의 activity vector</U>를 결정하게 된다. 보다 디테일하게 구현한 형태는 다음과 같다.

저자는 MNIST 이미지를 여러 time frame으로 늘려 사용하였다. 가장 <U>bottom layer</U>는 위의 figure에서 볼 수 있듯이 MNIST 이미지를 의미하고 가장 <U>top layer</U>는 각 이미지의 one-hot encoding label을 의미한다. 각 hidden layer는 $2000$개의 뉴런을 가지고 총 $2 \sim 3$개의 층을 가진다고 한다. 위의 그림을 기준으로 보면 input/output layer를 포함해서 총 $4$개의 layer가 있다고 생각하면 될 것 같다.

이 논문이 preliminary로 삼은 recurrent multilayer learning 논문은 'How to represent part-whole hierarchies in a neural network'이다([참고 링크](https://arxiv.org/pdf/2102.12627.pdf)). 각 시점에서 짝수 번째의 layer activity는 홀수 번째의 layer activity를 normalize한 결과(앞서 봤었던 <U>layer normalization</U>)를 기준으로 업데이트가 되며, 반대로 홀수 번째의 layer activity는 짝수 번째의 layer activity를 normalize한 결과를 기준으로 업데이트가 된다.

그러나 이러한 방식의 alternating(번갈아 가며 학습하는) 구조는 biphasic oscillation(서로 다른 상(phase)이 공존하는 형태를 의미하는 말인 것 같은데, <U>학습 과정의 불안정성</U>이라고 보면 될 것 같다)를 방지하기 위함이었지만, 이러한 학습법이 굳이 <U>불필요하다는 것</U>이 밝혀졌다. 약간의 정규화 장치를 포함한 synchronize(짝수/홀수 layer를 구분하지 않고 <U>한번에 layer를 학습</U>하는 것)이 효과적이었으며, 실제 실험에서는 모든 층의 layer를 동시에 학습하였고 앞서 말했던 biphasic oscilation을 방지하기 위해 new pre-normalized state와 previous pre-normalized state를 <U>$7 : 3$로 weight하는 방식</U>을 대신 사용하였다.

앞서 본 구조에 따라 MNIST가 학습되는 과정은 이와 같다. 먼저 각 이미지는 hidden layer를 단일 bottom-up pass를 통해 초기화하게되고,
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/220823188-bd5efd49-3dfe-4ef8-adc5-93e5961f0e4d.png" width="600">
</p>
그 후에 $8$번의 synchronous iteration(damping 적용)이 <U>단일 이미지에 대한 학습</U>을 진행하게 된다.
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/220824075-9118741a-72d0-4454-9a18-400776871989.png" width="600">
</p>
Test data에 대해서 performance를 측정할 때에는 $10$개의 <U>각 라벨에 따라</U> goodness 측정을 하게 되고, $3$에서 $5$ iteration 동안 <U>평균 goodness가 가장 높은 label</U>을 기준으로 예측을 진행한다. 그 결과 $1.31\%$의 test error를 보였다고 한다.
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/220824953-e237739d-d727-48b1-955d-7c6e6089680e.png" width="600">
</p>

Negative data는 single forward pass를 통해 모든 class에 대한 probability를 구한 뒤 각 probability에 따라 incorrect class를 고르는 식으로 결정하였으며, 이러한 방식이 <U>학습 과정에서 효과적</U>이라고 하였다. 아무래도 probability에 따라서 incorrect class를 정하게 될수록 <U>hard negative sample이 생성될 가능성이 높기 때문</U>이지 않을까 생각해본다.

---

# Predictions from spatial context as a teacher
Recurrent network 구조를 사용하게 되면 objective를 학습하는 과정은 위쪽 그리고 아래쪽 layer의 input과 <U>높은 agreement</U>를 가지게 된다(인접한 레이어끼리 의견 공유가 활발하다고 생각하면 된다). 즉 positive data에 대해서는 위쪽과 아래쪽 레이어와의 agreement가 크게, 반대로 negative data에 대해서는 작게끔 학습되는 구조가 되는데, 이는 <U>spatially local connectivity</U>를 가지는 네트워크 구조에서는 <U>좋은 property로 사용</U>될 수 있다.

위쪽 레이어로부터 내려오는 input은(top-down) prediction의 결과에 가까운 레이어로부터 오기 때문에 이미지를 기준으로 보다 넓은 영역에 의해 결정된 representation일 것이고, 이는 아래쪽 레이어로부터 올라오는 input(bottom-up)으로 하여금 어떠한 output을 만들어내야 하는지에 대한 내용이 된다. 즉 시간이 지남에 따라 <U>recurrent하게 forward forward algorithm을 적용하는 것</U>은 local representation을 학습하는 <U>bottom layers</U>가 prediction에 대한 contextual information을 가지고 있는 <U>top layers</U>로부터 영향을 받을 수 있게 되는 것이다. 반복된 input의 노출이 lower layer의 causality에 대한 문제를 줄여주고, layer 간 <U>의존성이 없던 forward 알고리즘</U>에게는 일종의 문맥상의 힌트를 줄 수 있게 된다는 것이다.

Positive data의 경우에는 <U>bottom-up information</U>(image에 대한 representation)이 prediction(label) information을 보다 <U>잘 반영하게끔</U> activity를 학습하고, 반대로 negative data의 경우에는 <U>bottom-up information</U>이 prediction(wrong label) information을 <U>cut-off(ignore)</U>하도록 activity를 학습하게 된다.

---

# CIFAR-10 dataset을 통한 실험
MNIST dataset의 경우 modality 특성상 단순한 형태를 가지다보니 forward forward algorithm으로도 충분히 성능이 보장되는 것을 확인할 수 있다. 하지만 CIFAR-10 dataset은 <U>이와는 다르게</U> $50,000$ 장의 $32 \times 32$ 크기의 RGB 이미지를 modality로 가지고, black border를 가지는 MNIST와는 다르게 background나 object의 형태가 보다 다양하기 때문에 <U>한정된 training data로</U> 모델링하기 힘들다는 특징이 있다. 그렇기 때문에 단순히 $3\sim 4$개의 hidden layer를 가지는 MLP 구조를 backpropagation 알고리즘으로 학습하게 되면 overfitting이 발생하는 등 학습이 제대로 진행되지 않는 것을 볼 수 있으며, <U>대부분의 결과</U>는 <U>convolutional</U> neural network 구조에 대한 내용이다.

앞서 설명했던 convolution 구조에서 얼핏 확인했겠지만, FF는 <U>weight sharing이 불가능</U>한 네트워크이다. 기존 CNN이라면 단일 convolutional layer에서의 parameter는 feature map의 모든 receptive에 동일하게 적용되겠지만, FF 알고리즘에서는 불가능하다. 그렇기 때문에 지나치게 weight 개수를 제한하지 않는 선에서 최대한 <U>적당한 갯수의 hidden unit를 적용한 backpropagation baseline과 비교</U>를 하였다.

네트워크는 총 $2 \sim 3$개의 hidden layer 수를 가지는데, 각 layer는 $3072$의 ReLU 노드로 구성된다. 각 hidden layer에서 output으로 나오는 <U>tensor의 차원을 모두 곱한 값</U>을 곧 노드 갯수라고 생각하면 된다($32 \times 32 \times 3 = 3072$). 연산이 되는 hidden unit은 $11 \times 11$의 receptive field를 가지기 때문에 각 point에서의 input 크기는 $11 \times 11 \times 3 = 363$이다. FF 알고리즘의 경우 sequential input에 대해 단일 레이어를 기준으로 top-down activity와 bottom-up activity에 대해 receptive를 가지는데, 가장 말단의 hidden layer를 제외하고는 모두 $11 \times 11$의 크기를, 가장 말단의 hidden layer의 경우 top-down activity에 대해서는 $10$의 input을 가지게 된다. 그림으로 간단하게 표현하면 다음과 같다. 

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/221398579-a7a1b8db-3f6d-4a69-89c1-d2823e08e9ea.png" width="800">
</p>

그림의 구조를 보면 마치 CNN(Convolutional Neural Network) 처럼 표현되어있지만, 사실은 <U>fully-connected되지 않은</U>(한정된 receptive field 크기를 가지는) <U>MLP 구조</U>로 볼 수 있다. 그리고 input/output resolution을 유지할 때 padding을 사용했던 CNN 방식과는 다르게, edge 부분의 hidden unit에 대해서는 receptive field를 <U>input에 맞게 truncate</U>해서 사용하게 된다.   
학습 과정에서 CIFAR dataset 특성 상 overfitting 문제가 생기는 것을 방지하기 위해서 정규화 방법으로는 <U>weight decay</U>를 사용했다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/221399147-72e210a6-165d-42a2-a991-d375ee2976e5.png" width="400">
    <img src="https://user-images.githubusercontent.com/79881119/221399178-0c5a71ff-e5af-448a-993c-8d285b9b9f7d.png" width="400">
</p>

같은 학습 구조에 대해 <U>BP와 FF algorithm</U>의 각 objective에 대해 <U>training/test error</U>를 비교하였다. 실험하는 과정이 앞서 설명한 image를 boring video로 간주하는 작업이기 때문에 각 이미지를 $10$ iteration 동안 네트워크에 통과시키며 activity를 연산하고, 이 중에서 goodness 기반 error가 가장 적은 $4 \sim 6$의 <U>energy를 축적</U>하여(summation or mean으로 간주하면 될 듯) 하는 방법을 사용하거나, 단순히 <U>single-pass softmax</U>를 보는 방법 두 경우에 대해 모두 실험하였다.

결과를 보면 hidden layer의 개수에 따라 성능 차이가 거의 없다고 볼 수 없고, FF 알고리즘이 BP 알고리즘에 test error 기준으론 크게 뒤쳐지지는 않지만 확실히 기존의 BP가 FF에 비해서는 <U>training error가 피팅되는 속도</U>가 더 빠른 것을 볼 수 있다.

---

# Sequence learning with string dataset
현재 아카이브에 올라온 FF algorithm paper는 <U>수정본</U>이고, 이전에 올라온 <U>draft 버전</U>([참고 링크](https://www.cs.toronto.edu/~hinton/FFA13.pdf))에 수록된 간단한 실험이 있다. 해당 내용에 대해 간단하게 요약하자면, 앞서 실험한 FF로는 연속된 이미지와 같이 video를 생성하기에는 아직 performance가 부족하지만, <U>문장을 완성하는</U> 형태의 <U>discretized sampling</U>은 상대적으로 구현하기 간단한 편에 속한다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/221448892-056b30f8-1fcf-4495-a378-6e32e371dfa2.png" width="400">
</p>

예를 들어 위의 그림에서 보는 것과 같이 공백을 포함한 $10$개의 character를 기준으로 <U>다음 character를 예측</U>하는 task를 생각해보자. 구현하기 가장 간단한 방법으로는 앞선 $10$개의 string character를 토대로 higher-order feature를 생성하는 앞단의 hidden layer와, 이 activity를 사용하여 <U>다음 character를 예측</U>하는 softmax(알파벳, 공백 혹은 마침표 등을 classification)를 고려할 수 있다. 실제 실험에서는 이솝 우화의 fable에서 각각 $100$개의 character로 구성된 $248$개의 string을 사용했으며, 공백이나 콤바, 세미콜론을 포함한 총 $30$개의 <U>정해진 문자 이외에는 모두 제거</U>하였다고 한다. 네트워크는 모든 hidden layer가 $2000$개의 ReLU 노드를 가지는 구조다.
Positive sample은 기존 string에서의 $10$개의 character를 그대로 가져온 경우가 되고(The wolf r), negative sample은 마지막 character를 이전의 $10$개의 prediction 중 하나의 character로 대체하는 것으로 구성하였다(The wolf h). <U>Sequence to sequence</U> 구조에서 주로 활용하는 <U>teacher forcing</U>이 되는 것이다(ground truth를 학습에 사용하여 수렴 속도를 높이는 방법). 다른 방법으로는 negative data를 아예 predictive model의 output으로만 구성하는 방법이 있는데, 네트워크가 가장 앞단의 $10$개의 character를 '기억한다'는 가정을 하는 것이다. 사실 이 부분이 기존 논문에서 가장 이해가 안되는 부분이였는데, 앞서 마지막 character만 바꾸어 teacher forcing을 하는 방식과는 다르게 직접 <U>predictive model의 generation</U>을 데이터로 쓰게 되면, sleep phase에서 <U>offline으로 negative data를 만들어낼 수 있다</U>는 것이다. 이 부분에 대한 내용은 <U>수정본에서 다시 설명</U>되어 있어서 확인해보았다. 나름대로 이해한 사실을 바탕으로 정리하면 다음과 같다.

---

# Sleep phase and Awake phase
Awake phase는 positive data로 학습되는 과정을 의미하고, Sleep phase는 negative data로 학습되는 과정을 의미한다. Draft 논문에서는 <U>실제 실험을 토대로 아래와 같은 결과</U>를 얻을 수 있었다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/221470160-05e65e73-1b99-435a-bc2e-5dc181ad6fe6.png" width="400">
</p>

파란색은 hidden representation 학습이 실제 estimation에 미치는 영향을 보기 위해 <U>랜덤하게 초기화한 hidden weight</U>를 freeze한 채로 linear classifier를 학습하게 된다. 검은색 선과 붉은색 선이 사실상 가장 중요한 부분인데, synchronous phase(검은색)에서는 실제 data에 해당되는 positive sample 그리고 학습 과정에서 network가 generation하는 sequence를 negative sample로 사용하여 positive gradient와 negative gradient를 함께 학습하는 것이다. 이와는 다르게 seperated phase(붉은색)에서는 처음에는 positive sample만 사용하여 네트워크를 학습시키고, 그 뒤에 학습된 네트워크가 만들어내는 negative sample로 네트워크를 학습하는 것이다. 학습 과정이 분리되었지만 실제 성능 상으로는 synchronize하는 것과 거의 유사하였고 저자는 여기서 <U>하나의 해석을 추가한 것</U>이다.

성능이 크게 악화되지 않다면, 굳이 online(데이터셋을 사용하여 학습하는 과정) 상에서 negative sample을 사용할 필요 없이 offline(데이터셋 없이 네트워크를 통해 만들어낸 임의의 sequence)를 이후에 학습하면 된다. 사람으로 치면 awake(깨어있는) 상태에서 감각과 interaction하는 현실을 경험하고, sleep(자고있는) 상태에서 <U>학습된 경험에 대한 disequillibrium을 조정</U>하는 과정으로 볼 수 있다. 그렇기 때문에 저자가 언급하는 <U>sleep phase</U>란 실제 데이터셋 없이 <U>negative sample에 의한</U> 최적화 과정이고, 반대로 <U>awake phase</U>란 실제 데이터셋으로 <U>positive sample에 의한</U> 최적화를 진행하는 과정이 된다. 좀 재밌었던 파트는 draft에서의 결과 그래프가 어째선지 reproduce가 안되었는데(그래서 수정본에는 결과 그래프가 없다), 프로그램상 버그인 것 같다고 하는 저자의 푸념섞인 글이었다. <U>학습이 안정적이지 못하다는 점</U>이 reproduction에 실패한 이유 같기도 한데, phase를 번갈아가며 학습하는 과정은 learning rate가 아주 작으며 momentum이 극도로 큰 경우에만 수렴이 된다고 한다. 저자는 이런 시행착오를 겪는 과정 자체가 FF 알고리즘이 biological하다는 증거로 간주하고, 실제로 <U>사람이 수면 부족을 겪는 것</U>처럼 positive phase만 학습시키다 보면 <U>성능이 떨어지는 상황</U>이 발생한다고 한다.

---

# 마무리하며
사실 논문이 여기서 끝나는 것은 아니고 FF 알고리즘의 학습법이나 컨셉에 대해서 조금 더 서술해주는 부분이 실험이나 구현 뒤쪽에 나와있다. Contrastive learning에 대한 내용 그리고 레이어 정규화 작업, 하드웨어와의 연관성 그리고 해당 연구를 기반으로 또다른 학습 가능성에 대해 future work를 제시한다. 이후에 다른 포스트에서 따로 다룰 예정이다.