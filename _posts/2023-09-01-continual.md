---
title: Continual learning/Lifelong learning의 개념과 방법론 총정리
layout: post
description: Continual learning/Lifelong learning
use_math: true
post-image: https://github.com/junia3/junia3.github.io/assets/79881119/8d6e9d28-e19a-464d-9c0a-291cc17729f5
category: paper review
tags:
- Continual learning
- Adaptation
- Deep learning
---

> 이 글은 survey 논문인 [A Comprehensive Survey of Continual Learning: Theory, Method and Application](https://arxiv.org/abs/2302.00487)를 각색 및 요약한 글입니다.
> 

# Continual Learning 이란?

### 인공지능과 유기체의 근본적 차이

인공지능의 주체가 되는 ‘모델’을 중심으로 돌아가는 지성 체계인 intelligent system에서 **학습**은 **주어지는 데이터 환경의 기본**이 된다. 외부의 자극이 달라지게 되면 사람을 포함한 다양한 유기체들은 지속적으로 이를 통해 정보를 수집하고, 기존 knowledge를 수정하는 형태로 업데이트하거나 축적해가는 과정을 거친다. 일반적인 오프라인 학습이 전제된 Static한 모델의 경우 학습 프레임워크과 인퍼런스 프레임워크가 분리되어있는 것을 알 수 있는데,  이는 결국 학습을 위해 일반화가 가능할 정도의 big data를 train domain으로서 정의한다는 점에서 명확한 한계를 가질 수 밖에 없다. 예컨데 단순히 categorize(혹은 classification) 문제만 하더라도, data augmentation이나 여러 모델 일반화를 위한 방법론을 적용하더라고 어찌되었든 ‘정해진 카테고리 내에서의 분류’라는 점과, ‘model representation에 open-world(실생활)의 모든 데이터를 내포할 수 없음’라는 점이 한계라고 볼 수 있다.

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/bf6b9ce3-23b8-46e3-a048-23a152d3a4ab" width="800">
</p>

물론 categorize 문제에서만 드러나는 문제점은 아니고, computer vision의 대표적인 task인 detection이나 segmentation에 대해서도 마찬가지이다. 물론 최근 연구에 따라 closed set을 사전 정의하지 않고 다양한 semantic information에 무관하게 작용할 수 있는 [SAM(Segment-Anything Model)](https://segment-anything.com/)와 같은 연구 방향도 제시되었지만, 결국 근본적으로 학습 데이터의 pool(범위)를 증가시켰을 뿐이지 사람이나 일반적인 유기체들이 하는 것처럼 학습 프레임워크과 인퍼런스 프레임워크가 통합된 구조는 아니라는 것을 알 수 있다.

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/0e458cd3-3c1b-4cfb-9997-354aa8dab631" width="800">
</p>

### Continual learning가 시사하는 문제점 그리고 딜레마

물론 모델 일반화에 사용되는 데이터가 증가할수록 그에 따른 performance도 증가하는 것은 당연하고, 이러한 연구 자체도 앞으로의 인공지능을 위해 발전을 이루어야한다. 하지만 언제까지나 우리가 추구하는 인공지능의 범위는 서비스를 제공하는 주체인 server 뿐만 아니라 edge device까지 포함할 것이고, 그렇게 된다면 모델 일반화에 사용되는 데이터 증가와 함께 수반되는 모델 파라미터의 증가를 감당할 수 없는 것은 당연할 것이다. 따라서 continual learning이 시사하는 문제점은 바로 다음과 같다.

1. 유기체의 knowledge 축적에 대한 natural한 특성을 고려해야한다.
2. 실생활에서 마주하는 인퍼런스 환경은 학습 데이터가 포함하지 못하는 영역들이 더 많다.
3. 따라서 미리 정해진 범위 내에서의 학습이 아닌, 지속적으로 변화하는 상황에 대한 학습법을 논하고자 한다.

Continual learning의 대표적인 문제점은 매우 명확하게도 불안정한 학습법과 관련된 문제, 그리고 기존 representation에 대한 망각 문제로 이어진다. 사람은 강아지란 존재를 알고 있으면서 고양이라는 새로운 개체에 대한 정보를 얻는다고 하더라도 강아지에 대한 정보를 잊어버리지 않는다. 심지어, 어떠한 형태의 새로운 정보는 기존에 가지고 있던 다른 정보의 이해를 도우며 서로 강화되기도 한다. 그러나 딥러닝이 모방한 뉴럴 네트워크 체계는 한정된 수의 parameter를 가지고 intractable한 posterior를 모방하는 과정이기 때문에  필연적으로 습득 가능한 지식에 한계가 있으며, 무엇보다 explicit하게 이전 데이터에 대한 정보가 주어질 수 없는 implicit한 학습법이므로 새로운 정보 습득이 이전 지식에 미치는 영향을 독립화할 수 없다는 단점이 있다. 이를 바로 ‘Catastrophic forgetting’이라고 부르며, 단순히 새로운 정보를 조금 학습했을 뿐인데도 기존 representation이 빠르게 붕괴하는 문제를 일컫는다. 그렇다고 해서 무작성 새로운 정보에 대한 습득을 막을 수는 없는 노릇이다. 간단한 설명을 위해 다음과 같이 면접 상황을 가정해보겠다.

> **면접관 :** A씨와 B씨, 각각 본인의 장단점을 말씀해주세요.
>
> **A :** 저는 새로운 환경에 빠르게 적응할 수 있다는 점이 장점입니다! 하지만 저는 기억력이 단점입니다. 그래도 제가 가진 강점으로 이를 잘 극복할 수 있습니다!
>
> **B :** 저는 가진 경험이 많아, 새로운 환경에서도 이를 토대로 적합한 결정을 내릴 수 있다는 점이 장점입니다! 하지만 저는 적응력이 단점입니다. 그래도 제가 가진 경험으로 이를 잘 극복해낼 수 있습니다!
>

A라는 사람은 새로운 환경에 빠르게 적응할 수 있다는 점이 강점이지만, 만약 기억력이 정말 부족한 사람이라면 (극단적인 경우를 가정하여 1시간으로 하자), 하루 전에 가르쳤던 업무 내용도 기억하지 못하기 때문에 했던 교육을 계속 다시 받아야한다는 점이 문제가 될 수 있다. 반면 B라는 사람은 가진 경험이 많아 지식이 풍부하다는 점이 강점이지만, 새로운 환경에 적응력이 정말 부족한 사람이라면 (극단적인 경우를 가정하여 단순 업무 하나를 가르치는데 2년이 걸린다고 하자), 업무 환경이 급격하게 바뀌는 상황에서는 큰 문제가 될 수 있다.

이처럼 **continual learning**도 딜레마가 존재한다. 새로운 데이터가 포함된 task에 최적화된 알고리즘을 짜게 된다면 기존 데이터가 포함된 tasks에 대한 성능이 바닥을 치게될 것이고, 그렇다고 해서 기존 데이터가 포함된 task를 잊지 않게끔 알고리즘을 짜게 된다면 새로운 데이터가 포함된 task에 대한 적응력이 현저히 줄어들 것이다. 이러한 딜레마를 “**Learning plasticity and Memory stability**”라고 부르며, plasticity는 빠른 적응력에 해당되는 단어, memory stability는 기존 지식 보존과 관련된 단어라고 생각해볼 수 있다.

### Continual learning의 방법론들

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/11bdd195-cf99-4e0b-bc7a-06a3cf99aa89" width="700">
</p>

단순히 plasticity 혹은 memory stability 사이에 균형을 맞춘다는 점만 고려하면, 모든 task 및 domain을 내포하는 continous한 상황 변화에 대해 일반화가 가능한 학습법을 추구하는 것이다. 가장 단순한 approach를 한다고 생각하면 기존 training sample에 새로운 task의 데이터셋을 추가하여 모두 학습하는 방법을 생각해볼 수 있지만, 연산 효율성이나 학습 데이터셋 자체가 가지는 라이선스 및 사생활 문제를 극복할 수는 없다. 만약 정말 continual한 환경에서 내 데이터가 모두 기록된다고 생각하면 마냥 마음 편하게 AI 서비스를 사용할 수 있는 사람이 많지는 않을 것이다. 사실 continual learning은 이러한 단순한 접근법을 사용하지 않고, **사용할 수 있는 자원(Usable Resource)을 최대로 활용하는 선**에서 소개된다. 따라서 continual learning이 추구하는 이상에 있는 방법은 오직 새로운 training sample만 가지고 학습하는 방법이라고 볼 수 있다. 위의 그림 중 **C**에서 볼 수 있듯이 continual learning은 딥러닝 학습 프레임워크에서 어느 부분에서 솔루션을 찾냐에 따라 방법론을 크게 분류할 수 있다. 각각의 방법을 다시 소개하겠지만 대충 한 문장으로 요약해보면 다음과 같다.

- Replay : 기존 task의 샘플을 다시 가져와서 최적화에 활용해보자
- Architecture : 학습되는 네트워크 구조를 적절히 변형하여 필요한 부분만 잘 학습시켜보자
- Representation : 변하는 상황에 대해서 robust한 representation을 잘 학습시켜보자
- Optimization : 기존 representation에 도움이 될 수 있는 방향으로 최적화를 진행해보자
- Regularization : 학습 시 정규화를 통해 학습 안정성을 챙겨보자

---

# Problem statement

논문을 보게 되면 분야에 따라 공통적으로 해결하고자 하는 문제가 있고, 조금 더 디테일하게 들여다보면 논문 자체에서 해결하고자 하는 보다 세밀한 문제를 찾을 수 있다. Continual learning을 메인으로 하는 페이퍼 등등에서 자주 보이는 notation에 대해 정리를 하고 넘어가는 것이 좋을 것 같다. Continual learning은 다양한 data distribution으로부터 학습하는 방법을 소개한다. 이때 학습되는 network를 $\theta$에 의해 parameterized되었다고 언급하고, $t$번째 task의 속하는 학습 배치는 $\mathcal{D}\_{t,b} =\\{\mathcal{X}\_{t,b}, \mathcal{Y}\_{t,b}\\}$로 표현한다. 해당 notation은 일반적으로 continual learning이나 domain adaptation과 같은 task에서 주로 사용되는 편인데, input data인 $\mathcal{X}\_{t,b}$와 그에 상응하는 ground truth(data label) $\mathcal{Y}\_{t,b}$ 를 포함하는 data 묶음을 표현하는 방식이다. 따라서 **data distribution의 변화**라는 상황이 포함하는 범위는 data 영역에서의 shift(fine detail부터 semantic한 부분까지 포함)가 될 수도 있으며 label 영역에서의 shift(long-tail problem, subclass 문제 등등)가 될 수도 있다. Task의 변화 $t \in \mathcal{T} = \\{1, \cdots, k\\}$는 task를 나타내는 지표로 구분할 수 있으며, batch의 변화에 대한 index는 $b \in \mathcal{B}_t$ 로 표현된다. 이를 통해 각각의 “Task”를 training sample인 $\mathcal{D}_t$가 대표하는 distribution인 $\mathbb{D}_t := p(\mathcal{X}_t, \mathcal{Y}_t)$로 명시할 수 있게 된다. Continual learning이 내포하는 범위 자체는 다음과 같이 굉장히 다양하고, 따라서 **“Continual Learning을 공부한다”**라는 점이 생각보다 다양한 setting을 포함한다고 볼 수 있다.

---

# Continual learning의 다양한 시나리오들

딥러닝 모델에 input으로 들어오게 되는 각각의 배치 단위에 따라서, 그리고 task의 구분 여부에 따라 일반적인 continual learning 시나리오를 구분하곤 한다. 각 시나리오를 소개함에 앞서 training/testing에 대한 셋팅은 앞서 소개한 problem statement의 모든 notation을 따르며, **‘요구되지 않는다(Not required)’**라고 표현한 부분은 필수가 아니라는 점에 대해 말하는 것이지, 해당 시나리오에서 무조건 없어야 되는 것은 아니라는 것을 짚고 넘어가도록 하겠다. 또한 굳이 명시되어있지 않다면 각 task는 모두 충분한 갯수의 라벨링된 학습 데이터를 가지며, 이는 곧 continual learning에서 굳이 명시하지 않는다면 supervised learning setting을 가정한다는 점을 시사한다. 아래에 표현한 시나리오의 큰 구분을 제외하고도 low-shot setting, open-world setting, un-/self-supervised setting 등등 다양한 연구들이 제안되었지만 그런 논문들 모두 커버하려면 글이 팔만대장경 급으로 길어지기 때문에 간단하게만 분류하도록 하겠다.

### **Instance-Incremental Learning (IIL)**

모든 학습 sample은 같은 task에 속하며, 배치 단위로 들어온다. 즉 학습 단계에서 단일 task의 sample이 배치 단위로 들어오는 상황이며, $\{\{\mathcal{D}\_{t, b}, t\}\_{b \in \mathcal{B}\_t}\}\_{t = j}$  인퍼런스 단계에서도 동일한 task를 가정한다($\\{p(\mathcal{X}\_t)\\}_{t = j}$). 이는 곧 일반적인 딥러닝 학습 알고리즘에서의 셋팅과 동일한 상황에서 input dataset만 continual하게 들어오는 상황을 가정하며, 단일 task를 가정하기 때문에 task specific identification($t$)이 요구되지 않는다 (Not required).

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/d9445b77-fad1-4694-830d-520579bcb104" width="700">
</p>

### **Domain-Incremental Learning (DIL)**

IIL과는 다르게 단일 task가 아닌 여러 tasks가 존재하는 상황을 전제로 한다. 모든 task들은 동일한 data label space를 가지고 input distribution만 다르다. 예컨데 CIFAR10, CIFAR10-C와 같은 관계라고 생각해볼 수 있다(같은 label space (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)를 가지지만 input distribution만 달라짐). 학습 단계에서는 여러 task의 sample이 배치 단위로 들어오는 상황을 가정하고, 이를 식으로 표현하면 다음과 같다.

\[
    \\{\mathcal{D}\_t, t\\}_{t \in \mathcal{T}}; p(\mathcal{X}\_i) \neq p(\mathcal{X}\_j) \text{ and }\mathcal{Y}\_i = \mathcal{Y}\_j \text{ for } i \neq j
\]

그리고 단일 task가 아님에도 불구하고 인퍼런스 단계에서 IIL과 마찬가지로 task specific identification($t$)이 요구되지 않는다($\\{p(\mathcal{X}\_t)\\}\_{t \in \mathcal{T}}$) (Not required).

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/0e70317d-9883-4ed4-9bbd-11b04906429c" width="700">
</p>

### **Task-Incremental Learning (TIL)**

각 task는 서로 disjoint한 data label space를 가진다(완전히 겹치지 않는 상황). 또한 이 셋팅에서는 task specific identity($t$)에 대한 정보가 training 및 testing에서 제공된다. 학습 단계에서의 상황을 식으로 표현하면:

\[
    \\{\mathcal{D}\_t, t\\}_{t \in \mathcal{T}}; p(\mathcal{X}\_i) \neq p(\mathcal{X}\_j) \text{ and }\mathcal{Y}\_i \cap \mathcal{Y}\_j=\emptyset \text{ for } i \neq j
\]

위와 같으며 인퍼런스 단계($\\{p(\mathcal{X}\_t)\\}\_{t \in \mathcal{T}}$)에서 이번에는 $t$를 사용할 수 있게 된다 (Available).

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/a11a5315-6b7a-44f0-ac4a-a06529a69d5e" width="700">
</p>

### **Class-Incremental Learning (CIL)**

각 Task는 서로 disjoint한 data label space를 가진다(완전히 겹치지 않는 상황). 그리고 Task specific identity($t$)는 오직 학습 시에만 사용 가능하다.

\[
    \\{\mathcal{D}\_t, t\\}_{t \in \mathcal{T}}; p(\mathcal{X}\_i) \neq p(\mathcal{X}\_j) \text{ and }\mathcal{Y}\_i \cap \mathcal{Y}\_j=\emptyset \text{ for } i \neq j
\]

따라서 기본 학습 단계에서는 위의 식을 그대로 따르며, 인퍼런스 단계($\\{p(\mathcal{X}\_t)\\}\_{t \in \mathcal{T}}$)에서는 $t$를 제한하게 된다 (Unavailable).

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/02e85bd5-8d2b-4e78-b044-61b292a22869" width="700">
</p>

### **Task-Free Continual Learning (TFCL)**

각 Task는 서로 disjoint한 data label space를 가진다(완전히 겹치지 않는 상황). 그리고 Task specific identity($t$)를 학습 그리고 인퍼런스 단계 둘 중 하나에서 절대로 사용할 수 없다.

\[
    \\{\\{\mathcal{D}\_{t, b}, t\\}\_{b \in \mathcal{B}\_t}\\}\_{t \in \mathcal{T}}; p(\mathcal{X}\_i) \neq p(\mathcal{X}\_j) \text{ and }\mathcal{Y}\_i \cap \mathcal{Y}\_j=\emptyset \text{ for } i \neq j
\]

따라서 기본 학습 단계에서는 기존 식과는 다르게 task 구분이 없이 배치 단위로 input을 구분하게 되며, 인퍼런스 단계($\\{p(\mathcal{X}\_t)\\}\_{t \in \mathcal{T}}$)에서는 $t$를 일부 상황에서 제한하게 된다 (Optionally available).

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/3cd3dfa3-ae34-4359-9e67-d98d47901353" width="700">
</p>

### **Online Continual Learning (OCL)**

각 Task는 서로 disjoint한 data label space를 가진다(완전히 겹치지 않는 상황). 그리고 학습 시 각 task의 모든 샘플은 one-pass로 최적화에 사용된다 (한번 최적화에 사용된 데이터 배치는 다시 input으로 사용되지 않음).

\[
    \\{\\{\mathcal{D}\_{t, b}, t\\}\_{b \in \mathcal{B}\_t}\\}\_{t \in \mathcal{T}},\lvert b \rvert = 1; p(\mathcal{X}\_i) \neq p(\mathcal{X}\_j) \text{ and }\mathcal{Y}\_i \cap \mathcal{Y}\_j=\emptyset \text{ for } i \neq j
\]

따라서 굳이 task를 구분하지 않더라도 모든 배치는 single-stream framework를 따르기 때문에 task에 무관한 학습이 된다.  그리고 인퍼런스 단계($\\{p(\mathcal{X}\_t)\\}\_{t \in \mathcal{T}}$)에서는 $t$를 일부 상황에서 제한하게 된다 (Optionally available).

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/0f93ad65-9aab-420e-8428-e096dabeed8a" width="700">
</p>

### **Blurred Boundary Continual Learning (BBCL)**

Task의 경계선이 애매한(blurry) 상황을 가정하고, disjoint(교집합이 아예 없는) data label이 아닌 일부 겹치는 상황에 대한 시나리오를 의미한다.

\[
    \\{\mathcal{D}\_t, t\\}_{t \in \mathcal{T}}; p(\mathcal{X}\_i) \neq p(\mathcal{X}\_j), \mathcal{Y}\_i \neq \mathcal{Y}\_j \text{ and }\mathcal{Y}\_i \cap \mathcal{Y}\_j=\emptyset \text{ for } i \neq j
\]

즉 data label space가 일부 겹쳐야한다는 점만 제외하면 class-incremental learning과  모두 동일하다. 따라서 인퍼런스 단계($\\{p(\mathcal{X}\_t)\\}\_{t \in \mathcal{T}}$)에서는 $t$를 사용할 수 없다(Available).

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/3047ed20-006d-4ec4-a632-e3378c8d4df0" width="700">
</p>

### **Continual Pre-training (CPT)**

사전 학습되는 data가 continual(sequence)로 도착하는 상황을 의미한다. 주된 목적은 downstream task의 성능을 올리는 것이다(good representation).

\[
    \\{ \mathcal{D}\_t^{pt}, t\\}\_{t \in \mathcal{T}^{pt}}, \text{ followed by a downstream task }j
\]

이번에는 pre-train 단계를 continual한 상황으로 가정했기 때문에 testing 단계에서는 단일 downstream task($t = j$)를 기준으로 삼게 된다. 따라서 downstream task의 task label인 $\\{p(\mathcal{X}\_t)\\}\_{t = j}$는 굳이 요구되지 않는다 (Not required).

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/be27724c-8ce4-4094-8db6-eb37f2e83bac" width="700">
</p>

---

# Continual Learning에서의 Evaluation metric

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/4642ae0e-1f36-4414-92d6-8934f3b6ad92" width="600">
</p>

특정 방법이 다른 방법보다 **‘유리하다’.** 혹은 contribution이 충분하다고 증명할 수 있는 방법은 performance가 우수한 것을 보이는 것이다. 이처럼 deep learning에서 성능 평가를 내릴 수 있는 각 상황별 evaluation metric이 굉장히 중요하게 작용하는데, continual learning과 같이 여러 상황이 동시에 주어지는 환경에서는 evaluation을 어떤 방식으로 진행할까? 앞서 continual learning을 소개할 때 간단하게 언급했던 딜레마인 “**Learning plasticity and Memory stability**”을 떠올려보자. Continual learning이 가지는 목적은 결국 **‘이전 representation을 얼마나 잘 유지하면서 현재 task의 성능을 잘 올릴 수 있는가?’**로 정리할 수 있다.  따라서 continual learning에서의 metric은 세가지 측면으로 정리될 수 있다:

1. 기존(Old) task들의 성능 평가 **: Overall performance**
2. 기존(Old) task에 대한 memory stability **: Memory stability**
3. 새로운(New) task의 성능 평가 **: Learning plasticity**

언뜻 보면 결국 1번이 2번이랑 같은 의미 아니냐고 할 수 있지만 각각의 evaluation에 대해 명확하게 살펴보면 다음과 같다.

### Overall performance

기존 task들의 성능 평가는 보통 avaerage accuracy (AA)나 average incremental average (AIA)로 나타낸다. 예컨데 $k$번째 task까지 incremental learning 이후 $j$번째 task의 accuracy 측정 값을 $a_{k, j} \in [0, 1]$ 이라고 하자. $a_{k, j}$를 측정하기 위해 상정하는 output space는 $j$번째 class만 포함할 수도, 아니면 
$j$번째 class까지의 모든 output space를 포함할 수도 있다. 예컨데 classifier head를 하나만 사용하는 CIL의 경우에는 $\cup_{i=1}^k \mathcal{Y}_i$를, classifier head를 여러 종류를 사용하는 TIL의 경우에는 $\mathcal{Y}_j$를 output space로 잡는다. 이때의 AA와 AIA는 다음과 같은 식으로 정리된다.

\[
    \text{AA}\_k = \frac{1}{k} \sum\_{j=1}^k a_{k, j},~\text{AIA}\_k = \frac{1}{k}\sum\_{i=1}^k \text{AA}_i
\]

### Memory stability

기존 task를 얼마나 잘 기억하고 있는지 측정하는 지표로는 보통 forgetting measure (FM)이나 backward transfer (BWT)를 사용한다. 전자의 경우 각 task에 대한 잊음(forgetting) 정도 $f_{j, k}$를 **과거에 획득한 성능 중 최대치**와 **현재 성능과의 차이**를 통해 계산한다.

\[
    f_{j, k} = \underset{i \in \\{1,\cdots, k-1\\}}{\max} (a_{i,j} - a_{k, j}),~\forall j < k
\]

그리고 $k$번째의 FM은 old tasks에 대한 forgetting 전반에 대한 평균으로 측정된다.

\[
    \text{FM}\_k = \frac{1}{k-1} \sum\_{j=1}^{k-1} f_{j, k}
\]

후자인 BWT의 경우에는 $k$번째 task가 이전 $k-1$번째 task까지의 성능에 주는 영향을 평균 내어 계산한다.

\[
    \text{BWT}\_k = \frac{1}{k-1}\sum\_{j=1}^{k-1} (a_{k, j}-a_{j, j})
\]

보통의 경우 $k$번째 task의 학습 이후 측정된 accuracy $a_{k, j}$가 각 task 학습 직후 측정된 accuracy $a_{j, j}$보다 낮기 때문에 negative BWT의 정도가 forgetting을 반영한다고 해석할 수 있다. 

따라서 overall performance, memory stability 모두 old task에 대한 성능을 반영하는 점에서 공통점을 가지지만, **overall performance**는 **성능 자체**를, **memory stability**는 **성능 변화**를 기록한다는 점에서 차이점이 있다.

### Learning plasticity

새로운 task를 얼마나 빠르게 학습하는지에 대한 지표가 되는 learning plasticity는 일반적으로 instransience measure (IM) 그리고 forward transfer (FWT)로 측정한다. IM은 네트워크가 새로운 task를 잘 배우지 못하는 정도를 의미하게 되고, joint training performance와 continual learning performance 사이의 차이로 정의한다.

\[
    \text{IM}\_k = a^\ast\_k - a\_{k, k}
\]

$a_k^\ast$는 기준이 되는 모델을 $k$번째 task까지의 데이터 $\cup\_{j=1}^k \mathcal{D}\_j$로 jointly(한꺼번에) 학습시켰을 때의 성능을 의미한다. 성능은 오직 $k$번째 데이터셋에 대해서만 구하게 된다. Jointly 학습시켰을 때보다 continual하게 학습을 시킬 경우 plasticity가 낮다면 해당 task에 대한 성능이 줄어들 것이기 때문에$(a^\ast_k > a\_{k, k})$ , 해당 지표가 클수록 **새로운 task에 잘 적응하지 못하는 정도로서 측정되는 것**이다. FWT는 이와는 반대로, 모든 old task가 $k$번째 task에 미치는 영향력을 평균으로 측정하게 된다.

\[
    \text{FWT}\_k = \frac{1}{k-1} \sum\_{j=2}^k (a\_{j,j} - \tilde{a}\_j)
\]

$\tilde{a}\_j$는 기존이 되는 모델을 $j$번째 task 데이터셋 $\mathcal{D}\_j$에만 학습시켰을 때의 성능을 의미한다. $k-1$번째까지 학습된 모델의 posterior가 prior로서 $k$번째 task를 마주하는 상황이 되었을 때, 해당 지표가 클수록 **이전 task가 현재 task의 performance에 미치는 영향력이 크다**고 볼 수 있다.

---

# Continual learning Methods

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/13409791-10cf-42c5-80ae-21b1fdfc27a8" width="1100">
</p>

결국 continual learning의 성능을 재는 방식은 $k$번째 task 자체에 대한 performance와, performance 변화 그리고 실제로 continual learning 방식의 효과성에 대한 입증으로 구성된다. 이를 위해 다양한 방향의 연구가 진행되었으며, 각각에 대해 윗부분에서 한 문장으로만 요약하고 넘어왔었다. 이를 도식화한 것이 바로 위에 보이게 되는 tree 구조가 되는데, 각 부분의 연구는 개별적으로 활발히 진행되었고, 모델 학습에서 타겟되는 부분에 따라 구분할 수 있었다. 모든 논문들을 이 글에서 다루기는 무리가 있을 것 같고, 간단한 논문 몇 개만 정리하는 것을 목표로 하는 중이다.

### Regularization based approach

일반적으로 생각해볼 수 있는 방법은 바로 old task와 new task 간의 균형을 위해 explicit한 regularization term을 더해주는 것이다 (아래 그림 참고).

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/d37cfd44-a611-4a62-8f30-042e14f399c5" width="750">
</p>

그림을 보면 알 수 있겠지만 정규화가 consistency하게 보고자 하는 위치에 따라 크게 두 방향으로 구분되는데, 어찌되었든 정규화를 위해서 공통적으로 old task에 대해 학습된 frozen model을 reference로 가지고 있어야한다는 점은 변하지 않는다.

정규화의 가장 첫번째 방법론은 weight regularization으로, 네트워크 파라미터의 변화를 선택적으로 정규화를 하는 방법이 된다. 일반적인 구현은 loss function에 각 parameter의 변화에 대한 quadratic penalty $(\lvert \theta - \tilde{\theta} \rvert^2)$를 추가한다던지, 그게 아니라면 각 parameter의 중요도(contribution/importance)에 따라 penalty를 주는 방법이 있을 수 있다. 예컨데 가장 대표적인 방법으로는 각 파라미터가 구현하는 probability surface $p(\Theta)$를 라플라스 근사하는 방법이 있는데, 이를 이해하기 쉽게 풀어서 설명해보도록 하겠다. 모든 deep neural network가 구성하는 함수는 특정 distribution에 대한 input/output 관계로 규명된다. 예컨데 우리가 간단한 classification task를 하고자 한다면 input image $X$에 대해 대응되는 output label $Y$가 있을 것이고, 해당 관계는 probabilistic modeling인 $p_\Theta(Y \vert X)$로 정의된다. 이를 곧 parameter에 대한 posterior로 표현하곤 한다. 뉴럴 네트워크는 사실상 $L$개의 연속된 레이어가 이전 레이어의 output에 대해 distribution mapping을 하는 구조로 구성되었을 것이고, 따라서 우리는 네트워크 전체가 아니라 이를 각 파라미터 단위로도 생각해볼 수 있다 $p_{\theta^l}(Z_{l} \vert Z_{l-1})$. 하지만 고차원의 뉴럴 네트워크를 다룰 때 ground truth가 되는 각 latent modality의 distribution을 우리가 직접 알 수는 없다는 점이 real-world에 대한 근본적 어려움으로 작용한다. 하지만 만약 log-likelihood를 전체 dimension에 대해 구하지 않고 라플라스 근사를 하게 되면, 적어도 우리는 해당 분포의 local point에서의 분포 형태를 가우시안 분포로 구할 수 있게 된다. 바로 이러한 관점에서 탄생한 것이 [EWC](https://arxiv.org/pdf/1612.00796.pdf)라는 논문이며, 실제로는 라플라스 근사를 하는 과정에서 필요한 헤시안 matrix를 직접 구하는 방법 대신 FIM(Fisher Information Matrix)을 통해 간접적으로 구하게 된다.

\[
    H(D\_k, \mu\_k) \approx \mathbb{E}\left(\nabla\_\theta \log p(\mathcal{D}\_k \vert \theta^l)\nabla\_\theta \log p(\mathcal{D}\_k \vert \theta^l)^\top\right)\vert\_{\theta = \mu\_k}
\]

그렇게 되면 FIM은 결국 log-likelihood를 가우시안 근사를 했을때 분포의 curvature 정보를 담게 되는데, 이를 토대로 중요한 parameter와 중요하지 않은 parameter를 구분할 수 있는 척도가 된다. 물론 이처럼 weight importance를 정하는 방식이 FIM에만 방법론이 고정된 것은 아니다. [Online EWC](https://arxiv.org/pdf/1805.06370.pdf)에서는 새로운 task에 대한 학습 및 이를 기존 representation에 추가하는 모듈 형태를 제시하며 online 환경에서 task agnostic하게 업데이트될 수 있는 weight importance 방식을 제안하였으며, [Synaptic Intelligence(SI)](https://arxiv.org/pdf/1703.04200.pdf) 논문에서는 실제 다음과 같이 parameter에 따른 biological한 plasticity framework를 딥러닝 학습 과정에 도입함으로써 regularization을 진행한다.

\[
\Omega_k^\mu = \sum_{\nu < \mu} \frac{\omega_k^\nu}{(\Delta_k^\nu)^2 + \xi}
\]

Denominator(분모) term에 존재하는 $\Delta_k^\nu$는 task index $\nu$를 기준으로 각 파라미터의 변화에 따른 trajectory를 의미하며, 온라인 환경에서 $\omega_k^\nu$는 각 parameter의 loss에 대한 gradient에 parameter 변화를 곱한 값을 지속적으로 accumulation해서 구하게 된다. 요약하자면 각 파라미터 단위로 정리되는 중요도는 loss function에 대해 parameter의 기여도를 구한 값을 catastrophic forgetting을 방지하기 위해 parameter 변화의 trajectory로 정규화한다고 볼 수 있다. 이러한 방법들은 모두 weight parameter에 대해 quadratic penalty term을 간접적으로 사용한다는 점에서 공통점을 가진다.

이러한 penalty term을 사용하지 않고 [factorized rotation을 기반](https://arxiv.org/abs/1802.02950)으로 parameter space를 FIM에diagonalize하는 방법이나, 

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/9f54fcc4-e315-4b12-a482-7d03a1cea77f" width="700">
</p>

Batch normalization이 포함된 구조에 유리한 [Kronecker-Factored Approximate Curvature](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lee_Continual_Learning_With_Extended_Kronecker-Factored_Approximate_Curvature_CVPR_2020_paper.pdf)이 제안되기도 하였다. 하지만 이런 기존 방법들은 모두 “이전에 학습된 old task parameter”를 기준으로 한다는 점에서 parameter 변화를 막는다는 공통적인 constraint를 가지고, 이는 곧 새로운 task에 adaptation이 적용되는 과정에서 보수적인 효과를 불러온다. 이러한 문제점을 해결하고자 expansion 및 renormalization 방법이 제안되기도 하였고, 이는 new task solution을 독립적으로 obtain한 뒤에, 이를 old model에 재배치하는 형태로 구현이 된다. [IMM(Incremental Moment Matching)](https://arxiv.org/pdf/1703.08475.pdf)이 초기 approach에 있는 논문인데, 논문 제목이랑 아래 figure에서도 볼 수 있듯이 old task와 new task 간의 moment matching을 통해 점진적으로 새로운 task의 representation을 추가해가는 전략을 취한다.

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/cfbf3e66-9cbc-4d08-9391-dc5949f42568" width="700">
</p>

이후 논문으로는 [ResCL](https://arxiv.org/abs/2002.06774)(IMM에 추가로 combination coefficient를 학습 가능하게 바꾼 것), Online EWC로도 유명한 [P&C](https://arxiv.org/pdf/1805.06370.pdf)을 볼 수 있다. P&C에서 주가 되는 메소드는 추가 network에 학습된 task를 기존 network에 distillation하는 과정인데, 이때 weight consolidation을 formulation하는 과정에서 online EWC를 사용한다. [AFEC(Active Forgetting)](https://arxiv.org/abs/2110.12187) 논문의 경우에는 forgetting rate(시냅스가 새로운 task를 받아들일 때 이전 정보와 conflict되는 정도를 active하게 설정)를 제안하였다. 이는 새로운 knowledge가 transfer되는 과정에서 잠재적으로 기존 representation과 공존했을 때 발생하는 negative transfer 문제를 다루고자 하였다. ResCL이랑 비슷한 approach라고 생각할 수도 있는데, plasticity 및 stability 간의 trade-off(old task와 new task loss 사이의 trade-off) 간의 균형을 위해 low-error path간에 [linear connector를 구성한 방법](https://arxiv.org/abs/2110.07905)도 제안되었다. 물론 parameter의 변화 자체를 규정하기보다는 learning rate를 줄이는 [NPC](https://arxiv.org/pdf/1907.13322.pdf)와 같은 논문들도 제시되었다. Penalty를 통해 간접적으로 weight update를 막는 방법 대신 important neuron의 학습을 막음으로써(freeze) hard regularization을 채택한 방법들도 존재한다.

정규화의 두번째 방법론은 function regularization으로, prediction function의 최종 output 혹은 그 중간의 output을 기준으로 정규화를 진행하는 방법이다. Weight 정규화랑은 관점이 조금 다른게 prediction layer에 대해 consistency를 보는 과정이 되므로 기존 model의 output을 teacher로 삼고 학습 output을 student로 맛아서 knowledge distillation을 진행하면서 새로운 task를 학습하는 것과 같다. 사실 조금 웃긴 관점인게 새로운 지식을 습득하는 주체는 학생이고 오히려 선생은 이전 지식을 유지하도록 가이드한다는 것이다. 가장 대표적으로는 [LwF(Learning without Forgetting)](https://arxiv.org/abs/1606.09282)을 예시로 들 수 있다. 이외에도 생성 모델을 기반으로 모델링하여 feature reconstruction의 변화를 줄이는 방법이라던지 replay-based method와 결합되어 이전 task의 샘플들을 정규화 과정에 도입하는 경우가 많다. Replay based approach는 바로 아래에서추가로 설명하도록 하겠다.  

### Replay based approach

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/99a86858-0814-492c-a79a-adffb32bb660" width="700">
</p>

새로운 task를 학습할 때 old data distribution을 recover하고 approximate하는 방법들도 생각해볼 수 있는데, 각각 replay하는 주체에 따라 세부적으로는 3가지로 구분한다.

첫번째는 **experience replay**다. 이는 작은 memory buffer를 가지고 여기에 약간의 이전 task에 대한 training sample을 저장하는 형태로 구현된다. 물론 당연하게도 저장 공간에는 제약이 있기 때문에, 얼마나 많은 정보를 저장할 것인지에 대한 문제를 해결하는 것이 주된 해결 방향으로 작용한다. 구성하는 과정(construction)에서 샘플 선정이 중요하며, 학습 과정마다 이를 업데이트하는 방식도 고려해야한다. 초창기 연구인 [reservoir sampling 방법](https://arxiv.org/abs/1902.10486)은 가장 간단하게 sample selection 과정마다 고정된 갯수만큼의 old sample을 유지하고 나머지를 랜덤하게 대체하는 방식을 사용한다. [Ring Buffer sampling 방법](https://arxiv.org/abs/1706.08840)은 이를 보다 발전시켜서 old training sample을 새로이 저장하는  과정마다 class마다 같은 샘플 수가 유지되도록 한다. [Mean-of-Feature 방법](https://arxiv.org/abs/1611.07725)에서는 각 class의 특징자 벡터의 평균을 일종의 prototype로 간주하여 이와 가장 closest(유사한) 샘플을 저장하는 방식을 사용하였다.

간단한 예시들을 열거하는 형태로 소개했는데, 이러한 내용들을 간단하게 보면 알 수 있듯이 샘플 선정 기준은 따로 정해진 것은 없고 empirical한 것을 알 수 있다. 예컨데 $K$-means를 사용할 수도 있으며, plane distance나 샘플의 entropy를 기준으로 thresholding할 수도 있다. 그러나 이러한 방법들은 모두 어느 정도 적당한 성능 선에서 가능성만 보여주었다. 이보다 좀 더 발전한 형태로는 gradient 기준으로 다양성을 최대화하는 방법이나([Gradient based sample selection](https://arxiv.org/abs/1903.08671)) entity간의 관계 조건인 cadinality constraint를 task performance와 [연관짓는 방법](https://arxiv.org/pdf/2006.03875.pdf), batch 단위로 gradient 유사성을 보는 방법이 있다.  또한 최적화가 가능한 방법으로 latent decision boundary를 조정하는 등 고정되지 않은 알고리즘으로서 정의된 approach가 발전하기 시작했다.

이와 parallel하게 샘플 저장 효율성과 관련된 방법도 동시에 발전하기 시작했는데, vector quantize 방법을 포함하여 sample point를 압축하여 저장하기도 했으며 augmentation을 사용하여 한정된 샘플에 다양성을 부여하기도 하였다. 추가적으로는 의도적으로 forgetting boundary(잊기 쉬운 샘플들)을 adversarial하게 합성하여 replay 효과를 높이는 방법도 존재한다.  이후에도 많은 연구들이 진행되었다. 그런데 여기서 들 수 있는 의문은 “그럼 굳이 샘플 저장하지 말고 feature 저장하면 안되는가?”인데, 이게 바로 다음에서 소개할 feature replay에 해당된다.

**Feature replay**는 experience replay에 비해 용량 측면에서 이점이 있으나 feature extractor가 업데이트되면서 같은 샘플에 대해서도 representation이 달라진다는 문제가 발생한다. 이는 곧 feature level에서의 replay가 catastrophic forgetting 위험성을 수반하는 이야기로 흘러가게 된다. 이러한 문제를 해결하려는 방법으로 old model과 new model 간의 feature distillation을 사용하게 된다. 또다른 approach는 experience replay를 기준으로 feature representation(평균이나 표준편차와 같은 통계)를 복구하는 방법을 사용하기도 한다. [RER 논문](https://openaccess.thecvf.com/content/CVPR2022/papers/Toldo_Bring_Evanescent_Representations_to_Life_in_Lifelong_Class_Incremental_Learning_CVPR_2022_paper.pdf)에서는 old sample의 representation을 저장해두고 이를 업데이트하면서distribution shift를 예측하는 방식을 사용한다. 이외에는 초기 layer를 고정시킨 채 중간 feature를 추출하여 이를 후반 layer를 업데이트하는 과정의 replay sample로 사용하는 방법도 있다.

마지막 방법으로는 **Generative replay**가 있는데, 이는 old task의 학습 데이터를 생성 모델을 사용하여 replay에 사용하는 전략을 취한다. 예컨데 새로운 task에 대해 생성된 데이터와 old task에 대해 생성된 데이터 간의 consistency를 사용한다던지 하는 방식이 있을 수 있다. VAE, GAN과 같은 다양한 구조가 사용된다. 방법 자체에 큰 차이는 없어서 간단하게 이쯤 언급하도록 하겠다. 결국 replay-based approach의 공통점은 task가 학습되면서 기존 학습에 사용되던 input sample 혹은 representation에 대한 정보를 직/간접적으로 사용하여 catastrophic forgetting을 방지한다는 것이다.

### Optimization based approach

앞서 소개했던 regularization 방법이나 replay는 결국 기존에 학습된 데이터를 implicit하게 혹은 explicit하게 활용할 수 있는 방법이었다. 이러한 “기존” 이라는 키워드에서 벗어나 explicit하게 최적화 구조를 바꾸거나 조정하는 방법이 소개되었으며, 이를 곧 optimization based approach라는 큰 틀로 묶을 수 있다.

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/18ddd51f-0828-45f8-91b8-ffe0ce6df738" width="700">
</p>

가장 흔한 아이디어는 gradient projection(그래디언트 사영)에 해당된다. 몇몇의 replay-based approach는 experience replay의 업데이트 방향에 따라 parameter update를 align하는 방식을 채택하였는데, 이를 통해 새로운 task에 대한 parameter 업데이트가 이루어질 때 기존 input이 구성하는 implicit한 공간 및 gradient 공간을 유지하는 효과를 가지고 올 수 있었다. 이러한 컨셉에서 replay라는 관점을 제거하여, [OWM](https://arxiv.org/abs/1810.01256)이나 [AOP](https://ojs.aaai.org/index.php/AAAI/article/view/20634)에서는 parameter update를 previous input space를 기준으로 orthogonal한 방향(input space를 유지하는 방향)으로 업데이트하는 전략을 취했다.  두 방법은 input space에 대한 orthogonality를 전략으로 취한 반면, [OGD](https://arxiv.org/abs/1910.07104)는 기존 학습 시의 parameter update를 보존한 뒤, 이후 task의 gradient optimization 방향을 이에 orthogonal한 방향으로 조정하는 전략을 통해 input space 대신 gradient space를 사용하게 된다. 이전에 소개했던 regularization based method인 Bayesian weight regularization과 gradient projection을 통합시킨 논문도 소개되었다.

여러 task에 대해 robust한 optimization이라고 하면 비슷한 관점으로 “meta-learning”을 떠올려볼 수도 있다. 아니나 다를까 meta-learning도 continual 방법론에 속하게 된다. [OML](https://arxiv.org/abs/1905.12588)은 메타러닝 학습 프레임워크를 제안함으로써 연속적으로 계산되는 샘플 input에 대해 학습 과정에서의 간섭을 줄이면서 online update에 도움이 되는 방법을 제안하였다. [ANML](https://arxiv.org/abs/2002.09571) paper에서는 이를 보다 확장하여 점진적으로 증가하는 task에 대해 context-dependant 정보를 함수화하여 특정 뉴런을 활성화하는 방식으로 학습을 진행하였다.

다른 방법론으로 고려해볼 수 있는 것은 generalization(일반화) 관점인데, 바로 loss landscape를 안정적으로 만드는 것이다. Task 및 Domain 관점에서 loss landscape가 안정적인 형태(curvature가 낮은, flat한 local minima)를 가질수록 adaptation에 도움이 된다는 관점에서 출발하게 된다(아래 그림 참고).

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/273e5cf2-917a-4252-affb-32468983fda3" width="700">
</p>

이러한 loss landscape를 고려하는 방법들은 대부분 optimizer에 관한 연구로 구성되며 [SGD](https://arxiv.org/abs/2006.06958)나 [Adam optimizer](https://arxiv.org/abs/2103.07113)에 솔루션을 제공한다.

### Representation based approach

사실 일반화 관점이라면 한번 더 고려해볼 수 있는 것이 robust representation이다. Domain generalization에서 얻고자 하는 효과에 가까운 솔루션이라고 볼 수 있다. 각 task에 specific한 representation이 아닌 일반화에 가까운 representation을 얻는 과정을 sparse(넓은 범위의 확률 분포를 커버하는) representation이라고 부른다.

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/a35ebc82-ddef-46a7-ab2c-3e525a34bd5a" width="700">
</p>

최근 meta-training이나 self-supervised learning(SSL)이 보여주는 promising한 결과들을 기반으로 많은 연구가 추가로 진행되었다. 최근 들어 transformer 기반 모델이 발전하기 시작하면서 large-scale pre-training의 효과 또한 입증되기 시작하였고, 결국 adaptation 관점보다는 generalization 관점에서 보다 넓은 범위의 representation을 커버하자는 의도의 approach 또한 continual learning에 제안되기 시작하였다. SSL이나 large-scale pre-training은 사실상 서로 같은 목적 및 학습 형태를 공유하는 점이 많고, 이는 large scale dataset이 라벨링이 힘들다는 점을 들 수 있다. 차이점이라고 한다면 SSL task는 주로 downstream task로 approch의 당위성을 증명한다.

SSL을 사용한 방법들로는 다음과 같은 논문들이 제안되었다. SSL approach를 사용한 논문들의 공통점은 “SSL로 학습된 representation”일수록 catastrophic forgetting에 강인하다는 점이다.  [LUMP](https://arxiv.org/abs/2110.06976)는 old task와 new task 간의 representation이 연속이라는 가정 하에 interpolation을 진행하는 방식을 사용하였다. [MindRed](https://arxiv.org/abs/2203.12710)는 저장된 old training sample을 사용하여 replay experience를 기존 representation으로부터 decorrelate시키는 방법을 사용했고, 이는 곧 새로운 task를 통한 학습이 보다 다양한 데이터로부터 오는 일반화 효과로 접근한 것을 알 수 있다. 다른 방법들로는 self-supervised loss를 distillation 방법론과 결합하여 현재의 representation을 이전 state로 mapping하거나 모델 간의 mapping을 하는 전략을 취하였다.

Pre-training을 도입한 방법들은 주로 continual learning 과정에서 얻는 representation의 이점보다는 사전 학습된 representation을 활용하여 여러 downstream continual setting에서 좋은 성능을 내고자 하는 것이 목적이다. 주로 대량의 데이터셋에 대해 학습된다던가, 보다 큰 파라미터 수를 가지는 larger model에 대해 학습된다던가 아니면 contrastive loss 등등 SSL 전략들과 함께 사용되었을 때 좋은 성과를 보였다. 이러한 방법론에서 가장 주된 문제점 및 해결 사항으로 제안되는 것이 이렇게 사전 학습된 representation을 continual 환경에서 어떻게 잘 전달하느냐인데, 이러한 전략은 사전 학습된 backbone 파라미터가 고정된 경우와 고정되지 않은 경우로 나뉜다. 고정된 backbone을 가지는 경우 parameter-efficient tuning 전략인 [Side-Tuning](https://arxiv.org/abs/1912.13503)과 같이 기존 parameter와 parallel하게 학습될 수 있는 방법이 제안되었다. [TwF](https://arxiv.org/abs/2206.00388)(Transfer without forgetting)은 마찬가지로 별도의 네트워크를 학습하는데, backbone의 knowledge를 레이어별로 distillation한다는 전략을 취한다. 이외에도 생성 모델의 representation의 각 layer로부터 task-specific 파라미터를 학습하는 구조나, Adapter를 기반으로 사전 학습된 transformer를 tuning하는 방법이 제안되기도 했다.

앞서 continual learning과 관련된 여러 task를 소개하는 과정에서 continual pre-training (CPT)를 언급했었는데, 사전 학습 시 사용되는 대용량의 학습 데이터를 일반적으로는 incremental한 방법으로 획득되다보니 마찬가지로 학습 과정에서 continual하게 학습한 뒤 이에 downstream performance가 향상되도록 하는 것이 주요 목적이다. 기존 연구 결과들에 따르면 continual learning을 진행하는 VL model에 대해 supervised 학습법보다 self-supervised 학습법이 promising한 결과를 보인다는 결과가 나타났다.

### Architecture based approach

위에서 언급한 방법들은 대부분 공유된 parameter(단일 model을 하나의 파라미터 단위로 생각했을 때)를 여러 task에서 잘 활용하는 방법에 대한 해결책이었다. 그러나 이렇게 단일 파라미터를 여러 task에서 employ할 경우 근본적으로 interference 문제를 해결하기 어렵다는 단점이 있다.

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/0fb6fabe-2b75-4d73-85ec-86e7e63779c2" width="700">
</p>

이런 기존 방법들과는 다르게 task-specific parameter를 공유되는 파라미터와 독립적으로 학습함으로써 간섭 문제를 해결하고자 한 방법이 바로 architecture로 접근한 논문들이다. Approach는 네트워크 구조가 고정되었느냐 아니냐에 따라 parameter-isolation 방법과 dynamic architecture 방법으로 구분된다. 그러나 최근에는 이런 식으로 분류하지 않고 parameter allocation, model decomposition 그리고 modular network 이렇게 세 가지로 분류해서 보는 듯하다.

**Parameter allocation**은 parameter isolation을 확장시킨 개념으로, 네트워크 전반에 걸쳐 각 task에 기여하는 parameter 부분집합(subspace)를 분리한다. 이때 architecture는 고정되어있을 수도 있으며 dynamic하게 변할 수도 있다. 고정된 네트워크 구조를 차용한 여럿 방법들의 경우에 학습하고자 하는 뉴런이나 파라미터를 각 task마다 분리해줄 수 있는 [binary mask를 explicit하게 학습하는 형태](https://www.notion.so/Continual-Learning-92b3c34e1e46474981dda12613c4639d?pvs=21)를 취한다. 이를 조금 다르게 틀어서 중요한 뉴런이나 파라미터자체를 identify하는 방법들이 사용되기도 하는데, [iterative pruning](https://arxiv.org/abs/1711.05769), [activation value](https://arxiv.org/abs/1903.04476) 혹은 [uncertainty estimation](https://arxiv.org/abs/1905.11614)이 이 방법들에 속한다. 네트워크가 담을 수 있는 용량이 한정되어있기 때문에, task가 진행되면 진행될수록 freeze하지 않은 parameter의 saturation이 발생하고, 이는 곧 획득할 수 있는 성능에 한계점이 있다고 볼 수 있다. 필연적으로 네트워크 파라미터를 적게 학습하는 전략을 취함으로써 성능 수렴으로 인해 발생하는 trade-off가 딜레마로 작용하는 상황이다. 이러한 문제를 줄이고자 dynamic하게 architecture를 변화하는 방법들(특히 expanding의 방향으로)이 제안되기 시작하였고, 이는 기존 네트워크 파라미터의 수용력이 새로운 task를 받아들일 정도로 충분치 않을 때 사용하기 적합하다.

**Model decomposition**은 model을 task-sharing(task 변화에도 무관한 파라미터)와 task-specific(task에 특성화된 파라미터) 성분으로 구분하는 approach다. 보통 task-specific component는 앞선 연구의 흐름에 따라 확장 가능한 network를 가정하는 것이 일반적이다. Task specific components를 구성하는 구조로는 parallel branches인 [ACL](https://arxiv.org/abs/2003.09553)과 같은 방법이라던지 adaptive layer인 [GVCL](https://arxiv.org/abs/2011.12328)이 주로 알려져있으며, 중간 feature map에 대한 [mask를 생성하는 generator를 고안](https://proceedings.neurips.cc/paper/2020/hash/b3b43aeeacb258365cc69cdaf42a68af-Abstract.html)하는 방법도 제안되었다. Feature mask를 model decomposition에 사용하는 것은 parameter spcae에서 동작하거나 앞서 parameter allocation에서 간단하게 언급했던 binary mask의 형태는 아니기 때문에 parameter allocation과는 좀 다르다고 할 수 있다.

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/aba52a4e-881a-4cac-80f1-1b578904327e" width="700">
</p>

**Modular network**는 parallel한 sub-network나 sub-module을 기존 네트워크 구조 상에 제안한 형태이다. Progressive Network 논문에서는 각 task마다 동일한 sub-network를 제안하고 adaptor connection으로부터 서로 다른 sub-network 끼리의 knowledge transfer를 수행하는 방법을 제안하였다. 다른 approach에서는 여러 parallel branch를 통해 candidate path(task 학습에 따른 경로를 의미함)을 설계하는 방식을 취한 뒤 가장 최적의 경로를 선택하는 접근도 포함한다. 후보군을 모집한다는 개념에서 [여러 sub-네트워크간의 앙상블](https://arxiv.org/abs/2207.06543)을 사용하기도 한다.

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/3f54ff8e-d63c-4351-9800-d3cddd414c7a" width="700">
</p>

---

# 결론 및 요약

딥러닝에는 정말 다양한 task들이 있고 이는 곧 딥러닝을 활용한 알고리즘을 실생활에 가깝게 묘사하기 위해 여러 가지 상황에 대한 assumption이 필수적이기 때문이라고 생각이 된다. 며칠 전 양자 컴퓨터와 관련된 영상에서, 지금의 binary 컴퓨터는 실제 자연을 전혀 묘사하지 못하기 때문에 같은 task를 수행하더라도 리소스가 천문학적으로 필요하다고 보았다. 아무튼 그만큼 전세계 인공지능 관련 연구자들이 인간이 아닌 환경에서 인간이 내리는 사고를 묘사하기 위해 많은 노력을 기울이고 있는 것 같다.

Continual learning은 실제로 뇌가 인지하고 사고하는 방식과는 다르다. 왜냐하면 딥러닝 모델이 학습하는 것은 경험에 대한 기억 그 자체가 아니라 해당 경험을 mapping할 수 있는 뉴런 사이의 representation이기 때문이다. 그렇기 때문에 보다 망각하기 쉽고, 가르치기 어렵다는 문제에 직면한다. Continual learning에서 주된 목적은 “이전에 가르친 내용을 망치지 않으면서 새로운 내용을 잘 집어넣는 방법”이다. 최근 Machine Unlearning이라는 새로운 task에서는 이러한 continual learning의 역과정에 대해서 다루는 듯하다. 딥러닝 모델은 파라미터 value가 곧 학습된 내용에 해당되기 때문에 이를 잘 분석하는 것이 곧 우리가 현재 상황에서 수행할 수 있는 최선의 explainable AI일 것이다.

가장 간단한 방법인 regularization부터 베이시안 모델링이나 생성 모델로부터 출발하여 수학적으로 접근한 여러 방법들, weight consolidation이나 parameter allocation과 같이 task에 따른 파라미터 분류를 통해 네트워크의 일부분만 학습하는 방법도 있었으며, 딥러닝 모델 구조는 처음부터 끝까지 한정적인 형태여야만 한다는 고정관념에서 벗어나 sub-module이나 parallel module, sub network를 기반으로 한 앙상블이나 knowledge transfer 등 간접적인 지식 전달을 목표로 하는 approach도 제안되었다. 가장 놀라웠던 점은 이번 글을 작성하면서 느꼈지만, 단순한 하나의 approach도 어떻게 생각하냐에 따라 굉장히 많은 솔루션으로 탄생할 수 있다는 점이었다. 어쩌다보니 급하게 continual learning을 공부하면서 얼렁뚱땅 정리를 하는 형태가 되었다. 나는 원래 continual learning을 하는 사람이 아니라서 이 글이 얼마나 제대로 된 정리가 되었을지는 모르겠다.