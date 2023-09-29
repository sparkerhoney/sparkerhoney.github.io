---
title: Parameter-free Online Test-time adaptation 논문 리뷰
layout: post
description: Online adaptation
use_math: true
post-image: https://user-images.githubusercontent.com/79881119/228143473-e28161a7-2ec0-4322-bf1b-9f617ab48a6d.gif
category: paper review
tags:
- AI
- Deep learning
- TTA
- Parameter free
---

# 딥러닝 연구의 한계점

**State-of-the-art(SOTA)**인 vision model을 학습하는 것은 연구 목적으로 사용할 때나, 실제 사업에 활용할 때 모두 cost가 많이 드는 작업에 해당된다.
심지어 <U>최근 연구 경향</U>을 보면 알 수 있듯이 보다 resource를 많이 사용하여(네트워크 크기나 데이터셋의 규모 모두에 해당) 다양한 분야에서 뛰어난 성능을 보여주기 시작했다. 하지만 일반적으로 NVIDIA나 Adobe research, FAIR와 같이 연구에 지원되는 리소스가 풍부하지 않은 보통의 연구 환경에서는 아무리 좋은 학습 결과를 보여주는 연구라 하더라도 reproduction이 불가능할 때가 많고, pre-trained network를 학습에 활용하고 싶다고 하더라도 제한된 리소스에 모두 **수용이 안되는 경우**가 발생한다. 그리고 가장 주요한 한계점은 사전 학습된 네트워크 파라미터가 공개되더라도 <U>학습된 데이터셋</U> 자체는 공개되지 않는 경우이다. 이번 논문도 **domain adaptation**의 특별한 케이스를 다루게 되는데, 만약 domain adaptation 환경에서 지속적으로 domain alignment를 위해 source dataset이 필요하다면 **online**(실제 deployment가 진행되는) 환경에서 사용할 수 없다는  뜻이기 때문이다. DA(Domain adaptation)이라는 task가 제안한 것이 딥러닝 네트워크의 학습된 knowledge를 효과적으로 transfer하여 다양한 실생활 예제에 적용될 수 있게 바꾸는 것인데 실제 학습 방법을 보면 그렇지 않다.
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/228143027-4697e2a1-b9e3-40b0-be70-5773b1bda50b.png" width="600">
</p>

---

# 효율적인 knowledge transfer가 필요한 이유

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/228143069-6fcfebdd-1113-45df-b38a-d43f19d1ef05.png" width="400">
</p>

최근 **OpenAI**의 **ChatGPT**가 유명해지기 시작했다. ChatGPT는 <U>Instruct(Human forcing)</U>을 통해 보다 사실적인 텍스트를 생성할 수 있게끔 강화학습을 적용하는 방법인 RLHF를 사용하여 학습하게 된다. 이때 학습의 기준이 되는 LM(Language model)이 그 유명한 GPT-3 혹은 가장 최근에 등장한 GPT-4이며, 연구 경향에 따라 천문학적인 규모의 파라미터 수를 가지는 transformer 구조의 언어 모델을 마찬가지로 천문학적인 규모의 데이터셋으로 학습하게 된다. AI 쪽으로는 업계에서 선두 그룹으로 달리고 있는 기업이기 때문에 대규모의 네트워크를 학습하는 과정에서 필요한 리소스를 감당할 수 있지만, <U>학습 과정에서 많은 자원이 사용되는 것</U>은 불가피하다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/228143031-ea355028-3002-47b3-9592-08d95f0308df.png" width="600">
</p>

보통 각 GPT-3를 학습할 때 생성되는 $\mathrm{CO_2}$가 대략 **뉴욕에서 샌프란시스코로 편도로 6번 비행**하는 동안 **생성되는 이산화탄소 양**에 맞먹는다고 한다. 지금도 GPT-4 이후로 GPT-5 등등 학습이 진행되고 있을 것이기 때문에, 이제는 단순히 뉴욕에서 샌프란시스코로 비행하는 수준보다 더 많은 양의 $\mathrm{CO_2}$가 생성되고 있을 것이라고 추측해볼 수 있다.

그리고 많은 자원이 사용되는 것을 넘어서 <U>네트워크 구조가 복잡해질수록</U> 시간 절약과 효율성을 위해 다양한 환경에서 적절하게 동작할 수 있는 ‘adaptation 방법’을 찾는 것이 점차 **많은 노력을 필요**로 하게 된다는 것을 암시할 수 있다.  대용량의 네트워크가 다량의 데이터셋에 대해 학습된 후, test 과정에서 **다양한 환경 상에 잘 적응할 수 있게** 만드는 것이 주요 해결과제인 것이다. 결국 사전 학습 과정을 adaptation과 분리해야하며, 이는 자원의 효율적 사용과 더불어 사전 학습 데이터(training dataset)에 대한 <U>license</U>나 <U>privacy</U> 또한 해결할 수 있는 연구 방향이 되는 것이다.

---

# Real world application의 특징

**TTA**는 이전에 다뤘던 글들 중 TENT, CoTTA, Contrastive TTA 등등에서 볼 수 있었던 것과 같이 <U>training dataset을 보지 못한 채</U>로 target dataset의 image sample만을 가지고 적절한 output distribution(likelihood)을 adaptation하는 과정이다.  TTA라는 task가 정의된 배경에는 보다 <U>real-world에 가까운 DA setting</U>이 있는데, 실제로 학습된 네트워크가 이용되는 상황은 일반적인 deep learning 학습 과정과 많은 차이가 있다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/228143036-d7c2f8da-425a-46e1-b2bc-9f63d70de90d.png" width="600">
</p>

우선 대부분의 **real-world** 상황에서는 adaptation이 **online**(deployment 과정에서 학습이 이루어지는 것)이라는 특징이 있다. Online learning은 단순히 test dataset을 학습에 활용한다는 범주가 아니라, 실제로 네트워크가 사용되는 환경의 데이터셋에 바로 적응하는 과정이라고 볼 수 있다.  예를 들어 **자율 주행 자동차**나 **드론**에 특정 vision model이 내장되어있는 상황을 가정해보자. Test-time에 네트워크에서 처리하는 test dataset은 pre-training 단계와는 다르게 <U>highly-correlated data</U> (non-i.i.d.)인 video stream이 처리된다. 독립 항등 분포(i.i.d)란, 같은 확률 분포를 가지면서 각자의 등장에 아무런 영향을 주지 않는 변수의 분포를 의미한다. 예컨데 주사위를 던지는 상황에서 바로 직전에 굴린 주사위가 $1$이 나왔다고 하더라도 바로 다음 차례에 굴린 주사위가 다시 $1$이 나올 확률이 $1/6$보다 작아지지 않는다는 것이다. 이처럼 **딥러닝 학습**이나 domain adaptation 학습 과정에서는 흔히 분포는 일정하지만($p(x)$), 각 분포가 배치 단위로 연산이 될 때 서로 <U>아무런 영향을 끼치지 않는 상황</U>을 가정한다($p(x_2 \vert x_1) = p(x_2)$).

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/228143038-b56b0829-e988-4015-9be9-ede4c11f976d.png" width="400">
</p>

하지만 연속되는 여러 프레임의 input을 처리할 때는 불가피하게 이전 instance가 다음 instance에 의존성을 주는 구조가 된다. 이러한 non-i.i.d. 데이터셋이 adaptation에 사용될 때 기존 TTA 방식은 misalignment, calibration 문제를 해결할 수 없다는 문제가 있다. 그리고 서로 아무런 영향을 끼치지 않는다는 사실을 배제하고도 real-world 환경에서 input dataset의 분포 $p(x)$가 일정하다는 것도 보장할 수 없는 것이 특징이다. 단순히 날씨 변화와 같은 **low-level domain shift**부터 시작해서, 주변 환경의 풍경이나 분위기가 바뀌는 **high-level domain shift**에 이르기까지 실제 환경에는 <U>고려해야할 변수가 늘어나기 때문</U>이다.

---

# 안정적인 test-time adaptation

결국 이런저런 말들로 real-world 환경의 특징을 언급했지만 간단하게 한마디로 표현하자면 ‘다양한 환경에서 안정적으로 작동하는 TTA 방식’을 찾는 것이 이 논문의 주된 목적이다. Test-time distribution이 달라지게 되면 상황에 따라 기존 TTA 방식에서 학습에 사용했던 방법들은 hyper-parameter에 취약하다는 특징이 있었고, 결국 미리 test-time condition에 대해서 알고 있는 상황이 아니라면 적용했을때 오히려 hypothesis를 망치는(collapse) 원인이 될 수 있다는 것이다. 따라서 이 논문의 저자들은 network의 parameter를 조정하는 방법 대신 model output에 대한 **manifold 정규화**를 통해 **latent assignment**를 진행하는 방법을 선택하였다. 이를 Manifold smoothness라 하는데, 보통 graph-clustering, semi-supervised learning이나 few-shot learning에서 사용되며, 보다 안정적인 솔루션을 찾기 위한 방법론으로 제시된다.

이 논문에서는 여러 방법들 중 Laplacian regularization을 latent correction term으로 사용하였으며, 딥러닝 학습법인 gradient optimization이 아니라 concave-convex function을 CCCP(Concave-convex procedure) 방법을 통해 직접 최적화하는 알고리즘을 사용하였다. **파라미터를 직접 건들지 않기 때문**에 hyper-parameter 의존성이 아예 사라지게 되었고, 단순히 output probability를 재배치함으로써 대부분의 domain shift 상황에서 안정적인 성능을 보여주었다고 한다.

---

# Related work

일반적으로 DA task는 가장 간단한 딥러닝의 assumption 중 하나인 <U>“학습할 때와 실제로 사용할 때의 distribution이 일치한다”</U>라는 제약을 풀기 위해 사용된다. 대체로 real-world application에서는 **머신러닝 책**에서 배운 상황들이 이상적으로 펼쳐지지 않기 때문에 distribution이 달라지더라도 명확하게 동작할 수 있는 방법을 찾는 것은 중요한 일이 되었다. 굉장히 초반의 domain adaptation은 training/test dataset에 모두 접근이 가능한 상황을 가정했었지만, 점차 UDA(Unsupervised domain adaptation)과 같이 target domain에 대한 supervision 없이 domain adaptation을 하는 약간 더 <U>현실적인 문제</U>로 넘어오기 시작했다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/228143044-5aacd8c4-e16d-4df4-b9fa-219fc49b4cfc.png" width="400">
</p>

UDA에는 여러 방법론이 사용되었지만 대표적으로 가장 많이 사용되는 방법은 **source** 및 **target distribution** 사이의 <U>divergence</U>(분포 거리)를 <U>좁히는 방법</U>을 사용하거나 domain discriminator(도메인 분류기)를 기준으로 gradient를 반대 방향으로 학습하여 feature extractor가 서로 다른 도메인의 input에 대해 <U>유사한 feature map을 생성하게끔</U> 학습하는 방법이 있다(DANN). 하지만 여전히 UDA는 adaptation을 하는 과정에서 target dataset과 더불어 source dataset을 사용한다는 점이 usability를 제한하는 요소가 되었다.

**Domain generalization(DG)**은 단일 도메인에 대한 최적화 대신 여러 target distribution에 대해 일반화가 가능한 네트워크를 학습하기 위해 정의된 task이다. 다양한 domain에 쉽게 적응하기 위해 augmentation이나 generative model을 통해 training data의 diversity를 높이고 <U>domain-invariant representation</U>을 학습시키거나, domain-specific/domain-agnostic component를 분리하는 방법을 사용한다. 결국 저자들이 하고자 하는 <U>‘어떠한 test domain에도 잘 적응한’</U> 네트워크를 구성하고자 하는 최종 목표에 대해서는 DG와 같지만, 이 논문은 adaptation 효율에 대해서 다룬다는 점에서 task 차이가 있고, 이러한 면에서 DG보다는 **source-free domain adaptation**에 가깝다고 할 수 있다. Class centroids의 moderate shift를 가정한 source to target adaptation 방법을 사용하거나, generative model을 이용하여 target domain에 가까운 샘플을 생성하는 등 여러 방법이 있다. Test-time의 unsupervised learning을 보조하기 위해 auxiliary task로 supervised learning과 self-supervised learning setup을 같이 최적화하는 **TTT(Test-time training)** 방법도 제안되었다.

기존의 DA보다는 <U>real-world setting</U>에 가깝다는 측면에서 위의 방법들이 가지는 장점이 있지만, **ad-hoc**(adaptation 과정에만 특화된) 학습 구조를 가진다는 점에서 한계점이 명확해진다. 예컨데 GAN, domain discriminator, auxiliary task 등등 결국 사전 학습된 네트워크의 가장 기본적인 supervised learning이 아니라 최적화 방법론이 추가되기 때문에 여러 네트워크 구조에 대해 적용 가능하거나 응용이 가능하지 않다는 것이다.

그런 측면에서 이 논문은 TENT paper에서 세팅한 <U>fully test-time adaptation scenario</U>에 가장 가깝다고 할 수 있다. TENT paper의 세팅은 이전에 작성한 글([참고 링크](https://junia3.github.io/blog/tent))에도 나와있듯이, model training 과정을 바꾸지 않은 채로 target domain에 대한 adaptation 성능을 높이고자 하였다. 이때 사용한 방법이 entropy minimization loss이고 모델의 파라미터 전체를 최적화하는 것이 아니라 일부(BatchNorm의 scale, bias($\gamma,~\beta$))를 최적화하는 방법을 사용했다. 이후 AdaBN 방법을 사용하는 연구나, mutual information objective를 최적화하는 SHOT과 같은 논문들도 제안되었다. 

<U>Parameter-free 논문</U>은 TENT와 SHOT의 motivation인 fully-test time adaptation이라는 점에서는 동일하지만, 직접 네트워크의 파라미터를 조정하는 방법이 아니라는 점과 다양한 test time distribution(online 환경), 특히 non-i.i.d. 시나리오까지 가정했다는 점에서 다른 연구라고 할 수 있다.

---

# Task 정의

TTA는 domain adaptation 단계에서 접근이 불가능한 <U>labelled source dataset</U>인 $\mathcal{D}\_s = \\{ (x, y) \sim p\_s (x, y) \\}$와 해당 데이터셋으로 학습된 <U>pre-trained parametric model</U> $q_\theta (y \vert x)$가 있는 상황을 가정한다. 여기서 $x$는 이미지를 의미하고 $y \in \mathcal{Y}$는 source classes 집합인 $\mathcal{Y}$의 각 원소들이 이미지마다 라벨링이 되어있는 상황이다. 해당 상황에서 unlabeled target dataset이 target distribution으로부터 무작위로 추출되고, 이를 $\mathcal{D}\_t = \\{ x \sim p_t (x) \\}$로 포현할 수 있다. 일반적인 covariate shift를 가정하면 image $x$를 조건부로 하는 $y$에 대한 확률 분포는 동일한 상황을 생각해볼 수 있다.

\[
    p_s(y \vert x) = p_t (y \vert x),~p_s(x) \neq p_t(x)
\]

이를 **posterior**로 갖는 **likelihood**와 **prior**에 대해 다시 표현하면,

\[
    \frac{p_s (x \vert y) p_s (y)}{p_s (x)} = \frac{p_t (x \vert y) p_t(y)}{p_t (x)}
\]

이와 같이 **likelihood**와 **prior**의 곱으로 표현 가능하다. 이미지인 $x$에 대한 분포는 서로 다른데 **posterior**가 <U>서로 같은 상황</U>을 가정하기 때문에 **likelihood**와 **prior**의 곱인 joint distribution 또한 차이가 발생하는 것을 알 수 있으며($p_s(x, y) \neq p_t(x, y)$), 이때 joint distribution 차이에 기인하는 원인은 <U>likelihood shift</U> ($p_s(x \vert y) \neq p_t(x \vert y)$) 혹은 <U>prior shift</U> ($p_s(y) \neq p_t (y)$) 두 경우가 있다. Likelihood shift 상황은 동일한 class에 대해 이미지가 바뀌는 상황을 생각해볼 수 있으며, prior shift는 class의 비율이 바뀌는 상황을 가정해볼 수 있다. 

논문에서 likelihood shift는 ImageNet to ImageNet-C(Corruption)으로 동일한 class set에 대해 이미지 분포가 바뀌는(gaussian, shot, impuse noise 등등) 상황을 가정하였고 이에 추가로 ImageNet-C to ImageNetC-16($1000$개의 class 중 일부인 $32$개의 class에 대해 superset $16$개의 class로만 구성) 상황을 가정하였다. Class 갯수가 달라지는 경우에는 superset이 포함하는 데이터셋 클래스에 대한 softmax prediction을 평균내는 방식(average pooling)을 사용한다. 다른 논문에서는 max-pooling을 통해 superset이 포함하는 카테고리 중 가장 최대의 확률값만 사용했는데, 이 논문에서는 average-pooling 방식이 보다 효과적이었다고 밝혔다. 뒤에서 추가로 experimental detail을 설명하겠지만, prior shift 상황으로는 class imbalance(이에 추가로 non-i.i.d.) 상황을 가정했다고 한다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/228143021-8d626e08-33b9-4761-a1e7-b8ed4b9fddfd.png" width="900">
</p>

Target distribution이 source에서 이동하는(shift) 상황을 가정하기 때문에 source dataset에 대해 학습된 parameteric model $q_\theta (y \vert x)$는 더이상 실제 domain-invariant distribution인 $p(y \vert x)$를 제대로 예측하지 못할 수 있다. **True source posterior**에 대해서는 $p_s(y \vert x) \approx q_\theta(y \vert x)$가 보장되지만 **true target posterior** $p_t(y \vert x) \approx q_\theta (y \vert x)$는 보장할 수 없다는 것이다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/228143050-575944b1-0e7e-4273-b0ab-204b62b1ee42.png" width="900">
</p>

실제로 target sample에 대한 <U>posterior가 제대로 예측되지 않는 경우</U>를 위와 같이 표현할 수 있다. TENT에서 사용된 entropy minimization을 non-i.i.d.인 상황에서 최적화하게 되면 원래 따라가야하는 분포인 실선과 다르게 linear classifier는 <U>단순히 현재 배치를 기준으로</U> 멀어지게끔 학습된다(엔트로피를 줄여야하므로). 물론 위의 예시는 극한의 상황(batch에 같은 class만 존재하는 상황)을 가정했지만, 2번째 batch까지 <U>녹색 샘플을 관측하지 못한 채</U> 분포의 boundary를 넘어가게 된다면 원래라면 두 분포 사이를 지나야하는 $q_\theta(y \vert x)$가 target posterior에 대해 제대로 예측하지 못한채로 **성능 복구가 불가능**해지는 것을 알 수 있다. 

---

# Parameter 학습의 위험성

<U>Domain에 무관하게</U> 제대로 된 분류가 진행되는 output space를 latent space $z \sim \mathcal{Z}$로 표현하겠다. TTA에서 기존 방법들은 network의 parameter를 잘 조정함으로써 test dataset을 통해 이상적인 $p(z \vert x)$를 찾고자 하였고, 이렇게 파라미터를 최적화하는 네트워크를 **NAM(Network Adaptation Methods)**이라고 통칭해보자. TENT, SHOT이 이러한 NAM에 해당되는데, NAM은 network를 학습이 가능한(adaptable) 파라미터인 $\theta^a$ 그리고 고정된(frozen) 파라미터인 $\theta^f$로 분류한 뒤 unsupervised loss $\mathcal{L}(x; \theta^a \cup \theta^f)$를 target data $x \sim p_t(x)$에 대해 계산한 값을 토대로 $\theta^a$를 최적화하게 된다. TENT의 경우 앞서 언급했던 내용과 같이 batch normalization parameter인 $\gamma, \beta$를 entropy minimization loss를 통해 최적화하는 방법을 사용하였고, SHOT는 convolutional filter를 mutual information maximization을 통해 최적화하는 방법을 사용하였다.

NAM은 직접 parameter를 최적화하기 때문에 보다 target sample의 prediction의 <U>성능을 높일 수 있는 가능성</U>이 있지만, 그와 반대로 <U>성능이 급격하게 나빠질 수 있는 가능성</U>도 공존한다(**High risk, high return**). 학습 가능한 파라미터 $\theta^a$를 한정된 target distribution의 일부에 대해 연속적으로 업데이트하는 과정에서 overspecialize(overfitting되거나 잘못된 수렴점으로 가는 현상)이 발생할 수 있는 것이다. 이러한 문제는 특히 hyper-parameter에 취약하다는 특징으로 드러나며, 이는 batch level에서 sample diversity가 부족해지는 non-i.i.d.인 실제 deployment 상황에서 더 두드러지게 나타난다. Sample diversity가 부족한 상황은 단순히 video streaming과 같은 자율주행 시나리오 뿐만 아니라, 높은 class imbalance가 생길 때에도(long-tail) 똑같이 적용될 수 있다. 예컨데 특정 class의 비율이 지나치게 높은 데이터셋이 있다면, 아무리 랜덤하게 샘플을 배치 단위로 분류하더라도 <U>특정 class가 dominant</U>한 현상을 막을 수 없기 때문이다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/228143055-a66e2f9b-06ef-4ae7-8e73-c7e463137d64.png" width="1100">
</p>

위의 그래프가 바로 **non-i.i.d.** 상황에서 생기는 문제점을 보여주는 그래프이다. 위의 그래프는 TENT에서 사용된 entropy minimization을 통해 최적화한 그래프를 그대로 보여주는데, learning rate에 상관없이 entropy minimization은 잘 진행되지만, 실제로 accuracy 그래프를 보게 되면 $\alpha = 0.001$인 상황을 제외하고는 모두 점차적으로 성능이 하락하는 모습을 보여준다. 앞서 본 **붉은색 분포와 녹색 분포** 그림에서 발생한 문제가 적용된 사례라고 볼 수 있다. 이러한 <U>hyper-parameter 의존성</U>은 미리 test-time scenario를 알아야하며, 성능 평가를 위해 label이 존재하는 상황을 가정해야한다는 큰 한계점이 존재한다. 실제로 hyper-parameter에 대한 의존성이 여러 domain shift 상황에서 어떤 risk를 가질 수 있는지 저자들은 총 $12$가지의 상황에 대한 성능 평가표를 matrix로 구성하였다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/228143056-556af371-9823-4447-9c4d-2ab1bdd18ef3.png" width="950">
</p>

- A : Independent and identical dataset(i.i.d.)
- B : Non-independent and identical dataset(non-i.i.d.)
- C : i.i.d. + prior shift(Long tail)
- D : non i.i.d. + prior shift(Long tail)

표에 나와있는 각 상황에 대한 **legend**는 위에 적힌 내용과 같다. $i$번째 row가 의미하는 바는 해당 줄 앞부분에 표시된 distribution에 대해 최적화된 hyper-parameter를 사용하여 다른 distribution(가로축에 표시된 distribution)에 최적화를 진행했을 때의 성능 향상과 하락을 adaptation 방법을 사용하지 않았을 때와 비교하여 측정한 행렬이다. 그래프를 보면 알 수 있듯이 <U>non-i.i.d.인 상황에서 성능 하락</U>이 많게는 $66\%$까지 발생하는 것으로 보이며, 저자들이 제시한 방법(LAME)이 훨씬 안정적인 것을 확인할 수 있다. 그렇다면 대체 저자들이 사용한 **LAME**이라는 방법은 어떤 원리인지 살펴보도록 하자.

---

# LAME(Laplacian Adjusted Maximum Likelihood)에 대하여

LAME에 대해 수식화하기 전에 간단하게 컨셉만 짚고 넘어가자면 LAME은 **classifier**가 내뱉은 <U>output probability</U>를 적절하게 **correction** 해주는 방법이며, feature extractor의 parameter를 바꾸는 방법이 아니라는 것이다. Source classifier의 parameter를 바꾸지 않음으로 인해 **error accumulation**이나 **knowledge accumulation**으로 인해 발생하는 문제를 방지할 수 있으며, <U>hyper-parameter tuning</U> 과정에서 <U>자유롭다</U>는 특징이 있으며 딥러닝 연산을 필요로하지 않기 때문에 연산 효율성이 올라간다.

### Maximum likelihood with pre-trained model for source dataset

 만약 target distribution $X \in \mathbb{R}^{N \times d} \sim p_t^N (x)$에서 추출된 batch 단위의 데이터 샘플이 있다고 생각해보자. 여기서 $N$은 sample 갯수를 의미하며 $d$는 feature dimension을 의미한다. 찾고자 하는 것은 latent assignment vector $\tilde{z}\_i = (\tilde{z}\_{ik})_{1 \le k \le K} \in \Delta^{K-1}$ 을 각 data point $x_i$에 대해 정의하여, 실제 distribution $p(z \vert x)$에 근접한 latent assignment를 찾고자 하는 것이다. $K$는 <U>class의 갯수</U>를 의미하고 $\Delta^{K-1} = \\{ \tilde{z} \in [0, 1]^K \vert 1^\top \tilde{z} = 1 \\}$은 <U>각 class에 대한 예측된 확률</U>을 의미한다. 단순하게 생각한다면 $\tilde{z_i} \in \Delta^{K-1}, \forall i$ 를 만족하면서 **log-likelihood를 최대화**하는 assignment $\tilde{Z} \in [0, 1]^{NK}$ 를 찾는 objective로 바꿔쓸 수 있다. 임의의 constant $C$에 대해 ,

\[
    \mathcal{L}(\tilde{Z}) = \log \left( \prod_{i=1}^N \prod_{k=1}^K p(x_i, k)^{\tilde{z_{ik}}} \right) = \sum_{i=1}^N \tilde{z}_i^\top \log (p_i) + C
\]

assignment vector $\tilde{z}\_i$가 $p\_i = (p(k \vert x_i))_{1 \le k \le K} \in \Delta^{K-1}$의 각 <U>class별 real probability</U>의 **log likelihood**와의 내적이 연산되고, 이때의 값이 최대가 될 때 <U>maximum log likelihood</U>가 달성된다고 볼 수 있다. 이에 추가적으로 assignment vector $z$가 over-confident(one-hot encoding 형태를 보이는 것)가 되는 문제를 방지하기 위해, $\tilde{z}_i$에 대한 negative-entropy regularization term도 더해주게 된다(Entropy가 0이 되는 문제를 막기 위해).

\[
    -\sum\_{i=1}^N \tilde{z}\_i^\top \log (p_i)+-\sum\_{i=1}^N \tilde{z}\_i^\top \log (\tilde{z}_i) = \sum\_{i=1}^N KL(\tilde{z}_i \vert \vert p_i),~(1^\top \tilde{z}\_i = 1,~\forall i)
\]

그렇게 되면 위와 같이 **실제 분포**($p$)와 **assignment 분포**($z$) 간의 <U>KL divergence</U> 식으로 표현되는데, 이게 또 잘 보면 $\tilde{z}\_i > 0$ 이라는 **constraints**도 자연스럽게 들어가있는 구조가 되어 최적화 식에서 이에 대한 **constraints**를 따로 고려할 필요가 없게 된다. 사실 위의 식을 최소화하는 solution은 $p\_i = \tilde{z}\_i$ 인 사실은 너무 자명하다. KL divergence 식은 무조건 0보다 크거나 같은 값을 가지는데, 두 분포가 서로 같을 때가 value가 $0$이 되어 최소가 되기 때문이다. 하지만 실제로 특정 샘플에 대한 확률값 $p\_i(k \vert x\_i)$ 자체는 **intractable**하기 때문에 이를 사용하지 않고 source network가 approximate한 likelihood인 $q_i = (q\_\theta(k \vert x_i))_{1 \le k \le K}$ 를 approximation의 기준으로 삼게 된다. 마찬가지로 식에서 $p_i \rightarrow q_i$로 치환하게 되면 optimal solution이 $q_i$가 되므로, 만약 source network가 target dataset에 대해 <U>예측한 likelihood가 정확하지 않다면</U>(실제 $p_i$와 큰 오차가 있다면) prediction이 바람직하지 않게 된다.

### Laplacian regularization

바로 위의 <U>optimal solution</U>이 가지는 오차를 줄이기 위해 제안되는 것이 바로 **Laplacian regularization**이다. Laplacian은 feature space에서의 point 사이의 거리를 latent assignment에 고려하게 되며, 가까운 거리에 있을수록 latent assignment 또한 유사하게 생성되게끔 하는 정규화 term이다. 기존에 Semi-supervised learning이나 graph-clustering에서는 labeling이 된 데이터 포인트들을 기준으로 supervised loss와 함께 최적화되거나 class-balance constraints를 주는 방식에 라플라시안 정규화가 사용되었는데, 이와는 다르게 TTA 문제에서는 <U>supervision</U>이나 <U>class balance</U>가 전혀 무관한 것을 알 수 있다(non-i.i.d. 이면서 unsupervised setting이기 때문). 따라서 이 논문에서 사용하는 Laplacian adjustment는 앞서 언급한 KL divergence 식(likelihood + entropy regularization)에 더해짐으로써 source에 대해 사전 학습된 $q_i$가 **제대로 예측하지 못하는 likelihood를 보조**하는 역할로 사용된다.

\[
    \mathcal{L}^\text{LAME} (\tilde{Z}) = \sum\_i KL(\tilde{z}\_i \vert \vert q\_i) - \sum\_{i, j} w_{ij}\tilde{z}\_i^\top \tilde{z}\_j
\]

Formulation이 마무리되었으니 식을 다시 한 번 해석하면 다음과 같다.

1. Maximum likelihood를 만족하는 $\tilde{z_i}$를 찾되, over-confident 현상을 방지하기 위해 KL divergence 식으로 정규화
2. 그런데 원래 식의 optimal solution $p_i$을 target data에 대해서 바로 찾을 수 없기 때문에 source에 최적화된 pre-trained network가 학습한 분포인 $q_i$를 사용하겠다.
3. 그런데 $q_i$는 target domain에서 오차율이 크기 때문에 여기에 추가로 각 배치 내에서 인접한 feature map끼리의 prediction을 유사하게 만드는 라플라시안 정규화를 사용하고자 함

$w_{ij} = w(\phi(x_i),~\phi(x_j))$는 학습된 feature extractor $\phi$로부터 추출된 feature map 사이의 affinity를 결과로 내뱉게 된다. Affinity가 높으면 더 큰 값을 가지기 때문에, 자동적으로 LAME을 최소화하기 위해서는 서로 유사한 feature map의 assignment 끼리의 dot product가 커져야한다($i, j$가 같은 카테고리에 속할 확률을 증가시켜야 한다).

그렇다면 이 loss를 최소화하는 optimal solution은 어떻게 구해야할까? 앞서 언급하기로는 해당 논문의 저자들은 function의 최솟값을 찾는 알고리즘을 gradient descent를 통한 parameter search를 사용하지 않았다. 그대신 probability $q_i$를 optimal solution으로 사용하는 것보다는 유의미한 $\tilde{z}_i$를 찾기 위해 Laplacian regularization을 진행하였다. 

결론부터 말하자면 저자는 LAME 식을 **convergence**를 보장하는 알고리즘인 <U>CCCP(Concave-Convex Procedure)</U>로 최적화를 진행한다. 각 iteration 마다 solution $\tilde{Z}^{(n)}$를 objective의 tight upper bound의 최솟값으로 설정한다. 이러한 방식을 통해 objective는 적어도 각 iteration 마다 증가하지는 않을 수 있게 된다. 말로만 설명하면 이해가 쉽지 않기 때문에 LAME 식을 풀어서 설명하면 다음과 같다.

### CCCP(Concave-Convex Procedure)

CCCP 알고리즘([참고 링크](https://www.cise.ufl.edu/~anand/pdf/cccp_nips.pdf))에 대해서 간단하게 소개하면 다음과 같다. Energy function $E(\overrightarrow{x})$가 energy function이고 bounded Hessian $\partial^2 E(\overrightarrow{x})/\partial \overrightarrow{x} \partial \overrightarrow{x}$를 가지고 있다고 생각해보자. 여기서 <U>energy function</U>이란 단순하게 말하자면 최소가 될 때가 가장 안정적인 상태가 되는 **모든 형태의 함수**를 의미하며, 우리의 경우에는 loss function을 이 energy function에 대입하여 생각해볼 수 있다. 임의의 energy function $E(\overrightarrow{x})$는 convex function과 concave function으로 분리할 수 있다. 아래 그림을 참고하여 증명 과정은 다음과 같다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/228143059-6dec76f2-8f56-46a4-b95c-f91f468f7bab.png" width="950">
</p>

**Proof 1.** Eigen value가 $\epsilon > 0$으로 bounded된 임의의 Positive definite Hessian을 가지는 convex function $F(\overrightarrow{x})$를 생각해보자. 사실상 결론에서 밝히겠지만 이 임의의 convex function은 energy function의 concave한 부분을 없애기 위한 역할을 수행한다.  그렇게 되면 $\lambda > 0$에 대해 positive definite한 $E(\overrightarrow{x}) + \lambda F(\overrightarrow{x})$를 정의할 수 있게 되며, 결국 이 함수는 convex가 된다(중간 그림). 그러면 자연스럽게 임의로 정한 <U>convex function</U>에 **negative value**를 곱한 $-\lambda F(\overrightarrow{x})$는 concave part가 되며, **energy function**은 <U>convex function</U>과 <U>concave function</U>의 합으로 표현 가능하다.

\[
    E(\overrightarrow{x}) = \left( \underset{\text{Convex part}}{E(\overrightarrow{x}) + \lambda F(\overrightarrow{x})}\right) + \left(\underset{\text{Concave part}}{- \lambda F(\overrightarrow{x})} \right)
\]

**CCCP**는 바로 이런 <U>energy function의 특성</U>을 이용하면 discrete iterative 알고리즘에서 다음과 같이 <U>monotonically decreasing</U>하는 solution $\overrightarrow{x}^{(t)}$를 찾을 수 있다고 언급한다.

\[
    \overrightarrow\nabla E_{vex}(\overrightarrow{x}^{(t+1)}) = -\overrightarrow \nabla E_{cave} (\overrightarrow{x}^{(t)})
\]

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/228143060-43f95210-c964-423b-b009-41017e6f9673.png" width="950">
</p>

**Proof 2.** Convex function과 concave function의 정의에 따라 미분 가능한 두 함수 $E_{vex}(\cdot)$와 $E_{cave}(\cdot)$에 대해 임의의 4개의 벡터 포인트 $\overrightarrow{x_1}$, $\overrightarrow{x_2}$, $\overrightarrow{x_3}$, $\overrightarrow{x_4}$에 대하여,

\[
\begin{aligned}
E_{vex}(\overrightarrow{x_2}) \ge& E_{vex}(\overrightarrow{x_1}) + (\overrightarrow{x_2} - \overrightarrow{x_1}) \cdot \overrightarrow \nabla E_{vex}(\overrightarrow{x_1}) \newline
E_{cave}(\overrightarrow{x_4}) \le& E_{cave}(\overrightarrow{x_3}) + (\overrightarrow{x_4} - \overrightarrow{x_3}) \cdot \overrightarrow \nabla E_{cave}(\overrightarrow{x_3})
\end{aligned}
\]

위의 두 식이 항상 만족한다. 이 상태에서 $\overrightarrow{x_1} = \overrightarrow{x}^{(t+1)}$,  $\overrightarrow{x_2} = \overrightarrow{x}^{(t)}$,  $\overrightarrow{x_3} = \overrightarrow{x}^{(t)}$, $\overrightarrow{x_4} = \overrightarrow{x}^{(t+1)}$로 두게 되면

\[
\begin{aligned}
E_{vex}(\overrightarrow{x}^{(t)}) \ge& E_{vex}(\overrightarrow{x}^{(t+1)}) + (\overrightarrow{x}^{(t)} - \overrightarrow{x}^{(t+1)}) \cdot \overrightarrow \nabla E_{vex}(\overrightarrow{x}^{(t+1)}) \newline
E_{cave}(\overrightarrow{x}^{(t+1)}) \le& E_{cave}(\overrightarrow{x}^{(t)}) + (\overrightarrow{x}^{(t+1)} - \overrightarrow{x}^{(t)}) \cdot \overrightarrow \nabla E_{cave}(\overrightarrow{x}^{(t)})
\end{aligned}
\]

아래와 같이 식이 바뀌게 되고, 부등호의 방향을 맞춰서 잘 더해보면 $0$이 아닌 constant $C = (\overrightarrow{x}^{(t+1)} - \overrightarrow{x}^{(t)})$에 대하여,

\[
E_{vex}(\overrightarrow{x}^{(t+1)}) + E_{cave}(\overrightarrow{x}^{(t+1)}) \le E_{vex}(\overrightarrow{x}^{(t)}) + E_{cave}(\overrightarrow{x}^{(t)}) + C \cdot \left(\overrightarrow\nabla E_{vex}(\overrightarrow{x}^{(t+1)}) +\overrightarrow \nabla E_{cave} (\overrightarrow{x}^{(t)}) \right)
\]

위와 같이 정리되기 때문에 $\overrightarrow\nabla E_{vex}(\overrightarrow{x}^{(t+1)}) = -\overrightarrow \nabla E_{cave} (\overrightarrow{x}^{(t)})$ 조건에 대해서 <U>항상 감소하는 solution</U>을 찾을 수 있게 되는 것이다.

이를 여러 iteration 반복하게 되면 어떠한 형태의 energy function에 대해서도 optimal solution에 가까워지는 방향을 정할 수 있다고 하는 것이 **CCCP 알고리즘** 방법이다.

### CCCP algorithm on LAME

다시 **LAME** 식을 가져와서 노려보도록 하자.

\[
    \mathcal{L}^\text{LAME} (\tilde{Z}) = \sum\_i KL(\tilde{z}\_i \vert \vert q_i) - \sum\_{i, j} w_{ij}\tilde{z}\_i^\top \tilde{z}_j
\]

오호라, $w_{ij}$는 인접한 feature 간의 affinity를 표현한 값이기 때문에 위의 식을 <U>Kronecker product</U> $\otimes$에 대해 다음과 같이 표현해볼 수 있겠다.

\[
\mathcal{L}^\text{LAME} (\tilde{Z}) = \sum_i KL(\tilde{z}_i \vert \vert q_i) - \tilde{Z}^\top (W \otimes I) \tilde{Z}
\]

그렇게 된다면 뒤쪽에 놓인 <U>matrix multiplication</U>은 임의의 $\tilde{Z}$에 대해 항상 양이 아닌 값을 가지게 된다.  왜냐하면 $W \otimes I$가 **positive semi-definite**임을 알고 있기 때문이다($W$가 positive definite이라면, $W \otimes I$는 positive semi-definite).

\[
\mathcal{L}^\text{LAME} (\tilde{Z}) = \left(\underset{\text{Convex part}}{\sum_i KL(\tilde{z}_i \vert \vert q_i)}\right) + \left(\underset{\text{Concave part}}{- \tilde{Z}^\top (W \otimes I) \tilde{Z}} \right)
\]

그렇다면 <U>CCCP의 증명</U>에 따라 위의 식은 $n$번째 예측된 $\tilde{Z}$에 대해 upper bound를 다음과 같이 설정할 수 있다.

\[
\mathcal{L}^\text{LAME} (\tilde{Z}) \overset{C}{\leq} \left(\underset{\text{Convex part}}{\sum_i KL(\tilde{z}_i \vert \vert q_i)}\right) - \left((W \otimes I) \tilde{Z}^{(n)}\right)^\top \tilde{Z}
\]

뒤쪽의 식은 Concave function을 $\tilde{Z}^{(n)}$위치에서 **선형 근사(1차 근사)한 테일러 함수**를 의미하기 때문에 선형성 때문에 <U>concavity를 잃는다</U>. 정확하게 말하자면 이제는 concave 함수이기도 하며 동시에 convex 함수가 된다. 아무튼 현재까지 예측된 $\tilde{Z}$만 알고 있다면, 다음 $\tilde{Z}$는 적어도 이전 함숫값보다는 크지 않은 convex function을 upper bound로 설정할 수 있게 된다는 것이다. 지금까지 한 내용을 간단하게 정리하면 아래와 같다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/228143009-8722e2d3-c316-4221-aa21-c4d1f3d2397d.png" width="950">
</p>

### Formulation for next $\tilde{Z}$

\[
\mathcal{L}^\text{LAME} (\tilde{Z}) \overset{C}{\leq} \left(\underset{\text{Convex part}}{\sum_i KL(\tilde{z}_i \vert \vert q_i)}\right) - \left((W \otimes I) \tilde{Z}^{(n)}\right)^\top \tilde{Z}
\]

이제 이 식을 최소화하는 것은 <U>Variational autoencoder</U>에서 **ELBO(Evidence Lower Bound)**를 최적화하여 간접적으로 $p(x)$의 log likelihood를 최대화하는 원리랑 똑같다고 볼 수 있는데, 

\[
\text{ELBO}\_{vae} \sim \log p(x) \ge E\_z(\log p(x \vert z)) - D_{KL}(q(z \vert x) \vert\vert p(z))
\]

차이점이라고 한다면 딥러닝 네트워크를 최적화하는 것이 아니라 실제로 strictly convex function의 <U>closed-form solution</U>을 구하고 싶은 것이다. 이때 사용할 수 있는 방법은 **KKT(*Karush-Kuhn-Tucker*)**로, 해당 upper bound는 주어진 제약식인 ‘assigned된 확률값을 모두 더하면 $1$이다’라는 조건을 가지면서 최소화하는 문제이기 때문에 다음과 같이 표현할 수 있다. 예컨데 $n$번째 솔루션인 $\tilde{Z}^{(n)}$의 upper bound가 아래와 같이 표현되기 때문에, 다음에 예측될 $\tilde{Z}^{(n+1)}$은 다음과 같은 constraint를 가지는 함수의 <U>minimization 솔루션</U>이 된다.

\[
\begin{aligned}
\underset{\tilde{Z}}{\min} \sum\_{i=1}^N KL(\tilde{z}\_i \vert\vert q\_i) - \sum\_{i=1}^N \sum\_{j=1}^N w_{ij}\tilde{z}\_i^\top \tilde{z}\_j^{(n)} \newline
\text{s.t}~~~\tilde{z}\_i^\top 1\_K = 1,~\forall i \in \\{ 1, \cdots, N \\}
\end{aligned}
\]

이에 대한 Lagrangian을 작성하면 아래와 같으며, 여기서 $\lambda = (\lambda_1, ~\lambda_2, \cdots,~\lambda_N)$는 배치 수 $N$개의 샘플에 대한 linear constraint에 곱해질 **Lagrange multiplier**이다.

\[
\begin{aligned}
\mathcal{L}(\tilde{Z}, \lambda) =& \sum\_{i=1}^N \tilde{z}\_i^\top \log (\tilde{z}\_i) - \sum\_{i=1}^N \tilde{z}\_i^\top \log (q_i) \newline
-& \sum\_{i=1}^N \sum\_{j=1}^N w_{ij} \tilde{z\_i}^\top \tilde{z}_j^{(n)} + \sum\_{i=1}^N \lambda_i (\tilde{z}\_i^\top 1_K -1)
\end{aligned}
\]

이제 이 Lagrangian식을 $\tilde{z_i}$에 대해 미분하게 되면 strict convex function에 대한 **FOSC**(first-order sufficient condition)으로 global optima를 찾을 수 있게 된다. 이게 $0$이 되는 부분을 찾으면 inner point로써 <U>strict convex function의 최솟값</U>이 된다.

\[
    \nabla\_{\tilde{z}\_i} \mathcal{L}(\tilde{Z}, \lambda) = (1 + \lambda\_i) 1\_K + \log (\tilde{z}\_i) - \log (q_i) - \sum_{j=1}^N w\_{ij} \tilde{z}\_j^{(n)}
\]

이게 $0$이 될 수 있게 만드는 $\tilde{z}_i$가 곧 $\tilde{z}_i^{(n+1)}$이다.

\[
    0 = (1+\lambda\_i)1_K + \log(\tilde{z}\_i^{(n+1)}) - \log (q_i) - \sum\_{j=1}^{N} w\_{ij} \tilde{z}\_j^{(n)}
\]

$z_i^{(n+1)}$를 구하기 위해 식을 정리하면 다음과 같다.

\[
\begin{aligned}
\log \left( \frac{\tilde{z}\_i^{n+1}}{q_i} \right) =& \sum\_{j=1}^N w\_{ij}\tilde{z}\_j^{(n)} - (1+\lambda_i)1\_K \newline
\tilde{z\_i}^{(n+1)} =& q_i \odot\exp \left( \sum\_{j=1}^N w\_{ij} \tilde{z}\_j^{(n)} -(1+\lambda_i) \right)
\end{aligned}
\]

이제 이 식에서 <U>Lagrange multiplier를 치환하기 위해</U> constraint인 $1_K^\top \tilde{z}_i^{(n+1)}=1$를 사용하게 되면,

\[
\begin{aligned}
1\_K^\top \tilde{z}^{(n+1)} =& 1\_K^\top \left( q\_i \odot \exp(\sum\_{j=1}^N w\_{ij} \tilde{z}\_j^{(n)}) \right) \exp (-(\lambda_i + 1)) = 1 \newline
\exp(\lambda\_i + 1) =& 1\_K^\top \left( q_i \odot \exp(\sum_{j=1}^N w_{ij} \tilde{z}\_j^{(n)}) \right) \newline
\therefore \lambda\_i =& \log \left(  \left( q_i \odot \exp(\sum\_{j=1}^N w_{ij} \tilde{z}\_j^{(n)}) \right)^\top 1_K\right) - 1
\end{aligned}
\]

위와 같이 정리할 수 있고, 해당 식을 다시 구했던 <U>solution에 대입</U>하게 되면,

\[
\begin{aligned}
\tilde{z\_i}^{(n+1)} =& q\_i \odot\exp \left( \sum\_{j=1}^N w\_{ij} \tilde{z}\_j^{(n)} -(1+\lambda\_i) \right) \newline
=& q\_i \odot\exp \left( \sum\_{j=1}^N w\_{ij} \tilde{z}\_j^{(n)} - \log \left(  \left( q\_i \odot \exp(\sum\_{j=1}^N w\_{ij} \tilde{z}\_j^{(n)}) \right)^\top 1\_K\right) \right) \newline
=& \frac{q\_i \odot \exp \left( \sum\_{j=1}^N w\_{ij} \tilde{z}\_j^{(n)}\right)}{\exp \left(\log \left(  \left( q\_i \odot \exp(\sum\_{j=1}^N w\_{ij} \tilde{z}\_j^{(n)}) \right)^\top 1\_K \right) \right)} \newline
=& \frac{q\_i \odot \exp \left( \sum\_{j=1}^N w\_{ij} \tilde{z}\_j^{(n)}\right)}{ \left(q\_i \odot \exp(\sum\_{j=1}^N w\_{ij} \tilde{z}\_j^{(n)}) \right)^\top 1\_K}
\end{aligned}
\]

위의 식과 같은 **final solution**을 얻을 수 있게 된다.

---

# Experiment setting

실험 셋팅 과정에서 가장 중요하게 생각한 점은 <U>model 및 domain indepence</U>이다. 본인들이 주장한 방식이 다양한 pre-trained model에 대해서도 좋은 성능을 보여줘야하고 그와 동시에 다양한 domain shift 환경에 적용할 수 있어야 하기 때문이다. 따라서 그만큼 저자는 다양한 네트워크에 대해 실험을 진행함과 동시에 다양한 adaptation scenario를 설정하여 TTA 성능을 비교하였다. 예컨데 네트워크의 경우에는 training procedure에 따라 구분하거나 네트워크 구조에 따라 구분하는 두 연구를 동시에 진행하였다(결과는 아래와 같음).

- Training procedure : ResNet의 경우 Microsoft Research Asia(MSRA)에서 release한 원본, Torchvision의 pre-trained model 그리고 SimCLR 방법을 통해 SSL 학습을 한 세 구조에 대한 representation을 비교하였다.
- Network itself : RN-18, RN-50, RN-101, EfficientNet(EN-B4), ViT-B

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/228143062-c4332a51-2a95-4870-b53f-479f15aa3aae.png" width="400">
    <img src="https://user-images.githubusercontent.com/79881119/228143063-5eb24702-51a7-453d-9b3e-acad97761958.png" width="400">
</p>

Validation 목적으로는 앞서 말했던 것과 같이 likelihood shift($p_s(x \vert y) \neq p_t(x \vert y)$) 상황을 가정하기 위해 ImageNet-$\text{C}$-Val을 생각하였고(ImageNet의 original dataset에 9가지의 perturbation을 가한 데이터셋), prior shift($p_s(y) \neq p_t(y)$)를 가정하기 위해서 ImageNet-$\text{C}\_{16}$ (기존 ImageNet class 중 $16$개의 superclass로 mapping)을 사용하게 되었다. ImageNet-$\text{C}\_{16}$ 데이터셋은 class diversity를 batch level에서 줄이기 때문에, 기존 파라미터를 조정하는 방식에서 처리하기 힘든 task에 해당된다. 보다 realistic prior shift를 구성하기 위해 의도적으로 Zipf distribution으로 class ratio를 바꾸는 방식을 사용하였다. 마지막으로는 <U>non-i.i.d. scenario를 구성</U>하기 위해 같은 corruption이 발생한 샘플이나 같은 class에 속하는 데이터셋을 의도적으로 같이 구성하는 방법을 사용하였다. Validation 과정에서 사용한 dataset이 총 3가지인데, 각각을 2가지의 prior shift 상황(Class imbalance의 유무) 그리고 sample 방법(non-i.i.d.의 유무)을 기준으로 총 $3 \times 2 \times 2 = 12$가지의 시나리오를 실험하였다고 볼 수 있다.

---

# Conclusion

우선 논문에서 가장 눈에 띄는 점은 optimization을 진행할 때 딥러닝에서 사용되는 parameter optimization을 전혀 사용하지 않고 test time adaptation을 진행한 것이다. 사실 TTA task에 대한 여러 논문들을 보다 보니까 결국 source-free 
UDA와 학습 과정에서 test 단계인지 train 단계인지 구분하는 것 이외에는 별다른 차이점을 느끼지 못했는데, 이번 논문에서는 <U>확실하게 online 상황을 가정</U>하고 target domain에 대한 사전 정보가 없는 경우를 가정했다는 점과 파라미터 수정이 필요없기 때문에 단일 process로 test time adaptation이 가능하다는 점에서 <U>연산 효율성이 높아질 수 있음</U>을 확인하였다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/228143065-63418b4c-6cca-4e23-b2af-f598058d5694.png" width="400">
    <img src="https://user-images.githubusercontent.com/79881119/228143066-05756d96-4d31-4a03-b4d0-d7727dcef640.png" width="400">
</p>

예를 들어 TENT는 parameter 조정 이후에 2nd forward pass가 필요하지만, LAME은 굳이 그럴 필요가 없다는 점이 장점이 된다. 그리고 학습되는 parameter가 GPU 기기에 수용될 필요가 없기 때문에 이에 대한 메모리 효율성도 함께 올라갈 수 있다.

그리고 parameter를 조정하는 방법들에 비해 다양한 domain shift 상황에서 평균적으로 **좋은 성능을 보여주는 것**도 확인할 수 있었다.
