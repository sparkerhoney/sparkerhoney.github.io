---
title: Consistency models 논문 리뷰
layout: post
description: Consistency model
use_math: true
post-image: https://user-images.githubusercontent.com/79881119/234610703-89f369d0-0008-4b8a-8207-d93263dd09a8.gif
category: paper review
tags:
- Score based model
- Consistency model
- PF ODE
- AI
- Deep learning
---

# 들어가며…

**Yang Song**씨의 논문은 항상 읽다보면 **피가 말린다**. 안그래도 수식이 방대한 딥러닝 세상에서 더욱 수식을 멋지게 활용(?)하여 **Appendix를 화려하게 채워주기 때문이다**. 바로 이전에 리뷰했던 논문인 score based diffusion의 기초 논문들 중 하나인 [‘Score-based generative modeling through stochastic differential equations’](https://junia3.github.io/blog/scoresde) 또한 Appendix가 굉장했던 기억이 있다. 아무튼 diffusion을 공부하는 사람이라면 대체 어디서부터 읽어야할지 막막하기도 하고, 가장 베이스라인이라고 여겨질 수 있는 DDPM이나 NCSN 등등을 읽다보면 대체 무슨 근본으로 이러한 수식을 전개하는거지 싶은 순간들이 온다. 본인은 diffusion을 공부하기 시작한 이후로 수없이 많은 기초 논문들, 블로그 및 유튜브와 Bishop의 pattern recognition 서적의 이런저런 수식들을 참고했었다. 다만 많은 시간동안 느꼈던 점은 제대로 이해하지 못하고 대충 넘어간 애들은 결국 내 것이 되지 못한 채 이후 논문들을 이해하는 과정에서 발목을 잡는다는 사실이었다. 또한 공부하면서 느꼈던 점은 생성 모델로서 무언갈 구현했다기 보다는 수학적 모델을 통해 생성 모델을 도출한다는 흐름이 논문의 수식 이해에 보다 도움이 되었다는 것이다. 

따라서 무작정 수식 전개를 이해하기보다는 근본적으로 diffusion이 대체 왜 생성 모델로 사용될 수 있는지, 그리고 그 <U>한계점과 해결책이 무엇인지</U> 이해하는 것이 가장 중요하다고 생각된다.

---

# Diffusion model의 문제점

여전히 **diffusion model**은 생성 속도가 느리다는 점을 극복하지 못했다. 근래에 이미지 쪽에서의 image manipulation이나 텍스트 쪽에서의 large language model based chatbot이 보다 다양한 사람들에게 서비스로 보급되기 시작한 이후, 사용자에게는 쾌적한 서비스의 공급/편의성이라는 측면과 사업자에게는 적은 리소스/비용이라는 이해관계가 맞붙기 시작했다. 결국 AI로 하여금 고퀄리티의 생산물을 만들어내는 것은 좋은데, 그게 오래 걸리면 무슨 소용일까. 마치 <U>배차간격이 긴 광역버스를 기다리는 퇴근길</U>과 같다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/234603450-76d2b35e-993c-42f7-aae3-de2ba2acc664.jpg" width="800">
</p>

GAN과 같은 implicit 생성 모델에 비해 가지는 모달리티의 안정성은 좋은데, 그걸 보장하기 위해서는 **trade-off**로 <U>시간과 연산량</U>을 지불할 수 밖에 없다는 것이다.

---

# 빠른 샘플링의 고안

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/234603561-58d2adc4-f683-4124-9bc2-1dda75a6f580.png" width="400">
    <img src="https://user-images.githubusercontent.com/79881119/234603471-f7852e02-c77f-4bfe-b43b-e43a19667484.png" width="700">
</p>

빠르게 생성한다는 것은 직관적으로 표현될 수 있다. 기존의 디퓨전이 **장인 정신**으로 한땀한땀(시간축 $t$에 따라서) 노이즈를 제거해가는 방식 대신에, **하이패스를 달아버려서** 한방에 노이즈로부터 샘플을 생성하는 것이다. 즉 기존의 GAN이 가능했던 빠른 샘플링인 $G(z)$를 디퓨전에 대해서도 가능하게 하고 싶다는 것이다. 이러한 생각에서 나온 방법들 중 일부는 다음과 같다.

- DDIM : Markovian process와 동일한 marginal likelihood를 가지는 Non-Markovian forward process를 정의하고, 이를 통해 샘플링 시퀀스의 time step을 간소화
- Diffusion model distillation : 샘플링 성능이 좋은 디퓨전 모델 ex) DDPM 을 사용하여 단일 step으로 좋은 샘플링이 가능하게끔 probability flow ODE를 학습

하지만 여전히 DDIM을 포함하여 probability flow ODE의 경우에도 샘플링의 속도를 빠르게 하면 할수록 발생하는 샘플 퀄리티의 하락을 무시할 수 없다. 샘플링 단계를 최소화하면서 샘플링 성능의 저하를 막는 것이 주요 포인트인데, 이게 기존 방식으로 해결하기에는 벅차다.  그나마 distillation 방법이 probability flow ODE에 대해 좋은 디퓨전 모델의 성능을 transfer하기 좋은 방법이긴 하지만, 결국 디퓨전 모델의 생성에 의존해야한다는 점 때문에 <U>학습 속도가 현저히 느려지게 된다</U>는 **bottleneck**에서 벗어날 수 없다.

---

# Related works

저자가 주장하는 consistency model의 개요는 이전의 Yang Song이 풀어냈던 diffusion 방식과 달라지지는 않았다. 이전 논문에서의 내용을 인용하면, 모든 diffusion process는 marginal likelihood $p(x_{0:T})$를 동일하게 가지는 ODE를 찾을 수 있다. 예컨데 원래의 diffusion SDE가 다음과 같다면,

\[
dx_t = \mu(x_t, t)dt + \sigma(t)dw_t 
\]

이 SDE와 동일한 marginal likelihood를 가지는 ODE는

\[
dx_t = \left( \mu(x_t, t) - \frac{1}{2}\sigma(t)^2 \nabla_x \log p_t(x_t) \right) dt
\]

위와 같이 표현할 수 있다. 해당 내용에 대한 증명은 **Appendix**로 Yang Song의 논문(SDE diffusion 논문)에 첨부되어 있다.

ODE로 변형했을 때 SDE에 대해 가지는 장점은 <U>stochastic한 diffusion coefficient를 가지지 않기 때문에</U>($dw$), probability flow ODE를 기준으로 starting point $x_0$를 잡는다면 미분 방정식의 solution이 그리는 trajectory를 따라가는 $x_T$까지의 모든 점 $x_t$에 대해 하나의 선으로 이을 수 있게 된다(아래 그림 참고). 확률 미분 방정식은 drift term이 방향만 정해줄 뿐, 실질적으로 뻗어나가는 구조는 랜덤한 요소가 좌우하기 때문에 starting point와 ending point만 알 뿐, 그 내부에서 각각의 $x_t$가 서로 교차하고 얽히는 과정을 알 수 없기 때문에 $1$대 $1$ mapping이 불가능하다는 단점이 있다. 하지만 ODE의 경우에는 trajectory를 그리는 요소에 시간축이라는 단일 변수가 관여할 수 있게 된다. 만약 특정 시점의 데이터인 $x_t$에 대해 score를 예측할 수 있는 모델인 $s_\phi(x, t) \approx \nabla \log p_t(x)$가 있다면, 위의 방정식은 perturbation kernel $p_t(x) = p_\text{data}(x) \otimes \mathcal{N}(0, t^2I)$에 대해 다음과 같은 form으로 나타낼 수 있다.

\[
\frac{dx_t}{dt} = -ts_\phi (x_t, t)
\]

참고로 이 configuration은 ‘Elucidating the design space of diffusion-based generative models’이라는 논문에 따른 것으로, **diffusion 확률 미분 방정식을 정의**할 때 drift term과 diffusion term을 디자인하게 되는 방식이 각 논문마다 다른 것을 알 수 있다. 위의 공식은 $\mu = 0, \sigma = \sqrt{2t}$를 따르는 미분 방정식의 solution이다.

고로 만약 $x_T\sim \mathcal{N}(0,T^2I)$를 정의하고 이에 따른 probability flow ODE $\frac{dx_t}{dt} = -ts_\phi (x_t, t)$를 풀어낸다면, $x_0$과 $x_T$를 잇는 하나의 trajectory를 구할 수 있게 된다. 미분 방정식을 푸는 방식은 Euler나 Heun solver와 같은 numerical 방법을 통해 함수의 형상을 예측하는 형태가 된다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/234603477-6678e3ad-a648-435c-89c2-bb7961322e32.png" width="">
</p>

이는 우리가 실제로 Analytic하게 풀어낼 수 없는(정해진 solution이 없는) 미분 방정식을 마주했을때, 아주 작은 변수의 변화에 대한 함숫값의 변화를 예측하는 과정을 의미한다. 그러나 그림을 보면 알 수 있듯이 실제로 numerical하게 풀어낸 미분 방정식의 해는 실제 solution과 오차가 클 수 밖에 없으며, 이는 시간 축이 길어지면 길어질수록, 샘플링 간격이 늘어나면 늘어날수록 variance가 높아지게 된다.

\[
\hat{x}_t,~t \in (0,~T)
\]

따라서 논문에서는 numerical instability를 보완할 목적으로 $t = \epsilon(0.002)$의 위치에서의 solution을 실제 데이터 샘플인 $x_0$에 근사한 값으로 간주했으며, time step의 총 수는 $T = 80$을 사용하였다.

Diffusion model은 결국 느린 sampling 속도가 가장 큰 문제점이라고 하였다. 물론 마찬가지로 ODE solver를 sampling에 사용하는 과정에서도 앞서 본 식과 같이 score model의 score 예측에 해당되는 $s_\phi(x, t)$가 발목을 잡게된다. 결국 numerical ODE solver 또한 퀄리티를 포기함으로써 속도를 증가시키는 방법이나 distillation을 사용할 수 밖에 없다.

하지만 이러한 노력에도 불구하고 기존 ODE solver는 꽤 좋은 퀄리티의 데이터를 생성하기 위해서는 단일 step으로는 불가능하다는 문제점이 발생하였다. Distillation을 하는 방식은 보통 DDPM과 같은 디퓨전의 prior에 의존하게 되는데, 결국 DDPM에서 각 time step에 대한 노이즈 데이터를 샘플링 해야하기 때문에 사용할 때 <U>연산량이 부담된다는 것을 해결할 수 없다</U>는 문제가 있다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/234603480-db2f2eb3-5e5e-4532-8aa1-70c66e000835.png" width="500">
</p>

바로 이러한 문제를 해결하고자 했던 논문 중 하나가 점진적으로 distillation을 수행하는 time step 수를 줄이는 progressive distillation이다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/234603487-edb4fe29-d5cb-402b-b7ef-0866993120e1.png" width="500">
</p>

처음부터 단일 trajectory을 모두 학습하려면 score 예측에 필요한 샘플이 그만큼 늘어나게 된다. 하지만 만약 여러 trajectory에 대해 부분적으로 학습된 ODE score estimator가 서로 연결되게끔 distillation 하면서 그 수를 줄여나가면, **굳이 처음부터 엄청난 수의 샘플을 사용하지 않고도** 충분히 좋은 성능을 보일 수 있다는 것이 그 방법이다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/234603492-70a828eb-da05-4c7f-bf2e-cfb1695ea092.png" width="600">
</p>

이 논문에서는 위와 같이 progressive distillation을 사용하지는 않지만 consistency distillation을 사용하여 ODE solver에 대한 예측과 prior에 대한 예측을 일치시키는 작업을 진행하게 된다. Consistency 논문에서 주된 비교 타겟으로 삼은 논문이 바로 [progressive distillation 논문](https://arxiv.org/pdf/2202.00512.pdf)이다.

---

# Consistency models

**Consistency model**을 언급할 적에 ‘새로운 생성 모델’이라고 언급하면서 논문이 시작된다. Consistency model은 diffusion process의 SDE를 기반으로 하는 probability flow ODE를 수학적 접근 프레임으로 삼는데, 이때 ODE를 풀어가는 방식에 만약 굳이 사전 학습된 DDPM에 의한 distillation이 불필요하게 된다면 이는 곧 scratch 부터 학습될 수 있는 새로운 생성 모델의 기본이 되는 것이다. 얼핏 보면 normalizing flow랑 비슷해보이기도 하지만 근본이 <U>디퓨전 확률 미분 방정식으로부터 출발</U>했기 때문에 확실히 다르다고 말할 수 있을 것 같다.

### Consistency model의 정의

 Probability flow ODE인

\[
dx_t = \left( \mu(x_t, t) - \frac{1}{2}\sigma(t)^2 \nabla_x \log p_t(x_t) \right) dt
\]

의 해가 되는 trajectory(궤도)를 $\\{x\_t \\}\_{t \in [\epsilon, T]}$ 라고 해보자. Consistency function은 함수 궤도 상에 있는 모든 점들을 $x_\epsilon$ 으로 한번에 보내는 함수를 의미한다.

\[
f : (x_t, t) \rightarrow x_\epsilon
\]

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/234603497-b5c391e5-7fc7-4c33-9e6b-63ad8529232d.png" width="550">
</p>

바로 위의 그림과 같이 표현할 수 있다. 초록색이 시간축 상에서 $\epsilon$부터 $T$까지 뻗어있는 PF ODE의 솔루션 궤도이며, 모든 시간축 상의 점들을 **태초마을로 귀환**시켜버리는 것이 논문에서 학습시키고자 하는 consistency model의 주된 목적이다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/234603447-6a5db911-d522-479e-96c9-321d0e009ac2.png" width="500">
</p>

그렇다는 의미는 다음과 같이 궤도 상의 모든 점은 함수 결과에 대해 consistency를 가진다고 볼 수 있다.

\[
f(x, t) = f(x_{t^\prime},t^\prime),~\forall t,~t^\prime \in \\{\tau \vert \epsilon \le \tau \le T\\}
\]

만약 <U>time argument가 고정</U>되어있다면(시간축 상에 발자국이 남아있다면), 역과정에 대해서도 invertible function이 된다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/234603500-a362aee4-fbfb-4943-9ff8-3cf4e9939f5b.png" width="600">
</p>

따라서 consistency model은 ODE를 통해 궤도를 예측하면서 남은 **발자국**의 출발점을 똑같은 곳인 $x_\epsilon$으로 보내는 과정이 된다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/234603457-969dd89b-f352-45d3-952e-bbc823803b5e.gif" width="500">
</p>

### Parameterization

딥러닝에서 가장 중요한 것은 학습이 가능하게끔 함수 parameter를 설정해주어야 한다는 점이다. 모든 형태의 consistency function은 boundary constraints(가장자리 조건)을 다음과 같이 가진다. 상당히 심플한데,

\[
f(x_\epsilon, \epsilon) = x_\epsilon 
\]

쉽게 말하자면 **태초 마을($x_\epsilon$)**에서 귀환($f(\cdot)$)을 하면 **태초 마을이** 나와야한다는 것이다. 굉장히 당연한 조건이라고 생각이 들 수도 있지만 현재 다루고 있는 내용이 미분 방정식의 solution인 연속 함수에 대한 내용이기 때문에 constraint를 제대로 설정하는 것이 매우 중요하다. 하는 방법은 총 두가지가 있을 수 있는데, 첫번째로는 다음처럼 함수를 case로 분류하거나

\[
f_\theta(x, t) = \begin{cases}x,&t = \epsilon \newline F_\theta(x, t),& \epsilon <t<T \end{cases}
\]

Skip point인 $t = \epsilon$에서 $c_\text{skip}(\epsilon) = 1$ 이고 $c_\text{out}(\epsilon) = 0$인 **미분 가능한 함수**를 통해 구현하는 방법이 있다.

\[
f_\theta(x, t) = c_\text{skip}(t)x + c_\text{out}(t) F_\theta(x, t)
\]

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/234603502-8ca6c9df-af07-4480-836c-c110b22cdca2.png" width="900">
</p>

표는 consistency model과 같은 방법론으로 접근한 여러 방법들에 대해 [관련 논문](https://arxiv.org/pdf/2206.00364.pdf)에서 참고하였는데, 보게 되면대부분 두번째 방법을 사용하는 것을 알 수 있고, 이 논문에서도 마찬가지로 두번째 방법을 사용하였다.

### Sampling

 잘 학습된 consistency model $f_\theta(\cdot, \cdot)$이 있다고 가정하게 되면, 단순히 알고 있는 prior로부터 샘플링을 진행한 뒤

\[
\hat{x}_T \sim \mathcal{N}(0, T^2I)
\]

그대로 함수(딥러닝 모델)에 넣으면 구할 수 있게 된다.

\[
\hat{x}\_\epsilon = f\_\theta(\hat{x}_T, T)
\]

따라서 single step generation을 할 수 있게 되는 것이다. 근데 만약 이런저런 이유로 **consistency 모델을 사용**하여 기존 디퓨전 모델과 같이 **multiple step generation**을 하고 싶다면 단순히 <U>태초 마을로 데려갔다가 다시 노이즈를 더했다가 하는 과정</U>을 반복하면 된다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/234603506-0f846807-dea8-4e77-9a81-f576867dc715.png" width="600">
</p>

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/234603509-ca5b4bde-cd26-46a0-be61-591b039491b8.png" width="550">
</p>

### Zero shot data editing

이와 같은 consistency model의 특징(prior를 기준으로 data와 대응되는 궤도 상의 어떤 점에서 출발하더라도 원래의 $x_0$로 수렴하는 성질)을 사용하게 된다면 image editing이나 manipulation을 zero shot으로 수행할 수 있다. 가장 간단하게 생각해볼 수 있는 것은 GAN, VAE와 같은 latent variable model에서 할 수 있는 interpolation이다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/234603454-c1fe1ac7-aacf-4aed-9040-bb4ada32befa.jpg" width="600">
</p>

Laent와 생성되는 sample이 parameter로 구성된 implicit decoder의 출력이 되는 GAN이나 VAE의 경우에는 샘플 $x_0$를 만들어내는 latent $z_0$ 그리고 샘플 $x_1$을 만들어내는 latent $z_1$사이의 보간을 통해 중간 이미지($\text{Image}(x_0, x_1)$)를 생성할 수 있고, 이는 곧 특징자 벡터를 자유롭게 사용하여 생성되는 이미지를 바꿀 수 있다는 장점이 된다.

\[
F_\Theta(\alpha \cdot z_0 + (1-\alpha) \cdot z_1) = \text{Image}(x_0, x_1) 
\]

확률 미분 방정식에서의 diffusion process를 그대로 사용하는 DDPM baseline의 경우에 prior sample인 $x_T$와 이에 대해 생성한 샘플 $F_{\Theta_{1:T}}(x_T) = x_0$이 $1$대 $1$ 대응이 아니라는 점을 생각해보자. 하나의 latent sample $x_T$가 포함된 모달리티에서 이에 대응될 수 있는 dataset 모달리티 샘플 $x_0^1, \cdots x_0^N$ 은 Markov process를 전제로 샘플링하기 때문에 latent interpolation이 image에서 유의미한 interpolation으로 이어지지 않는다는 문제가 있다. 그런데 이를 consistency model과 같이 Probability flow ODE의 solution에 대해 풀게 된다면 $x_T$는 더이상 data modallity에 대해 one to many mapping이 아니게 된다. 따라서 GAN이 가지는 장점 중 하나인 latent manipulation을 통한 <U>이미지 manipulation이 용이</U>하다는 특징을 가져갈 수 있다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/234603514-6ef113b5-a82d-4ef0-9626-8f4dbf318c19.png" width="600">
</p>

또한 추가적으로 sample의 modality와 더불어 condition이 들어가는 경우에도 zero-shot으로 사용할 수 있다는 장점이 발생한다. 예컨데 좋은 성능의 image inpainting, colorization 그리고 super-resolution 등등을 수행할 수 있는 디퓨전 기반의 모델은 모두 해당 task에 대한 목적을 가지고 explicit하게 학습이 전제되어야한다. 하지만 앞서 말했던 것과 같이 consistency model은 어떠한 수준의 noise에서도 $x_\epsilon$을 복구할 수 있게끔 학습되기 때문에 여러 noise level에 대한 denoising이 가능하며,

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/234603517-ef598c62-a4a4-4883-93b3-60bb501bbae1.png" width="700">
</p>



이를 다르게 생각한다면 어떠한 input이 들어가더라도**multiple step generation**을 수행하게 되면 임의의 input에 대해 그 시작점을 찾을 수 있게 되는 것이다(condition이 들어갈 때는 단순히 prior sampling 부분만 스킵하면 될 것 같음). 만약 input이 **grey image**라면 이를 consistency model에 대해 multistep(노이즈를 더하고 $x_0$를 예측하고를 수차례 반복)을 적용할 수 있다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/234603527-bcd67b05-4c1c-4cc0-b062-a8db6a1fc8fe.png" width="700">
</p>

이렇듯 딱히 <U>condition에 대해 따로 학습할 필요가 없다</U>는 부분은 아래와 같이 inpainting, super-resolution 그리고 SDEdit(painting to image)와 같은 task에 자연스럽게 사용될 수 있다는 장점을 부여해준다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/234603536-91cb5030-69f6-4067-8cc9-495050027f7b.png" width="700">
</p>

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/234603538-1e75c3b7-b61c-4239-96b2-62d87dc09eb1.png" width="700">
</p>

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/234603541-223f5440-e14f-4bf1-b06b-ff6a180c6380.png" width="700">
</p>

---

# Train consistency model

저자들이 앞서 밝힌 것과 같이 consistency model을 학습하는 방법으로는 꽤나 좋은 pre-trained source model의 score를 사용하여 distillation하는 방식과, 처음부터 학습하는 방식이 있다고 했었다. 그 두가지 방법에 대해 각각 소개하면 다음과 같다.

### Distillation을 통해 학습시키기

\[
dx_t = \left( \mu(x_t, t) - \frac{1}{2}\sigma(t)^2 \nabla_x \log p_t(x_t) \right) dt,~~\frac{dx_t}{dt} = -ts_\phi (x_t, t)
\]

앞서 봤던 PF ODE식을 생각해보자. 여기서 좌측 식만 보게되면 실제 데이터 분포에 대한 score를 구할 수 없기 때문에 학습된 네트워크의 score prediction을 대입하면 우측 식과 같이 empirical PF-ODE를 문제로 가져올  수 있다. 시간축 $[\epsilon,~T]$을 $N-1$개의 sub-interval로 분리한다고 생각해보자. 자르는 기준선에 대한 boundary condition $t_1 = \epsilon$ 그리고 $t_N = T$에 대해 증가하는 sequence $[t_1,~t_2,~\cdots,~t_N]$를 정할 수 있다. 시간축을 나누는 기준은 임의로 정할 수 있지만, 관련 논문 중 하나의 setting을 따라갔다.

\[
t_i = \left(\epsilon^{1/\rho} + \frac{i-1}{N-1} (T^{1/\rho} - \epsilon^{1/\rho}  )\right)^\rho,~\rho = 7
\]

물론 <U>샘플링이 촘촘할수록</U> numerical ODE solver가 실제 solution에 가까워지기 때문에 $N$의 값이 클수록 더 정확한 예측을 할 수 있게 된다. 아무튼 이렇게 solver가 예측한 특정 시점에서의 함숫값을 $\hat{x}_{t_n}^\phi$라 한다면,

\[
\hat{x}\_{t_n}^\phi := x\_{t_{n-1}} + (t_n - t_{n-1}) \Phi(x\_{t\_{n+1}}, t\_{n+1}; \phi)
\]

단일 step ODE solver의 update function인 $\Phi(\cdots; \phi)$에 대해 예측된 다음 함숫값은 위와 같다.  이때 $\phi$라는 파라미터가 ODE solving에 관여하는 이유는 지금 적용하고자 하는 score estimator가 empirical PF ODE를 풀고자하며, 이는 곧 사전 학습된 score estimator를 사용할 것임을 알려준다. Numerical ODE solver 중 가장 흔히 사용할 수 있는 <U>오일러 방식</U>을 적용하면 위의 식은,

\[
\hat{x}\_{t_n}^\phi := x\_{t\_{n-1}} - (t_n - t\_{n-1})t\_{n+1} s_\phi(x_{t\_{n+1}},~t\_{n+1})
\]

간단하게 이처럼 표현할 수 있다. 그런데 사실 SDE를 PF-ODE로 바꾸면서 생기는 오차는 실제 score estimate function 과의오차와 부합하게 된다. 따라서 이 부분에 대한 connection을 해주기 위해서 강제로 $1$ to many mapping을 만들어줄 수 있다.

\[
x \sim p_\text{data},~x = x+\eta \text{(Gaussian noise)} 
\]

이런 식으로 설정한 data point $x$를 기준으로, PF ODE 상의 인접한 data point $(\hat{x}\_{t_n}^\phi,~x_{t\_{n+1}})$를 구할 수 있고, 이때 $x_{t_{n+1}}$은 SDE의 transition kernel에 따라 $\mathcal{N}(x,~t^2\_{n+1}I)$의 분포에서 샘플링하게 된다(대충 아래 그림).

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/234603543-2607ff4a-647b-474c-835e-eb9551974072.png" width="600">
</p>

그리고 이렇게 샘플링한 adjacent point들에 대해 consistency network를 학습한다. 학습 방법은 간단하게 두 인접한 sample point(하나는 forward SDE에 따라 샘플링, 하나는 이렇게 샘플링된 애를 score estimator와 numerical ODE solver를 통해 궤도 예측)를 각각 네트워크에 통과한 결과가 서로 같게끔하면 된다.

\[
\mathcal{L}^N_{CD}(\theta, \theta^-;\phi) := \mathbb{E}(\lambda(t_n)d(f_\theta(x_{t_{n+1}}, t_{n+1}),~f_{\theta^-}(\hat{x}^\phi_{t_{n}},t_n)))
\]

$\lambda(\cdot)$는 시간에 따른 kenrel 분포 변화때문에 loss에 weight를 주기 위한 term이고 $d$는 두 예측 사이의 거리 metric, 학습의 주체가 되는 $\theta$가 student parameter로 loss에 대한 gradient descent를 받게 되고  $\theta^-$는 teacher parameter로 student parameter를 EMA 방식으로 가져간다. 흔히 알고있는 distillation 방법이랑 동일하다. 거리 메트릭은 이것저것 다 가능한데 이미지 생성에 주로 사용되는 MSE, L1 그리고 LPIPS를 해당 논문에서는 모두 실험했으며 weight term인 $\lambda(\cdot)$는 <U>심플하게 $1$로 고정해서 사용하는 것</U>이 모든 task 및 dataset에 대해 괜찮은 성능을 보였다고 밝힌다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/234603546-c1e191df-b7c5-4db4-9c1b-8208cd6cc737.png" width="500">
</p>

지금껏 정리한 학습 과정을 나타낸 **알고리즘 pseudo code**는 위와 같음. Numerical ODE가 가지는 bounded condition(numerical하게 푼 solution이 실제 solution과 가지는 오차가 특정 범위 내에 존재한다는 가정)과 consistency network $f$가 가지는 Lipshitz condition을 만족한다는 조건 상에서 loss function의 supremum 또한 수렴한다는 증명을 할 수 있다. 이는 곧 **empirical PF ODE(**Consistency model**), 보다 엄밀히 말하자면 consistency network**가 distillation되는 실제 SDE 궤도에 따라 Numerical ODE와 함께 수렴이 가능하다는 증거가 된다.

\[
\begin{aligned}
&\text{If local error uniformly bounded by }O((t_{n+1} - t_n)^{p+1}),\newline
&\sup_{n, x} \parallel f_\theta(x, t_n) - f(x, t_n;\phi)\parallel_2 = O((\Delta t)^p)
\end{aligned}
\]

해당 내용은 논문의 Appendix A.2 절에 수록되어있는데, 증명법은 간단하게 귀납법을 사용하면 가능하다(증명은 이 글에서 생략하겠다).

참고로 논문에서는 학습 주체가 되는 $f_\theta$를 student network가 아닌 online(학습되는) network, 그리고 EMA로 파라미터를 받는 $f_{\theta^-}$를 teacher network가 아닌 target(목적이 되는) network라고 이름지었다. Consistency distillation loss는 무한히 증가하는 time step sample $N$에 대해 학습될 때 target과 online parameter를 같게 만들 수 있으며, 이는 곧 distillation의 주체가 되는 **consistency network**가 완벽하게 <U>모든 정보를 이어받았다</U>고 이해할 수 있다.

### Isolation(단독으로) 학습시키기

위에서 소개한 방법은 consistency network를 score network의 정보와 ODE solver를 사용하여 어떤 식으로 consistency loss를 수렴시킬 수 있는지에 대해 증명하는 과정이었다. 이번에는 consistency model이 기존 diffusion 방식에서 벗어난 PF ODE 자체로의 가능성을 보여주며, 새로운 생성 모델의 시작이라는 기준이 된 학습법에 대해 언급하도록 하겠다.

Distillation 방식의 경우 사전 학습된 diffusion process model이 필요하고, 이를 통해 score estimation $s_\phi(x, t)$를 미분 방정식의 한 요소로 사용할 수 밖에 없었다. 만약 consistency model을 단독으로 학습시키고자 한다면 <U>해당 의존성을 없애버려야한다</U>(아래의 식에서 $\nabla_x \log p_t(x_t)$를 구해야함).

\[
dx_t = \left( \mu(x_t, t) - \frac{1}{2}\sigma(t)^2 \nabla_x \log p_t(x_t) \right) dt
\]

이를 score estimator 없이 구하는 방법은 다음과 같다.

구하고 싶은 **score**를실제 data의 **marginal distribution**에 대해 역으로 projection하면 적분식이 나온다.

\[
\nabla \log p_t(x_t) = \nabla_{x_t} \log \int p_\text{data}(x) p(x_t \vert x) dx
\]

그리고 $\log$에 대한 미분은 closed form으로 정리된다.

\[
\nabla \log p_t(x_t) = \frac{ \int p_\text{data}(x) \nabla_{x_t}p(x_t \vert x) dx}{\int p_\text{data}(x)p(x_t \vert x)dx}
\]

그리고 확률 분포 $p(x_t \vert x)$에 대한 미분은 log likelihood $\log (p(x_t \vert x))$에 대한 미분으로 치환할 수 있다.

\[
\nabla \log p_t(x_t) = \frac{ \int p_\text{data}(x) p(x_t \vert x)\nabla_{x_t}\log p(x_t \vert x) dx}{\int p_\text{data}(x)p(x_t \vert x)dx}
\]

분모와 분자를 정리하게 되면,

\[
\nabla \log p_t(x_t) = \frac{ \int p_\text{data}(x) p(x_t \vert x)\nabla_{x_t}\log p(x_t \vert x) dx}{p_t(x_t)}
\]

그리고 $x_t$는 적분의 주체가 되는 변수 
$x$와 상관없기 때문에 상수로 취급 가능하다.

\[
\nabla \log p_t(x_t) = \int\frac{  p_\text{data}(x) p(x_t \vert x)}{p_t(x_t)}\nabla_{x_t}\log p(x_t \vert x) dx
\]

앞부분의 식은 Bayes’ rule에 따라 조건부의 위치가 바뀌게 되고,

\[
\nabla \log p_t(x_t) = \int p(x \vert x_t) \nabla_{x_t} \log p(x_t \vert x) dx
\]

이는 $x_t$를 조건으로 하는 확률 분포에 따른 $x$에 대해 평균을 구하는 것과 같다.

\[
\nabla \log p_t(x_t) = \mathbb{E}(\nabla_{x_t} \log p(x_t \vert x) \vert x_t)
\]

조건부 확률은 diffusion process에서 가우시안 커널로 정의가 되었기 때문에

\[
-\mathbb{E}\left(\frac{x_t - x}{t^2} \vert x_t\right)
\]

이처럼 근사시킬 수 있다. 물론 가지고 있는 샘플 내에서 평균을 구하는 과정이 되기 때문에 numerical error는 존재할 수 밖에 없다. 아무튼 이렇게 구한 score를 사용하게 되면 score estimation을 해주는 pre-trained network 없이 샘플링이 가능하고, 이 샘플들을 통해 consistency network 학습이 가능하다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/234603552-c24a46a3-00b3-4db2-942f-329b570711f5.png" width="500">
</p>

길게 증명과정이 있었지만 process는 간단하게도 인접 샘플들을 모두 사전 정의한 diffusion SDE에 따라 생성, 이를 사용하여 consistency model을 학습시키게 된다. 해당 process를 따르는 consistency network 학습 loss는 다음과 같이 변한다. 다변수 standard gaussian 변수 $z \sim \mathcal{N}(0, I)$에 대해,

\[
\mathcal{L}^N_{CT}(\theta, \theta^-) := \mathbb{E}(\lambda(t_n)d(f_\theta(x + t_{n+1}z, t_{n+1}),~f_{\theta^-}(x + t_nz,t_n)))
\]

이와 같고, 마찬가지로 해당 loss를 수렴시키는 과정이 distillation loss를 수렴시키는 것과 결과적으로 동일함을 증명할 수 있다. 이 부분 증명이 진짜 중요하긴 한데 Taylor expansion을 통해 $o(\Delta t)$ 에 대한 term을 뽑아내는 방식으로 증명이 이루어져서 수식 길이가 너무 길어서 이것도 이 글에서는 패스.. 증명은 Appendix에서 Theorem 2.를 참고하면 된다.

---

# Experiments

실험은 비교적 직관적으로 각 factor에 대해 차례대로 실험해서 좋은 factor를 선별하는 과정을 거치게 된다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/234603556-63854500-a4b1-40fa-80b2-513fe79bd466.png" width="900">
</p>

우선 (a)를 보게 되면 <U>LIPS loss가 가장 효과적인 distance metric</U> $d(\cdot)$임을 알 수 있고, 바로 다음 실험인 (b)를 보게 되면 LPIPS를 고정 metric으로 활용하는 식으로 단계단계 실험을 진행한다. (b)에서는 solver에 대한 학습 효과를 보는데, 1차 근사만 고려하는 오일러보다는 2차 근사를 고려하는 Heun이 좀 더 좋은 성능을 보이는 것을 알 수 있다.

(c)에서는 앞서 bias를 줄이기 위해 테스트한 time step sample 수 $N$에 대한 경향성을 <U>조금 더 촘촘하게</U> 늘려서 실험했는데, 당연하게도 $N$이 커질수록 성능이 좋아진다. 이건 수식 증명에서도 볼 수 있는 내용. 
$N$이 어느 정도 증가하면 그 이후로는 <U>성능 수렴이 발생하는 것</U>도 함께 확인할 수 있다. 아무래도 numerical ODE에 따른 성능 향상의 bottleneck이지 않을까 생각해봄.

(d)는 마지막으로 CT를 사용한 학습 과정인데, 일단 FID가 현저히 떨어지는건 어쩔 수 없는 한계점. CT의 경우에는 CD와는 다르게 특정 <U>numerical ODE solver에 성능이 좌우되지 않기 때문</U>에(학습에 사용되는 샘플링은 사전에 정의된 커널로 함) solver를 사용할 필요가 없다. CT의 경우에는 distillation이 사용되지 않기 때문에 $N$에 대한 효과가 두드러졌는데, 예컨데 $N$이 너무 작으면 빠른 수렴은 가능했지만 샘플링 성능이 구리고 키우면 수렴은 좀 느려지지만 그대신 샘플링 성능은 오른다. 이 두 가지 장점을 같이 사용하기 위해 $N$을 <U>조금씩 증가시키면서 학습</U>시키는 방법(보라색)을 고안하였고, EMA factor $\mu$또한 이에 맞춰 점차 증가시키는 방법을 사용하였다. 그래프를 보면 빠른 성능 수렴 + 높은 샘플링 퀄리티(FID)를 보이는 것을 확인할 수 있다.

---

# 결론

실험 결과에는 few-step image generation, direct generation 및 zero-shot image editing과 관련된 여러 결과들이 첨부되어있다. 아무래도 consistency network가 첫번째로 empirical PF ODE의 수렴을 이용하여 한 번에 샘플링이 가능한 네트워크 학습을 고안한 만큼, 앞으로 기존 diffusion 샘플링이 하지 못했던 빠른 샘플링과 관련된 새로운 방향이지 않을까 생각된다. 만약 해당 방법론이 stable diffusion과 같은 zero-shot text to image generation과 결합되어 높은 퀄리티의 샘플링이 가능하다면 <U>새로운 사업상의 게임 체인저</U>로 등장할 수 있지 않을까 싶다.
