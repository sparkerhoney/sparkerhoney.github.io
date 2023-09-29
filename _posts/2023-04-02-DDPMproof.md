---
title: DDPM 수식 증명만 죄다 조져버리기
layout: post
description: paper mathematics
use_math: true
post-image: https://user-images.githubusercontent.com/79881119/232323912-84c15e40-ee39-4783-aea4-b3880fcf3dfa.gif
category: paper review
tags:
- DDPM
- Generative model
- AI
- Deep learning
---

# 들어가며...

정말 수식 증명만 조지는 내용이라 DDPM 자체에 대한 내용은 이전 글을 참고하면 좋다([참고 링크](https://junia3.github.io/blog/DDPM)). 근데 지금보니까 저 글을 쓸때도 완벽하게 이해하고 쓴 건 아닌 것 같다는 생각.. DDPM 논문 링크는 [여기](https://arxiv.org/abs/2006.11239). 이 어려운 벌써 논문이 3년이나 됐나 싶다.

# Forward and Reverse process

이미지 $x_0$에 작은 variance($\beta$)의 **가우시안을 아주 조금씩** 계속($T$만큼) 더해가다보면 최종 output $x_T$은 $x_0$와 같은 spatial dimension을 가지는 가우시안 분포가 된다. 논문에서는 각 process에서의 variance를 스케줄링하여 **고정값으로 사용**.

### Forward process $q(x_t \vert x_{t-1})$

\[
q(x_t \vert x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)
\]

그런데 알다시피 실제로 딥러닝을 학습시키려면 stochastic한 샘플링은 미분이 불가능하기 때문에 reparameterization trick을 사용한다. 각 time step에서 노이즈를 더할 때 이전 step의 output을 $\sqrt{1-\beta_t}$ 만큼 scaling 해주는 걸 알 수 있는데, 이렇게 하면 변수의 variance를 $1$로 유지할 수 있게됨.

\[
\left(\sqrt{1-\beta_t}\right)^2+\left(\sqrt{\beta_t}\right)^2 = 1
\]

수학적 귀납법을 통해 살펴보게 되면, $\alpha_t = 1-\beta_t$에 대해서

\[
\text{For}~\epsilon \sim \mathcal{N}(0, I), x_{t} = \sqrt{\alpha_t}x_{t-1} + \sqrt{1-\alpha_t}\epsilon
\]

일반적인 $t$ 번째 step에서의 식이 위처럼 표현되니까

\[
x_t = \sqrt{\alpha_t}\sqrt{\alpha_{t-1}}x_{t-2} + \sqrt{\alpha_t}\sqrt{1-\alpha_{t-1}}\epsilon_{t-2} + \sqrt{1-\alpha_t}\epsilon_{t-1}
\]

위와 같이 $t>1$인 상황에서 한단계 더 확장해서 표현 가능하고, 뒤의 epsilon들을 잘 합치게 되면,

\[
\sqrt{\alpha_t}\sqrt{1-\alpha_{t-1}}\epsilon_{t-2} + \sqrt{1-\alpha_t}\epsilon_{t-1} \rightarrow \left(\left(\sqrt{\alpha_t}\sqrt{1-\alpha_{t-1}} \right)^2 + \left(\sqrt{1-\alpha_t} \right)^2\right) \epsilon
\]

그래서 이게 잘 정리가 된 것이 다음과 같은 식.

\[
q(x_t \vert x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)
\]

이걸 또다시 **reparameterization**으로 표현하면 loss에서 써먹을 수 있게된다. 논문 전개과정에 주구장창 나오는 식.

\[
x_t := \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon,~\epsilon \sim \mathcal{N}(0, I)
\]

### Reverse process $p_\theta(x_{t-1} \vert x_t)$

Forward process의 posterior를 모방하도록 학습될 녀석. 분포를 예측하는 과정임. 정방향에 유사하게 역방향을 학습하고 싶다? 어디서 본 것 같은 워딩인데 자세히 노려보면 VAE에서의 접근과 같기 때문에 결국 얘도 lower bound를 최적화해야함.

\[
p_\theta(x_{0 : T}) := p(x_T) \prod_{t=1}^T p_\theta(x_{t-1} \vert x_t)
\]

$x_T$부터 샘플링하는 과정을 위의 joint distribution probability로 표현이 가능하고,

\[
p_\theta(x_{t-1} \vert x_t) := \mathcal{N}(x_{t-1}; \mu_{\theta}(x_t,~t), \Sigma_\theta (x_t,~t))
\]

Diffusion process의 가정인 ‘아주 작은 가우시안을 더해가는 과정’의 역과정은 ‘아주 작은 가우시안을 빼가는 과정’과 동일하기 때문에 각 포인트에서 **노이즈를 예측한 다음**에 빼주면서 샘플링이 가능하다.

위에서 식으로 전개한 forward process와 지금 언급하고 있는 reverse process를 ‘같게’ 만드는 것이 loss function이 될 것이다.

---

# Loss function

### Evidence lower bound(ELBO) in DDPM

\[
\mathbb{E}(-\log p\_\theta (x_0)) \leq \mathbb{E}\_q \left( -\log p_\theta(x_0 \vert z) -\log \frac{p_\theta(z)}{q_\phi(z \vert x_0)} \right)
\]

위의 식이 VAE에서 온건데, ELBO의 앞쪽 term은 $z$로부터 $x$를 reconstruction하는 부분에 대한 decoder 학습을 담당하고, 뒤쪽 term은 gaussian과 같은 pre-defined 분포로 가정한 $p_\theta(z)$와 encoder의 output $q_\phi(z \vert x)$의 분포가 가까워지게끔 하는 KL divergence에 해당된다. 위의 term에서 reconstruction과 KL divergence에서의 decoder 부분을 갈아끼우게 되면,

\[
\mathbb{E}(-\log p_\theta (x_0)) \leq \mathbb{E}\_q \left( -\log p_\theta(z) -\log \frac{p_\theta(x_0 \vert z)}{q_\phi(z \vert x_0)} \right)
\]

이처럼 되고, latent $z$를 time step에 대한 변수로 치환하면서 encoder의 변수를 제거하게 되면(DDPM에서는 parameteric 학습을 encoder에 대해 진행하지 않기 때문) 우리가 얻고자 하는 DDPM의 ELBO를 표현할 수 있다.

\[
\mathbb{E}(-\log p_\theta (x_0)) \leq \mathbb{E}\_q \left( -\log p_\theta(x_T) -\log \frac{p_\theta(x_{0:T-1} \vert x_T)}{q(x_{1:T} \vert x_0)} \right)
\]

그리고 각 process의 joint는 Markov process이기 때문에 summation으로 풀어쓸 수 있음.

\[
\mathbb{E}(-\log p_\theta (x_0)) \leq \mathbb{E}\_q \left( -\log p_\theta(x_T) -\log \prod_{t \ge 1} \frac{p_\theta(x_{t-1} \vert x_t)}{q(x_t \vert x_{t-1})} \right)
\]

뒤쪽 $\log$랑 production 위치를 바꾸면 summation이 됨.

\[
\mathbb{E}(-\log p_\theta (x_0)) \leq \mathbb{E}q \left( -\log p_\theta(x_T) - \sum_{t \ge 1}\log \frac{p_\theta(x_{t-1} \vert x_t)}{q(x_t \vert x_{t-1})} \right)
\]

그렇다면 굳이 왜 기존의 **VAE랑 다르게 위치를 바꾸었냐**라고 본다면 직관적으로는 forward process와 reverse process를 같게끔 학습하기 위함이라고 말할 수 있지만 보다 엄밀하게 표현하자면 VAE에서 학습시키는 decoder는 **implicit conditional probability**인 $p_\theta(x_0 \vert z)$를 학습하는 것이 주된 목적이었다면 Diffusion에서 학습시키는 decoder는 **diffusion process**가 나타내는 **미분 방정식을 학습**하는 것이 주된 목적이 되기 때문이다. 

헌데 여기서 문제가 발생하는 것은 encoder 역할을 수행했던 diffusion process $q$는 이전 step의 sample에 대해 노이즈를 더하는 과정 $q(x_t \vert x_{t-1})$을 수행했기 때문에 실제로 학습해야하는 그 역과정 $q(x_{t-1} \vert x_t)$의 분포는 알 수 없다는(intractable) 것.

\[
q(x_{t-1} \vert x_t) = \frac{q(x_t \vert x_{t-1})q(x_{t-1})}{q(x_{t})}
\]

본인은 이걸 한동안 이해 못했던거 같은데 쉽게 이해하는 법은 다음과 같음.

$x_{t-1}$에다가 더하는 노이즈를 알고 있기 때문에 $x_t$에 대한 확률 분포는 normal distribution으로 명확하게 알 수가 있음$(q(x_t \vert x_{t-1}))$ 근데 실제로 각 time step에서 노이즈가 더해진 애들의 분포 자체는 알 수 없으니까 $(q(x_t),~q(x_{t-1}))$ Bayes 식에서 intractable한 term이 두 개나 생겨서 분포에 접근하기가 힘듦.

그렇기 때문에 위의 식에 모든 term에 대해 $x_0$을 조건부에 추가하면 항상 분포를 알 수 있게 되므로(normal distribution의 누적이니까) tractable하게 바꿀 수 있다.

\[
\begin{aligned}
q(x_{t-1} \vert x_t,x_0) =& \frac{q(x_t \vert x_{t-1},x_0)q(x_{t-1},x_0)}{q(x_{t},x_0)} \newline
=& \frac{q(x_t \vert x_{t-1},x_0)q(x_{t-1}\vert x_0) q(x_0)}{q(x_{t} \vert x_0) q(x_0)} \newline
=& q(x_t \vert x_{t-1},x_0) \times \frac{q(x_{t-1} \vert x_0)}{q(x_t \vert x_0)}
\end{aligned}
\]

$x_0$에 조건부가 된 식이라서, $p_\theta(x_t \vert x_{t-1})$와 달라진 것 아니냐는 생각이 들 수도 있지만 위의 식은 $t > 1$인 모든 latent에 대해 $x_0$에 독립 분포를 가진다는 특징이 있다(마르코프 프로세스니까). 그래서 최종 마무리가 된 loss function을 다음 식처럼 나타낸다.

\[
D_{KL}(q(x_T \vert x_0) \vert\vert p_\theta(x_T)) -\sum\_{t > 1} D_{KL} (q(x_{t-1} \vert x_t, x_0) \vert\vert p_\theta(x_{t-1} \vert x_t)) -\mathbb{E}\_q\left(\log p_{\theta}(x_0 \vert x_1) \right)
\]

유도 과정은 앞서 본 식에서 $t>1$인 부분에 대해 posterior를 조건부로 바꿔서 tractable하게 만들 수 있으니까, 정리하게 되면 다음과 같이 나온다.

\[
\begin{aligned}
\mathcal{L} \le& \mathbb{E}\_q\left(-\log (p_\theta(x_T))-\sum_{t=2}^T \log \frac{p_\theta(x_{t-1} \vert x_t)}{q(x_t \vert x_{t-1})} -\log \frac{p_\theta(x_0 \vert x_1)}{q(x_1 \vert x_0)} \right) \newline
\le& \mathbb{E}\_q\left(-\log (p_\theta(x_T))-\sum_{t=2}^T \log \left( \frac{p_\theta(x_{t-1} \vert x_t)}{q(x_{t-1} \vert x_t, x_0)} \times \frac{q(x_{t-1} \vert x_0)}{q(x_t\vert x_0)}\right) -\log \frac{p_\theta(x_0 \vert x_1)}{q(x_1 \vert x_0)} \right) \newline
\le& -\sum_{t > 1} D_{KL} (q(x_{t-1} \vert x_t, x_0) \vert\vert p_\theta(x_{t-1} \vert x_t))+\mathbb{E}\_q\left(-\log (p_\theta(x_T))-\log \frac{q(x_1 \vert x_0)}{q(x_T \vert x_0)}-\log \frac{p_\theta(x_0 \vert x_1)}{q(x_1 \vert x_0)} \right) \newline
\le& D_{KL}(q(x_T \vert x_0) \vert\vert p_\theta(x_T)) -\sum_{t > 1} D_{KL} (q(x_{t-1} \vert x_t, x_0) \vert\vert p_\theta(x_{t-1} \vert x_t)) -\mathbb{E}\_q\left(\log p_{\theta}(x_0 \vert x_1) \right)
\end{aligned}
\]

### ELBO into objective function

위의 식은 수식일 뿐이고 실제로 최적화 과정에서 사용되는 식은 이를 가우시안 분포에 대해 다시 정리한 식이다. 위의 ELBO를 보게 되면 크게 세 부분으로 정리되는데, 이 중 가장 앞부분의 KL divergence는 diffusion process를 통해 **자연스럽게 만족되는 식**(gaussian 분포를 따라가는 것)이기 때문에 실제 loss function에서 제외된다. 원래 VAE에서는 해당 부분이 encoder의 output을 가우시안 분포로 정규화하는 KL divergence term으로 사용되는데, diffusion에서는 굳이 $q$를 학습시킬 필요가 없다보니 빠진다고 생각할 수 있다.

\[
D_{KL}(q(x_T \vert x_0) \vert\vert p_\theta(x_T))
\]

그리고 가장 마지막 부분인 log likelihood 식이 가장 이해하기 어려운데, 이를 먼저 언급하고 넘어가도록 하겠다.

\[
\mathbb{E}\_q\left(\log p_{\theta}(x_0 \vert x_1) \right)
\]

이 식을 다시금 이해하자면, noise가 아주 조금 더해진 $x_1$에서 $x_0$로 reconstruction할 때의 모든 확률을 구해야한다. 중간 과정의 경우에는 분포 사이의 KL divergence 식이기 때문에 확률 분포를 굳이 확률값으로 매핑할 필요가 없었지만, 이 식에서는 확률값 자체를 구해야하기 때문에 적분을 취해야한다.

우선 단순화시켜 생각하기 위해, $\log$ likelihood의 likelihood 자체만 따로 빼내서 보도록 하자.

\[
p_\theta(x_0 \vert x_1)
\]

$p_\theta$는 언급했던 것과 같이 $x_1$를 통해 $x_0$을 구성하는 각 요소들을 추출할 확률 분포를 구하는 diffusion process의 역과정을 학습한 neural network이다.

\[
p_\theta(x_0 \vert x_1) \sim \mathcal{N}(x_0;~\mu_\theta(x_1, 1), \sigma_1^2)
\]

확률값을 구하기 위해서는 이미지 상의 ‘모든 픽셀’에 대해 주어진 확률 분포를 적분해야만 구할 수 있다. 이때, 이미지가 $[-1, 1]$로 정규화된 상태라고 생각한다면 가우시안 분포에서 $-1$보다 작은 value는 모두 $-1$로 확률을 매핑하고 $1$보다 큰 value는 모두 $1$도 확률을 매핑한다고 생각할 수 있다. 또한 원래 이미지 전체는 $0 \sim 255$ 의 discrete RGB value로 표현되기 때문에 다음과 같이 변수를 $1/255$로 끊어서 확률매핑이 가능하다.

예를 들어 $x_1$에서 $x_0$으로 매핑될 때, ground truth에 따르면 $0$이라는 값이 나와야한다고 가정해보자.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/232325472-ed7e91ec-cd58-4826-8641-e3e110aee05d.png" width="1000"/>
</p>

그렇다면 $p_\theta(x_0 \vert x_1)$는 해당 값 근방을 적분 구간으로 삼아 확률분포를 적분한 값으로 정의할 수 있는 것이다(아래 그림에서 노란색 부분의 넓이).

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/232325475-c42d8a7f-800e-4567-91a2-f2951bcf2314.png" width="900"/>
</p>

이를 모든 픽셀에 대해 곱하게 되면, 전체 픽셀 ($1\sim D$)에 대한 joint probability를 구할 수 있다.

\[
p_\theta(x_0 \vert x_1) = \prod_{i=1}^D \int_{\delta_{-}(x_0^i)}^{\delta_+(x_0^i)} \mathcal{N}(x; \mu_\theta^i (x_1, 1), \sigma_1^2) dx
\]

이때 적분 구간인 $[-\delta_-(x_0^i),~+\delta_+(x_0^i)]$는 앞서 말했던 것과 같이 $-1$보다 작은 값들과 $1$보다 큰 값들을 각각 원래 이미지의 $-1$과 $1$에 매핑하고 나머지는 $1/255$만큼 간격을 준다고 생각할 수 있다.

\[
\delta_+(x) = \begin{cases}
\infty,&\text{if }x=1 \newline
x+\frac{1}{255},&\text{if }x<1
\end{cases},~~\delta_-(x) = \begin{cases}
-\infty,&\text{if }x=-1 \newline
x-\frac{1}{255},&\text{if }x>-1
\end{cases}
\]

### 중간과정 Optimization($L_{1:T-1}$)

결국 위에서 쭉 말한 내용은 $L_T$는 굳이 필요없다는 내용이었고 $L_0$는 확률 분포의 적분을 통해 구할 수 있다는 말이었다. 그렇다면 실제로 더해진 노이즈를 예측해서 빼는 과정인 $L_{1:T-1}$를 KL divergence 식으로 구하면 어떻게 될까?

\[
\sum_{t > 1} D_{KL} (q(x_{t-1} \vert x_t, x_0) \vert\vert p_\theta(x_{t-1} \vert x_t))
\]

해당 식에서 $q(x_{t-1} \vert x_t, x_0)$를 가우시안 분포로 나타내는 증명과정은 다음과 같다. 앞서 유도한 바와 같이 기존의 prior를 다음과 같이 고쳤었다.

\[
q(x_{t-1} \vert x_t,x_0) = q(x_t \vert x_{t-1},x_0) \frac{q(x_{t-1} \vert x_0)}{q(x_t \vert x_0)}
\]

그리고 우리는 식을 구성하는 각각의 분포를 다음과 같은 tractable한 가우시안 형태로 알고 있다.

\[
\begin{aligned}
q(x_t \vert x_0) \sim& \mathcal{N}(\sqrt{\bar{\alpha}\_t}x_0, (1-\bar{\alpha}\_t)I) \newline
q(x\_t \vert x\_{t-1}) \sim& \mathcal{N}(\sqrt{1-\beta_t}x\_{t-1},\beta_tI)
\end{aligned}
\]

오호라 위의 식을 그대로 가우시안 식으로 바꿔쓰게 되면 가우시안 앞부분에 붙는 상수는 무시하고 exponential 안의 식만 따로 정리할 수 있다.

\[
\begin{aligned}
q(x\_{t-1} \vert x_t, x_0) \propto& \exp \left(-\frac{1}{2} \left( \frac{(x_t - \sqrt{\alpha_t}x_{t-1})^2}{\beta_t}  + \frac{(x_{t-1} - \sqrt{\bar{\alpha}_{t-1}}x_0)^2}{1-\bar{\alpha}\_{t-1}} - \frac{(x\_t - \sqrt{\bar{\alpha}\_t}x_0)^2}{1-\bar{\alpha}\_t} \right) \right) \newline
=& \exp \left(-\frac{1}{2} \left( \frac{x\_t^2 - 2\sqrt{\alpha_t}x\_t x\_{t-1}+\alpha\_tx\_{t-1}^2}{\beta\_t}  + \frac{x\_{t-1}^2 - 2\sqrt{\bar{\alpha}\_{t-1}}x\_0x\_{t-1} + \bar{\alpha}\_{t-1}x\_0^2}{1-\bar{\alpha}\_{t-1}} - \frac{x\_t^2 -2\sqrt{\bar{\alpha}\_t}x\_0x\_t + \bar{\alpha}\_tx\_0^2}{1-\bar{\alpha}\_t}\right) \right)
\end{aligned}
\]

조금 복잡해 보이긴 하는데 여기서 $x_{t-1}$ 부분에 대해서 quadratic(2차 함수) 형태가 되도록 공통 변수로 묶어주게 되면,

\[
=\exp \left(-\frac{1}{2} \left( ( \frac{\alpha_t}{\beta_t} +\frac{1}{1-\bar{\alpha}\_{t-1}} )x\_{t-1}^2 - (2\frac{\sqrt{\alpha}\_tx\_t}{\beta_t} + 2\frac{\sqrt{\bar{\alpha}\_{t-1}}x\_0}{1-\bar{\alpha}\_{t-1}})x_{t-1} + C(x\_t, x\_0) \right) \right)
\]

위와 같이 정리할 수 있다. 해당 식을 가우시안 분포라고 생각하게 되면 평균은 $ax^2+bx+c$  형태의 이차식에서 $-b/2a$와 같기 때문에

\[
\begin{aligned}
\tilde{\mu}(x\_t,~x\_0) =& \frac{\frac{\sqrt{\alpha}\_tx\_t}{\beta\_t} + \frac{\sqrt{\bar{\alpha}\_{t-1}}x\_0}{1-\bar{\alpha}\_{t-1}}}{\frac{\alpha\_t}{\beta\_t} +\frac{1}{1-\bar{\alpha}\_{t-1}}} = \left( \frac{\sqrt{\alpha}\_tx\_t}{\beta\_t} + \frac{\sqrt{\bar{\alpha}\_{t-1}}x_0}{1-\bar{\alpha}\_{t-1}}\right) \times \frac{\beta\_t (1-\bar{\alpha}\_{t-1})}{1-\bar{\alpha}\_t} \newline
=& \frac{\sqrt{\alpha}\_t(1-\bar{\alpha}\_{t-1})}{1-\bar{\alpha}\_t}x\_t + \frac{\beta\_t \sqrt{\bar{\alpha}\_{t-1}}}{1-\bar{\alpha}\_t}x\_0
\end{aligned}
\]

위와 같이 나타낼 수 있고 분산은 $1/(-2a)$과 같기 때문에

\[
\tilde{\sigma}\_t^2 = \beta\_t \times \left( \frac{1-\bar{\alpha}\_{t-1}}{1-\bar{\alpha}\_t} \right)
\]

이처럼 나타낼 수 있다. DDPM 논문에서는 해당 부분을 제대로 증명하지 않고 넘어가서 갑자기 붕 떴던 기억때문에 남겨놓는다.

이제 실제로 $p\_\theta(x\_{t-1} \vert x_t)$와 $q(x\_{t-1} \vert x\_t, x\_0)$ 간의 KL divergence를 최소화하는 식을 살펴보면 $\mathcal{N}(x\_{t-1}; \mu_\theta(x\_t,t),~\Sigma\_\theta(x_t, t))$는 해당 논문에서 미리 스케쥴링해서 학습할 파라미터로 설정하지 않았기 때문에 $\tilde{\sigma}\_t^2 = \beta\_t \times \left( \frac{1-\bar{\alpha}\_{t-1}}{1-\bar{\alpha}\_t} \right)$라고 할 수 있다. 실제 실험에서는 단순히 $t$번째 step에서의 variance인 $\beta_t$를 $\sigma_t^2$로 사용해도 큰 차이가 없다고 언급하는데, 이는 사실상 variance 누적곱에 해당되는 $\bar{\alpha}\_t = \prod\_{\tau=1}^t(1-\beta\_\tau)$ 가 step 수가 크기 때문에 $t$에 따라 큰 차이를 보이지 않기 때문이라고 생각했다(아니라면 말고). 암튼 결론은 $\mu_\theta(x\_t,~t)$만 예측하면 된다는 소리..

 KL divergence 식을 가우시안에 적용하게 되면 exponential이 벗겨지면서 다음과 같이 간단하게 표현 가능하다. 아래 식에서 $C$는 파라미터에 무관한 모든 term을 의미한다. 

\[
L_{t-1} := \mathbb{E}\_q \left( \frac{1}{2\sigma\_t^2} \vert\vert \tilde{\mu}\_t(x\_t, x\_0) - \mu\_\theta(x\_t, t)  \vert\vert^2 \right)+C
\]

근데 이 식을 그대로 쓰면 stochastic 부분땜에 미분이 불가능하다. 써 있는 $x_t$를 죄다 reparameterization($x_t = \sqrt{\bar{\alpha}\_t}x\_0+\sqrt{1-\bar{\alpha}\_{t}}\epsilon$)으로 바꾸면 된다. 근데 이렇게 쓰면 너무 장황하니까 $x\_t(x\_0,\epsilon)$으로 표현하고자 함. 참고로 $\tilde{\mu}\_t$에 있는 $x\_0$도 process 상에서 $x\_t$를 통해 예측되는 $x\_0$이기 때문에 reparameterization하여 $x\_0 = \frac{x\_t}{\sqrt{\bar{\alpha}\_t}}-\frac{\sqrt{1-\bar{\alpha}\_t}}{\sqrt{\bar{\alpha}\_t}}\epsilon$로 쓰면 된다.

\[
L_{t-1} -C = \mathbb{E}\_q \left( \frac{1}{2\sigma_t^2} \left\vert\left\vert \tilde{\mu}\_t\left(x\_t(x_0,\epsilon), \frac{x\_t(x\_0,\epsilon)}{\sqrt{\bar{\alpha}\_t}}-\frac{\sqrt{1-\bar{\alpha}\_t}}{\sqrt{\bar{\alpha}\_t}}\epsilon\right) - \mu\_\theta(x\_t(x\_0,\epsilon), t) \right\vert\right\vert^2 \right)
\]

앞서 구했던 식에 대입하면,

\[
\begin{aligned}
\tilde{\mu}(x\_t,~x\_0) =& \frac{\sqrt{\alpha}\_t(1-\bar{\alpha}\_{t-1})}{1-\bar{\alpha}\_t}x\_t + \frac{\beta\_t \sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}\_t}x\_0 \newline
=& \frac{\sqrt{\alpha}\_t(1-\bar{\alpha}\_{t-1})}{1-\bar{\alpha}\_t}x\_t(x\_0,\epsilon) + \frac{\beta_t \sqrt{\bar{\alpha}\_{t-1}}}{1-\bar{\alpha}\_t}\left(\frac{x\_t(x_0,\epsilon)}{\sqrt{\bar{\alpha}\_t}}-\frac{\sqrt{1-\bar{\alpha}\_t}}{\sqrt{\bar{\alpha}\_t}}\epsilon\right) \newline
=& \frac{1}{\sqrt{\alpha\_t}}x_t(x_0, \epsilon)-\frac{\beta\_t}{\sqrt{(1-\bar{\alpha}\_t)\alpha\_t}}\epsilon \newline
=& \frac{1}{\sqrt{\alpha\_t}}\left( x\_t(x\_0, \epsilon)-\frac{\beta\_t}{\sqrt{1-\bar{\alpha}\_t}}\epsilon\right)
\end{aligned}
\]

위와 같다. 따라서 다시 식에 해당 값을 대입하게 되면,

\[
\mathbb{E}\_q \left( \frac{1}{2\sigma_t^2} \left\vert\left\vert \frac{1}{\sqrt{\alpha_t}}\left( x\_t(x\_0, \epsilon)-\frac{\beta_t}{\sqrt{1-\bar{\alpha}\_t}}\epsilon\right) - \mu\_\theta(x\_t(x\_0,\epsilon), t) \right\vert\right\vert^2 \right)
\]

요렇게 됨. 그런데 여기서 네트워크는 굳이 $x_t(x_0, \epsilon)$를 구할 필요가 없게 된다. 왜냐면 time step-$t$에서는 이미 input으로 $x_t$가 주어지고, $x_{t-1}$를 만들기 위한 분포 예측 과정이므로 forward 과정에서 stochastic하게 더해진 epsilon 부분만 예측하면 되는 것.

\[
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}}\_t} \epsilon\_\theta(x_t, t)\right)
\]

따라서 $x_t$에서 $x_{t-1}$를 샘플링할 때 계산하는 것은 $x_{t-1}$에 대해 예측된 평균인 $\frac{1}{\sqrt{\alpha\_t}}\left( x\_t - \frac{\beta\_t}{\sqrt{1-\bar{\alpha}}\_t} \epsilon\_\theta(x\_t, t)\right)$에 사전에 정의된 variance term인 $\sigma_tz$, $z\sim\mathcal{N}(0, I)$을 더하면 된다.

\[
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left( x_t - \frac{\beta\_t}{\sqrt{1-\bar{\alpha}}\_t} \epsilon\_\theta(x_t, t)\right)+\sigma\_tz
\]

그렇기 때문에 $L_{1:T-1}$에 대한 loss는 아주 간단하게 표현 가능하다($x_t$에 대한 부분은 제외하고 예측하면됨).

\[
\mathbb{E}\_{x\_0,\epsilon} \left( \frac{\beta\_t^2}{2\sigma\_t^2 \alpha\_t(1-\bar{\alpha}\_t)} \parallel \epsilon - \epsilon\_\theta(\sqrt{\bar{\alpha}\_t}x\_0 + \sqrt{1-\bar{\alpha}\_t}\epsilon, t) \parallel^2\right)
\]
