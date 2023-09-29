---
title: Improved DDPM + Diffusion beats GAN + Classifier free diffusion guidance 논문 리뷰
layout: post
description: Improve diffusion models
use_math: true
post-image: https://user-images.githubusercontent.com/79881119/235350592-1e8e47d9-7d84-48bc-b0eb-b8d2a499600b.gif
category: paper review
tags:
- Score based model
- Diffusion model
- Generative model
- AI
- Deep learning
---

# 들어가며…

이번에 리뷰할 논문은 총 3개 시리즈로, 전체적으로 디퓨전 모델(DDPM, DDIM)에 기반하여 **작성되었다는 점**이 공통점이고, 모두 diffusion model의 sampling quality를 높이기 위한 노력으로 이어진다고 볼 수 있다. 그 중 [Improved DDPM](https://arxiv.org/abs/2102.09672)의 경우 DDPM에서 가장 baseline 실험만 진행된 점에 추가로 몇몇의 modification을 통해 샘플의 log likelihood를 높일 수 있음을 보인 논문이며 [Diffusion beats GAN](https://arxiv.org/abs/2105.05233) 논문은 보다 다양한 architecture ablation과 classifier guidance를 제시하여디퓨전 모델이 GAN 이상의 샘플링 퀄리티를 보일 수 있음을 보여주었다. 마지막으로 [classifier-free diffusion](https://arxiv.org/abs/2207.12598)은 앞서 언급한 classifier guidance가 가지는 한계점과 문제점을 언급하며 conditional generation의 장점과 classifier의 explicit한 학습으로부터 자유로워질 수 있는 방법을 제시한다. 증명할 부분 자체는 많지 않기 때문에 한번에 다루는 것이 좋을 것 같다고 생각하여 정리해보려고 한다.

---

# Improved DDPM

DDPM이 <U>새로운 생성 모델의 학습법</U> 및 <U>샘플링 모델</U>로 딥러닝 씬에서 주목받기 시작하면서 CIFAR-10이나 LSUN같은 데이터셋 외에도 샘플의 diversity가 다양한 ImageNet과 같은 데이터셋의 학습에도 유용할 지 의문이 발생하기 시작했다. GAN의 경우에는 빠른 샘플링 속도 및 높은 샘플링 퀄리티로 주목을 받아왔었지만 학습의 불안정성과 샘플 생성 시 다양성이 떨어진다는 문제를 해결하기 힘들었고, VAE와 같은 likelihood 방법은 안정성과 샘플 다양성을 보장할 수 있지만 그 대신 샘플링 속도나 퀄리티가 GAN에 비해 떨어진다는 문제를 해결하기 힘들었다. 이런 와중에 DDPM이 등장한 것이다. 디퓨전 모델도 지금도 그렇긴 하지만 만능이라고 할 수 없었다. 새로운 방법론으로 제시가 되었을 뿐, 네트워크 구조나 학습법에 대한 future work는 여전히 과제로 남아있는 상태였던 것이다.

이 논문에서는 크게 두 가지 방법을 통해 샘플링 퀄리티를 높인다. 첫번째는 hybrid objective로, 기존 DDPM에서는 Variational lower bound(VLB)를 수정한 simplified loss를 사용했으나, 이 논문에서는 여기에 추가로 VLB loss를 사용하여 최적화를 하였다.

두번째는 고정된 variance가 아난 학습된 variance를 사용하였고, 이를 통해 DDPM이 수백의 forward process를 통해 좋은 퀄리티의 샘플들을 만들었던 점과 비교하여 더 적은 forward pass($50$)로도 이를 달성할 수 있었다. DDIM([블로그 글 참고](https://junia3.github.io/blog/ddim))에서는 non-Markovian process에 기반한 새로운 샘플링 방법을 고안했었는데, 이와는 다르게 DDPM의 Markovian process 자체는 유지하는 방향으로 연구를 진행한 것이다.

### 기존 DDPM

DDPM 공식에 대한 모든 정리는 이전 게시글에 다뤄놓았기 때문에 증명 부분을 제외하고 보면 다음과 같다. DDPM의 기본 원리는 data distribution $x_0 \sim q(x_0)$가 있을 때, forwarding noise process를 아주 작은 가우시안 노이즈를 더해가는 방식으로 정의한다.

\[
q(x_t \vert x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)
\]

이때 충분히 큰 시간 $T$ 동안 잘 scheduling된 $\beta_t$가 있다면 parameterized prior를 $x_T \sim \mathcal{N}(0, I)$에서 샘플링할 수 있게된다. 즉, forward process에 의해 점차 가우시안 노이즈에 가까워진다는 것이다. 그렇다면 이 조건부 확률의 반대 방향인 $q(x_{t-1} \vert x_t)$를 알 수만 있다면 임의의 가우시안 노이즈로부터 $x_0$를 샘플링할 수 있게 되는데, 식을 보면 알 수 있겠지만 이 부분은 <U>tractable하지 않다</U>.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/235350616-84dda90c-0784-4dd0-a515-78d964a55385.png" width="600">
</p>

그렇기에 파라미터를 가진 neural network를 통해 각 step의 noised sample $x_t$로부터  $p_\theta(x_{t-1} \vert x_t)$를 예측하고, 이를 통해 점차 노이즈를 제거하여 $x_0$를 샘플링하고자 하는 것이다. 그리고 아주 작은 가우시안 노이즈를 더하는 과정의 역과정은 곧 <U>가우시안 노이즈를 빼는 과정</U>으로 approximate이 가능하다.

\[
p_\theta(x_{t-1} \vert x_t) := \mathcal{N}(x\_{t-1}; \mu_\theta (x_t, t), \Sigma_\theta (x_t, t))
\]

그리고 이를 최적화하는 variational lower bound 공식은 다음과 같이 정리된다.

\[
\mathcal{L} 
\le D_{KL}(q(x_T \vert x_0) \vert\vert p_\theta(x_T)) -\sum_{t > 1} D_{KL} (q(x_{t-1} \vert x_t, x_0) \vert\vert p_\theta(x_{t-1} \vert x_t)) -\mathbb{E}\_q\left(\log p_{\theta}(x_0 \vert x_1) \right)
\]

맨 앞부분은 충분한 시간 $T$에 대해 임의의 데이터셋 $x_0$를 perturbation하게 되면 자연스럽게 만족하는 식이므로 $0$에 가깝다고 생각할 수 있다. 따라서 최적화에 필요한 식이 아니게 된다. 중간의 식은 역과정을 예측하는 네트워크가 forward process의 posterior를 잘 따라갈 수 있게끔 설정한 KL divergence 식이 된다. 마지막으로 $x_1$에서 $x_0$를 생성하는 과정은 $256$의 RGB 데이터로 구성되는 이미지의 확률을 projection하기 위해 설정된 식이다. 디테일한 증명 및 loss 각 term에 대한 설명은 DDPM 게시글에서 확인하면 된다([참고 링크](https://junia3.github.io/blog/DDPMproof)).  아무튼 이 식을 단순화하여 나타낸 것이 곧 다음과 같은 simplified loss이다.

\[
\begin{aligned}
&\mathbb{E}\_{x_0,\epsilon} \left( \frac{\beta\_t^2}{2\sigma\_t^2 \alpha_t(1-\bar{\alpha}\_t)} \parallel \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}\_t}x\_0 + \sqrt{1-\bar{\alpha}\_t}\epsilon, t) \parallel^2\right) \newline
\approx& \mathbb{E}\_{t, x_0, \epsilon} \left( \parallel \epsilon - \epsilon_\theta(x\_t, t) \parallel^2\right) 
\end{aligned}
\]

### DDPM의 log-likelihood 증가시키기

DDPM에서 흔히 생성된 샘플의 퀄리티를 평가하는 메트릭인 FID나 IS(Inception Score)는 좋은 수치를 보여주었으나 log-likelihood 수치는 잘 달성하지 못한 모습을 보여주었다. Log likelihood는 generative model이 data distribution의 mode를 얼마나 잘 반영하는지 나타내는 지표이다. 쉽게 설명하자면  log-likelihood를 최적화하는 것이 곧 generative model로 하여금 data distribution의 전체적인 형태를 잘 잡아내도록 하게 할 수 있다. 아무리 샘플을 잘 만들어내더라도 실제 데이터 분포의 일부분만 잘 반영하는 네트워크는 해당 데이터를 ‘잘’ 만들어낸다고 판단하기 어렵기 때문이다. DDPM이 대체 왜 log likelihood를 제대로 반영하지 못하는지에 대해 분석한 것이 바로 이 논문이며, log likelihood를 높이고자 방법을 찾고 이를 적용하는 것이 <U>실제 샘플링 퀄리티에 큰 도움이 될 수 있다</U>는 전개 방향이 된다.

### Learnable Standard deviation(Variance)

DDPM에서는 저자가 사전에 정의한 variance를 고정으로 사용한다($\sigma_t^2I$). 여기서 신기한 점은 $\sigma^2_t$를 $\beta_t$를 정의하는 것과 forward process의 posterior로 유도된 $\tilde{\beta}\_t = \frac{1-\bar{\alpha}\_{t-1}}{1-\bar{\alpha}\_t} \beta_t$를 사용했을 때의 샘플링 퀄리티가 큰 차이가 없다는 사실이다. 차이가 난다고 하면 전자를 사용하면 $q(x_0)$에 대해서 $t$번째 kernel의 variance가 isotropic Gaussian이 되고 후자를 사용하면 delta function이 된다는 점이다. $\beta_t$와 $\tilde{\beta}_t$를 variance가 가질 수 있는 양단의 기준점이라고 한다면, 왜 해당 파트가 샘플링 성능에 큰 영향을 끼치지 않는지에 대한 이유가 중요해진다. 사실 이 부분에 대한 논의를 기존 DDPM 게시글에서 제대로 언급하지 못했었는데, 그때 나름대로 생각했던 말을 인용하자면

> 실제 실험에서는 단순히 $t$번째 step에서의 variance인 $\beta_t$를 $\alpha_t^2$로 사용해도 큰 차이가 없다고 언급하는데, 이는 사실상 variance 누적곱에 해당되는 $\bar{\alpha}_t$가 step 수가 크기 때문에 $t$에 따라 큰 차이를 보이지 않기 때문이라고 생각했다(아니라면 말고).
> 

라고 했었다. 자기 피드백을 해보자면 한 10% 정도만 맞는 말을 한 것 같다. 실제로 Improved DDPM 저자가 분석한 내용을 살펴보도록 하자.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/235350647-b458c10f-a842-40eb-ab2e-6550b8fff47c.png" width="500">
</p>

Diffusion step이 커질수록 $\beta_t$와 $\tilde{\beta}_t$ 값이 거의 동일해진다. 이는 결국 diffusion step이 점차 커지면 커질수록 $\sigma_t$를 설정하는 것은 샘플 퀄리티에 큰 영향을 끼치지 않을 것이라는 사실을 의미한다. 두 값이 유의미한 차이를 보이는 곳은 $t = 0$ 근방인데, 이 부분에서는 이미지가 <U>거의 완성된</U>(small noise) 영역이기 때문에 샘플 퀄리티가 크게 차이나지 않게 된다.

즉, Diffusion step 수가 늘어날수록 $\Sigma_\theta(x_t, t)$를 바꾸는 것은 image distribution를 결정하는 과정에 크게 영향을 주지 못한다. 그렇다면 결국 $\sigma_t$를 고정하는 것이 diffusion process에서 최선이라는 것이라는 걸 언급하고 싶은 것일까(?)는 아니다. 아래 그래프를 보도록 하자.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/235350618-13bb612e-572c-41b2-9c6d-ac0d3c79dcf5.png" width="500">
</p>

Log likelihood를 최적화하는 방법은 diffusion process에 맞는 VLB(Variance Lower Bound)를 최적화하는 과정인데, 실질적으로 노이즈에 가까운 뒷부분($T = 4000$)보다 샘플에 가까운 앞부분($T=0$)에 올수록 loss가 차지하는 중요도(importance)가 올라가게 되는 것을 볼 수 있다. 이는 DDPM이 log likelihood를 효과적으로 개선시키지 못한 이유 중 하나가 바로 variance의 양단이 차이가 많이 나는 앞부분에서 제대로 $\Sigma_\theta(\cdot)$를 적용하지 못한 것으로 해석할 수 있고, DDPM에서 굳이 loss term에 넣지 않았던 $\Sigma_\theta(x_t, t)$ 또한 예측이 필요하다는 사실을 보여준다. 논문에서는 만약 $\Sigma_\theta(x_t, t)$를 예측하고자 하는 범위가 너무 작다면($t$가 어느 정도 증가하고 나면 $\log$ 범위에 대해서도 infimum/supremum의 차이가 거의 안남) neural network로 이를 예측하는 것이 어려울 것이라고 보았다(변동이 크면 수렴에 방해될 수 있기 때문). 따라서 그대신 variance를 $\beta_t$와 $\tilde{\beta}_t$의 보간을 통해 parameterization하였다. 네트워크는 <U>하나의 dimension마다 특정 요소를 내뱉는</U> interpolation용 vector $v$를 예측하고, 해당 output은 variance를 다음과 같이 interpolate한다.

\[
\Sigma_\theta(x_t, t) = \exp(v\log \beta_t + (1-v)\log \tilde{\beta}_t)
\]

앞서 말했던 것처럼 $\beta_t$ 자체를 interpolation 하는 것보다 $\log$를 씌워서 interpolation 하는 것이 numerical 관점에서 안정적이기 때문에 위와 같이 수식화된다.  참고로 $v$는 꼭 내적에만 국한되지 않고 $0\sim1$ 이외의 값들을 가질 수 있게 설정되었지만, 학습 후에 네트워크의 동작을 보았을 때 실제로 <U>외적을 예측하는 경우는 없었다</U>고 한다. Simplify된 loss는 VLB loss에서 variance에 대해 normalize되는 부분을 모두 무시하기 때문에 parameterized된 variance를 고려할 수 없게 된다. 따라서 $L_\text{simple}$ 대신 variance에 대한 고려를 할 수 있는 $L_\text{VLB}$를 weighted summation한 loss를 사용하였다.

\[
L_\text{hybrid} = L_\text{simple} + \lambda L_\text{vlb}
\]

물론 DDPM에서 학습이 용이했던 이유 중 하나는 $L_\text{simple}$의 영향도 있기 때문에 이를 방해하지 않도록 $\lambda = 0.001$의 작은 값을 사용하였으며, $\mu_\theta(x_t, t)$ term이 $L_\text{vlb}$에 영향을 받지 않도록 해당 loss를 최적화할 때는 stop gradient를 사용하였다. 즉 $\Sigma_\theta$는 $L_\text{hybrid}$를 통해 simplified loss에 guided된 상태로 안정적인 variance 학습을 할 수 있게 하며 그와 동시에 $\mu_\theta$는 오로지 simplified loss로만 최적화하게 된다.

### Better noise scheduling

DDPM에서 사용했던 **noise scheduling**를 기준으로 새로운 noise scheduling 방법도 제시한다. Linear scheduling 방식은 고차원(높은 resoltuion) 이미지에는 잘 적용되지만, 오히려 저차원($32 \times 32$) 이미지에는 효과적이지 않은 것을 알 수 있다. 이는 forward process에 의해 gaussian noise에 가까워지는 속도가 low resolution image의 경우 더 심하기 때문으로,

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/235350619-7cdd3163-4cfc-49e5-b7e3-c82f25024500.png" width="700">
</p>

위의 그림을 보게 되면 **linear scheduling**(upper row)을 적용했을 경우에 몇 prcoess가 지나지 않아도 <U>이미지 정보가 거의 유실되는 것</U>을 볼 수 있다. 샘플링을 잘하기 위한 diffusion process를 학습하려면 variance의 누적인 $\bar{\beta}_t$가 보다 단계적으로 샘플들을 noisy하게 만들어야한다는 것이다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/235350620-a77002e6-61e7-41d9-bb59-f901518a64ac.png" width="500">
</p>

실제로 Linear schedule로 학습된 애들은 diffusion process를 $20\% \sim 30\%$까지 skip하더라도 FID에 큰 손실이 없는 것을 알 수 있고, 이는 diffusion step 수가 늘어나는 것과 샘플링 퀄리티가 좋아지는 것에 아무런 도움이 되질 않는다는 것을 암시한다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/235350622-3af084f8-02bc-4480-8f3f-1b4c297d4346.png" width="500">
</p>

따라서 저자는 cosine schedule 방식을 사용하였고, 이를 통해 보다 점진적으로(linear하게) 줄어드는 형태의 variance를 구현할 수 있었다고 한다.

\[
\bar{\alpha}_t = \frac{f(t)}{f(0)},~f(t) = \cos \left( \frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)^2
\]

이 정의에 따르면 $\beta$가 $t = T$에 가까워질수록 $1$에 지나치게 가까워진다는 문제가 발생하는데, 이러한 singularity(계속 같은 modality에서 샘플링 되는 문제)를 없애기 위해 $0.999$보다는 커지지 않도록 clip해서 썼다고 한다. 마찬가지로 작은 offset $s$를 사용함으로써 $t = 0$ 근방에서 너무 작아지지 않도록 설정해주었는데, 이때 픽셀의 bin(확률 분포에서 확률로 넘어가는 공식)을 고려하여 $1/127.5$로 설정하였으며, 이는 곧 $s = 0.008$이라는 값으로 결정된다.

### Gradient noise 줄이기

Hybrid loss를 사용한 이유는 다음과 같다. 사실 대놓고 log likelihood를 줄이고자 한다면 VLB로 optimize하는게 가장 좋을 것이다. 허나 결과는 그리 호락호락하지 않았더라..

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/235350623-087f6934-e107-411a-8b5e-44bbd8c12fd5.png" width="500">
</p>

실제로 <U>log likelihood</U>에 관련이 있는 loss는 VLB loss 그 자체인데 막상 그래프를 보면 hybrid보다 훨씬 수렴시키기 어렵고 noisy한 학습이 진행되는 것을 볼 수 있다. 애초에 전반적으로 hybrid가 더 낮은 loss curve를 보여주는 것을 볼 수 있다. 상식선에서 VLB loss를 사용했을 때 왜 더 log-likelihood가 나쁘게 나오는지 저자들은 하나의 가설을 세웠고, 이는 VLB loss로부터 오는 gradient가 Hybird에서 오는 gradient보다 **noise가 많다는 사실**이었다(아래 그림 참고). 

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/235350624-bfe1dd94-a7a0-420b-9fa6-356a3e639339.png" width="500">
</p>

예컨데 $L_\text{VLB}$는 simplified loss와는 다르게 각 step마다 magnitude가 다르다. 즉, loss가 어느 time step $t$를 기준으로 계산되냐에 따라 차이가 나므로, 단순히 $t$를 <U>uniform하게 샘플링하는 것이 VLB objective에 도움이 되지 않는다</U>는 것이다. 이를 완화하고자, 다음과 같은 importance sampling을 제시하였다.

\[
\begin{aligned}
&L_\text{VLB} = \mathbb{E}_{t \sim p_t} \left( \frac{L_t}{p_t} \right) \newline
&\text{Where }p_t \propto \sqrt{\mathbb{E}(L_t^2)} \text{ and }\sum p_t = 1
\end{aligned}
\]

일종의 focal loss와 비슷하다고 생각할 수 있는데, 학습 시 $10$개의 previous loss에 대한 기록을 통해 지속적으로 업데이트되는 각 loss의 확률에 따라 loss sampling이 된다고 생각하면 된다. 물론 처음에는 각 $t$에 대해 $10$개의 샘플이 모일 때까지는 uniformly 추출하는 과정을 거친다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/235350625-0d11d863-2ce3-4458-a2da-2e5a73ce5828.png" width="500">
</p>

논문을 읽다보니 느낀 것은, Improved DDPM에서 집중한 것은 어떻게 likelihood based network에 비교하여 DDPM의 log likelihood를 효과적으로 개선시킬까에 대해 다룬 논문인 것 같다. Variance를 parameterize하는 과정에서 FID도 꽤나 많이 올라간 듯하다.

---

# Diffusion beats GAN

다음 논문은 GAN의 샘플링 퀄리티를 넘어설 수 있는(사실 DDPM baseline은 ImageNet과 같은 복잡한 데이터셋에 대한 샘플링은 성능이 상대적으로 좋지 않았다) 방법을 제시한 ‘Diffusion beats GAN’ 논문이다. 디퓨전을 굉장히 사랑하는 듯한(?) OpenAI에서 낸 자극적인 제목의 논문이다 보니 <U>디퓨전이 유명해지게끔 한 논문들 중 하나</U>로 손꼽힌다. 기존 GAN 중 SOTA였던 BigGAN의 생성 성능보다 좋은 ImageNet 샘플링 성능을 보여주었으며, unconditional/conditional image generation 모두 다뤘다는 점이 인상깊은 논문이다.

### Why diffusion is not good enough?

분명 디퓨전이 sample diversity도 좋고, 학습 안정성이 높은데도 불구하고 <U>왜 샘플링 성능이 충분히 올라오지 못했을까?</U> 이에 대해서 저자들이 문제점을 가설로 설정하고, 그리고 이 가설을 풀어나가는 과정이 곧 이 논문의 핵심이라고 할 수 있겠다. 저자들이 생각한 diffusion이 다른 generative model에 비해 여전히 좋은 샘플링을 내지 못하는 이유는 다음과 같다.

1. GAN(Generative Adversarial Networks)의 경우 diffusion에 비해 오랜 연구가 진행되었고, 이에 따라 최적의 네트워크 구조나 학습법, 하이퍼 파라미터 등등 리서치가 충분히 진행되었기 때문이다.
2. GAN은 fidelity를 높이는 대신 diversity가 trade off로 지불되었기 때문에, 샘플링 성능 자체만 놓고 보자면 GAN을 이기기 힘들다는 것이다.

결국 이유를 분석하자면 DDPM은 생성 모델로서 연구된 기간이 아직 짧다는 점(실제로 diffusion beats GAN 논문은 DDPM 이후 약 $1$년 뒤에 나온 논문), 그리고 GAN은 샘플링되는 데이터의 다양성을 포기하는 대신 높은 퀄리티의 데이터를 만들게끔 설계되었다는 점이다. 따라서 이 논문에서는 GAN이 가지는 두 장점(최적의 네트워크 구조 + 샘플링 퀄리티)를 diffusion에 접목시키고, 아예 GAN을 뛰어넘겠다는 포부를 담고 시작하게 된다.

### Background

이 논문에서 background로 사용된 기본 프레임워크는 DDPM인데, 이에 추가로 위에서 설명한 improved DDPM(trainable variance) 그리고 빠른 샘플링을 위해 제시된 DDIM(Denoising  diffusion implicit models)를 메인으로 한다. 앞에서도 언급했지만 Improved DDPM 논문에서도 적은 time step을 통한 높은 샘플링을 획득할 수 있었지만 이를 해결하는 방식이 ‘학습 과정을 바꿨다는 점’이고, DDIM은 이와는 다르게 동일한 marginal distribution을 가지는 non-Markovian process를 기반으로  ‘샘플링 과정을 바꿨다는 점’에서 서로 다른 연구라고 할 수 있다. [DDIM에 대한 글](https://junia3.github.io/blog/ddim)은 본인 포스팅에도 있기 때문에 미리 읽고 오는 것을 추천한다.

결론부터 말하자면 학습 방법은 Improved DDPM의 hybrid loss를, 샘플링의 경우 50 step보다 작은 sequence를 통해 생성할 경우에는 DDIM을 적용하게 된다. 사실 이 내용은 위에서 미처 설명하지 못한 <U>Improved DDPM에서의 실험</U>과 관련이 있다(아래 그래프 참고).

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/235350626-de4e219c-8302-40b1-b59d-44430094be02.png" width="">
</p>

해당 그래프를 보게 되면 약 $50$ step 이후로는 DDIM의 샘플링 퀄리티가 더 좋아지는 것을 볼 수 있다. 

### Sample quality metrics

Sampling quality를 측정하는 대표적인 방식은 기존 GAN에서 사용하는 IS, FID가 있지만, 여전히 모두 완벽하지 않고 단점이 있다는 치명적인 문제를 안고 있다. 사실상 생성 모델 연구가 **정성적 평가**로는 설득력을 가지는데 그에 비해 **정량적 평가**로 <U>설득력을 가지기 힘든 이유 중 하나</U>라고 볼 수 있다. 정말로 두 이미지 중에서 ‘잘 만든’ 이미지를 평가하는 것은 특이점을 넘어서게 되면 사실상 큰 의미가 없기 때문이다. 특히나 GAN과 같이 adversarial network로 학습하는 경우 fake sample이 gradient에 attack을 수행하기 때문에 자연스럽게 FID score가 높아질 수 밖에 없다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/235350613-d14a2024-bd45-4d3f-a4c2-565c297fc8bf.png" width="700">
</p>

\[
IS(G) = \exp(\mathbb{E}\_{x \sim G})(D\_{KL}(p(y \vert x, p(y))))
\]

Inception score는 위와 같이 측정된다. $p(y)$는 실제로 생성되는 샘플들이 전체 class에 대해 고르게 잘 만들어내는지 측정하고 $p(y \vert x)$는 생성된 샘플의 퀄리티를 측정한다. 하지만 IS가 반영하지 못하는 것은 각 클래스 별로 다양한 이미지를 생성하지 못하는 상황이다. 예컨데 CIFAR-10 dataset에 대해 10개의 클래스 각각 한가지 샘플만 찍어내더라도 그 퀄리티가 높으면 IS 상으로는 흡족한 결과가 나오게 된다(collapse를 판별할 수 없음). 이를 극복하기 위해 inception network를 사용하여 layer에서의 feature를 사용, 평균 및 공분산을 사용하여 다변수 가우시안 분포를 모델링하는 FID 방식이 소개되었다.

\[
FID(x, g) = \parallel \mu_x - \mu_g \parallel^2_2 + Tr(\Sigma_x + \Sigma_g - 2(\Sigma_x \Sigma_g)^{1/2})  
\]

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/235350630-f9820d26-0213-40c2-b50f-cc206ca1f2e2.png" width="650">
</p>

그리고 또다른 방법으로는 precision recall metric으로 sample fidelity(precision)과 sample diversity(recall)을 분리하는 방법을 제시한 논문도 있다. 예컨데 모델이 학습한 implicit probability가 $P_g$이고 실제 샘플의 분포가 $P_r$이라고 했을 때, 모델이 생성한 샘플 중 실제 샘플의 분포 내에 들어가는 정도를 측정하는 것이 샘플링 성능이랑 관련이 있고 이와는 반대로 실제 분포의 샘플 중 모델이 생성한 샘플에 들어가는 정도를 측정하는 것이 샘플링 다양성과 관련이 있다.

\[
(Precision) = \frac{TP}{TP+FP}
\]

True positive($P_r$에 해당되는 샘플이면서 $P_g$에 포함되는 것)  + False positive($P_r$에 해당되는 샘플이 아닌데 $P_g$에 포함되는 것) 중 True positive($P_r$에 해당되는 샘플이면서 $P_g$에 포함되는 것)의 비율이 샘플링 퀄리티와 직결되고,

\[
(Recall) = \frac{TP}{TP+FN}
\]

True positive($P_r$에 해당되는 샘플이면서 $P_g$에 포함되는 것) + False negative($P_r$에 해당되는 샘플이지만 $P_g$에 포함되지 않는 것) 중 True positive($P_r$에 해당되는 샘플이면서 $P_g$에 포함되는 것)의 비율이 샘플링 다양성과 직결된다고 해석하면 된다. 이 논문에서는 Precision, IS를 fidelity를 측정하는 목적으로, Recall을 diversity를 측정하는 목적으로 사용하였다.

### Architecture improvement

앞서 DDPM을 baseline으로 하는 디퓨전 연구가 가지고 있던 한계의 원인 중 하나가 네트워크 구조에 따른 충분한 리서치가 진행되지 않은 점을 들 수 있다고 했다. 이에 저자들은 diffusion model에서 sampling quality를 높일 수 있는 구조를 다음과 같이 서칭하였다.

- Depth(네트워크 깊이) 대비 Width(채널 수) 를 늘린다. 이때 모델 크기는 상대적으로 일정하게 유지하게끔 증가시킨다.
- Attention head의 갯수를 늘린다(베이스라인이 되는 UNet의 residual block에 attention이 들어간다).
- Attention을 원래 $16 \times 16$의 feature map level에만 적용했었는데, 이걸 $32 \times 32$, $8 \times 8$의 feature map에도 적용한다.
- Activation upsampling 및 downsampling 시에 BigGAN의 residual block을 사용한다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/235350631-2f69801a-2bb9-49d4-95de-77ef0cd49169.png" width="700">
</p>

- Residual connection을 $\frac{1}{\sqrt{2}}$만큼 수행한다.

비교를 위해 ImageNet $128 \times 128$ 크기의 이미지에 대해 $256$의 batch size, $250$의 sampling step으로 통일하고 FID를 기준으로 실험을 진행하였다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/235350632-8ba35f51-f417-4e88-b698-f16e3c56fb8c.png" width="500">
    <img src="https://user-images.githubusercontent.com/79881119/235350633-37db8891-bdad-42e5-8fec-a90d7b99533e.png" width="500">
</p>

좌측 테이블에서는 rescaling 부분을 제외하고는 모든 구조적 제안이 FID 성능을 높이는데 기여하는 것을 볼 수 있다. 또한 아래 그래프에서 보게 되면 depth를 증가시키는 선택 또한 성능 향상에 도움이 되는 경향을 보았지만, 학습 시간이 지나치게 증가한다는 문제 때문에 더이상 실험을 진행하지 않았다고 한다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/235350635-ce72acaf-3c88-4dc0-9b36-8ac82719d400.png" width="750">
</p>

그리고 attention configuration에 대한 실험도 진행하였는데, 실험 결과를 보게 되면 head의 개수를 늘리고 각 head의 channel 수를 줄이는 것이 가장 좋은 FID를 보여주었다. 그래프에서 확인해보면 $64$ channel을 사용할 때가 학습 속도 면에서 가장 성능 효율이 좋았기 때문에 이를 사용하게 되었다. 신기하게도 이러한 구조적 장점(성능 경향성)은 transformer의 구조와 동일하다고 한다.

### Adaptive group normalization

AdaGN이라고 불리는 이 친구는 time step과 class embedding을 각 residual block에 stylization해주기 위해 사용되었다. 예컨데 hidden layer activation $h$가 있고 time step과 class embedding의 linear projection $y = [y_s,~y_b]$가 있을 때,

\[
\text{AdaGN}(h,~y) = y_s \cdot\text{GroupNorm}(h) + y_b
\]

위와 같이 정의된다. 아마도 StyleGAN을 읽어본 사람이라면 GroupNorm 부분만 제외하고는 AdaIN과 동일한 것을 확인할 수 있다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/235350636-39e8c778-bbe4-4e29-b846-2fdbfefed0c5.png" width="">
</p>

AdaGN이 좋은 성능을 보이는 것을 기준으로 모든 네트워크 학습에 사용하였지만, 특별히 ablation을 진행한 결과는 위와 같다. 아무튼 위의 여러 과정을 거쳐 결정된 네트워크 구조는 다음과 같다.

- 각 resolution마다 2개의 residual block(BigGAN)을 가지며, width도 resolution에 맞게 조정됨
- Attention head마다 $64$의 channel 수를 가지는데, resolution $32, 16, 8$에 모두 attention layer가 있음
- BigGAN residual block을 upsampling, downsampling할 때 사용하며 AdaGN이 들어가서 timestep과 class embedding을 넣어줌

### Classifier guidance

GAN과 같은 아키텍쳐에서 conditional image synthesis가 label이 한정된 데이터셋에는 높은 퀄리티를 보장할 수 있는 방법 중 하나로 증명되었다. 예컨데 GAN을 하나의 확률 분포라고 생각하면 단순히 실제 데이터인지 아닌지 구분하는 것보다 discriminator가 $p(y \vert x)$가 explicit하게 정보를 주는 것이 각 label에 맞는 이미지를 잘 생성할 수 있게끔 generator를 유도할 수 있다는 것이다.

근데 생각해보면 앞서 우리는 이미 AdaGN을 통해 class embedding을 time step과 더불어 일종의 style 정보로 넣어주었다는 사실이 있다. 하지만 class embedding을 넣어주는 과정은 실제로 discriminator의 정보를 explicit하게주는 것과 차이가 있다. 따라서 저자는 해당 부분에 대한 방법을 발전시켜서 실험을 진행한다. 예컨데 사전 학습된 classifier가 있다고 생각해보자. 이 classifier는 각 time step $t$에 해당되는 noisy image $x_t$에 대해 classification task에 학습된 상태로 가정한다($p_\phi(y \vert x_t,~t)$).  이때의 log likelihood gradient $\nabla_{x_t} \log p_\phi(y \vert x_t,~t)$를 diffusion sampling의 guidance로 사용하겠다는 것이다. 이 부분에서 DDPM sampler인 Markovian process에 적용될 수 있는 conditional guidance와 DDIM sampler인 non-Markovian process에 적용될 수 있는 conditional guidance를 구분하여 설명한다. 각각을 수식으로 보면 다음과 같다.

### Conditional reverse noising process

각 noised image에 대해서 사전 학습된 pre-trained classifier network $p_\phi(y \vert x_t,~t)$는 diffusion pcoess에 완전히 explicit한 정보이기 때문에 다음과 같이 normalizing factor $Z$에 대해 constant 취급이 가능하다. 자세한 증명은 논문 Appendix에 있으므로 여기서는 생략.

\[
p_{\theta,\phi}(x_t \vert x_{t+1} , y) = Zp_\theta (x_t \vert x_{t+1}) p_\phi(y \vert x_t)
\]

여기서 원래의 공식을 recall해보자면 일반적인(class condition 없는) diffusion process는 다음과 같이 정의가 되었었다. 각 time process에 대해 예측된 $\mu, \Sigma$에 대해서,

\[
\log p_\theta(x_t \vert x_{t+1}) = -\frac{1}{2}(x_t-\mu)^\top \Sigma^{-1} (x_t - \mu)+C
\]

이때 상대적으로 $\log p_\phi(y \vert x_t)$가 가지는 curvature가 $\Sigma^{-1}$에 비해 작을 것으로 예상된다. 이에 대한 해석은 다음과 같다. $\log p_\theta(x_t \vert x_{t+1})$은 $1/2\parallel \Sigma \parallel$을 계수로 갖는 quadratic function이다. 따라서 이 quadratic function의 곡률을 결정하는 부분이 곧 $\Sigma$의 크기와 연관이 있는데, diffusion step 대부분에서 $\Sigma$는 $0$에 가까운 작은 값을 가지게 되므로 계수가 매우 커지게 된다. 따라서 $p_\phi$가 가지는 function이 이에 비해 적은 곡률을 가질 것이라고 가정할 수 있게 되는 것이다.
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/235352170-23a9d778-f0ab-4fc6-ad1b-3cbe2efa7517.png" width="500">
</p>
이렇게 되면 $x_t = \mu$인 점에서(diffusion reverse process의 quadratic function의 꼭짓점 부분) classifier guidance 부분을 테일러 1차 근사를 통해 나타낼 수 있다.

쨌든 샘플링되는 파트는 $\mu$가 메인인데, 어차피 그 부분에서 $p_\theta$에 대비해서 $p_\phi$가 가지는 <U>곡률이 상대적으로 매우 작기 때문에</U> 무시할 수 있다는 개념이다.

\[
\log p_\phi(y \vert x_t) \approx \log p_\phi(y \vert x_t) \vert_{x_t = \mu}+(x_t - \mu)\nabla_{x_t} \log p_\phi (y \vert x_t) \vert_{x_t = \mu} = (x_t - \mu)g+C_1
\]

여기서의 $g$는 $x_t = \mu$에서의 classifier에 의한 log likelihood의 gradient와 같다. 이를 위의 공식에 대입하게 되면,

\[
\begin{aligned}
\log(p_\theta(x_t \vert x_{t+1}) p_\phi(y \vert x_t)) \approx& -\frac{1}{2}(x_t-\mu)^\top \Sigma^{-1} (x_t - \mu)+(x_t - \mu)g + C_2 \newline
=& -\frac{1}{2}(x_t-\mu-\Sigma g)^\top \Sigma^{-1} (x_t - \mu -\Sigma g)+ \frac{1}{2}g^\top \Sigma g + C_2 \newline
=& -\frac{1}{2}(x_t-\mu-\Sigma g)^\top \Sigma^{-1} (x_t - \mu -\Sigma g)+ C_3 \newline
=& \log p(z) + C_4,~z \sim \mathcal{N}(\mu + \Sigma g, \Sigma)
\end{aligned}
\]

결국 classifier에 의한 guidance는 샘플링할 때 gradient 방향을 틀어준다고 생각할 수 있다(drift 조정).

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/235350637-0aa972ae-3e28-4476-99b9-afa4de6b4442.png" width="700">
</p>

### Conditional sampling for DDIM

그러나 위의 샘플링 방법은 drift를 조정해주는 과정이 들어가고, 수식 상에서 <U>Markov process임을 가정</U>하고 있으므로 DDIM처럼 deterministic한 sampling을 하는 경우에는 사용할 수 없다. 

\[
x_{t-1} = \sqrt{\bar{\alpha}\_{t-1}}\underset{\text{predicted }x\_0}{\left( \frac{x\_t - \sqrt{1-\bar{\alpha}\_t}\epsilon_\theta^{(t)}(x\_t)}{\sqrt{\bar{\alpha}\_t}} \right)} + \underset{\text{direction pointing to }x\_t}{\sqrt{1-\bar{\alpha}\_{t-1} - \sigma\_t^2} \cdot \epsilon\_\theta^{(t)}(x\_t)} + \underset{\text{random noise}}{\sigma\_t z},~z \sim \mathcal{N}(0, I)
\]

위의 식을 보면 알 수 있듯이 deterministic DDIM은 $x_0$로부터 $x_t$를 예측하는 형태로 샘플링이 진행되다보니 $x_t$에 대한 classifier gradient를 적용할 수가 없게 되는 것이다. 여기서 바로 이전에 살펴봤던 논문인 SDE와 diffusion model을 연결했던 논문이 힘을 발휘한다. 해당 내용도 포스팅되어있다([참고 링크](https://junia3.github.io/blog/scoresde)). 해당 논문에서 VP-SDE라고 명시된 확률 미분 방정식에 대해 보면 다음과 같다. 예컨데 원래의 DDPM은 다음과 같은 process를 통해 샘플링을 진행한다.

\[
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}}\_t} \epsilon\_\theta(x\_t, t)\right)+\sigma_tz
\]

그런데 이때, 이 식을 score estimate function $s_{\theta^\ast}(\cdot)$에 대한 확률 미분 방정식으로 포현하면 다음과 같다.

\[
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} (x_i + \beta_i s_{\theta^\ast}(x_i, i)) + \sqrt{\beta_i}z_i
\]

고로 기존의 ancestral sampling에서 벗어나서 score function을 time $t$에 대해 정의할 수 있게 된다. 

\[
\nabla\_{x_t} \log p_\theta (x_t) = -\frac{1}{\sqrt{1-\bar{\alpha}\_t}}\epsilon\_\theta (x_t)
\]

이를 앞서 정의했던 $p(x_t)p(y \vert x_t)$의 score function에 적용하게 되면 다음과 같다.

\[
\begin{aligned}
\nabla_{x_t}\log \left( p\_\theta(x_t)p\_\phi(y \vert x\_t)\right) =& \nabla_{x_t} \log p\_\theta(x_t) + \nabla_{x\_t} \log p\_\phi(y \vert x_t) \newline
=& -\frac{1}{\sqrt{1-\bar{\alpha}\_t}}\epsilon\_\theta(x\_t) + \nabla\_{x\_t} \log p\_\phi(y \vert x\_t)
\end{aligned}
\]

즉, epsilon을 다음과 같이 새롭게 정의할 수 있게 된다. 앞서 DDPM의 경우와 동일하게 gradient를 바꾸는 느낌이다.

\[
\hat{\epsilon}\_\theta(x\_t) := \epsilon\_\theta(x\_t) - \sqrt{1-\bar{\alpha}\_t}\nabla\_{x\_t} \log p\_\phi (y \vert x\_t) 
\]

### Classifier gradient scaling

Classfier $p_\phi$에 의한 score guide를 주기 위해서는 classification model을 학습시켜야 한다. Classifier architecture는 UNet model의 downsampling 부분에서 추출된 feature map에 attention pooling($8 \times 8$)을 통해 최종 output을 추출하게 된다. Classifier는 각 노이즈 스텝에 대해 분류할 수 있어야하므로 각각의 time step에 대한 noised input을 학습하게 된다. 학습 이후에는 앞서 언급한 gradient 영향을 주면서 샘플링을 진행한다고 보면 된다.

초반 unconditional ImageNet model(class condition을 따로 embedding으로 주지 않은 네트워크)로 실험했을때, classifier guidance $s$를 $1$보다 크게 하지 않으면 원하는 class의 샘플이 나올 확률이 절반으로 뚝 떨어지는 것을 확인하였고, 심지어 이 확률로 샘플을 만들어도 <U>시각적으로 그다지 해당 클래스의 범주에 속하지 않는 것</U>을 확인하였다. 

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/235350638-8036a8c0-dedf-424f-9130-085fcdee73e5.png" width="">
</p>

예컨데 “Pembroke Welsh corgi”의 class에 대한 scale을 $1.0$으로 주었을 때(좌측) 제대로 생성되지 않던 웰시코기 이미지가 $10.0$으로 키웠을 때 유의미하게 좋아지는 것을 볼 수 있다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/235350639-d330f2db-523f-4e8c-b4a9-6c419e2ff940.png" width="">
</p>

위의 표에서 주목할 점은 unconditional model에 classifier guidance를 충분히 큰 값으로 주게 되면(guidance = $10.0$) conditional model에 필적하는 FID 및 IS를 보여주는 것을 확인할 수 있다.

논문에서 추가로 언급한 내용 중에 low resolution image를 condition으로 하는 2-stage diffusion process를 사용했을 때 BigGAN의 성능을 넘어선 것을 알 수 있는데, 여전히 샘플링 속도가 문제가 된다는 점과 classifier training으로부터 자유롭지 않기 때문에 labeled sample에 한정된다는 문제가 발생한다.

---

# Classifier-free diffusion guidance

### Low Temperature Sampling

Classifier guidance 논문은 classifier에 의한 gradient 조절을 통해 샘플의 <U>다양성을 조금 희생</U>하는 대신 **fidelity**를 얻을 수 있었다. Classifier guidance의 주목적은 샘플링의 다양성보다 샘플링의 퀄리티에 대한 연구라는 것은 앞선 설명을 통해 명확해졌을 것이다.

이처럼 샘플링의 다양성과 퀄리티에 대한 trade-off는 GAN을 비롯한 generative model에서 이미 연구가 된 바가 있다. 이러한 방법론들을 ‘Low temperature sampling’이라고 하는데, 해당 용어는 <U>energy based model</U>인 볼츠만 머신에서 파생된 것이다.

Samping이 되는 prior를 에너지에 기반한 state의 집합($S(\tau)$)이라고 생각해보자. 예컨데 에너지가 높은 상태는 불안정하기 때문에 그만큼 존재할 수 있는 state도 많아진다. 온도가  높아지면 높아질수록($\tau \uparrow$) 샘플링 다양성이 증가한다는 경향성과 묶어서 생각할 수 있다. 이와 반대로 에너지가 낮은 상태($\tau \downarrow$)는 안정적이기 때문에 그만큼 존재할 수 있는 state의 영역이 줄어든다. 샘플링 다양성이 감소하는 대신, 한정된 state에서 더 많은 샘플링을 통해 state 밀도를 높일 수 있기 때문에 더욱 그럴 듯한 샘플을 만들어내는 fidelity라는 경향성과 묶어서 생각해볼 수 있다.

이렇듯 “Low temperature sampling”은 <U>다양성을 희생하는 대신 fidelity를 높이는 전략</U>으로, truncation trick을 쓰는(feasibility가 높은 영역에서 샘플링하는 전략) 방법을 사용하거나 Glow와 같은 autoregressive model에서 부적절한 샘플들을 rejection하는 전략들을 사용하는 등이 이러한 방법론의 한 메소드로 제시가 된다.

Diffusion beat GANs라는 논문이 제시되면서 해당 문제에 대해 두 가지 접근법을 제시했으나(각 process마다 gaussian noise를 줄이는 방법/Score 예측을 줄이는 방법), 두 방법 모두 그다지 효과적이지 못했다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/235350640-205bdc8e-b12b-4f64-8ed1-b59488f5effd.png" width="">
</p>

Temperature가 낮아질수록 fidelity가 좋아지거나 predicion이 좋아지는 경향을 보여야하는데 전혀 그렇지 못한 것을 확인할 수 있다. 따라서 class guidance scale $s$를 통해 이를 trade-off로 조절할 수 밖에 없었다.

### Classifier 없는 guidance?

이는 마치 다음과 같다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/235351438-cac45ec0-b464-4ade-b500-725a40fc9e16.png" width="">
</p>

논문에서 task를 정하는데 있어 <U>이러한 pipeline이 된 맥락</U>은 다음과 같다. Classifier guidance는 diffusion model의 학습 pipeline을 보다 복잡하게 만든다. 왜냐하면 앞서 언급했던 바와 같이 UNet 형태의 diffusion model을 학습하면서 각 time step에서의 noised sample의 downsampled feature을 토대로 classifier를 따로 학습해야하기 때문이다. 사전 학습된 classifier를 사용할 수 없다는 문제는 아무리 time step을 최소화한다고 하더라도 학습을 복잡하게 만드는 과정으로 나타난다.

또한 classifier guidance sampling은 image classifier를 속이는 형태의 gradient based adversarial attack으로 해석할 수 있다. 결국 FID나 IS와 같은 metric은 어쩔 수 없이 classifier-based metric인데, 샘플링 과정에서 classifier를 잘 속이도록(classifier 상으로 유의미한 image가 나오도록)하는 과정은 FID나 IS와 같은 <U>metric을 높이기 위한 직접적인 목적 함수</U>가 되기 때문이다. 고로 정말 classifier guidance라는 방법이 샘플링 효과를 높일 수 있는 방법이었기 때문에 FID나 IS score가 좋은게 아니라, 방법 자체가 metric을 개선시키기 좋은 환경이므로 성능을 높일 수 있지 않았나라고 판단한 것이다. 참으로 똑똑한 사람들.. 참고로 diffusion beat GANs는 OpenAI에서 쓴 논문이고 얘는 Googlebrain에서 쓴 논문이다. 이정도면 거의 세기의 대결……

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/235350617-fa8547b8-cacd-4eac-91bc-9f3192cf3877.jpg" width="">
</p>

### Background

학습 방법은 의외로(?) 심플하게 정리된다. 물론  기본 학습 셋팅 자체가 DDPM과는 약간 다르기 때문에 동일한 수식으로 비교하기는 애매하지만 방법에 대해서만 언급하면 다음과 같다. 학습은 continuous time diffusion model을 학습한다. 데이터셋 $p(x)$으로부터의 샘플 $x$와 정해진 하이퍼파라미터 범위$\lambda \in [\lambda_{\min},~\lambda_{\max}]$의 latent인 $z_\lambda$에 대해서, forward process $q(z \vert x)$는 variance preserving(VP) markov process로 표현할 수 있다.

\[
q(z_\lambda \vert x) = \mathcal{N}(\alpha_\lambda x, \sigma_\lambda^2I), \text{ where }\alpha_\lambda^2 = 1/(1+e^{-\lambda}),~\sigma_\lambda^2 = 1-\alpha_\lambda^2
\]

Continuous한 임의의 $z_\lambda$에 대해 위와 같이 marginal을 정의하게 되면, 인접한 latent에 대한 조건부 그래프는 다음과 같이 표현 가능하다.

\[
q(z_\lambda \vert z_{\lambda^\prime}) = \mathcal{N}((\alpha\_\lambda/\alpha\_{\lambda^\prime})z\_{\lambda^\prime}, \sigma\_{\lambda \vert \lambda^\prime}^2) , \text{ where }\lambda < \lambda^\prime,~\sigma^2\_{\lambda \vert \lambda^\prime} = (1-e^{\lambda-\lambda^\prime})\sigma\_\lambda^2
\]

$\lambda$를 실제로 계산하게 되면 $\alpha_\lambda$와 $\sigma_\lambda$에 대해 데시벨 단위의 SNR과 같은 맥락으로 표현이 가능하기 때문에, 이전 process의 input을 signal로서 점차 줄여가면서 더해지는 노이즈를 증가시키는 방식을 표현한 것을 알 수 있다. 이를 input $x$에 대해 조건화하여 Bayes’ rule을 사용하여 posterior로 바꾸는 과정과 이를 통해 parameterized reverse process $p_\theta$와의 loss를 구하는 과정은 DDPM과 동일하므로 따로 언급하지는 않겠다. 결국 학습하고자 하는 네트워크는 다음과 같은 목적함수를 가진다.

\[
\mathbb{E}\_{\epsilon, \lambda}(\parallel \epsilon_\theta(z\_\lambda)- \epsilon \parallel_2^2)
\]

$\epsilon \sim \mathcal{N}(0, I)$이며 $z_\lambda = \alpha_\lambda x + \sigma_\lambda \epsilon$로 추출하게 된다. Continous function에 대한 score mathinc으로 학습이 진행된다고 보면 될 것 같다. $p(\lambda)$가 일정하면 평소에 보는 variational lower bound 식이 되는데 저자들은 classifier guidance 논문에서 밝힌 것처럼 cosine schedule에서 아이디어를 얻어 사용했다고 한다. 해당 내용은 위에서 언급했던 바와 같이 보다 점진적으로 감소하는 noise를 구현하여 네트워크가 모든 노이즈 분포에 대해 골고루 학습될 수 있도록 하는 것이다.

\[
\begin{aligned}
&\lambda = -2\log \tan(au+b),~u \sim \mathcal{U}(0, 1) \newline
&a = \arctan (e^{-\lambda_{\min}/2})-b,~b = \arctan(e^{-\lambda_{\max}/2})
\end{aligned}
\]

### Classifier guidance

앞서 low temperature sampling에서 언급했던 바와 같이 GAN이나 Flow based model의 경우에 FID score와 IS 간의 trade-off를 할 수 있다는 장점이 있지만, 이를 디퓨전 모델에 가져오는 것이 상당히 힘들다고 언급했었다. 가장 주된 이유 중 하나는 prior를 만드는 과정이 diffusion process로 고정되기 때문이다. 이러한 비슷한 효과를 주기 위해 앞서 리뷰했던 classifier guidance 논문에서는 diffusion score에 noised image에 대한 classifier guidance를 주는 모델링을 통해 해결하고 하였다.

\[
\hat{\epsilon}\_{\theta, \phi}(z\_\lambda, c) := \epsilon\_\theta(z\_\lambda, c) - w\sigma\_\lambda\nabla\_{z\_\lambda} \log p\_\phi (c \vert z\_\lambda) 
\]

해당 모델링에서 classifier의 영향력을 행사하는 $w$가 곧 probability의 scale factor로, log likelihood에 대해 보다 생성되는 데이터가 해당 label을 가지는 이미지 범주에 들게끔 학습시키기 때문에 diversity를 희생하고 fidelity를 높이는 기능을 한다.

\[
\tilde{p}\_{\theta, \phi}(z\_\lambda \vert c) \propto p\_\theta(z\_\lambda \vert c)p\_\phi(c \vert z_\lambda)^w
\]

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/235350641-c99563f7-e436-48f9-bd57-fd7b2b53deb6.png" width="">
</p>

이에 대한 효과는 위와 같은 toy experiment에 대해 관찰하게 되면 더 명확하게 드러나는데, 각각의 가우시안 분포가 classifier guidance에 의해 멀어질수록 구분이 잘 되는 특징은 높아지지만 그에 비해 각 분포가 차지하는 부피는 줄어드는 것을 볼 수 있다.

### Classifier free guidance

Classifier guidance가 앞서 본 실험에서와 같이 IS와 FID 간의 trade off를 잘 보여주기는 했지만, 그럼에도 불구하고 완벽하지는 않은 low temperature sampling이며 가장 큰 문제는 image classifier로부터 자유롭지 못하다는 것이다. 저자들이 주장하는 classifier free guidance 방법은 기존의 $\epsilon_\theta(z_\lambda, c)$를 $\hat{\epsilon}\_{\theta, \phi}(z_\lambda, c)$ 로 auxiliary하게 바꾸지 않더라도 classifier guidance와 같은 효과를 주고 싶게 한다는 것이다. 가장 큰 차이는 classifier parameter $\phi$의 의존성을 없애고 싶은 것이다.

따라서 논문에서는 classifier를 사용하는 대신, unconditional diffusion model $p_\theta(z)$ 그리고 conditional model $p_\theta(z, c)$를 함께 학습하는 전략을 취한다. 이때 개별적인 네트워크를 구성하고 각각을 훈련시키는 것이 아닌, 두 probability 모두를 parameterize하는 방법을 생각해낸다. 그 방법은 다음과 같다.

1. Unconditional model은 class identifier $c$ 대신 $\emptyset$을 null token으로 넣어준다. 즉, $\epsilon_\theta(z_\lambda) = \epsilon_\theta(z_\lambda, \emptyset)$
2. $p_\text{uncond}$ 만큼의 hyperparameter probability 만큼 null class sample을 생성 및 학습에 사용하여 unconditional model 학습에 사용한다.
3. Conditional과 unconditional의 weight를 다음과 같이 벡터로 조정한다. $\tilde{\epsilon}(z_\lambda, c) = (1+w)\epsilon_\theta(z_\lambda, c) - w\epsilon_\theta(z_\lambda)$

해당 식은 classifier gradient $\phi$에 대한 식이 전혀 포함되지 않기 때문에 기존 논문에서 했던 approximation(테일러 1차 근사)와 같은 문제에서도 해결된다. 또한 gradient를 직접 건드는 샘플링이 아니므로 adversarial attack이 아니다.

### 실험 결과

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/235350642-63f54983-7e25-4ea0-9cd0-43d5de9a6a11.png" width="">
</p>

완전히 unconditional이랑 conditional이랑 동일한 확률로 샘플링할 줄 알았는데 실제 결과를 보니 $0.5$가 마냥 좋지는 않아보인다. 아무튼 해당 논문에서는 총 3개의 확률에 대해 실험을 진행하였다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/235350644-5713b6a9-d3c6-40c4-b0cd-d243a8273876.png" width="500">
    <img src="https://user-images.githubusercontent.com/79881119/235350645-6227a31e-e75a-4de2-b3d6-123d1f194454.png" width="500">
</p>

---

# 결론

총 3개의 논문에 대해 봤는데 각각 논문들이 문제시한 점이 어느 정도 이어진다고 생각해볼 수 있다. 가장 먼저 improved DDPM에서는 단순히 DDPM의 기존 방식이 왜 샘플링 성능이 좋지 않은지를 여러 요소들을 종합적으로 판단 후에 이런저런 실험을 진행한 것이 특징이라고 할 수 있을 것 같다.

그와는 별개로 classifier관련 두 논문 중 첫번째인 OpenAI의 논문은 classifier의 guidance를 사용하게 되면 GAN이나 flow based model에서 가능한 고퀄의 샘플링이 가능하다는 점에 집중했으며 그와 동시에 diffusion model architecture를 최적화하는 연구를 진행했다는 점이 contribution이 될 것 같다.

마지막으로 classifier guidance free 논문은 굳이 classifier 학습이 없이도 class condition을 주고 학습시키거나 주지 않고 학습시키는 동시 최적화를 통해 디퓨전 단일 네트워크가 unconditional diffusion model $p_\theta(z)$ 그리고 conditional model $p_\theta(z, c)$ 모두 학습할 수 있으며, 이를 기반으로 classifier guidance의 score estimation을 classifier parameter $\phi$에 무관하게 구성할 수 있음을 입증하였다.