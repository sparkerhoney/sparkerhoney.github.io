---
title: Zero-shot Text-Guided Object Generation with Dream Fields에 대하여
layout: post
description: zero-shot scene generation with clip
use_math: true
post-image: https://user-images.githubusercontent.com/79881119/217227526-9d8ff3cc-1d07-4a22-9ddc-4d7fdbc07f54.gif
category: paper review
tags:
- nerf
- zero shot
- scene generation
---

# 들어가며 …

이 논문에서 **가장 핵심이 되는 키워드**만 따로 생각해보면, text representation을 통해 학습된 multi-modal image 관계를 사용하여 3D object를 렌더링하는 것이다. 3D generation 혹은 rendering의 경우 획득이 비교적 간단한 데이터셋인 이미지와는 다르게 captioning된 3D dataset이 필요하다. 이는 3D generation network로 하여금 한정된 갯수(pool)의 category만 생성할 수 있다는 한계와 문제점을 가진다(ex. [ShapeNet](https://arxiv.org/abs/1512.03012)).

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/217228269-31afad43-1d31-4c0a-9976-eeff42540e1f.png" width="400"/>
</p>

그렇기 때문에 한정된 구조나 texture를 가지는 기존 방식에서 벗어나, 웹 상에서 대량으로 수집된(대략 4억 장의 text prompt-image pair) [WebImageText dataset](https://github.com/google-research-datasets/wit)에 학습된 [CLIP 네트워크](https://arxiv.org/abs/2103.00020)를 사용하여 generation에 guidance를 주는 방법을 선택하였다. 구체적인 학습 과정은 뒤에서 마저 설명하겠지만 간단하게 컨셉만 보면, 여러 방향에서 수집된 camera view에 대해 최적화된 Neural Radiance Field(NeRF)가 target caption과의 유사성이 높게끔 학습시킨다. 이때 유사성에 대한 guidance로 사용하는 것이 바로 **CLIP network**이며, 단순히 CLIP network를 통한 loss를 주게 되면  3D 구조가 무너지거나 fidelity가 악화되는 문제가 발생하였기 때문에 여기에 추가로 간단한 geometric prior를 주었다. **Geometric prior**에는 sparsity를 유발하는 transmittance에 regularization,  scene bound, 그리고 새로운 MLP 구조가 포함된다. 

---

# Why zero-shot is important?

논문을 읽기 시작하면서 **가장 근본적인 질문**이 문득 떠올랐다. 사실 충분히 Neural Field에 대한 연구는 진행되었고, 주어진 dataset만 있다면 high fidelity의 3D representation을 렌더링하거나 새로운 각도나 방향에서의 image를 생성하는 것은 그리 어렵지 않을 것이다. 그리고 zero-shot에 대해서 SOTA 연구들을 살펴보면, 그 성능이 fully supervised learning을 진행한 연구에 비해 지나치게 떨어지는 경우도 있다. 그렇기 때문에 zero-shot 논문들을 보게 되면 **연구의 필요성**에 대한 설명이 필수적이라는 것을 알 수 있다. 3D object model은 흔히 Game이나 Virtual reality를 쉽게 접할 수 있는 환경에서 많이 사용된다. Unity나 언리얼 엔진같은 프로그램을 다뤄봤다면 알 수 있겠지만 우리가 가상 현실이나 게임에서 볼 수 있는 모든 형태의 3D object, 하물며 2D object 조차도 digital software(Blender나 Maya 등등)에 의존해서 생성되고 여기에 texture(무늬)를 입히는 과정도 디자인 작업이 필요하다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/217228280-adf8cbe4-089f-4d9c-8520-10ae5c91075e.png" width="400"/>
</p>


사실상 NeRF와 같은 연구에서 사용되는 모든 형태의 object도 이에 기반한 작업물이고, 앞서 설명했던 WebImageText dataset과는 다르게 인터넷 상에서 무료 에셋으로 열려있는 3D 오브젝트는 퀄리티의 일관성이나 다양성 측면에서 쉽게 **수집하기 어렵다**는 문제가 있다. 딥러닝을 하기 위해서는 데이터셋이 필수적인데, 데이터셋을 모으고자 그래픽 디자이너만 몇 만을 고용해서 오브젝트를 만들고 있을 수도 없고, 디자이너들 각각의 스타일도 모두 다르기 때문에 균일한 distribution을 구성할 수 없다는 현실에 부딪힌다.

그렇기 때문에 데이터셋 수집이 어려운 환경에서 **접목시킬 수 있는 여러 방법론**(meta learning, low-shot learning, domain generalization, unsupervised-domain adaptation 등등) 중 이 논문에서는 zero-shot learning을 적용하고자 한 것이고, 위에서 설명한 내용이 사실상 이 논문에서 task를 정의하기 전에 설정한 기존 연구들의 problem 혹은 limitation이라고 볼 수 있다.

---

# Challenging in multimedia application

NeRF와 같은 연구 목적이라면 단순히 3D dataset을 구성하는 과정에서 몇 가지의 category만 구성하고, 각 카테고리에 맞는 오브젝트의 형태만 구성하면 된다. 하지만 만약 오브젝트를 생성하는 목적이 실제 멀티미디어 환경에서 소비자를 만족시킬 목적이라면 의자와 같은 단순한 사물도 목적에 따라(소파, 벤치, 휠체어 등등) 혹은 재료에 따라 texture가 달라지고 이를 구성해야하는 어려움이 따른다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/217228284-122e98ee-493f-4cec-b8ac-78d00445e54e.png" width="300"/>
    <img src="https://user-images.githubusercontent.com/79881119/217228288-b8e3512a-1b8f-413b-bad8-a12e5f886c0a.png" width="900"/>
</p>

기존 approach들이 3D dataset을 단순히 **point cloud**나 **voxel grid**, 혹은 **triangle mesh**로 형태만 표현하던 것과는 다르게 시각적인 방향에 대한 3D geometry와 texture를 고려해야한다는 어려움이 생긴 것이다.

---

# Automatically generate open-set 3D models

따라서 저자들이 연구하고자 한 dream fields 연구는 이러한 기존 방식들의 제약으로부터 벗어나기 위해 open-set의 natural language prompt로부터 image representation을 학습하고, 이를 실제로 다양한 zero-shot task에 적용했을때 좋은 성능을 보였던 CLIP을 사용하였다. Dream fields는 NeRF를 학습하는 과정에서 scene의 geometry와 color 모두 perceptual metric을 최대화하는 방향으로 학습되며, 논문 제목에서 알 수 있듯이 학습에 3D training dataset이 아예 사용되지 않는다. NeRF 연구에서는 다양한 방향에서 획득한 RGB photo를 ground truth로 학습하여 새로운 방향에서의 image reconstruction을 진행한다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/217228289-d2db45e9-7a88-411d-a44d-f53a51e0bd67.png" width="700"/>
</p>


따라서 위에서 보는 바와 같이 학습된 $F_\theta$가 implicit 3D space를 학습하고, 딥러닝이 내포하는 특정 object에 대한 representation을 viewing point, direction을 통해 샘플링하여 새로운 각도에서의 이미지를 생성할 수 있다는 것이다. NeRF의 장점은 interpolation이 부드럽고 색 변화에도 자연스럽게 대처할 수 있다는 점이다. NeRF의 퀄리티를 높이기 위해 이후 여러 연구들이 진행되었지만, 모든 연구들의 공통점은 description으로부터 novel image를 생성하는 것이 불가능하다는 것이다. Input image들이 존재하고, 이에 기반하여 새로운 각도의 이미지를 렌더링할 수는 있으나 **text description**을 3D semantic feature로 사용할 수 없다. CLIP의 image-text representation과 NeRF의 volumetric rendering 과정 모두 미분 가능하기 때문에  이른바 ‘dream fields’는 zero-shot으로도 충분히 학습이 가능하다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/217228291-99eb8691-35bd-4c4c-a80b-42bdfe2f116f.png" width="500"/>
</p>


그러나 아무런 regularization 없이 CLIP representation(text prompt에 의한 supervision)만 활용하여 학습하다보니 다양한 **artifact**가 발생하였고, 이를 해결하기 위해 **geometric constraints**를 추가했다고 한다.

---

# Contribution

이 논문이 가지는 contribution은 비교적 명확하기도 하고, 실제로 저자들이 introduction에 작성하였다. 해당 내용을 간단하게 요약해보면,

- Image/Text pair로 학습된 CLIP 모델을 통해 3D shape이나 multi-view dataset 없이도 NeRF를 최적화할 수 있었다.
- Zero-shot description에 대해 다양한 3D object generation이 가능하다.
- Geometric priors를 적용했을 때 fidelity가 상승하였다.

이 중 마지막 **geometric prior**과 관련된 내용은 사실 실험을 진행하면서 제안된 sub-contribution으로 보는 것이 더 적절할 것 같고, 위의 두 내용이 이 논문의 가장 중요한 포인트라고 생각한다.

---

# Related works

해당 연구는 [DeepDream](https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html)에서 가장 큰 영감을 받았다고 한다. 사실 본인은 이 부분은 살짝 too much하다고 보았는데, 아무래도 NeRF를 사용해서 완전히 새로운 object를 만들어내고자 하는 것은 비교적 최근 연구인 CLIP을 기반으로 하지 않았더라면 아이디어 빌드업이 불가능할 정도로 어려운 연구라고 생각했기 때문이다.  Neural network를 통해 원하는 이미지를 생성하고자 하는 연구들은 많이 진행되었다. 대표적으로 pre-trained된 generative network를 사용하여 추가 학습 없이 image를 생성하는 것은 GAN inversion이나 latent optimization 등등 많은 연구가 진행되어왔다. 이 논문에서 CLIP을 제외하고는 가장 유의미한 관계를 가지는 연구가 style transfer의  관점에서 진행한 differentiable image parameterization라고 한다.

솔직히 이 부분 보면서 어이가 없었던 것은 논문을 읽다보면 알 수 있겠지만 CLIP을 사용한 것 자체에서 기존 style transfer 연구들과는 차별성을 두어야하고 오히려 본인들이 작성했던 paper보다 해당 paper에 insight를 줄 수 있는 논문이 훨씬 많아 보이는데도 불구하고 인용수를 늘리기 위해서 작성한 느낌이 강하게 들었다. 고의가 아니라면 미안하지만 본인은 related works의 초반부가 대체 왜 들어갔는지 이해가 안된다. 아무튼 계속하자면, 기존의 style이나 content 기반의 loss를 image-text loss로 대체하여 text prompt에 기반한 style 및 content에 대한 generation의 controllability를 늘릴 수 있었다. 왜냐하면 style transfer의 관점으로 접근했을 때는 target이 되는 style image feature가 필요하고, 실제로 style이 적용되었을 때 quality 또한 안정적이지 않기 때문에 문제였지만 CLIP에 기반한 style transfer는 따로 style image가 필요 없고 단순히 description만 사용하기 때문이다. [기존 방식](https://distill.pub/2018/differentiable-parameterizations/)의 경우 한정된 geometry(예를 들면 토끼 object)나 optimized된 texture에만 한정된 transfer를 사용했지만 dream fields 연구를 통해 open-ended text guided generation이 가능하게 되었다.

물론 해당 paper 이외에도 CLIP을 사용한 3D generation 연구들 중  [CLIP-Forge](https://arxiv.org/abs/2110.02624)도 있지만, 해당 연구에서는 geometry만 생성하는 decoder를 사용했고 ShapeNet category에 대해서만 guidance를 주었다는 점에서 out of domain generation이 불가능하다는 문제가 있다. [Text2Shape](http://text2shape.stanford.edu/)연구는 아래에 보이는 것과 같이 text-conditional WGAN 학습을 통해 voxelized objects를 생성하는 task를 진행했지만, 마찬가지로 각 ShapeNet 카테고리를 생성하는 task에서 벗어나지 못한 점과 NeRF와는 다르게 voxel 특성상 한정된 resolution을 가지는 문제가 있었다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/217228296-61ab9531-1744-4afd-afed-18d554b37d55.png" width="1200"/>
</p>


실제로 voxel 기반의 shape generation을 보면 그다지 성능이 좋지 않은 것을 확인할 수 있다.

Related works에 재밌는 연구도 있는데, MIT 연구실에서 한 사람이 혼자 진행한 연구 중  ‘[Evolving evocative 2D views](https://arxiv.org/pdf/2111.04839.pdf)’는 약 4쪽짜리 페이퍼로 아카이브에 올라가있다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/217228302-dadb4e0a-b689-4857-ad05-5ad936f86c04.png" width="800"/>
</p>


이 연구에서는 3D supershapes(3D 공간에서의 실제 형태)를 하나의 viewpoint에 대해 projection한 2D view를 최적화하는 loop 알고리즘을 통해 CLIP score를 맞추고, manually coloring하는 과정을 거친다. 이외에도 CLIP을 사용한 다양한 연구들이 소개되는데, 예를 들어 human SMPL model의 vertices나 textures를 수정하여 stylize하는 [ClipMatrix](https://arxiv.org/abs/2109.12922)연구나, signed-distance fields를 editing할 수 있는 인터페이스를 구축한 연구도 있다. 이미지의 경우에는 더 다양한 연구들이 진행되었는데, SIREN network의 weight를 CLIP을 통해 학습함으로써 image generation에 적용하거나, VQGAN-CLIP, StyleGAN-CLIP 등등 image generation에서 사용될 수 있는 다양한 네트워크를 CLIP과 함께 사용한 실험들이 있다. 물론 2D image에 대한 generative model를 NeRF와 함께 사용하는 3d-aware GAN과 같은 연구들도 진행되었지만, 대부분의 연구는 open-set text에 대한 generation에 대한 능력이 배제된 것을 알 수 있다.

---

# Backgrounds

### NeRf

NeRF는 사실 앞서 논문 리뷰로 따로 다루기도 했고, 하도 유명한 논문이다보니 사전 지식에 대해서 아는 사람들이 많을 것 같다. NeRF는 scene의 density나 color를 MLP를 통해 학습하게 되고, 이때 network에 query로 들어가는 것이 특정 3D point의 좌표인 $(x,~y,~z)$와 viewing direction $(\theta,~\phi)$이다. Notation을 보고 어느 정도 짐작은 가겠지만 네트워크를 학습하는 과정은 canonical space와 spherical space의 좌표계를 동시에 사용하며, 이때 canonical space를 implicit하게 학습하는 MLP는 해당 위치의 density 혹은 transmittance를 연산하는데 집중하게 되고 spherical space는 해당 위치의 color(RGB value)를 연산하는데 집중하게 된다. 이렇게 학습된 파라미터를 통해 어떤 각도에서나 물체가 보이는 모습을 2D image로 렌더링할 수 있게 된다.

보다 단순화된 형태로, 이 논문에서는 MLP가 3D position인 $x$를 input으로 받은 뒤 각 위치에 대한 density $\sigma_\theta(x)$와 color $c_\theta (x)$를 output으로 내보내게 된다. 이렇게 MLP를 통해 추출된 정보를 기반으로, 특정 viewpoint에서의 image를 다음과 같은 식을 토대로 렌더링하게 된다. 각 픽셀은 다음과 같이 ray $r(t)$에 따라 volume rendering equation 결과값을 기반으로 색이 정해진다.

\[
    C(r, \theta) = \int_{t_n}^{t_f} T(r, t) \sigma_\theta (r(t)) c_\theta (r(t)) dt  
\]


위의 식에서 $T(r, t)$는 transmittance로, $t_n$에서 출발한 ray가 $t$위치에 도달할 때까지 absorb(object에 의해 decaying)되지 않을 확률을 의미한다. 즉, scene boundary $t_n$ 부터 $t$ 까지 물체가 존재하지 않을 확률값으로 해석하면 된다.

\[
T(r, t) = \exp\left( -\int_{t_n}^t \sigma_\theta (r(s))ds \right)
\]

그러나 실제로는 이와 같이 ray를 따르는 모든 point를 기준으로 적분할 수 없기 때문에 다음과 같이 ray를 smaller segments $(t_{i-1} \leq r_i < t_i)$로 분리하여, 각 segment 내에서는 $\sigma$와 $c$가 어느 정도 constant하게 유지된다고 가정을 한다. 만약 point가 아주 세밀하게 나눠진다면 실제로 이 가정을 따라갈 수 있게 된다.

\[
C(r, \theta) \approx \sum_i T_i (1 - \exp (-\sigma_\theta (r(t_i))\delta_i))c_\theta (r(t_i))
\]

\[
T_i = \exp \left( -\sum_{j < i} \sigma_\theta (r(t_j))\delta_j \right),~\delta_i = t_i - t_{i-1}
\]

MLP parameter $\theta$와 pose $p$에 대한 setting만 있다면, 각 pixel에 대한 ray를 샘플링하여 color 및 density를 계산할 수 있고 이를 모두 rendering하면 특정 픽셀에서의 color $C(r, \theta)$를 계산할 수 있고 image $I(\theta, p)$를 구할 수 있게 된다.

또한 MLP를 사용했을 때, 3D representation에 대한 implicit한 학습은 어렵기 때문에 feature의 representation을 고차원으로 올려주기 위해 positional encoding을 더해준다.

\[
\gamma (x) = (\cos (2^l x),~\sin (2^l x))^{L-1}_{l=0}
\]

$L$은 positional encoding의 level을 의미한다. 실제 구현 단계에서는 [mip-NeRF](https://arxiv.org/abs/2103.13415)에서 제시된 integrated positional encoding(IPE)를 사용했다고 한다.

### IPE(Integrated Positional Encoding)

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/217228304-e1060464-073b-4717-8264-dacccf47592a.png" width="700"/>
</p>


IPE의 컨셉은 위와 같은 사진에서 확인해볼 수 있다. NeRF의 경우에는 모든 frequency를 동일하게 encoding하는데, 이렇게 되면 좌측 이미지에서 보이는 것과 같이 고차원의 encoding value는 aliasing이 발생하게된다(encoding frequency보다 sampling rate이 더 작아지는 경우 발생). 그렇기 때문에 우측 이미지와 같이 샘플링 영역에 대해 gaussian과 같이 적용시켜, 샘플링하는 region이 동일하게 취급당할 수 있게끔 해준다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/217228305-fc2e71d1-590a-4195-b89e-f7bc210faca6.png" width="300"/>
</p>


카메라의 center of projection $o$로부터 pixel이 존재하는 평면에 수직인 방향(normal vector)를 축으로 뻗어나가는 원뿔을 생각해보자. 원뿔의 꼭짓점은 점 $o$이며 $d$벡터가 원뿔의 높이를 대변한다고 생각할 수 있다(아래 그림 참고).

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/217228310-064b35c9-b705-44a5-9959-3d4156c97254.png" width="700"/>
</p>


그렇다면 픽셀이 존재하는 평면과 평행하면서 center of projection $o$로부터 일정 거리 떨어진 곳($o+d$)에 하나의 image plane을 생각해볼 수 있고, 이때 해당 pixel의 반지름(혹은 width)을 $\dot{r}$이라는 parameter로 정의할 수 있다. 이때 위의 그림에서 보이는  $t_0$와 $t_1$ 사이의 position $x$들은 다음과 같이 정의된다.

\[
F(x, o, d, \dot{r}, t_0, t_1) = \left(t_0 < \frac{d^\top(x-o)}{\vert \vert d \vert \vert_2^2} < t_1 \right) \& \left( \frac{d^\top (x-o)}{\vert\vert d \vert\vert_2 \vert\vert x-o \vert\vert_2} > \frac{1}{\sqrt{1+(\dot{r}/\vert\vert d \vert\vert _2)^2}}  \right) 
\]

위의 내용은 사실 모든 $x$에 대해 샘플이 $n$번째 원뿔 요소에 포함되는지 여부고, 이를 실제로 positional encoding에 적용할 수 있어야한다. 가장 간단한 방법은 앞서 소개한 positional encoding $\gamma (x)$를 모든 point에 대해 구한 후, 특정 영역 내의 point를 평균낼 수 있다.

\[
\gamma^* (o, d, \dot{r}, t_0, t_1) = \frac{\int \gamma (x) F(x, o, d, \dot{r}, t_0, t_1) dx}{\int F(x, o, d, \dot{r}, t_0, t_1) dx}
\]

하지만 공식에서 볼 수 있듯이 결국 모든 샘플에 대해 평균내는 과정도 intractable하기 때문에 결국 평균 연산 또한 효율적으로 계산해서 approximation해야하는 문제가 발생한다. 그렇기 때문에 가장 우측에 보이는 것과 같이 $\gamma(x)$의 expectation을 따르는 multivariate gaussian을 가정하는 integrated positional encoding을 사용하였다.

### Image-text models

CLIP과 같은 image-text를 함께 활용한 모델은 다음과 같이 간단한 공식으로 표현할 수 있다. 각 모델은 이미지 인코더인 $g$와 텍스트 인코더인 $h$로 구성된다. 각각의 인코더는 modality에 따른 data를 embedding space로 mapping하는 역할을 한다. 만약 이미지 $I$에 대응되는 text prompt $T$가 있다면, image embedding $g(I)$와 text embedding $h(T)$는 서로 유사한 방향 벡터를 가지게 된다. 여기서 방향 벡터라는 의미는 CLIP space를 상정했을때, 텍스트와 이미지 각각 특정 벡터로 치환되는데, 그와 동시에 L2 norm이 $1$이 되게끔 normalize를 진행한다. 실제 CLIP에서 classification을 진행할 때 코드는 다음과 같다.

```python
# Prepare the inputs
image, class_id = cifar100[3637]
image_input = preprocess(image).unsqueeze(0).to(device)
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)

# Calculate features
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)

# Pick the top 5 most similar labels for the image
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(5)

# Print the result
print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")
```

중간에 ‘top 5 most similar labels for the image’ 부분에서 확인할 수 있듯이, `.encode_image()` 메소드와 `.encode_text()` 메소드를 통해 인코딩된 embedding을 normalize하여 비교하는 것을 볼 수 있다. 만약 크기를 정규화하지 않는다면 실제 cosine similarity(distance)를 제대로 반영하지 못할 수 있기 때문이다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/217228312-43dc415c-c74d-43ef-b957-85011bccdb8e.png" width="500"/>
</p>


$N$개의 text-image pair가 주어지게 되면 서로 대응되는 $N$개의 쌍에 대해서는 positive pair로, 나머지 $N^2-N$개의 쌍에 대해서는 negative pair로 간주하여 contrastive learning을 진행한다. InfoNCE를 대칭 matrix 형태로 연산하게 되므로 마치 text embedding에 대한 information과 image embedding에 대한 information 사이의 관계(mutual information)을 극대화하는 문제로 치환된다.

---

# Methods

### Object representation

NeRF의 scene representation 학습 방법을 토대로 설계되면서 dream fields는 MLP의 parameter set $\theta$를 통해 neural network의 query인 position $x$에 대한 output인 $\sigma_\theta (x)$와 $c_\theta(x)$를 추출하게 된다. 따라서 NeRF space는 MLP 내부에 implicit하게 내장되며, 이는 3D space의 각 부분에서의 density와 color로 대표되는 가상의 공간을 만든 것과 같다. 앞서 설명했던 바와 같이 원래 NeRF 논문과는 다르게 simplified된 버전으로 camera의 viewing direction과는 무관한 네트워크를 가정하였으며, 이는 viewing direction을 사용했을 때의 장점이 없었기 때문이라고 한다.  카메라의 pose $p$를 알게 되면 해당 카메라가 존재하는 평면 상의 모든 pixel에 대한 ray r$(t)$를 가정할 수 있고, 각 ray에 대한 sampling equation

\[
T_i = \exp \left( -\sum_{j < i} \sigma_\theta (r(t_j))\delta_j \right),~\delta_i = t_i - t_{i-1}
\]

을 통해 이미지 $I(\theta, p)$를 렌더링할 수 있다. 샘플링 갯수는 곧 fidelity와 관련이 있고, 논문에서는 $192$를 고정값으로 사용했다고 한다.

### Objective function

실질적으로 dream fields가 최적화될 수 있는 object function은 의외로 간단하다. NeRF space는 MLP parameter $\theta$에 대해 미분 가능하고 마찬가지로 CLIP encoder $g, h$ 또한 freeze된 상태로 학습에 관여하지만 여전히 backpropagation을 통해 gradient 연산이 가능하다. 따라서 CLIP loss를 기반으로 end-to-end 학습이 진행되는 과정에서 parameter $\theta$에 적용되는 유의미한 loss는

\[
\mathcal{L}_\text{CLIP}(\theta, \text{pose } p, \text{caption } y) = -g(I(\theta, p))^\top h(y)
\]

위와 같다. Camera pose $p$에 대해 rendering된 이미지 $I$를 이미지 인코더에 통과시킨 임베딩이, 원하는 text prompt와 유사한 관계를 갖게끔 학습시키게 된다(cosine similarity를 최대화). 사용하는 image encoder, text encoder는 CLIP 원본 논문의 ViT 기반을 사용하기도 했으며, [LiT baseline](https://arxiv.org/abs/2111.07991)도 사용했다고 한다.  Few-shot NeRF 학습에 대해서 다룬 논문인 [DietNeRF](https://arxiv.org/abs/2104.00677)에서는 rendering된 이미지와 real image의 유사도를 계산하는 방법을 사용했는데, 이 논문에서는 real image가 아닌 단순히 caption과의 비교를 통해 zero-shot(object photo가 아예 없는 상황) task도 가능케한 것이다. 물론 DietNeRF 그림을 보면 알겠지만 CLIP image encoder를 활용한 consistency loss를 사용한 점에서는 CLIP embedding space를 low-shot learning에 활용했다는 유사점을 볼 수 있다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/217228314-1ec75432-97c3-4f06-8e04-25734551295e.png" width="400"/>
</p>


### Challenges with CLIP guidance

하지만 CLIP은 단순히 이미지와 텍스트 묘사가 얼마나 유사한 지에 대한 정보만 줄 뿐이지, text 자체가 image의 디테일에 대한 semantic information을 내장하고 있지는 않다. 이런 문제는 NeRF를 최적화하는 과정에 있어 더 두드러지는데, multiple direction에서 보여지는 이미지가 주어졌을 경우에는 유의미한 3D representation을 학습할 수 있었던 네트워크는 많은 supervision에 대해서 학습할 수 있기 때문에 다음 그림과 같이 spurious density 문제(novel view에 대해서 제대로 된 representation이 학습되지 못하고 흩어지는 현상)을 최소화할 수 있게 된다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/217228317-b95e1d77-e74e-4352-b935-1344dfa2706b.png" width="400"/>
</p>


하지만 zero-shot task로 상정한 지금, 오로지 네트워크가 학습에서 의존할 수 있는 supervision은 text prompt 뿐이므로 NeRF 학습 상황이 굉장히 unconstrained한 문제가 생긴다. 최적화 이론에서 optimization을 진행할 함수나 집합이 너무 방대하고 복잡할 경우, PCA와 같이 주성분 방향만 찾거나 penalty(regularization)를 주어 한정된 feasibility를 가지도록 하여 searching space를 최소화하는 방법을 사용한다. NeRF에서 다량의 이미지를 학습에 사용하는 것, 딥러닝에서 large dataset을 통해 네트워크를 학습하는 과정도 결국 비슷한 맥락이다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/217228320-f2c77e3a-52dd-49b5-8228-a9dc4a1e958e.png" width="500"/>
</p>


서로 다른 각도에서 촬영한/획득한 이미지가 많으면 많을수록 네트워크로 하여금 각도에 따른 viewing image에 대한 constraints가 증가하는 것이다. 실질적으로 $N$개가 있을 때 네트워크가 상호작용하는 과정을 여러 번 거친다고 가정했을 때, $1$개의 이미지만 추가되는 과정은 신경망 연산에는 $N^l$ 이상의 학습 효과를 줄 수 있기 때문이다. 여기서 $l$은 레이어 수를 의미한다. 따라서 zero-shot learning에 대해 단순히 text prompt만 적용하는 것은 아래 그림과 같이 좋은 결과를 보여주지 못했다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/217228322-b8424c53-3796-4677-81bf-25e400e2eb57.png" width="500"/>
</p>


가장 눈에 띄는 문제로는 density가 붕 뜨는 region(비어있는 부분)이 생긴다던가, object 부분에 density가 집중되어야 선명한 fidelity의 object가 생성되는데 그냥 camera space 전반에 산재한다는 문제가 발생하였다.

### Pose sampling

Random crop과 같이 image dataset에 흔히 적용하는 augmentation 기법은 이 논문이 참고한 style transfer 연구였던 deep dream 등의 연구에서 image generation 성능을 높이는 방법으로 사용되었다. Image augmentation은 그러나 in-plane의 2D transformation에서만 적용될 수 있다는 문제가 있다. Dream fields는 3D data augmentation을 사용하기 위해 각 training iteration마다 서로 다른 camera pose extrinsics를 샘플링하는 방법을 사용하였다. Scene을 중심으로 $360^\circ$ 의 azimuth에서 uniformly sample하였고, 각 training iteration은 object을 서로 다른 방향에서 본 모습을 샘플링하게 된다. MLP는 서로 같은 scene representation을 공유하고 있기 때문에, 단순히 camera sampling을 하는 것만으로도 object geometry 성능에 큰 향상을 줄 수 있다.

단순히 카메라의 회전각 이외에도 camera elevation(focal length 조절, object와의 거리) 또한 augmentation에 사용될 수는 있었으나 굳이 사용하지는 않았다고 한다.

### Encouraging coherent objects through sparsity

Near field artifacts(field 경계에 artifact가 생기는 것)이나 spurious density(듬성듬성하게 샘플링되는 것) 문제를 해결하기 위해서 dream fields rendering 과정에서 opacity regularization을 진행했다고 한다. Opacity는 앞서 설명했던 transmittance와 연관되는 내용인데, transmittance는 ray $r$이 $t$부터 plane 근처 $t_n$ 사이를 이동하는 과정에서 물체에 흡수되지 않을 확률이다(밀도가 높은 점이 있으면 흡수율이 높다). 저자들이 가정한 것은 total transmittance를 $N$개의 sampling segment를 통과하는 동안의 joint transmittance라고 가정하였다. 식이 동일하니까 해석하는 과정에서는 큰 문제가 없다.

\[
\begin{aligned}
T_i =& \exp \left( -\sum_{j < i} \sigma_\theta (r(t_j))\delta_j \right) \newline
=& \prod_{j=1}^{i} \exp \left( -\sigma_\theta(r(t_j))(t_j - t_{j-1}) \right)
\end{aligned}
\]

그런 뒤, transmittance loss를 다음과 같이 정의하였다.

\[
    \begin{aligned}
        \mathcal{L}\_T =& -\min (\tau, \text{mean}(T(\theta, p))) \newline
        \mathcal{L}\_\text{total} =& \mathcal{L}_\text{CLIP} + \lambda \mathcal{L}_T
    \end{aligned}
\]

여기서 $\tau$는 저자들이 설정한 target transparency가 된다. 만약 joint probability가 높아진다면 그만큼 spurious density가 적다는 의미이고, 반대로 낮아진다면 ray 전반에 걸쳐 density가 산재한다는 뜻이기 때문에 해당 loss term을 최적화하는 것이 object에 대한 정규화 방법이 될 수 있다. 따라서 초반 안정적인 학습을 위해 $\tau$를 $88\%$로 두고 학습시킨 뒤, $500$ iteration 이후에는 $40\%$로 감소시켜 completely transparent(object가 아예 안생겨버리는 현상)을 방지하였다. Focal length에 대해서 $\tau$를 scaling하는 것이 서로 다른 object distance에 대해서 잘 적용되었다고 한다. 그리고 렌더링 과정에서 단순히 white 혹은 black background에 대해 진행할 경우 transmittance가 수렴하더라도 scene이 background에 치우치는 문제가 발생하였고, 이를 줄이기 위해서 background를 랜덤한 이미지로 설정하는 것이 object 학습에 도움이 되었다고 한다. Dream fields는 gaussian noise, checkerboard 그리고 random fourier texture를 background로 사용하고 이를 random gaussian noise로 smoothing하여 사용했다고 한다. 

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/217228324-8d62abcd-d50f-4404-b5eb-ec212b29e11e.png" width="600"/>
</p>


사용된 prompt는 “An illustration of a pumpkin on the vine”이다. 실제로 렌더링된 모습을 보면 regularization이 적용될 경우 sparse한 image 경향성이 줄어드는 것을 확인할 수 있고, 추가적으로 white background에서 augmented background로 수정했을 때 sharper object가 생성되는 것을 볼 수 있다.

### Localizing objects and bounding scene

Neural Radiance Fields를 image reconstruction에 학습시킬 때, scene content는 NeRF에서 주로 사용되는 데이터셋과 같이 중앙 부분에 align되는 것이 일반적이다. Dream fields는 굳이 center에 놓이지 않더라도 3D object의 중앙 부분을 예측할 수 있으며, ray shifting도 그에 맞게 유지할 수 있음을 확인하였다. Origin을 찾는 과정은 rendering된 object의 무게중심을 exponentially moving하는 방법을 사용했고, object가 지나치게 치우치는 것을 방지하기 위해 density를 masking함으로써 일정 크기의 cube 내에 object를 위치시켰다.

### Neural scene representation architecture

NeRF는 8개의 layer를 가진 MLP를 사용했고, 모두 같은 width(channel 수)를 가지고 있으면서 RGB value를 내보내기 위한 두 개의 추가 layer가 있었다(아래 그림 구조 참고).

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/217228327-5dba54b4-f3af-45c7-9b7a-1d8d4a93eacf.png" width="700"/>
</p>


해당 논문에서는 위의 구조 대신 residual MLP structure를 사용했고, residual connection은 two dense layer마다 반복해서 사용했다고 한다. 모델 구조가 논문에 그림으로 첨부가 되어있지 않아서 official 코드를 찾아보니 아래와 같이 나와있었다. Google이다 보니 `pytorch`가 아니라 `flax, jax` 모듈을 사용하였다.

```python
class MipMLPLate(nn.Module):
  """MLP architecture."""
  activation: str, features_early: Sequence[int], features_residual: Sequence[Sequence[int]]
  features_late: Sequence[int], fourfeat: bool, max_deg: int, use_cov: bool, dropout_rate: float

  @nn.compact
  def __call__(self, mean, cov=None, x_late=None, decayscale=1., *, deterministic):
    """Run MLP."""
    # Integrate the positional encoding over a region centered at mean.
    if not self.fourfeat:
      # Axis-aligned positional encoding.
      feat = 2**np.arange(self.max_deg)[:, None, None] * np.eye(3)
      feat = feat.reshape(-1, 3)
    else:
      # Random Fourier Feature positional encoding. Fix the PRNGKey used for the
      # fourier feature basis so the encoding does not change over iterations.
      fourfeat_key = random.PRNGKey(124124)
      dirs = random.normal(fourfeat_key, (3, 128))
      dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
      rads = 2 ** (self.max_deg * random.uniform(fourfeat_key, (128,)))
      feats = (rads * dirs).astype(np.int32)
      feats = np.concatenate([np.eye(3), feats], 1).astype(np.float32)
      feat = feats.T

    mean_proj = (mean[Ellipsis, None] * feat.T).sum(-2)
    if self.use_cov:
      cov_diag_proj = ((cov[Ellipsis, None] * feat.T).sum(-2) * feat.T).sum(-2)
      decay = np.exp(-.5 * cov_diag_proj * decayscale**2)
    else:
      # Disable IPE
      decay = 1.
    x = np.concatenate([decay * np.cos(mean_proj),
                        decay * np.sin(mean_proj)], -1)

    # Network
    activation = nn.__getattribute__(self.activation)
    for feat in self.features_early:
      x = activation(nn.Dense(feat)(x))
      x = nn.Dropout(self.dropout_rate)(
          x, deterministic=deterministic)

    for feat_block in self.features_residual:
      h = nn.LayerNorm()(x)
      for l, feat in enumerate(feat_block):
        h = nn.Dense(feat)(h)
        h = nn.Dropout(self.dropout_rate)(
            h, deterministic=deterministic)
        if l < len(feat_block) - 1:  # don't activate right before the residual
          h = activation(h)
      x = x + h

    if x_late is not None:
      x = np.concatenate([x, x_late], axis=-1)
    for feat in self.features_late[:-1]:
      x = activation(nn.Dense(feat)(x))
      x = nn.Dropout(self.dropout_rate)(
          x, deterministic=deterministic)
    x = nn.Dense(self.features_late[-1])(x)  # don't activate output
    return x
```

`jax` 모듈에서의 `np`는 `numpy`와 동일하고 마찬가지로  `flax.linen` 의 `nn`은 `pytorch`의 `torch.nn` 과 유사하게 동작한다고 보면 된다. Residual layer는 2 layers마다 적용하였고, layer normalization을 각 residual layer 앞쪽에 적용해주는 것이 효과적이었다고 한다. 또한 API에 구체적으로 나타나있지는 않지만 bottleneck 구조와 같이 중간 channel을 expansion하는 과정이 있다고 생각하면 된다.

또한 vanishing gradient 문제가 highly transparent scenes(대부분의 rendering이 0가 될 경우)에서 발생할 수 있게 되므로 ReLU activation function을 적용하는 대신 Swish를 사용하였으며 density $\sigma_\theta$를 softplus function을 통해 rectify했다고 한다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/217228329-5722b141-5cec-4e42-8a78-5c926ea4e878.png" width="400"/>
</p>


\[
f(x) = x \cdot \sigma(x),~\sigma(x) = \frac{1}{1+e^{-x}}
\]

---

# Results

### Quantitative results on geometric priors

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/217228331-f17970bf-ae50-4883-9765-f6f54d2e8791.png" width="400"/>
    <img src="https://user-images.githubusercontent.com/79881119/217228340-29b945ee-0cb9-4b02-9a32-8d2b3fc017fa.png" width="900"/>
</p>


이 논문에서 실제로 사용했던 geometric prior들이 COCO caption에 대해 생성된 이미지를 retrieval할 수 있는지  측정하였다. 논문에서 사용한 metric들이 적용이 될 때마다 생성한 샘플에 대해 retrieval 정확도가 올라가는 모습을 볼 수 있다. 또한 regularization이 사용될 때마다(논문에서 제시한 transparency에 대한 정규화) retrieval 성능이 올라가는 것을 볼 수 있다.

### Compositional generation

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/217228342-09cbf4d6-be6e-4086-98dc-e34b2a8a5fe2.png" width="900"/>
</p>


해당 그림에서는 cherry picking하지 않은 dream fields의 샘플들을 보여준다. Shape과 material에 대해 독립적으로 다르게 구성한 object를 생성하고 결과를 확인해보았다. DALL-E와 같은 네트워크 또한 caption에 대해 충분히 좋은 이미지를 생성할 수 있지만, 이 논문과 같이 3D object를 직접 생성해내지는 못한다는 한계가 있다.

### Regularization strength

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/217228352-e6281836-ba10-4ed6-bc3f-808d6c190523.png" width="800"/>
</p>


그리고 transparency에 대한 strength를 크게 주면 줄수록 object가 컴팩트한 사이즈를 가지는 것을 확인할 수 있다. 위의 object를 생성한 description은 ‘A cake toped with white frosting flowers with chocolate centers’이다.

---

# Discussion and Limitations

Dream fields는 zero-shot을 처음으로 NeRF에 효과적으로 적용했던 논문이라고 볼 수 있지만, 여러 한계점이 존재한다. 첫번째는 generation 과정이 iterative optimization을 필요로 하기 때문에, 샘플링을 하는 시간이 오래 걸린다는 점이다. 논문 저자들은 meta-learning이나 amortization 방법을 통해 추후에 이를 가속화할 수 있다고 제시한다.

두번째로는 모든 perspective에 대해 동일한 prompt를 사용한다. 이는 object의 서로 다른 면에 비슷한 무늬가 반복되게하는 문제를 만들었다고 한다(일종의 texture collapse 문제 같은 것들이 발견된 듯하다). 카메라 위치에 따라 충분히 캡션이 달라질 수 있는 상황을 가정하지 못한 점이 문제라고 한다. 예를 들어 같은 물체라도 모든 면에서 보았을 때 동일한 형태가 되는 texture가 아니라면 앞면에서 봤을 때와 뒷면에서 봤을 때 묘사할 수 있는 appearance가 달라질 수 있기 때문이다.

마지막으로는 CLIP 네트워크 자체가 가지는 문제다. 아무리 CLIP이 image to text representation을 잘 학습했더라도, ground truth training image의 일부에 대해서도 잘못된 score rendering을 하는 경우가 생긴다. CLIP 네트워크의 성능이 generation에 영향을 미칠 수 있다는 점이다. Pre-trained model에 의존하는 것은 네트워크가 내포하고 있을지도 모르는 bias를 감수하겠다는 것이다.

위의 내용은 저자들이 밝힌 한계점이고, 본인이 말하고 싶은 한계점은 논문 자체의 writing에 대한 부분이다. Style transfer 논문이 아예 related works에 포함이 되지 않는 것은 아니고 각 paper에서 사용되던 방법 중 일부가 적용된 사항도 있지만(background rendering 등), 실제로 이 논문을 작성하면서 main inspiration을 준 것은 아니라고 본다. 사실 논문을 끝까지 읽고서는 어느 정도 납득이 갈 수 있는 내용이었지만 related works 도입부부터 갑자기 현재 task와는 너무 동떨어진 것 같은 paper가 등장해서 혼란스러움을 겪었다. 사실 이 한계점은 개인적인 견해이기 때문에 크게 중요하진 않다고 생각한다.
