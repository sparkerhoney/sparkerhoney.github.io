---
title: (StyleCLIP, StyleGAN-NADA) CLIP based image manipulation에 대하여
layout: post
description: Image manipulation, VL contrastive
use_math: true
post-image: https://user-images.githubusercontent.com/79881119/215319341-b7c913a7-26c9-4bc4-bab9-a19281714eeb.gif
category: paper review
tags:
- Multimodal
- VL
- StyleGAN
- Image manipulation
- Text(prompt) guidance
---

# StyleGAN
[StyleGAN](https://arxiv.org/abs/1812.04948)의 등장은 사실상 생성 모델을 style based approach로 해석한 첫번째 논문이라고 볼 수 있다. StyleGAN 논문을 이 글에서 자세히 설명하지는 않겠지만 가볍게 요약하자면, constant vector로 표현된 하나의 feature map 도화지가 있고($4 \times 4 \times 512$), 이 도화지에 latent vector로부터 추출된 style 정보들(평균값인 $\mu$와 표준편차인 $\sigma$)를 affine layer를 통해 얻어내어 이전의 feature map을 modulation하면서 점차 사이즈를 키워나가는 구조다. 점차 feature map의 spatial resolution을 키운다는 점에서 타겟팅이 된 논문 구조인 [PGGAN](https://arxiv.org/abs/1710.10196)도 같이 읽어보면 좋다.

---

# Image manipulation

아무톤 styleGAN의 디테일한 방법론에 대해서 논하고자 하는 것은 아니고, styleGAN이 가져온 이후 연구들의 동향에 대해서 살펴볼 수 있다. 무작위로 뽑아낸 latent vector $z$ 혹은 $w$로부터 '<U>style 정보</U>'를 얻어낼 수 있다는 것은 반대로 생각해서 특정 이미지를 주었을 때, 해당 이미지를 만들어낼 수 있는 latent vector $z$ 혹은 $w$를 추출할 수 있다는 말과 동일하다.   
이러한 GAN inversion 개념은 image manipulation에 큰 동향을 불러왔으며, 특히나 styleGAN의 경우 high-resolution image synthesis가 가능케 한 논문이었기 때문에 <U>고퀄리티의 이미지 조작이 가능하다는 점</U>이 main contribution이 되었다. 이에 여러 이미지 조작 논문들이 나왔으며, 해당 내용에 대해 궁금한 사람들은 본인이 작성한 [image manipulation 포스트](https://junia3.github.io/blog/imagemanipulate)를 참고하면 좀 더 좋을 것 같다.   
StyleCLIP도 결론부터 말하자면 사전 학습된 StyleGAN을 활용한 Image manipulation이라는 task에 대해서 다룬 내용이고, 이 논문에서는 기존 방식과는 다르게 VL contrastive 논문인 CLIP을 활용한 <U>text guidance</U>가 보다 image manipulation에 조작 편의성을 가져다줄 수 있다는 점에 집중하였다.

---

# 기존 방법들의 문제점은?

앞서 언급하기도 했지만 <U>image manipulation task</U>에 대한 연구 동향은 대부분 사전 학습된 styleGAN latent space에서 유의미한 semantical information을 찾고, 이를 조작하는 방식으로 진행된다. 그러나 직접 latent space를 searching하는 과정 자체는 구체적으로 이미지 상에서 어떤 스타일을 변화시킬지도 모르고, 무엇보다 사용자가 원하는 이미지 조작이 이루어지기 힘들다는 것이다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/215320268-1ddc118f-2e02-4fe1-be7a-919196a3d324.png" width="900"/>
</p>

예를 들어 위와 같은 이미지를 보면, 평범한 표정의 여자 이미지를 만들어내는 latent vector $w_1$이 있고, 이와는 다르게 놀라는 표정의 여자 이미지를 만들어내는 latent vector $w_2$가 있다고 해보자. 단순히 latent space를 searching하는 과정에서는 '놀라는 표정'을 만들어낼 수 있는 latent manipulation 방향을 알 수 없기 때문에 random하게 찾는 과정을 거칠 수 밖에 없다. Latent vector는 $512$ 크기의 차원 수를 가지기 때문에 supervision이 없다면 인간이 직접 찾아야하는 번거로움을 피할 수 없다. 이러한 방법을 사용한 것이 [GANspace](https://arxiv.org/pdf/2004.02546.pdf), [Semantic face editing](https://arxiv.org/pdf/1907.10786.pdf) 그리고 [StyleSpace 분석](https://openaccess.thecvf.com/content/CVPR2021/papers/Wu_StyleSpace_Analysis_Disentangled_Controls_for_StyleGAN_Image_Generation_CVPR_2021_paper.pdf) 논문들에 해당된다. 이러한 문제들을 해결하기 위한 방식이 attribute에 대한 classification을 guidance로 삼는 [InterfaceGAN](https://arxiv.org/pdf/2005.09635.pdf)이나 [StyleFlow](https://arxiv.org/pdf/2005.09635.pdf)가 제안되었다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/215321251-44e2194d-1ae1-4fc8-857e-ac61137e92e4.png" width="900"/>
</p>

만약 latent space에서 특정 element를 바꿨을 때의 attribute 변화가 '얼굴 표정'임을 원한다면 해당 attribute의 변화를 최대화하는 쪽으로 latent manipulation을 진행하게 되는 것이다. 이외에도 parametric model인 3DMM을 활용하여 3D face mesh에 consistent한 sampling을 진행하는 [StyleRig](https://arxiv.org/pdf/2004.00121.pdf) 논문 등등이 소개되었다.   
어찌되었든 기존 방식들은 정해진 semantic direction을 찾아가는 과정을 바꾸지는 못했고, 이는 사용자의 image manipulation 방식에 제한을 둘 수밖에 없었다(latent space를 searching하거나, classifier에 의한 guidance). 기존 방식에서 벗어나 latent space에 mapping되지 않은 direction에 대해서 searching하는 과정을 고려하면(StyleRig), <U>manual effort</U>와 충분한 갯수의 <U>annotated data</U>가 필요한 것을 알 수 있다.

---

# CLIP을 활용한 image manipulation
따라서 저자들은 방대한 text-image web dataset으로 prompt 기반 image representation을 학습한 VL contrastive model의 성능에 집중하였고, 해당 네트워크가 <U>prompt based zero-shot image classification</U>에서도 성능을 입증했던 바와 같이 마찬가지로 image manipulation 과정에서도 어느 정도 자유도를 보장할 수 있다고 생각하였다. CLIP 네트워크와 기존 framework인 styleGAN을 혼용하는 방식은 여러 가지가 있을 수 있지만, 저자들은 이 논문에서 총 세 가지의 방법들을 언급하였다.

1. 각 이미지 별로 text-guided latent optimization을 진행하는 방법. 단순히 CLIP model을 loss network(latent vector를 trainable parameter로 gradient를 보내고, clip encoder의 결과와 text encoder의 결과 사이의 similarity를 loss로 주게 되면 latent 최적화가 진행되는 방식)
2. Latent residual mapper를 학습하는 방법. 이 과정에서는 latent vector를 직접 최적화하는 것이 아니라 latent mapper를 학습하여, $\mathcal{W}$ space의 특정 latenr를 CLIP representation에 맞는 또다른 latent vector로 mapping한다.
3. 이미지에 상관 없이 text prompt에 맞는 style space를 학습하는 방법. 이 방법은 styleGAN이 가지고 있는 $\mathcal{W}$ space의 disentanglement 특징을 그대로 유지하면서 style mapping을 하고자 하는 것이 주된 목적이다.

---

# Related works

## Vision-Language tasks
가장 먼저 언급할 수 있는 related works로는 <U>vision + language</U> representation에 대한 내용이 될 것이다. Vision과 Language가 융합된 연구에는 language based image retrieval(원하는 부분 찾아내기), image captioning(이미지를 묘사하는 캡션 달기) 혹은 visual question answering(시각적인 정보를 토대로 질문에 대한 답변하기) 등등 여러 가지 task가 있다. BERT 모델이 여러 language task에 대해 좋은 성능을 보였던 바를 토대로, 최근 VL 방법들은 대부분 joint representation(이미지와 텍스트 간의 유의미한 관계)를 학습하기 위해 <U>transformer backbone</U>을 활용하기 시작했다. CLIP도 마찬가지로 <U>transformer backbone</U> 기반으로 학습되었다고 생각하면 된다. 물론 image encoder는 ResNet50 구조를 차용한 구조의 경우 완전한 transformer라고 볼 수는 없지만, attention pooling을 통해 어느 정도 embedding을 추출하는데 있어 transformer와 유사하게 동작한다고 할 수 있다. 

## Text-guided image generation and manipulation
물론 image generation에 대한 연구도 multimodal로 진행되었던 적이 있다. 예를 들어 conditional GAN 구조를 사용하면서, 이때 condition vector를 text embedding을 활용하여 학습하는 형태가 될 수 있다. 가장 초창기 연구([논문링크](https://arxiv.org/abs/1605.05396))의 경우 아직 transformer 아키텍쳐가 발전되기 전이었기 때문에 단순한 RNN 구조를 차용하여 text embedding을 추출, 이를 generator와 discriminator 학습에 있어서 conditional 요소로 사용하게 된다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/215374864-a3774c59-1894-4e93-925b-bbba30f72354.png" width="900"/>
</p>

이후에는 multi-scale GAN 구조를 활용하여 image quality를 올리거나, attention mechanism을 text와 image feature 간에 활용하는 형태의 연구도 진행되었다([논문링크](https://arxiv.org/pdf/1711.10485.pdf)).

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/215375227-be754bf6-97c0-4b3a-9ca0-dadbcd63ae91.png" width="900"/>
</p>

앞서 설명한 간단한 내용들은 image generation에 해당되고, 우리가 지금 포커싱하는 것은 특정 image에 대해서 text가 이미지 수정에 대한 supervision을 줄 수 있는 image manipulation과 관련된 이야기를 해볼까 한다. 사실 image manipulation이라는 task는 generation보다 까다롭기 때문에 위에서 언급한 것과 같이 <U>딥러닝 초창기</U>부터 연구가 진행되어오던 것은 아니다. 그럼에도 GAN 구조를 활용한 초창기 연구들이 있었다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/215375676-9fc38293-eff9-41d8-bb78-8b525cf9322c.png" width="900"/>
</p>
GAN encoder/decoder structure를 활용한 초반 여러 방법들의 경우 image와 text description 각각의 semantic을 disentangle하는데 초점을 맞추었다. 비교적 가장 최근의 image manipulation 논문 중 하나라고 볼 수 있는 [ManiGAN](https://arxiv.org/pdf/1912.06203.pdf)은 ACM이라는 text-image <U>affine combination module</U>을 제안하여 manipulated된 이미지의 퀄리티를 높일 수 있었다. 구조를 보게 되면 styleGAN에서 사용되는 affine을 기반으로 한 style modulation 과정과 상당히 닮아있는 것을 확인할 수 있다. 이러한 앞선 연구들의 특징은 styleGAN base로 접근하지 않고 대부분 basic GAN approach를 사용했지만, StyleCLIP 논문에서는 StyleGAN approach를 사용했다는 점이 차이점이 될 것 같다.   
물론 StyleGAN을 사용한 approach는 StyleCLIP이 처음은 아니다. [TediGAN](https://arxiv.org/pdf/2012.03308.pdf)도 마찬가지로 StyleGAN의 approach를 사용했는데, StyleCLIP 논문에서의 방법과 차이가 있다면 text를 StyleGAN의 latent space로 mapping하는 encoder를 학습하고, 이를 통해 text 기반으로 찾은 style space와 image의 style space를 융합하고자 한 것이다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/215376914-5c2dedbc-189c-49b9-b9c0-8054b37e78d8.png" width="600"/>
</p>

어찌보면 TediGAN의 approach는 StyleGAN에서 8개의 MLP로 생성한 $\mathcal{W}$-space 구조가 normal distribution $\mathcal{Z} \sim \mathcal{N}(0,~I)$에 기반하기 때문에, image modality와 text modality의 차이가 있다고 하더라도 <U>encoder에 의한 implicit functional mapping이 가능</U>하다고 생각했던 것 같다. 사실 이러한 방법이 조금은 더 text와 image 간의 유의미한 representation을 찾는 방법이라고 생각되지만, CLIP의 성능 덕분인지 TediGAN보다는 StyleCLIP의 성능이 더 좋았다고 한다. 이외에도 DALL-E 방식도 있지만 GPU가 어마무시하게 사용된다는 점에서 text to image manipulation의 용이성이 떨어진다는 점이 있다. StyleCLIP처럼 논문 형태로 나온 것은 아니고 단순히 온라인 포스팅으로 StyleGAN과 CLIP을 image manipulation에 사용한 방식이 있다([참고 링크](https://towardsdatascience.com/generating-images-from-prompts-using-clip-and-stylegan-1f9ed495ddda)).

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/215377921-95be986c-42e7-436a-811b-640742392199.png" width="500"/>
</p>

위의 그림을 잘 보게되면 결국 StyleCLIP에서 image manipulation approach로 생각한 latent optimization 과정이 거의 똑같은 것을 알 수 있다. 하지만 위의 방식은 이미지를 처음부터 생성하는 관점에서의 latent optimization이고, 이 논문은 특정 이미지를 만들어내는 style space의 latent vector에 대해서 optimization을 진행, <U>image manipulation</U>에 보다 초점을 맞췄다는 점이 차이가 될 수 있겠다.

## Latent space image manipulation
Image manipulation이라는 task에서 pre-trained styleGAN generator를 사용하는 방법은 이미 다양한 논문들이 존재한다. StyleGAN의 intermediate style space인 $\mathcal{W}$는 disentanglement 등 image manipulation 관점에서 도움될 특징이 많기 때문이다. 보통 특정 이미지를 latent space로 매핑한 뒤에, manipulated image의 representation으로 guidance를 주는 방법이 일반적인데, 여기서 image annotation을 supervision으로 사용하는 approach와 직접 meaningful direction을 찾는 approach로 나눌 수 있다. 대부분의 styleGAN을 base로 한 manipulation 연구들에서는 $512$ 차원의 $\mathcal{W}$-space vector를 사용하거나, 이를 확장시켜서 각 feature level에 따라 style를 넣는 $\mathcal{W}+$ vector를 사용한다. 그러나 [stylespace 분석 논문](https://arxiv.org/pdf/2011.12799.pdf)에서 stylespace $\mathcal{S}$를 사용하는 것이 더 좋다는 주장을 하였고, 기존 space보다 disentanglement에 효과적임을 보여주었다.
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/215379873-de5b104f-831d-4240-bf46-97d1a6705691.png" width="700"/>
</p>
앞서 말했던 바와 같이 StyleCLIP 논문에서는 총 세가지의 optimization 방법을 제시했는데, 다시 언급해보면

1. 각 이미지 별로 text-guided latent optimization을 진행하는 방법. 단순히 CLIP model을 loss network(latent vector를 trainable parameter로 gradient를 보내고, clip encoder의 결과와 text encoder의 결과 사이의 similarity를 loss로 주게 되면 latent 최적화가 진행되는 방식)
2. Latent residual mapper를 학습하는 방법. 이 과정에서는 latent vector를 직접 최적화하는 것이 아니라 latent mapper를 학습하여, $\mathcal{W}$ space의 특정 latenr를 CLIP representation에 맞는 또다른 latent vector로 mapping한다.
3. 이미지에 상관 없이 text prompt에 맞는 style space를 학습하는 방법. 이 방법은 styleGAN이 가지고 있는 $\mathcal{W}$ space의 disentanglement 특징을 그대로 유지하면서 style mapping을 하고자 하는 것이 주된 목적이다.

위와 같다. 따라서 latent optimization을 직접 진행하게 되는 1번과 2번 방법은 기존 StyleGAN space인 $\mathcal{W}$, 그리고 $\mathcal{W}+$를 그대로 사용하게 되고, 3번의 경우 $\mathcal{S}$ space를 새롭게 학습하는 과정이 된다(input agnostic).

---

# StyleCLIP text-driven image(latent) manipulation

논문에서는 위에서 설명한 세 가지 방법에 대해 따로 실험을 진행하였다. 가장 간단한 방법은 각 source image와 원하는 style에 대해 묘사하는 text prompt를 기반으로 최적화하는 것이다. 다만 이러한 방법은 hyperparameter에 대해 불안정한 학습이 진행될 수 있기 때문에 각 샘플 별로 mapping network를 학습시켜서 latent space를 mapping하는 방법을 사용할 수도 있다. 이를 <U>local mapper</U>라고 부르는데, mapper의 목적은 latent space에서 이미지를 생성하는 각 latent vector를 최적화하는 step을 inference하여 한 번에 mapping할 수 있게끔 학습하고자 하는 것이다. 각 mapper의 학습에는 단일 text prompt에 대해 각 image sample의 latent starting point를 기준으로 학습하게 된다.   
하지만 저자들이 실험한 결과를 바탕으로는 이러한 각 manipulation에 대한 mapper 학습을 할 경우 mapper의 step이 유사해진다는 것(manipulation 다양성이 떨어짐)과 disentanglement를 제대로 활용할 수 없다는 점이 문제가 되었다. 따라서 input agnostic(샘플의 starting point에 상관없이) mapping network를 학습하는 방법을 사용했으며, 이때의 global manipulation direction은 style space $\mathcal{S}$에서 계산되었다고 한다.
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/215386009-5771a369-9e87-4e98-8bea-427fe25502e4.png" width="900"/>
</p>

## Latent optimization
Latent optimization은 상당히 간단한 식으로 표현된다. 딱히 길게 풀어쓸 내용은 아니기 때문에 수식을 먼저 보고 해석하는 식으로 마무리하겠다.

\[
    \begin{aligned}
    &\arg \min \_{w \in \mathcal{W}+} D\_\text{CLIP} (G(w), t) + \lambda\_\text{L2} \parallel w - w_s \parallel_2 + \lambda\_\text{ID} \mathcal{L}\_\text{ID} (w) \newline
    &\mathcal{L}\_\text{ID} (w) = 1- \left< R(G(w_s)), R(G(w)) \right>
    \end{aligned}
\]

위에서 $R$은 ArcFace network(얼굴 인식)의 pretrained된 feature extractor로 보면 되고, $\left< \cdot, \cdot \right>$ 연산은 곧 두 생성된 이미지에 대한 feature embedding 사이의 cosine similarity를 계산한 것과 같다. Input image는 [e4e](https://arxiv.org/abs/2102.02766)를 기반으로 $w_s$로 mapping된다고 생각하면 된다. $D\_{\text{CLIP}}$은 보이는 바와 같이 생성된 이미지에 대한 <U>CLIP image embedding</U>과 <U>text embedding</U> 사이의 코사인 거리를 좁혀주는 역할을 하게 되고, 나머지 term은 <U>원본 이미지를 얼만큼 유지할 지</U>(contents 유지력)에 대한 지표가 된다. 해당 유지력에 대한 weight는 $\lambda\_\text{L2},~\lambda\_\text{ID}$로 조절할 수 있다.

## Latent mapper
그러나 위에서 보이는 latent optimization 과정은 각 input image와 text prompt에 대해 변동성이 크기 때문에 각 process마다 적절한 hyperparamer 조정 과정이 필요하다. 그렇기 때문에 다음으로 생각해볼 수 있는 방법은 latent starting point $w$에 상관없이(input image를 고정하지 않고), text prompt $t$에 대한 latent mapper $M_t$를 학습하여 $\mathcal{W}+$로의 manipulation step을 학습하고자 하는 것이다. 즉, pretrained styleGAN 기반으로 latent를 최적화하던 과정을 <U>하나의 함수로 구현</U>한 것이 mapper라는 개념이다.   
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/215380004-e5e82b34-8b57-44fe-9211-6dc1013b3178.png" width="900"/>
</p>
StyleGAN의 각 layer level에 따라 조절하는 image의 detail이 달라지기 때문에 coarse, medium 그리고 fine에 대한 mapper를 다르게 설정하였다. 모든 mapper는 StyleGAN의 mapper layer 수의 절반인 4개의 layer를 가지는 fully-connected networks를 사용했다. 보다 디테일하게 살펴보면, latent mapper는 latent $w_s$를 다이렉트로 최적의 latent vector $w$로 매핑하는 형태가 아니라, <U>변화량을 예측</U>하는 네트워크가 된다.

\[
    \mathcal{L}\_\text{CLIP}(w) = D\_\text{CLIP}(G(w + M_t (w)), t)    
\]

이외에는 이전에 사용했던 loss term과 모두 동일하다. 이때, $M_t(w) = w - w_s$와 같다고 생각하면 된다. 논문에서는 $\lambda\_\text{L2} = 0.8$ 그리고 $\mathcal{L}\_\text{ID} = 0.1$을 사용했다.

\[
    \mathcal{L}(w) = \mathcal{L}\_\text{CLIP}(w) + \lambda\_\text{L2} \parallel M_t(w) \parallel_2 + \lambda\_\text{ID} \mathcal{L}\_\text{ID} (w)
\]

그런데 고정으로 사용한 것은 아니고 <U>샘플마다 다른 parameter를 사용한 것</U>도 있다고 한다. 사실 여기서 이미 <U>논문의 limitation</U>이 보인다고 할 수 있는게, text based approach를 사용했으면 어느 정도 **latent searching** 과정에서의 편의성은 보장되었지만 **hyperparameter searching** 과정에서의 편의성은 보장되지 않았기 때문에 결국 거기서 거기라고 생각했다.
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/215389280-7f136251-9a53-4692-ab5a-1c4265161981.png" width="700"/>
</p>


## Global direction
Latent mapper는 optimization 과정에 비해 빠른 inference time을 보장했지만, fine-grained disentangled manipulation(디테일한 부분들을 서로 구분지어 manipulation)하는 과정이 기존 방식에서는 어렵다는 것을 알게 되었고, 이는 결국 $\mathcal{W}+$ spcae 자체가 가지고 있는 한계점으로 분석되었다. 따라서 논문에서는 앞서 소개했던 style space $\mathcal{S}$에서 global direction을 찾고자 하였다. 만약 $s \in \mathcal{S}$가 style space에서의 code라고 해보자. 그리고 $G(s)$는 해당 style code를 기반으로 생성된 image라고 생각해볼 수 있다. Manipulation을 원하는 attribute에 대한 text prompt가 있다고 생각하고, 여기서 style code $s$의 manipulation direction인 $\Delta s$를 찾고자 한다. 즉, 우리가 원하는 attribute를 최대화하는 방향으로 $G(s + \alpha \Delta s)$라는 이미지를 만들고 싶다는 것이다. 여기서 style direction $\Delta s$는 <U>manipulation을 원하는 attribute를 제외하고는 다른 attribute를 변화시키지 않아야한다</U>. 변화의 정도를 step size $\alpha$가 결정하게 된다.   
먼저 CLIP text encoder를 사용하여 joint language-image embedding으로부터 $\Delta t$를 구하고, 이를 다시 manipulation direction $\Delta s$로 mapping하는 것이다. Text에 대한 direction $\Delta t$는 natural language를 text prompt engineering하여 구하고, 이에 해당되는 $\Delta s$는 각 style channel이 target attribute에 끼치는 영향을 고려하여 설정된다. 이를 보다 자세히 풀어쓰면 다음과 같다. 예를 들어 image embedding에 대한 manifold $\mathcal{I}$와 text embedding에 대한 manifold인 $\mathcal{T}$가 있다고 했을 때, image에서의 semantic changes를 이끌어내는 방향 벡터와 text에서의 semantic changes를 이끌어내는 방향 벡터는 서로 어느 정도 collinear(large cosine similarity)를 가질 것으로 예상되고, 특히 각각을 normalization을 거친다면 거의 동일할 것으로 예상된다.   
따라서 원본 style의 이미지 $G(s)$와 변화된 style에 대한 이미지 $G(s + \alpha \Delta s)$에 대해 각각의 $\mathcal{I}$ manifold에서의 embedding을 $i$, $i+\Delta i$로 표현할 수 있다. 두 이미지의 CLIP 상에서의 차이를 $\Delta i$로 표현할 수 있기 때문에, 스타일 변화에 대한 text embedding 벡터인 $\Delta t$가 있다면 둘 사이의 유사도를 통해 global direction을 찾을 수 있게 된다.

##### Natural language to $\Delta t$

사실 이 부분은 대부분의 CLIP based approach에서 모두 사용하는 코드/방법이기 때문에 알아두는 것이 좋은데, 그 이유는 특정 text prompt에 대한 특징을 단일 image 하나에 mapping하는 것은 사실상 이미지와 텍스트 간의 modality 차이 때문에 바람직하지 않다는 점 때문이다. 만약 '안경을 쓴 사람'이라는 image description 하나로는 정확하게 어떤 머리를, 어떤 얼굴을 그리고 어떤 성별의 사람이 안경을 썼는지 표현할 수 없기 때문에 흔히 image와 text 사이에는 one to one mapping function이 불가능하고, 결국 학습된 CLIP space 또한 이러한 manifold 간의 차이가 있다고 말할 수 있다.   
이런 inconsistency를 줄이기 위해서 제시된 방법 중 하나가 바로 ImageNet zero-shot classification에서 사용된 template이고, 물론 지금 task는 해당 dataset 기반이 아니지만 여전히 text prompt augmentation 관점에서는 유용하기 때문에 사용하였다. 사용하는 방법은 다음과 같다.


```python
imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    ...
    ...
]
```

위와 같이 중괄호 안에 들어갈 text prompt 부분을 남겨둔 채로 약 $80$개의 prompt engineering template를 사용한다. 일종의 앙상블 혹은 정규화라고 보면 되는데, 각 embedding을 평균내는 것으로 embedding space 상에서의 한 점을 mapping하는 것이 아닌, probability distribution을 mapping하는 효과를 줄 수 있다.

##### Channelwise relevance

그리고 사실 앞서 말했던 사항은 $\Delta i$와 $\Delta t$의 관계이고 실상 우리가 접근해야하는 style code $s$를 어떤 식으로 manipulation해야하는지 언급하지 않았는데, 보통 style code를 channel 별로 구분해서 원하는 attribute를 변화하고자 할 때 다음과 같은 메커니즘을 사용할 수 있다. 만약 각 <U>style code의 coordinate</U> $c$를 증가시키거나 감소하여 생성한 이미지에 대해 <U>CLIP image manifold</U> $\mathcal{I}$에서의 변화를 $\Delta i_c$라고 하자. 만약 이미지가 coordinate 상관없이 $\Delta i$라는 방향으로 변화했을 때, <U>style coordinate $c$에 대한 relevance</U>($R_c$)는 $\Delta i_c$를 $\Delta i$에 projection한 것과 같다.

\[
    R_c (\Delta i) = \mathbb{E}\_{s \in \mathcal{S}} \left( \Delta i\_c \cdot \Delta i \right)    
\]

실제로는 100개의 image pair를 사용하여 평균 변화량에 대해 예측하도록 하였다. 우선 각 image pair는 $G(s \pm \alpha \Delta s_c)$로 설정되었으며, 여기서 $s_c$는 $c$ coordinate를 제외하고는 모두 $0$인 벡터이고, $c$ coordinate은 해당 channel의 standard deviation으로 설정한다. 각 채널에 따른 relevance $R_c$를 계산했을 때 만약 relevance가 threshold인 $\beta$보다 작아진다면 해당 채널을 무시한다. Threshold parameter는 <U>entanglement를 얼마나 허용할 지</U>에 대한 지표가 된다.

\[
    \Delta s = \begin{cases}
        \Delta i_c \cdot \Delta i, & \text{if }\vert \Delta i_c \cdot \Delta i \vert \ge \beta \newline
        0, & \text{otherwise}
    \end{cases}    
\]

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/215396601-781f1bdf-d647-4ed8-bc39-532e271cb8e0.png" width="1200"/>
</p>

---

# 결론

이 논문의 main contribution이라고 하면 stylespace를 적절하게 사용하여 CLIP embedding을 어떤 방식으로 image manipulation에 사용할 지 다양한 방법을 적용했다는 점과 text embedding을 직접 활용하는 것이 아니라 CLIP space에서의 유사도를 활용하여 style transfer를 진행했다는 점이 될 수 있다. 그러나 결국 <U>hyperparameter에 취약</U>하여 각 sample마다 clip에 대한 <U>attribute change가 안정적으로 일어나지 않을 수 있다는 점</U>과, 서로 유의미한 관계에 있는 object가 아니라면(호랑이와 사자) <U>image manipulation이 어렵다는 점</U>을 한계점으로 생각해볼 수 있다.   
특히 StyleGAN이 학습된 baseline인 얼굴 이미지와 비슷한 형태의 modality에는 style mixing이 자유로운데, 그에 반해 CLIP space는 보다 broad하고 diversity가 보장된 embedding representation을 활용할 수 있다. 그냥 본인 생각을 소신있게 풀어보자면, StyleGAN baseline이 style mapping이라는 관점에서 image manipulation 연구에 최적화가 되어있지만, 어찌보면 그 때문에 다양한 연구나 많은 paper가 나오지 못하는 원인일 수도 있겠다고 생각해본다.

---

# StyleGAN-NADA

StyleCLIP과 접근법은 비슷하지만, latent manipulation으로 접근했던 것과 다르게 layer finetuning으로 접근한 [styleGAN-NADA](https://arxiv.org/pdf/2108.00946.pdf) 방식도 있다. 결국 이 논문에서 해결하고자 하는 task도 image generator를 이미지 manifold 상의 특정 target domain으로 mapping하는 과정에서 <U>text prompt만 가지고 guidance(supervision)</U>을 줄 수 있다면, 굳이 style에 대한 image를 supervision으로 사용하지 않고도 domain에 한정된 image manipulation에서 벗어나 <U>다양한 형태의 스타일링이 가능</U>하다는 것이다. 아래는 실제로 StyleGAN-NADA 방법을 통해 다양한 스타일로 변환된 결과를 보여준다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/215648852-17c8a3ef-4706-4bd5-ac3b-63b845322d30.png" width="500"/>
    <img src="https://user-images.githubusercontent.com/79881119/215648859-836d6a4d-0823-4d12-8d9e-7e1778baf484.png" width="500"/>
</p>

**StyleCLIP**이 가졌던 단점 중 하나는 $\mathcal{W}+$ space, $\mathcal{S}$ space 모두 결국 사전 학습된 StyleGAN의 domain 내에서 image manipulation이 진행되기 때문에 in-domain이라는 문제가 해결될 수 없다는 점이었다. 그러나 해당 논문이 접근했던 방식과 더불어 CLIP의 text guiding이 쉽게 적용된 것은 아니었다. 자칫 잘못된 방법으로 최적화를 진행하게 되면 adversarial solution(실제로 현실적인 image manifold에서 벗어나 artifact가 많이 생기는 현상)으로 **target domain이 학습될 수 있기 때문**이다. 뒤에서도 추가로 계속 설명하겠지만 최적화에 사용된 loss 형태는 StyleCLIP에서 사용한 방식과 유사하게 CLIP embedding space에서 **text direction**과 **image direction**을 맞춰주는 것이다(나란히).

---

# Related works

그동안 text guided image synthesis 관점에서 다양한 연구들이 진행되어왔다. 사실 CLIP이라는 논문은 학습 과정에서 image와 text의 관계에 대해 초점을 맞춘 것 뿐이지만, 이 연구는 text prompt를 활용한 image synthesis나 manipulation task에도 다양하게 활용될 수 있었다. StyleCLIP 방법에서도 확인할 수 있지만, CLIP을 사용하여 StyleGAN과 같은 사전 학습 모델에 대한 최적화 관점으로 접근하는 방식이 대부분 latent optimization 방법이고, <U>특정 이미지를 생성하는 latent code를 찾고자 하는 것</U>이 주된 목적으로 작용했다. 하지만 StyleGAN-NADA에서는 이런 방법들 대신 <U>image generator 자체를 text prompt guidance를 통해 최적화</U>하여, 한정된 domain에서 벗어난 image manipulation이 가능하게 하였다.

그리고 무엇보다 text-guided synthesis와 관련되어 찾아볼 task는 한정된 데이터를 기반으로 generator를 학습시키는 연구들이다.  일종의 few-shot learning을 generator에 적용하고자 했던 연구들은 적은 수의 데이터가 generator를 overfitting시키거나 mode-collapse(샘플의 다양성이 떨어지는 현상)를 일으킬 수 있는 문제가 있었기 때문에 augmentation 방법을 사용하거나 auxiliary task를 사용하여 representation을 보다 풍부하게 학습하고자 하였다. Style-NADA에서는 이러한 데이터의 부족함 때문에 생기는 overfitting이나 mode-collapse를 걱정할 필요 없이, 어떠한 데이터도 사용하지 않고 단순히 CLIP based text prompt만 guidance로 사용하게 된다.

추가로 이 논문을 이해하기 위해 StyleGAN, StyleCLIP에 대한 이해가 필요하지만 이 부분은 이미 작성한 내용이므로 넘어가도록 하겠다. StyleCLIP 관련 내용은 위에서 확인해볼 수 있고, StyleGAN 관련 내용은 게시글로 포스팅하였다([참고 링크](https://junia3.github.io/blog/gan2)).

---

# CLIP based guidance

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/215648862-2005c3e6-3ee6-4692-b814-eee2b7aff28b.png" width="500"/>
</p>

결론적으로 논문에서 사용한 방식을 간단한 그림으로 나타낸 것이 위의 figure이다. Source domain에 대해 사전 학습된 generator $G$를 copy한 뒤, 하나는 <U>frozen</U>하여 계속 source domain의 이미지를 만들게 하고 다른 하나는 <U>fine tuning</U>하여 target domain을 만들게 하되, 이때의 guidance를 CLIP loss로 주는 방식이다.

물론 저자들은 StyleCLIP에서 겪었던 시행착오와 동일하게 Global loss부터 시작하여 왜 directional CLIP loss가 중요한지 설명하였다. 사실 해당 부분이 **CLIP based style transfer, image manipulation 논문**에서 가장 중요한 점이라고 생각되어 디테일하게 짚고 넘어가고 싶었다.

### Global loss

가장 쉽게 생각할 수 있는 것은 generator가 생성한 이미지와 target이 되는 text prompt 사이의 CLIP loss를 최적화하는 방식이다.

\[
    \mathcal{L}\_\text{global} = D_\text{CLIP} (G(w), t_\text{target})
\]

Latent code $w$가 주어졌을 때, image generator $G$에 의해 생성된 이미지를 image encoder $E_I$에 통과시켜 나온 임베딩과, target text prompt를 text encoder $E_T$에 통과시켜 나온 임베딩 사이의 코사인 유사도를 계산한다. StyleCLIP과 다른 점은 위의 loss에서 최적화하고자 하는 parameter가 $w$가 아닌 $G$의 parameter라는 점이다. 아무튼 가장 간단하게 생각할 수 있는 방법이지만, 위의 방법을 사용하게 되면 <U>adversarial solution</U>으로 최적화가 되는 문제가 발생한다. Generator parameter가 고정되지 않고 학습이 되는 상황에서 Wassertein loss나 adversarial loss와 같이 real image manifold를 유지해줄 수 있는 방법이 없기 때문에 이러한 문제가 발생한다. 또다른 문제는 이러한 loss term은 <U>mode collapse</U>를 일으킨다는 것인데, 바로 다음과 같은 그림을 보면 이해가 쉽다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/215648868-84893a52-5139-413f-8a42-45c5777f728f.png" width="500"/>
</p>

예를 들어 **고양이 이미지**가 target domain에 해당되고 **강아지 이미지**가 source domain에 해당된다고 하자. 둘을 각각 CLIP space(image embedding) 상에 올려놓은 것이 붉은색 점들과 푸른색 점들이다. 그리고 **‘고양이’라는 text prompt**에 대한 CLIP space(text embedding)상의 점과 **‘강아지’라는 text prompt**에 대한 CLIP space(text embedding)상의 점을 각각 표시한 것이 청록색/보라색 점에 해당된다.

단순히 생성된 이미지가 text prompt에 부합하고자 한다면 학습은 (b)에서 보는 바와 같이 진행된다. 모든 강아지 이미지가 단순히 text prompt에 가장 가까운 image embedding을 만들게끔 하는 것이 minimizer가 되기 때문에, 붉은색 점들이 그리는 분포와 같이 <U>다양한 고양이 이미지를 생성하지 못하고</U> 녹색 분포와 같이 협소한 공간에서 이미지를 생성하게 된다. 이를 mode-collapse라고 부른다.

### Directional CLIP loss

위와 같은 문제가 발생할 수 있기 때문에 StyleCLIP에서 사용한 방법과 같이 global direction approach를 사용하게 된다. Global direction approach란, 이미지를 target text prompt에 가까워지도록 움직이는 것이 아니라 source text에서 target text로의 방향만 알려주고, 이를 source image 벡터에 더하게 되면 방향에 대한 정보만 더해줄 수 있기 때문에 샘플의 다양성은 유지할 수 있다는 것이다. 

\[
    \begin{aligned}
        \Delta T =& E_T(t_\text{target}) - E_T (t_\text{source}) \newline
        \Delta I =& E_I (G_\text{train}(w)) - E_I (G_\text{frozen}(w)) \newline
        \mathcal{L}_\text{direction} =& 1-\frac{\Delta I \cdot \Delta T}{\vert \Delta I \vert \vert \Delta T \vert}
    \end{aligned}
\]

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/215648873-6a890147-2e22-4e6a-a66e-629cb218cb7f.png" width="400"/>
    <img src="https://user-images.githubusercontent.com/79881119/215648877-d6a9ac55-27b5-4fa5-ab6d-2cf4085ff31a.png" width="500"/>
</p>

이 방법에 대해 잘 나타낸 것이 위에 보이는 그림이다. 이전과는 다르게 방향을 일치시켜주는 것만으로도 충분히 target domain에 유사한 이미지를 생성할 수 있는 것을 확인할 수 있다.

---

# Layer freezing

Domain shift가 texture를 바꾸는 방식(예를 들어, 실제 사진을 그림처럼 바꾸는 과정)의 경우 같은 training scheme을 적용하더라도 mode collapse가 발생하거나 overfitting이 발생했다고 한다. 기존의 few-shot domain adaptation 방법들에서 network weights의 일부를 제한하는 형태를 통해 synthesized result의 퀄리티를 높였는데, 저자들은 이 방법이 zero-shot인 이 task에서도 적용될 수 있을 것이라 생각했다. <U>파라미터 수를 줄이고 최적화하는 방법</U>을 통해 간접적으로 학습되는 network의 크기를 줄일 수 있고, 적은 데이터셋에 대해 overfitting을 방지할 수 있는 <U>정규화 작업</U>이 된다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/215648878-55b6b464-864e-46b3-bde7-8dbda2792575.png" width="500"/>
</p>

그렇다면 학습할 레이어는 어떻게 고를 수 있을까? 

이를 설명하기 위해서 저자들은 **본인들이 아이디어를 빌드업한 과정**을 차례대로 설명해준다. StyleGAN을 읽어봤다면 알겠지만 각 위치에 들어가는 style code가 서로 다른 semantic attribute에 영향을 끼친다. 예를 들어 위의 그림에서 $w_1$, $w_2$는 상대적으로 coarse feature를 담당하는 latent code가 될 것이고, 그와 반대로 $w_l$에 가까워질수록 fine feature를 담당하는 latent code가 될 것이다. Image manipulation 논문에서 좋은 효과를 보였던 $\mathcal{W}+$도 어찌보면 각 layer 층에서 style에 영향을 미치는 style code를 서로 다르게 사용하였기 때문이라고 할 수 있다. 이처럼 특정 스타일의 이미지 혹은 도메인의 이미지를 만들기 위해서는 여러 개의 $w_i \in \mathcal{W}$ 중 가장 도메인 변화에 큰 영향을 끼치는 layer를 위주로 fine-tuning하는 것이 효과적이고, high quality 이미지를 만들 수 있는 방법이다.

우선 $k$개의 layer를 골라야하기 때문에 랜덤한 갯수 $N$개의 latent code를 $M(z) = w$의 형태로 추출해내고, 이를 $\mathcal{W}+$로 각 레이어에 대한 style code로서 replicate하게 된다. 이 내용은 본인 게시글 중 image manipulation 관련 글을 보면 보다 이해가 쉬울 것이다([참고 링크](https://junia3.github.io/blog/imagemanipulate)). 그런 뒤 $i$번의 iteration을 통해 StlyeCLIP의 latent-code optimization을 진행하고(여기서는 학습되는 parameter가 generator가 아니라 latent code라고 보면 된다), 각 latent code의 변화를 측정한다. 그렇게 가장 많이 변화한 $w$ code를 기준으로 $k$개의 layer를 고르고, 이를 학습에 사용한다.

---

# Latent-Mapper mining

앞서 말했던 방법들을 통해 생성되는 이미지의 정규화를 효과적으로 진행했음을 알 수 있다. 그러나 이러한 방법들은 generator로 하여금 완전히 target domain을 생성하게끔 학습하지 못하게 했는데, 예를 들어 dog를 cat으로 바꾸는 task에서, fine-tuning을 거친 네트워크가 강아지와 고양이 모두 생성하는 네트워크가 되거나, 혹은 강아지랑 고양이 그 사이의 애매한 이미지를 만들게 된다는 것이다. 이러한 문제들을 피하기 위해 StyleCLIP의 latent-mapping 방식을 같이 사용하여, latent code를 cat과 관련된 region으로 옮기는 작업을 추가했다고 한다.

---

# 결론

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/215648880-5fcf3a4d-9426-46d1-ab6b-9fc4d76902c0.png" width="1000"/>
</p>

최적화 과정에 있어서 해당 논문은 <U>image embedding, text embedding</U> 모두 guidance로 사용할 수 있고, 같은 최적화 방법이 두 guidance 모두에서 잘 작용했다고 한다. 이 논문이 가지는 contribution은 StyleCLIP과는 다르게 StyleGAN의 generator 부분의 parameter를 효과적으로 fine-tuning하는 방식을 사용했고, 이를 통해 기존에 할 수 없었던 out of domain style 적용이 효과적으로 이루어질 수 있음을 보여주었다. 다만 최적화 과정에서 latent code를 배제했을 때 domain이 섞이는 현상을 해결하기 위해 latent mapping 방식을 같이 사용한다던지, 학습할 layer를 mining하는 과정에 시간이 걸린다는 점이 문제가 될 수 있다. 그럼에도 불구하고 **CLIP을 사용하여 기존 style mixing 방식과는 다르게 접근**했다는 점이 인상깊었다.
