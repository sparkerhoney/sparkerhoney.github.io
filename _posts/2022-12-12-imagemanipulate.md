---
title: GAN을 활용한 이미지 조작(image to image 그리고 GAN inversion까지)
layout: post
description: Image translation/manipulation
post-image: https://user-images.githubusercontent.com/79881119/235353241-4dfff373-a961-4d00-966a-46f8356692dc.png
category: paper review
use_math: true
tags:
- AI
- deep learning
- generative model
- adversarial training
---

Generative model인 GAN은 여러 방면에서 활용될 수 있다.   
- 대표적인 Image synthesis(합성)의 경우 Texture synthesis([PSGAN](https://arxiv.org/abs/1909.06956), [TextureGAN](https://openaccess.thecvf.com/content_cvpr_2018/papers/Xian_TextureGAN_Controlling_Deep_CVPR_2018_paper.pdf), [Texture Mixer](https://arxiv.org/abs/1901.03447))이 있으며,   
- Image super resolution(화질 높이는 것)([ProGAN](https://arxiv.org/abs/1710.10196), [Progressive face super-resolution](https://arxiv.org/abs/1908.08239), [BigGANs](https://arxiv.org/abs/1809.11096), [StyleGAN](https://arxiv.org/abs/1812.04948))이 있다.
- Image impainting이라는 task는 미완성된 그림이나 사진을 완성하는 작업으로, [Deepfillv1](https://arxiv.org/abs/1412.7062), [ExGANs](https://arxiv.org/abs/2009.08454), [Deepfillv2](https://arxiv.org/abs/1806.03589), [Edgeconnet](https://arxiv.org/abs/1901.00212), [PEN-Net](https://arxiv.org/abs/1904.07475)이 있다.
- Face image synthesis는 얼굴 이미지 합성과 관련된 task로, [ELEGANT](https://openaccess.thecvf.com/content_ECCV_2018/papers/Taihong_Xiao_ELEGANT_Exchanging_Latent_ECCV_2018_paper.pdf), [STGAN](https://arxiv.org/abs/1904.09709), [SCGAN](https://arxiv.org/abs/2011.11377), [Example guided image synthesis](https://arxiv.org/abs/1911.12362), [SGGAN](https://ieeexplore.ieee.org/document/8756542), [MaskGAN](https://arxiv.org/abs/1907.11922)
- Human image synthesis로는 사람의 포즈나 전체적인 윤곽을 생성하는 task가 된다. [Text guided](https://arxiv.org/abs/1904.05118), [Progressive pose attension](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhu_Progressive_Pose_Attention_Transfer_for_Person_Image_Generation_CVPR_2019_paper.pdf), [Coordinate-based](https://openaccess.thecvf.com/content_CVPR_2019/papers/Grigorev_Coordinate-Based_Texture_Inpainting_for_Pose-Guided_Human_Image_Generation_CVPR_2019_paper.pdf), [Semantic parsing](https://arxiv.org/abs/1904.03379)

하지만 오늘 살펴볼 내용은 이것과는 다르게 이미지를 바꾸는 작업, 즉 image manipulation과 관련된 것들을 볼 예정이다. 오늘 게시글과 관련된 내용들을 언급해보자면
1. Image to image translation([CycleGAN](https://arxiv.org/abs/1703.10593), [MUNIT](https://arxiv.org/abs/1804.04732), [DRIT](https://arxiv.org/abs/1808.00948), [TransGaGa](https://arxiv.org/abs/1904.09571), [RelGAN](https://openreview.net/pdf?id=rJedV3R5tm))
2. Image editing([SC-FEGAN](https://openaccess.thecvf.com/content_ICCV_2019/papers/Jo_SC-FEGAN_Face_Editing_Generative_Adversarial_Network_With_Users_Sketch_and_ICCV_2019_paper.pdf), [FE-GAN](https://ieeexplore.ieee.org/document/9055004), [Mask-guided](https://arxiv.org/abs/1905.10346), [FaceShapeGene](https://arxiv.org/abs/1905.01920))
3. Cartoon generation([CartoonGAN](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.pdf), [PI-REC](https://arxiv.org/abs/1903.10146), [Internal Representation Collaging](https://arxiv.org/abs/1811.10153), [U-GAT-IT](https://arxiv.org/abs/1907.10830), [Landmark Assisted CycleGAN](https://arxiv.org/abs/1907.01424))

---

# Image translation
위에서 많은 task를 언급했던 것은 GAN으로 이만큼이나 많이 할 수 있다는 걸 보여줄라고 한 것이고, 사실 이 게시글의 주 목적은 단순히 image manipulation과 관련된 초기 논문 아이디어에서 insight를 얻어보기 위함이다.   
Image-to-image translation이라 함은 input image로부터 output 이미지를 생성하는 task가 되며, 이때 output과 input은 서로 어떠한 관계에 놓이게 된다.   
이를 테면 computer vision이나 machine learning task에서 주로 나오는 semantic labeling이나 boundary detection이 될 수도 있고,
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209128127-87d17af5-7a5c-4286-88b0-892aacb811df.png" width="400"/>
    <img src="https://user-images.githubusercontent.com/79881119/209128133-59673788-fece-4176-bab6-2485f6049932.png" width="400"/>
</p>
Computer graphics나 computational photography에서 다루는 image colorization, super-resolution이 될 수도 있다.
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209128137-a759fc58-54ff-4881-af7c-579c27016684.png" width="400"/>
    <img src="https://user-images.githubusercontent.com/79881119/209128139-4ab35598-1900-41eb-bac2-c097da070b6d.png" width="400"/>
</p>

즉, 해당 task의 supervision은 source로부터 target을 만드는 과정이 되며, generator $G$는 source domain $S$의 이미지를 사용하여 target image $T$를 만들게끔 학습된다.   
어떠한 문제가 되던, 위와 같은 task는 다음과 같은 objective로 귀결된다.
- Objective function $\mathcal{L}$ 설정
- Training data $(x, y)$ 설정
- Network $G$ 학습하기
- Image translation $G(S) = T$ 정의하기

\[
    \arg \min_{\mathcal{F}} \mathbb{E}_{x,y}(\mathcal{L}(G(x), y))    
\]

그러나 기존의 GAN 방식을 바로 적용하기에는 문제가 있는데, 이는 바로 <U>image generation의 mode(modality)를 제어할 수 없다는 것</U>이다. 즉 우리는 아무런 이미지만 만들면 되는 게 아니라 input으로 넣은 이미지의 translation 버전을 원하는데, 이걸 generator가 인지할 수 없다는 것이 첫번째 문제다. 다음은 generator로 생성한 이미지의 low-resolution 문제가 있다.   
이러한 task를 다룬 총 세 개의 대표적인 paper가 바로 pix2pix, cycleGAN 그리고 pix2pixHD이다. 그 중 가장 유명한 논문인 pix2pix와 cycleGAN에 대해서 먼저 살펴보도록 하자.

---

# pix2pix: Image-to-Image Translation with Conditional Adversarial Nets
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209128143-1afe1947-10ea-4004-a5a1-2e6034374d6d.png" width="600"/>
</p>
pix2pix는 대표적인 image to image translation을 GAN으로 해결한 연구이다. 가장 유명한 figure인 sketch to real image에 대한 framework는 위와 같다.   
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209128145-2aa6c562-4816-434c-9417-047d39df9ac2.png" width="600"/>
</p>
만약, generator가 단순히 sketch image를 가지고 real image를 만들어내는 task에 대해서 생각해보면, 위의 그림과 같이 '그럴듯한 가방'을 만들어내는 것도 가능하지만 그럴 경우 실제 sketch와의 correspondence 문제까지 고려해야한다. 즉, 위쪽 row에 대해서는 sketch 부분에 잘 맞게끔 이미지가 생성되지만, 아래쪽의 row에서는 sketch는 거의 무시한 채 이미지를 생성해낸다.   
이러한 문제를 기존 GAN loss에서는 고려할 수 없었으며(아래쪽 식을 참고),
\[
    \begin{aligned}
        &\min_G \max_D V(D,G) \newline
        V(D,G) =& \mathbb{E_{x \sim p_{data}(x)}}(\log D(x)) + \mathbb{E_{z \sim p_z(z)}}(\log (1-D(G(z))))
    \end{aligned}    
\]
이는 generator가 만든 이미지에 대해 loss를 적용할 때 단순히 real distribution에 의한 결과인지 cross-entropy로 구분했기 때문이다. 물론, 이 식을 그대로 적용하지는 않고 input으로 sketch image를 주기 때문에 데이터셋 $(x, y)$를 적용한 GAN loss를 고려해보면,

\[
    \begin{aligned}
        &\min_G \max_D V(D,G) \newline
        V(D,G) =& \mathbb{E_{x \sim p_{data}(x)}}(\log D(y)) + \mathbb{E_{z \sim p_z(z)}}(\log (1-D(G(x, z))))
    \end{aligned}    
\]

Generator에 input $x$가 latent vector $z$와 함께 주어지는 구조인 걸 확인할 수 있다. 그러나 이러한 식은 실질적으로 discriminator가 generator에게 줄 수 있는 학습 정보는 "진짜같은" 이미지인지에 대한 loss 뿐이므로 correspondence를 해결할 수 없다는 문제가 생긴다.   
그래서 제시된 objective function이 conditional GAN의 방법을 이용한 loss이며, 여기에 추가적으로 MAE(Mean absolute error)를 생성된 이미지와 정답(GT) 이미지 사이에 줌으로써 low resolution result 문제와 correspondence 문제 모두 해결하려 했다.

\[
    \begin{aligned}
        &\arg \min_G \max_D \mathcal{L} (G,D) + \lambda \mathcal{L}(G) \newline
        \mathcal{L}(G, D) =& \mathbb{E}(\log D(x, y))+\mathbb{E}(\log(1-D(x, G(x, z)))) \newline
        \mathcal{L}(G) =& \mathbb{E}(\parallel y-G(x, z) \parallel_1)
    \end{aligned}  
\]

위에 표현된 식에서 $\mathcal{L}(G, D)$는 conditional GAN loss에 해당되며, 앞서 설명했던 것과 같이 discriminator에 input 정보를 함께 줌으로써 생성된 이미지가 입력된 이미지에 대한 조건을 가지게끔 해준다. 그러나 해당 loss는 content가 유지된다는 보장을 줄 수 없기 때문에 여기에 추가적으로 $\mathcal{L}(G)$로 표현된 식을 통해, 원본 이미지와 유사하게 생성되게끔 만들어준다. $\lambda = 100$ 정도로 크게 생각해주면 된다.
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209128147-a661f2ef-4c41-4177-ac9d-f97edfccd894.png" width="600"/>
</p>

생성자 구조는 이와 같이 [U-Net](https://arxiv.org/abs/1505.04597) 형태를 이용하였다. 또한 이 논문에서 특별한 점은 $x$에 추가적으로 latent vector $z$를 넣어주지 않고, decoder part의 dropout으로 해당 stochastic한 부분을 충당할 수 있다고 한다. 그래서 사실상 loss 식에서 표현된 $z$를 따로 넣어주지는 않는다. 추가로 넣어주더라도 별로 효과적이진 않다고 판단했다. 즉, 넣어주어도 결국 $z$에는 아무런 영향을 못받는다고 했다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209128149-7dd5735c-0f0b-4e9f-abfd-af21d2df494c.png" width="600"/>
</p>

각 loss term에 의한 효과를 보여주는 그림이다. L1 loss를 쓰지 않았을 때는 GT의 전체적인 틀을 잃어버리는 문제가 발생하고, cGAN loss를 쓰지 않았을 때는 blurry한 결과가 나오는 것을 확인할 수 있다. Discriminator 구조는 appendix에 따로 나와있는데, 간단하게만 설명하면 모든 ReLU는 LeakyReLU(기울기 0.2)를 사용하였고 [DCGAN](https://arxiv.org/abs/1511.06434)에서와 같이 첫번째 layer에서는 BatchNorm을 사용하지 않았다. 이러한 pix2pix는 두 개의 paired dataset만 있다면 한쪽을 source, 다른 쪽을 target으로 삼아서 다양한 image to image translation task에 적용될 수 있다는 장점이 있다.   
그러나 여기서 생길 수 있는 문제는, 과연 input-output 간에 paired dataset이 없다면 어떻게 될까이다. 이를테면 sketch dataset에 그에 상응하는 사물 이미지가 있어야 sketch to image generation 학습이 가능하고, scene에 대한 depth map이 존재해야 depth estimation 생성이 가능하기 때문이다. 즉 한계점은 pair를 모으기 힘들고, 몇몇 task에 대해서는 아예 불가능할 수도 있다는 것이다. 바로 이러한 관점에서 제시된 것이 [cycleGAN](https://arxiv.org/abs/1703.10593)이다.   
이를테면 사진이 있는데, 그걸 모네 화풍으로 바꾸고 싶다고 하자. 모네의 화풍에 대한 image를 구하기 위해 우리가 직접 사진을 찍을 수도 없고, 정말 그림을 그린 그 풍경이 현재도 존재한다고 가정할 수도 없다. 심지어 조경이나 날씨 등 환경도 달라지고, 동일 시간대 내에서 사진을 찍은게 아니라면 paired dataset을 구축할 수 없다. ~~그렇다고 모네를 환생시켜서 그림 그려달라고 할 수도 없는 노릇~~

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209128152-72f039eb-b155-4cb6-9885-bb78eacdf647.png" width="600"/>
</p>
이 그림을 보게 되면, pix2pix는 paired dataset에 대해서만 적용될 수 있고, 우측과 같은 unpaired dataset에서는 구현이 불가능한 것을 알 수 있다. 그렇다면 cycleGAN의 intuition을 알아보도록 하자.

---

# CycleGAN: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks
생성 과정을 간단하게 번역 과정으로 생각해보자. 영어를 한국어로 번역하고, 번역된 한국어를 다시 영어로 번역하면 원래의 문장이 나와야한다. 바로 이것이 cycleGAN의 주된 메커니즘이며, 여기서 번역기 기능을 하는 것이 generator라고 생각하면 된다.
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209128154-d19c06b6-27cf-44a6-939c-5d59d9e055aa.png" width="600"/>
</p>
예를 들어 말 이미지에서 얼룩말 이미지로 바꾸는 task에 대해서 설명하면, 말을 얼룩말로 바꾼 이미지를 다시 말 이미지로 되돌렸을 때 원래의 말 이미지가 나와야 한다는 것이다. 이러한 방법을 통해 unpaired dataset을 가지고 있더라도 원래의 컨텐츠를 유지하면서 이미지를 생성할 수 있게 된다는 것이다.
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209128157-9abd55c3-9674-4c03-8e01-2b219a7019c5.png" width="1000"/>
</p>
이 위에 나타난 그림이 되게 중요한데, cycleGAN의 프레임워크를 이 그림만 보면 모두 이해할 수 있기 때문이다. $X$를 한쪽 도메인이라고 생각하고 $Y$를 다른쪽 도메인이라고 생각하자. 여기서 도메인이 의미하는 것은 데이터셋이 포함되는 하나의 집합이다.   
$X$ 도메인에 포함된 데이터셋 $x$와 $Y$ 도메인에 포함된 데이터셋 $y$에 대해서, 각 방향에 대한 generator를 함수로 표현할 수 있다. $X$ 도메인의 데이터셋을 받아 $Y$ 도메인의 데이터를 생성하는 네트워크를 $G$라고 하고, 이렇게 생성된 $G(x)$를 $\hat{y}$라 표현한다. 마찬가지로 $Y$ 도메인의 데이터셋을 받아 $X$ 도메인의 데이터를 생성하는 네트워크를 $F$라고 하고, 이렇게 생성된 $F(y)$를 $\hat{x}$라 표현한다. 두 네트워크에 대해 $X$ 도메인 이미지에 대해 F, G를 최적화하는 과정은 다음과 같다. 우선 $X \rightarrow Y$로 생성된 이미지에 대한 adversarial loss는 다음과 같이 표현된다. 

\[
    \mathcal{L}_{GAN}(G, D_Y, X, Y)    
\]

Adversarial loss에서 각 notation이 의미하는 바는 다음과 같다.
- $X$ 에서 $Y$ 이미지를 생성하는 forward process에 대해서,
- $Y$ 도메인의 실제 데이터와 가짜로 생성된 데이터를 비교하는 discriminator $D_Y$가 있고,
- 가짜로 데이터를 생성하는 generator $G$에 대해서 adversarially 최적화를 하겠다.

요약하자면, $D_Y$와 $G$가 서로 경쟁하면서 학습하는 구조가 된다. 이렇게 되면 문제는 $D_Y$에 대해서나 $G$에 대해서는 학습이 가능한데, 역과정에 대해서는 학습이 안된다. 그렇기 때문에 우리는 추가로 경쟁 구조를 하나 더 만들 것이다.

\[
    \mathcal{L}_{GAN}(F, D_X, Y, X)    
\]

위의 식에서 각 notation이 의미하는 바는 다음과 같다.
- $Y$ 에서 $X$ 이미지를 생성하는 reverse process에 대해서,
- $X$ 도메인의 실제 데이터와 가짜로 생성된 데이터를 비교하는 discriminator $D_X$가 있고,
- 가짜로 데이터를 생성하는 generator $F$에 대해서 adversarially 최적화를 하겠다.

요약하자면 $D_X$와 $F$가 서로 경쟁하면서 학습하는 구조가 된다. 이제 모든 domain에 대한 generator, discriminator 학습이 가능해진다.
그리고 추가적으로 여기에 앞서 언급한 cycleGAN에서의 주요 intuition인 <U>"다시 돌아왔을 때 원래와 같아야 함"</U>을 적용한 cyclic loss는 다음과 같다. 정말 간단하게도 다시 역방향 generator를 사용해서 생성된 이미지를 기준으로 원래 이미지와의 L1 loss를 계산한다.
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209128161-3a5726a6-706f-4190-a084-7686f293a540.png" width="600"/>
</p>

구현은 pix2pix에서와 동일한 generator과 discriminator를 사용했으나, cycleGAN original paper에서는 Instance Normalization을 사용했으며 modified ResNet based generator를 사용한 점이 살짝 다르다.
 <p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209128162-454b45de-740b-4f14-ad57-bc636cbe9f30.png" width="800"/>
</p>
위의 그림이 cycleGAN에서 사용된 generator 구조라고 보면 된다.
학습 알고리즘을 글로 풀어쓰면 다음과 같다.

1. $x$, $y$ 두 개의 이미지를 서로 다른 도메인 $X$, $Y$로부터 하나씩 가져온다.
2. 각각 방향에 맞는 generator를 통과시켜 fake image를 얻는다. $(x,~y) \rightarrow (G(x),~F(y)) = (\hat{y},~\hat{x})$
3. 각각 도메인에 맞는 discriminaor를 통해 discriminator loss를 계산한다. $D_X$는 $(x, x')$을, $D_Y$는 $(y, y')$을 토대로 계산한다.
4. 다시 각자의 방향으로 가는 generator를 통과시켜 fake image에 대한 reconstructed image를 얻는다. $(\hat{y},~\hat{x}) \rightarrow (F(\hat{y}),~G(\hat{x})) = (\tilde{x},~\tilde{y})$
5. 다시 reconstructed된 이미지에 대해서 cyclic L1 loss를 계산해준다. 계산은 $(x,~\tilde{x})$ 그리고 $(y,~\tilde{y})$에 대해서 진행한다.
6. Generator loss를 최종적으로 계산한다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209128166-90c08835-28d1-4694-91e3-48724217422a.png" width="700"/>
    <img src="https://user-images.githubusercontent.com/79881119/209128164-6af25b70-ff59-423a-b9bf-8a785910a622.png" width="450"/>
</p>

위의 좌측 이미지를 보면 알 수 있듯이, 해당 task는 굉장히 다양한 형태로 구현할 수 있고, 서로 연관짓고 싶은 domain에 대한 데이터만 있으면 어떠한 학습도 가능하다는 장점이 있다. 그러나 limitation으로 등장한 것은 GAN 자체가 사물에 대한 인식을 할 수 없기 때문에 얼룩말로 바꾸는 task와 같은 경우 배경에 무늬가 들어가거나, 심지어 사람이 타고 있다면 사람에도 얼룩말 무늬가 들어가는 일이 발생한다. 보통 모든 데이터셋에 말을 타고 있는 사람이 있다면 이런 일은 일어나지 않겠지만, 학습할 때 사람이 추가로 들어있는 경우가 거의 없을 경우에 inference하면 이와 같은 artifact가 발생하게 된다.

---

# GAN inversion
다음으로 볼 내용은 [GAN inversion](https://arxiv.org/pdf/2101.05278.pdf)이다. GAN을 뒤집는다라는 간단한 제목으로 소개되는데, 해당 task는 image manipultation에 있어 다음과 같이 접근한다.
 <p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209128169-d2f99af6-41d4-48e6-82e1-2f906f2c6267.png" width="700"/>
</p>
학습된 decoder(혹은 generator $G$)에 대해 $z$라는 latent space 상의 한 점은 fake image인 $x = G(z)$를 만들어낸다. 흔히 latent vector를 뽑는 과정은 Normal distribution으로부터 추출한다($z \sim \mathcal{N}(0, I)$).   
그렇다면 이러한 접근은 어떨까? Real image가 있는데, 이 real image와 비슷한 이미지를 만들어낼 수 있는 $z$를 찾는 것이다. 이를 수식으로 표현한 것이 바로
\[
    z^\* = \arg \min_z (G(z),~x)    
\]
따라서 우리는 **이상적인 $z^\*$를 찾고 싶고**, real image $x$와 거의 똑같은 이미지를 만들어낼 수 있는 $z$를 찾을 수 있으면, 이 <U>$z$를 조금씩 바꿔가면서 image editing이 가능</U>하다는 것이다. Concept은 GAN을 이해할 수 있다면 받아들일 수 있을 정도로 simple하다.   
이러한 image manipulation에서 활용되기 쉽게 여러 도메인에 대한 style을 학습시킨 pre-trained styleGAN이 있다. 다음 깃허브 링크를 참고하면 좋을 것 같다. [사전 학습된 StyleGAN 링크](https://github.com/justinpinkney/awesome-pretrained-stylegan)   

---

# Image to styleGAN
위에서 언급했던 것과 같이 사전 학습된 StyleGAN이 있다고 하자, image to style이란 GAN이 주어진 이미지를 styleGAN의 latent space로 보내는 task다. StyleGAN에 대해서는 이미 다뤄서 대충 알고 있겠지만, $\mathcal{Z}$ space에서 $\mathcal{W}$로 보내는 MLP 구조가 있다. 그런데 [Image to style](https://arxiv.org/pdf/1904.03189.pdf)에서는 이를 좀 다르게 $\mathcal{W^+}$ space로 보낸다. 기존의 $\mathcal{W}$ space에서의 벡터는 <U>똑같은 벡터를 18개 복사</U>하여 각 styleGAN layer에 넣어주는데, 이와는 다르게 $\mathcal{W^+}$ space에서는 $18 \times 512$ 크기의 벡터를 업데이트하는 것이기 때문에, 보다 세부 조정된 inversion이 가능하다. 즉, <U>18개의 서로 다른 row vector들이</U> style layer에 들어간다고 생각하면 된다.
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209128172-15bc984f-5d16-48cd-a13e-cf8e6eb26e9d.png" width="700"/>
</p>
대충 그림으로 나타낸 것이 바로 위쪽 그림이다. StyleGAN에 의해 임의로 생성되는 이미지를 사용하는게 아니라, latent를 최적화하여 그럴싸한 원본 이미지를 만드는 latent를 찾은 뒤에, 그 latent를 활용하여 image를 editing하겠다는 전략이다.   
그렇다면 $\mathcal{W^+}$를 어떤 식으로 최적화할까?

1. Initial latent code $w^\*$로부터 시작한다.
2. Image $I^\* = G(w^\*)$를 생성한다. 여기서 $G$는 사전 학습된 styleGAN generator이다.
3. $I^\*$를 원래 생성하고 싶었던 reference 이미지 $I$와 비교하고, loss function $\mathcal{L}$을 계산한다.
4. 위에서 정의한 Loss function을 토대로 처음에 guess한 latent code $w^\*$를 gradient descent 방법을 사용하여 업데이트한다.
5. 위의 과정을 몇번의 iteration을 통해 반복한다.

위의 과정을 보면 성능에 중요한 영향을 미치는 것은 딱 두 가지로 정할 수 있다. 애초에 decoder는 미리 학습된 styleGAN과 같은 generator를 사용하기 때문에 딱히 건드릴 순 없고, image 특성을 잘 반영해서 생성해줄 latent space를 잘 찾는 것과 그 latent space에 속해있는 latent vector를 찾아낼 수 있는 loss function이다.   
초기화의 경우 무작위의 latent를 샘플링하는 방법도 있으나, 이전 글에서 styleGAN에서 소개했던 바와 같이 **sampling이 분포가 희박한 곳에서 발생할 경우에 문제가 발생한다**. 이게 무슨 의미냐면, initial point에서 gradient descent가 일어나는 feasible direction이 최적화가 힘든 길이면 원하는 이미지가 잘 생성되지 않을 수도 있다. 그렇기 때문에 단순히 랜덤으로 초기화하는 것보다는 평균치를 의미하는 mean latent code(mean face)로부터 시작하는 것이 낫다는 것이다. 또 loss function에 대해서 언급하자면, 단순히 MSE를 계산하는 것은 pixel-wise error를 계산하기 때문에 high quality embedding을 찾기 힘들다. 왜냐하면 MSE loss가 가지는 값이 실제로 그 이미지를 대변할 수 없기 때문이다. 그렇기 때문에 perceptual loss를 사용하여 latent space를 잘 찾을 수 있게 보조해주는 loss를 생각해볼 수 있다.
\[
    w^\* = \min_w L_\text{percept} (G(w), I)+\frac{\lambda_{mse}}{N} \parallel G(w) - I \parallel_2^2   
\]
[Perceptual loss](https://arxiv.org/abs/1603.08155)는 pre-trained VGG-16(ImageNet에 대해 사전 학습된 네트워크)의 feature extraction 부분을 활용하며, 두 이미지 간의 hidden feature 사이의 유사도를 측정한다.
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209128173-bbb6ae26-9699-4b74-9900-cc8bf25f4108.png" width="700"/>
</p>
위의 그림이 perceptual loss가 제시된 논문에서 발췌한 그림이다. Style에 대한 정보를 VGG-16을 활용한 loss를 토대로 최적화하였고, 이를 컨셉으로 생각하게 되면 우리가 하고자 하는 이미지의 content나 style을 이해하고 최적화하는 상황에서 사용될 수 있음을 시사한다.

\[
    L_\text{percept}(I_1, I_2) = \sum_{j=1}^4 \frac{\lambda_j}{N_j} \parallel F_j (I_1) - F_j (I_2) \parallel_2^2
\]

---

# Various applications
다음부터는 여러 적용 방법들에 대해서 소개소개..

## Morphing
Morphing은 image processing technique으로, 흔히 한쪽 이미지에서 다른쪽 이미지로 점진적 변화를 할 때 사용한다. 두 개의 이미지 $I_1$ 그리고 $I_2$에 대해 각각 latent space로 embedding된 코드 $w_1$와 $w_2$를 사용, morphing은 다음과 같이 두 latent code의 convex set의 임의의 점을 가리킨다.

\[
    w = \lambda w_1 + (1-\lambda)w_2,~\lambda \in (0, 1)
\]

이렇게 구한 $w$로 morphed image $G(w)$를 구하면 된다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209128176-ffdf81a3-daa6-4137-845b-307a053eea45.png" width="700"/>
</p>
좌측 이미지와 우측 이미지의 morphed image가 $\lambda$에 따라 나타난 그림은 위와 같다.

## Expression transfer
이 개념은 사실상 latent arithmetic와 같다. 예를 들어 이런 개념이다.   
- 무표정의 고양이 사진($I_1$)이 있고, 이를 latent에 embedding한 vector $w_1$이 있다고 하자.
- 무표정의 강아지 사진($I_2$)가 있고, 이를 latent에 embedding한 vector $w_2$가 있다고 하자.
- 웃는 강아지 사진($I_3$)가 있고, 이를 latent에 embedding한 vector $w_3$가 있다고 하자.
- 그렇다면, $w = w_1 + \lambda(w_3 - w_2)$를 통해 만든 이미지는 어떤 모습일까?

사실 이쯤만 되어도 어느 정도 감이 좋은 사람이라면 **"웃는 고양이 사진"** 이 의도한 정답이라는 것을 알아채주셨을 것이다. 결국 expression transfer에서 하고자 하는 것은 latent에서의 연산이 해당 feature를 반영하는 latent space에서의 방향이라는 것.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209128177-8eba5ab9-dab9-4e4f-b516-91e97c694788.png" width="700"/>
</p>

## Style transfer
StyleGAN에서 했던 style transfer 방식과 완전 동일한데, 다만 이걸 $\mathcal{W}$ space가 아니라 $\mathcal{W^+}$ space에서 하고자 하는 것이다.

- Image $I_1$에 대해 최적화 및 임베딩된 $18 \times 512$ 크기의 latent vector $w_1$를 생각할 수 있다.
- Image $I_2$에 대해 최적화 및 임베딩된 $18 \times 512$ 크기의 latent vector $w_2$를 생각할 수 있다.
- 중간 부분까지 $w_1$를 가져다 쓰고, 그 다음부터는 $w_2$를 가져다 쓴다. 즉 새로운 latent code $w$는 $w_1$과 $w_2$의 몇개의 row를 concatenate한 구조가 된다(아래 그림 참고).

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209128180-092a4a59-8600-408f-8f84-49cbd36033eb.png" width="700"/>
</p>

그렇게 되면 다음과 같이 정말 의미 없는 두 도메인에 대해서도 style transfer가 일어날 수 있다. 단순히 얼굴 이미지에 대한 style synthesis를 제시했던 styleGAN에서 더 확장성 있는 연구 가능성을 제시해준 것과 같다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209128183-9bb9979c-9bd6-440c-a30a-dc5eb64832ae.png" width="700"/>
</p>

이후 [Image2StyleGAN++](https://arxiv.org/abs/1911.11544)를 통해 더 확장성 있는 application을 보여준다. mask based style transfer, image impainting 그리고 local edit 등등 여러 application이 제시가 되었다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209128190-242c9a82-0e6f-445a-bbcb-f75adee7d0cd.png" width="400"/>
    <img src="https://user-images.githubusercontent.com/79881119/209128193-3b0bb6ea-3799-4501-b91d-04a49314c345.png" width="400"/>
    <img src="https://user-images.githubusercontent.com/79881119/209128194-337469ac-f014-4894-9d17-d82394f7d058.png" width="400"/>
    <img src="https://user-images.githubusercontent.com/79881119/209128196-c247f424-e564-424a-81bd-9c15601f7bcb.png" width="400"/>
</p>