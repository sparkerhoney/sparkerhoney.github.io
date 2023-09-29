---
title: ControlNet 논문 이해하기 및 사용해보기
layout: post
description: Controllable stable diffusion
use_math: true
post-image: https://user-images.githubusercontent.com/79881119/235468213-09aaeaec-4cd4-4503-8e67-6b279557a6d5.png
category: paper review
tags:
- Diffusion model
- Generative model
- Controllable diffusion
- AI
- Deep learning
---

# 들어가며
ControlNet의 논문 제목 풀네임은 'Adding conditional control to text-to-image diffusion models'이다. 이른바 <U>ControlNet</U>이라고 불리는 이번 연구는 사전 학습된 large diffusion model을 어떻게 하면 **input condition**에 맞게 <U>효율적인 knowledge transfer</U>이 가능할지에 대해 논의한 페이퍼이다.  Diffusion model이라는 말이 들어갔지만 기존에 리뷰했던 디퓨전 베이스 페이퍼와는 완전히 다른 방향의 연구에 해당된다. 오히려 최근 LLM(Large Language Model)을 파라미터 효율적으로 학습하는 연구 방향인 Parameter efficient fine tuning과 연결짓는 편이 더 합리적이다. 실제로 코드를 받아서 실험해보았을 때 저자들이 제시한 ControlNet 구조를 학습시키는 과정은 서버용 GPU가 아닌 <U>개인 GPU로도 충분히 학습 가능</U>하며, 가장 눈에 띄는 장점은 ControlNet은 어떠한 input condition에 대해서도 학습이 가능하다는 점이다. 방법론으로 들어가게 되면 ControlNet의 가장 메인 포인트라고 할 수 있는 ‘zero convolution’이 등장하는데, 과연 어떠한 방식으로 input condition을 자유롭게 조정할 수 있게 되었는지 차근차근 살펴보도록 하자.

---

# Input condition in diffusion models

Input condition을 diffusion model에 주는 방식은 사실 이미 존재했었다. 아직 본인 블로그에서는 요즘 가장 핫한 stable diffusion의 근간이 되는 연구인 [latent diffusion 논문](https://arxiv.org/abs/2112.10752)을 따로 다루지는 않았지만 간단하게 소개하자면, 

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/235462245-c6711e21-c0b8-435f-80e9-09fea31ea502.png" width="700">
</p>

예컨데 이미지를 <U>유의미한 semantic 정보만 유지</U>하고 이미지 생성에 크게 필요하지 않은 high frequency feature를 거르는 vector quantized encoder/decoder를 학습한 상태로 생각하자(즉, 이미지 $x$를 작은 크기의 resolution을 가지는 latent image로 축소한다고 생각하면 된다). 이렇게 축소된 latent를 diffusion process를 통해 복구하는 과정을 학습하는 것이 우리가 일반적으로 이해하고 있는 **DDPM** 혹은 **DDIM**의 학습 및 샘플링 프로세스이다.

우리가 기존에 살펴본 내용 중에서 attention pooling에 시간 정보와 class label 정보를 projection embedding으로 넣어주는 방법론이 있었다([diffusion process conditioning 논문 리뷰글](https://junia3.github.io/blog/diffusionpapers)). 이를 확장시켜 생각하면, 만약 특정 목적을 가지고 condition을 임베딩으로 사영시킬 수 있는 task specific encoder $\tau_\theta$만 있다면, 각 디퓨전 모델 학습 시에 $\tau_\theta$를 통해 추출된 condition vector를 attention layer를 통해 조건화해줄 수 있다. 예컨데 만약 다음과 같은 이미지와 텍스트 description 쌍이 있다고 생각하면(출처 : BLIP 논문),

> Description : The car is driving past a small old building
> 

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/235462200-a917f64d-6bd3-4c05-9f05-29ef14451152.png" width="400">
</p>

CLIP의 text encoder와 같은 <U>임의의 텍스트 인코더</U>를 통해 추출한 embedding을 이미지 생성 시(reverse process)에 조건부로 넣어주게 되면 해당 디퓨전 모델은 샘플링 과정에서 prior에 prompt 조건만 추가해주게 되면 text to image task를 수행할 수 있게 되는 것이다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/235462194-52d7d549-92c8-4f76-a8eb-a31384addbbd.png" width="800">
</p>

단순히 prompt를 통해 위와 같은 고퀄리티의 이미지를 만들 뿐만 아니라, <U>다양한 모달리티에 대한 학습된 encoder</U>만 있다면 attention pooling 조건화를 통해 diffusion process를 학습시킬 수 있다.

# Naive conditioning의 단점

이러한 방법들이 가지는 문제점은 상당히 명확하다.

첫번째로는 diffusion model이 특정 condition에 맞게 학습되려면 그만큼 score network가 <U>해당 condition을 이미지 생성에 잘 반영</U>해야하는데, 이를 달성하기 위한 **학습 데이터**가 상당히 <U>많이 필요하다는 것</U>이다. 예컨데 Vision-Language(VL) task는 멀티모달에서 활발히 연구가 되었기 때문에 CLIP, ALIGN과 같은 대량의 데이터셋이 구축되었지만 다른 모달리티(pose to image, semantic to image 등등)은 그렇지가 않다는 것이다. 실제로 LAION-5B와 같이 stable diffusion의 학습 base가 된 데이터셋에 비해서 object shape나 pose 같이 특정 목적성을 가진 데이터셋은 여러 가지 한계점 때문에 대량으로 구축하기 힘들기 때문이다. 대략 <U>수만배 정도 차이</U>가 난다.

두번째로, 이미지 생성이나 manipulation 같은 processing 과정이 대량의 데이터를 통해 솔루션을 획득하는 과정은 굉장히 리소스가 많이 든다는 점이다. 첫번째 문제였던 데이터 갯수의 차이를 극복하더라도 <U>사전 학습된 네트워크를 학습하는 것은 장벽</U>으로 작용하게 된다.

마지막으로 processing 과정은 problem 정의에 있어 그 형태의 boundary를 예측할 수 없을 정도로 다양하고, 더욱이 발전할 수 있다. 즉 한계가 없는 문제를 해결하는데 있어 greedy한 선택만 취하게 된다면(디퓨전 프로세스를 제한하거나 attention activation을 바꾸는 것) 이는 결국 고차원의 이해가 필요한 작업들(depth, pose 등등)에는 최적화가 힘들다는 것을 의미한다.  말이 조금 복잡하게 표현된 것 같은데 이를 latent diffusion의 방법론을 통해 다시 한 번 언급하자면, latent diffusion process는 사전 학습된 task specific encoder의 embedding output에 conditioning을 의존하게 되므로(embedding을 단순히 샘플링 부분에 넣어주는 과정을 통해 constraints를 줌) 보다 다양한 task에 대한 학습 과정에서 최적의 선택이 아닐 수 밖에 없다는 것이다. 고로 <U>end-to-end 학습을 할 수 있는 방법을 강구</U>해야한다. 아래의 그림과 같이 기존 방식은 conditioning part와 실제 디퓨전 모델 학습이 end-to-end가 아닌 분리된 형태를 가진다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/235462202-abcf9761-fbf6-41cd-b966-3d012c7433e9.png" width="1000">
</p>

---

# ControlNet, end-to-end neural network

따라서 논문이 문제로 삼은 기존 conditioning의 한계점을 극복하기 위해 저자는 새로운 <U>transfer learning 구조를 제안</U>하였다.  ControlNet을 간단하게 묘사하면 다음과 같다.

- diffusion model의 parameter를 복사하여 새로운 학습 프레임워크를 원래 parameter와 병렬로 구성한다. 이를 각각 “trainable(학습 가능한)  copy”와 “locked(학습 불가능한) copy”라고 부른다.
- Locked copy는 기존 network의 성능인 이미지 생성에 필요한 representation을 유지하고 있다고 생각할 수 있다.
- Trainable copy는 conditional control을 위해 여러 task-specific dataset에 대해 학습되는 프레임워크다.
- Locked copy와 Trainable copy는 zero convolution을 통해 서로 연결된다. Zero convolution 또한 학습 가능한 레이어에 속한다.

대충만 쭉 묘사했는데 사실 이 부분은 그림을 보면 이해가 쉽다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/235462204-48bf3f2f-bc94-423d-a3f7-a6f4a900382a.png" width="700">
</p>

$x$가 들어가서 $y$가 나오는 구조는 diffusion process에 접목시키게 되면 특정 시점의 noised latent vector $z_{t}$가 input으로 들어가서 다음 시점의 noised latent vector $z_{t-1}$를 예측하는 것과 같다. 회색으로 된 neural network는 원래의 diffusion model로 파라미터가 고정된 채 변하지 않게끔 하면 사전 학습된 디퓨전 모델의 <U>이미지를 만드는 성능을 해치지 않고</U> 가만히 놔둘 수 있다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/235462207-4ecbc531-2eae-4da1-a9be-b2dcfc6cca9c.png" width="500">
</p>

좌측의 얼어있는 친구는 가만 놔두고 우측의 불타는 친구만 condition에 대해 학습한다고 생각하면 된다. Trainable copy이므로 fine-tuning 과정인데 원래의 parameter를 최대한 손상시키기 않겠다는 의도가 보이는 학습 구조가 된다.

---

# Method

그렇다면 구체적으로 어떻게 해당 학습이 효과적으로 conditioning을 할 수 있는지 수식적으로 살펴보도록 하자. 예컨데 conditioning을 하는 neural network block은 흔히 우리가 알고있는 resnet에서의 bottleneck block이나 transformer의 multi-head attention block을 생각하면 된다.

2D(이미지와 같은 형태)의 feature를 예시로 들어보자. 만약 feature map $x \in \mathbb{R}^{h \times w \times c}$가 정의되어 있다면, neural network block $\mathcal{F}_\Theta(\cdot)$는 블록에 포함되는 parameter $\Theta$를 통해 input feature map $x$를 transform하게 된다.

\[
y = \mathcal{F}_\Theta(x)
\]

 바로 이 과정이 앞서 그림에서 봤던 (a)에 해당된다. 이제부터 해당 parameter $\Theta$는 잠궈놓을 것이다(학습하지 않을 것). 그리고 이를 똑같이 복사한 trainable parameter $\Theta_c$는 잠궈놓은 친구와는 다르게 input condition $c$를 input으로 받아 학습에 사용될 것이다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/235462208-12e86232-15ef-4f0f-b15d-f4d66b69869a.png" width="500">
</p>

참고로 더해지는 부분에 대해서는 네트워크가 <U>activation을 저장해놓을 필요가 없기 때문에</U> 학습 시에 메모리를 $2$배로 가질 필요성도 없어진다. Backpropagation을 통해 계산된 gradient는 학습 가능한 모델에 대해서만 optimization을 진행할 것이기 때문이다.

### Zero convolution

이때 더해질 때 바로바로 이 논문에서 가장 중요한 녀석인 zero convolution이라는 개념이 사용되는데, 각 neural block의 앞/뒤로 하나씩 붙는다고 생각하면 된다. 앞/뒤에 붙는 녀석들을 각각 $\mathcal{Z}\_{\Theta_1}(\cdot), \mathcal{Z}\_{\Theta_2}(\cdot)$라고 해보자. 물론 zero-convolution은 feature map의 크기를 변화시키면 안되기 때문에 $1\times 1$ 크기를 가지는 convolution이며 weight와 bias 모두 zero로 초기화된 상태로 학습이 시작된다.

위의 그림대로 원래의 output $y$에 conditioning 함수를 거친 output을 더하면 다음과 같다.

\[
y_c = \mathcal{F}\_\Theta(x) + \mathcal{Z}\_{\Theta_2}(\mathcal{F}\_{\Theta_c}(x + \mathcal{Z}\_{\Theta_2}(c)))
\]

여기에서 대체 왜 weight 및 bias가 $0$으로 초기화된 ‘Zero convolution’이 사용되었는지 이유가 등장한다. Zero-convolution은 weight 및 bias가 모두 $0$이므로, input에 상관없이 처음엔 모두 $0$을 output으로 내뱉는다.

\[
\begin{cases}
\mathcal{Z}\_{\Theta_1}(c) = 0 \newline
\mathcal{F}\_{\Theta_c}(x+\mathcal{Z}\_{\Theta_1}(c)) = \mathcal{F}\_{\Theta_c}(x) = \mathcal{F}\_{\Theta}(x) \newline
\mathcal{Z}\_{\Theta_2}(\mathcal{F}\_{\Theta_c}(x + \mathcal{Z}\_{\Theta_2}(c))) = \mathcal{Z}\_{\Theta_2}(\mathcal{F}\_{\Theta_c}(x)) = 0
\end{cases}
\]

즉 처음에는 $y_c = y$로 시작하게 된다. 해당 내용이 암시하는 것은 training이 시작되는 당시에는 ControlNet 구조에 의한 input/output 관계가 사전 학습된 diffusion의 input/output과 전혀 차이가 없다는 것이고, 이로 인해 optimization이 진행되기 전까지는 neural network 깊이가 증가함에 따라 영향을 끼치지 않는다는 것을 알 수 있다.

### Gradient flow in zero convolution

$1 \times 1$ convolution 구조를 가지는 zero convolution에 대한 연산 과정에 local gradient를 유도할 수 있다. 예컨데 input feature map $I \in \mathbb{R}^{h \times w \times c}$가 있을때 forward pass는

\[
\mathcal{Z}(I,; \\{W, B\\})\_{p, i} = B_i + \sum_{j}^c I_{p, i}W_{i, j}
\]

이처럼 표현되고, zero convolution은 최적화 전까지는 $W = 0, B = 0$이기 때문에 $I_{p, i}$가 $0$이 아닌 모든 point에 대해서

\[
\begin{cases}
\frac{\partial \mathcal{Z}(I; \\{W, B\\})\_{p, i}}{\partial B_i} = 1\newline
\frac{\partial \mathcal{Z}(I; \\{W, B\\})\_{p, i}}{\partial I_{p, i}} = \sum_{j}^cW_{i,j} = 0 \newline
\frac{\partial \mathcal{Z}(I; \\{W, B\\})\_{p, i}}{\partial W_{i, j}} = I_{p, i} \neq 0
\end{cases}
\]

위와 같이 정리된다. Input에 대한 gradient는 $0$으로 만들지만 weight나 bias에 대한 gradient는 $0$이 아니기 때문에 학습이 가능하다. 왜냐하면 first step만 지나게 되면 Hadamard product 기호인 $\odot$에 대해

\[
W^\ast = W-\beta_\text{lr} \cdot \frac{\partial \mathcal{L}}{\partial \mathcal{Z}(I; \\{W, B\\})} \odot \frac{\partial \mathcal{Z}(I; \{W, B\})}{\partial W} \neq 0
\]

$0$이 아닌 weight를 만들기 때문에 바로 다음 step에서는

\[
\frac{\partial \mathcal{Z}(I; \\{W^\ast, B\\})\_{p, i}}{\partial I\_{p, i}} = \sum\_{j}^cW^\ast_{i,j} \neq 0
\]

학습이 잘된다.

---

# Stable diffusion + ControlNet

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/235462212-cafb40f1-f222-4560-9491-52d370dd512f.png" width="">
</p>

위에서 설명한 구조를 기존 stable diffusion에 구현한 구조는 위와 같다. Loss는 기존 diffusion algorithm에 task specific condition $c_f$만 추가된 형태가 된다.

\[
\mathcal{L} = \mathbb{E}\_{z_0, t, c_t, c_f, \epsilon \sim \mathcal{N}(0, 1)}\left( \parallel \epsilon - \epsilon_\theta(z\_t, t, c\_t, c\_f) \parallel_2^2 \right)
\]

---

# 결과

### Canny Edge

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/235462216-b7291a87-32f6-4ac2-983f-b7193936fd35.png" width="">
</p>

### Hough Line

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/235462217-4d045efa-eddb-4450-9ec2-92a3ad0cb070.png" width="">
</p>

### Scribble

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/235462218-4986d394-2c30-4e23-b3fd-18f65c34bcb9.png" width="">
</p>

### HED edge

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/235462222-f9f9d56d-ee9f-4e1c-af07-9fb29cd10880.png" width="">
</p>

### Pose

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/235462224-f1f5b4d8-579f-473e-b033-aad1b955a387.png" width="">
</p>

### Segmentation

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/235462226-5209cf47-c377-400e-add5-07738a63644b.png" width="">
</p>

### Depth

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/235462230-9f60b1a4-d1d7-41c3-a5d8-a3b74f07af57.png" width="">
</p>

### Cartoon line drawing

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/235462232-8dbc6b0e-ad70-41f6-aa4b-fb2d91c40f60.png" width="">
</p>

---

# Official Code로 직접 실행해보기

현재 official code는 [깃허브 소스](https://github.com/lllyasviel/ControlNet.git)로 제공되고 있다. 엥간하면 로컬 서버에서 돌아가기는 하는데 안정적으로 돌릴라면 서버에서 돌리는게 좋다. 여기다가 실행법은 올리겠지만 원본 페이지에 들어가서 ⭐ 한번씩 눌러주면 좋을 것 같다. 다음 repository를 클론 후

```bash
git clone https://github.com/lllyasviel/ControlNet.git
```

Conda 가상 환경을 설치해준다.

```bash
cd ControlNet
conda env create -f environment.yaml
conda activate control
```

그런 뒤 사용하고자 하는 모델과 stable diffusion을 [Hugging Face Page](https://huggingface.co/lllyasviel/ControlNet)로부터 다운받으면 된다. 다운받는 위치는 ControlNet/models에 stable diffusion ckpt를 넣고 detector를 ControlNet/annotator/ckpts에 넣으면 된다.

### Detector(모두) 다운받는 코드(ControlNet 레포지에서 실행)

```bash
cd ./annotator/ckpts
curl -LO https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/body_pose_model.pth
curl -LO https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/dpt_hybrid-midas-501f0c75.pt
curl -LO https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/hand_pose_model.pth
curl -LO https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/mlsd_large_512_fp32.pth
curl -LO https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/mlsd_tiny_512_fp32.pth
curl -LO https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/network-bsds500.pth
curl -LO https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/upernet_global_small.pth
```

굳이 다 다운받고 싶지 않으면 원하는 파일에 대한 curl만 실행하면 된다.

### Models(모두) 다운받는 코드(ControlNet 레포지에서 실행)

```bash
cd ./models
curl -LO https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_canny.pth
curl -LO https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_depth.pth
curl -LO https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_hed.pth
curl -LO https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_mlsd.pth
curl -LO https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_normal.pth
curl -LO https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_openpose.pth
curl -LO https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_scribble.pth
curl -LO https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_seg.pth
```

마찬가지로 굳이 다 다운받고 싶지 않으면 원하는 파일에 대한 curl만 실행하면 된다.

### 데모 버전 API 실행하기

원하는 모델을 실행하는 코드는 간단하게

```bash
python gradio_어쩌구2어쩌구.py
```

를 실행하면 되는데, 만약 서버컴에서 이걸 실행하고 로컬에서 접속하고 싶다면 코드를 살짝만 바꿔주면 된다. 예컨데 모든 `gradio_어쩌구2저쩌구.py` 파일 코드를 보게 되면 가장 마지막 줄에

```python
block.launch(server_name='0.0.0.0')
```

요 친구가 있는데 이걸 다음과 같이 바꿔주면 된다.

```python
block.launch(server_name='0.0.0.0', share=True)
```

본인은 대충 

```python
python gradio_scribble2image_interactive.py
```

이걸 실행해보겠다. 제대로 실행되면 다음처럼 나온다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/235462235-7dd62696-6f57-4ae8-a0ee-640a2b1cf4e8.png" width="">
</p>

대강 public URL은 72시간 동안 유효하다는 뜻, 본인은 연세 vpn으로 서버컴에 접속한 상태지만 노트북으로 들어가보겠다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/235462236-0d8e2563-80b2-45a8-9b61-46aa33e51c64.png" width="900">
</p>

다음과 같은 화면이 뜬다. 실제로 잘 되는지 확인해보자.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/235462237-2678713e-f6ad-441a-aed9-0debfa373a47.png" width="700">
</p>

비루한 그림실력.. 힘내라 ControlNet

Run 버튼을 누르자 DDIM sampler가 동작하기 시작한다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/235462239-0268dd2a-2e93-45fa-aff1-f9261b635f62.png" width="">
</p>

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/235462241-f63bdb2f-ddd6-437f-9aa0-e27094bbe81c.png" width="">
</p>

그림을 못그려도 인생 살기 큰 문제 없다는 긍정적인 희망이 생기는 논문이었다... 암튼 이렇게 하면 된다. 넉넉잡아 10기가 이상의 GPU면 다 돌아가는 듯하다.