---
title: Better speech synthesis through scaling
layout: post
description: Paper review
use_math: true
post-image: https://th.bing.com/th/id/OIG.rZM9TiRbekMeMmrI0F9f?pid=ImgGn
category: statistics
tags:
  - Data
  - Science
  - Statistics
---
# Better speech synthesis through scaling
## Abstract

이미지 생성 분야에서의 **autoregressive transformers**와 **DDPMs**의 적용에 영감을 받아, 이 연구에서는 이미지 생성 분야의 발전을 음성 합성에 적용하는 방법을 설명합니다.<br>
결과로 "**TorToise**"라는 다양한 목소리의 *Text-To-Speech이 제안*되었습니다.<br>

- **코드 및 가중치:** [GitHub 링크](https://github.com/neonbjb/tortoise-tts)<br>

## Background
### Text-To-Speech

TTS 연구는 주로 효율적인 모델의 개발에 초점을 맞추고 있으며, 이는 상대적으로 **작은 데이터셋**에서 훈련되었습니다.<br>

- *이러한 선택의 원인*:<br>
	1. **대규모로 배포할 수 있는 효율적인 음성 생성 모델**을 구축하고자 하는 욕구(desire)<br>
	2. 매우 큰 규모의 변환을하는 음성 데이터셋의 부재<br>
	3. TTS에서 전통적으로 사용되는 Encoder-Decoder 모델 아키텍처의 **확장**<br>

#### Neural MEL Inverters

**현대의 TTS 시스템**은 **MEL 스펙트로그램(melspectrogram)** 으로 인코딩된 음성 데이터를 기반으로 작동합니다.<br>

![Understanding the Mel Spectrogram | by Leland Roberts | Analytics Vidhya |  Medium](https://miro.medium.com/v2/resize:fit:1400/1*zX-rizZKXXg7Ju-entot9g.png)<br>

- *Mel 스펙트로그램(melspectrogram)*<br>
	- MEL 스펙트로그램은 음성 신호나 오디오 신호를 시각적으로 표현하는 방법 중 하나입니다. <br>
	- 주파수 영역에서의 신호의 진폭 또는 에너지를 시간에 따라 표시합니다. <br>
	- MEL 스펙트로그램의 특징은 다음과 같습니다:<br>

		- **MEL 스케일**: MEL 스케일은 인간의 귀가 주파수를 인식하는 방식을 모방한 것입니다. <br>
			- 인간의 귀는 낮은 주파수에서는 주파수 변화를 더 잘 인식하지만, 높은 주파수에서는 그렇지 않습니다.<br> MEL 스케일은 이러한 인간의 청각 특성을 반영하여 주파수를 비선형적으로 변환합니다.<br>
    
		- **시간 vs 주파수**: MEL 스펙트로그램은 수평축으로 시간을, 수직축으로 MEL 스케일로 변환된 주파수를 나타냅니다. <br>
			- 색상 또는 밝기는 해당 시간과 주파수에서의 신호의 강도를 나타냅니다.<br>
    
		- **응용 분야**: MEL 스펙트로그램은 음성 인식, 음성 합성, 음악 분석 등 다양한 오디오 처리 작업에서 널리 사용됩니다.<br>
<br>
- **MEL 스펙트로그램의 장점**: 이러한 인코딩 방식을 사용하는 주된 이유는 MEL 스펙트로그램이 공간적으로 **매우 압축**되어 있기 때문입니다. <br>
	- 예를 들어, [Tacotron](https://github.com/NVIDIA/tacotron2)에서 사용되는 MEL 구성은 22kHz에서 샘플링된 원시 오디오 파형 데이터에 대해 256배의 압축률로 작동하지만, 그 데이터에 포함된 대부분의 정보를 포함하고 있습니다.<br>
<br>
- **MEL 스펙트로그램 디코딩 연구**: *MEL 스펙트로그램을 오디오 파형으로 다시 디코딩하는 고품질의 방법*을 찾기 위한 연구가 많이 이루어졌습니다. <br>
	- 이 작업을 수행하는 합성기는 일반적으로 "**vocoder**"라고 불리지만, 이 논문에서는 이를 "**MEL inverter**"라고 일반적으로 언급합니다.<br>
<br>
- **신경망 기반의 현대 MEL Inverters**: 신경망을 기반으로 한 현대의 MEL inverter는 매우 정교합니다. <br>
	- 이것들은 사람의 귀로는 녹음된 파형과 거의 구별할 수 없는 파형을 생성하며, 그들의 훈련 세트 외부에서도 매우 일반화될 수 있습니다. <br>
	- 이 연구를 활용하여, [Univnet(Kim, 2021)의 구현](https://github.com/maum-ai/univnet)을 **TTS 시스템의 마지막 단계**로 사용합니다.<br>
  
---

## Image generation

TTS 시스템은 주로 **지연 시간(latency)** 에 중점을 두지만, 다른 분야에서는 그렇지 않았습니다. <br>

	➡️ 사용자는 application에서 즉시응답을 바라기 때문에 최대한 rapidly generation이 전통적으로 중요했습니다.
<br>
**However**, image generation에서는 샘플링 시간에 관계없이 **고품질 결과를 생성하는 모델을 훈련**하는 데 더 많은 집중이 이루어졌습니다. <br>

- **지연시간(latency)**<br>
	- TTS(Text-to-Speech) 시스템은 텍스트를 음**성으로 변환하는 기술**입니다. <br>
		- **사용자가 입력한 텍스트**를 실시간으로 사람처럼 **발음하는 음성으로 바꾸는 것이 주 목적**입니다.<br>
  <br>
	- "**지연 시간**" 또는 "**latency**"는 시스템이 어떤 입력을 받았을 때 그에 대한 **반응이나 출력이 나타나기까지의 시간을 의미**합니다. <br>
		- **For example**, TTS 시스템에서 **사용자가 텍스트를 입력**하면 그 **텍스트를 음성으로 변환하는 데 걸리는 시간**이 바로 ==지연 시간==입니다.<br>
<br>
	- TTS 시스템에서는 이 지연 시간이 매우 중요합니다. <br>
		- **because**, *사용자는 텍스트를 입력한 직후 빠르게 음성 응답을 기대하기 때문*입니다. <br>
		- 특히 실시간 응용 프로그램(예: 음성 도우미, 네비게이션 시스템 등)에서는 빠른 응답 시간이 필수적입니다.<br>
<br>
	- **So**, TTS 시스템은 "지연 시간(latency)"에 중점을 둡니다. <br>
		- 이는 시스템이 사용자의 요구에 신속하게 응답하도록 설계되었음을 의미합니다. <br>
		- 이를 통해 사용자 경험이 향상되며, TTS 기술이 다양한 실시간 응용 분야에서 활용될 수 있게 됩니다.<br>
<br>
이 논문에서는 `image generation`의 몇가지 연구 분야를 살펴보았습니다.<br>
<br>

---

##### **DALL-E**

[DALL-E는 OpenAI](https://openai.com/dall-e-2)에서 개발한 모델로, 텍스트 설명을 바탕으로 **이미지를 생성**하는 능력을 가지고 있습니다.<br>
이 모델은 *Autoregressive Decoder*를 사용하여 **텍스트를 이미지로 변환**합니다.<br>

- **자동 회귀(Autoregressive)** : 주로 시계열 데이터에서 *다음 데이터 포인트*를 *현재 및 이전 데이터 포인트*를 기반으로 예측하는 방법을 의미합니다.<br>
	- 자동 회귀 모델은 주로 시계열 예측에 사용되지만, 딥러닝 및 **자연어 처리 분야**에서도 널리 사용됩니다.<br>
  <br>
- **자동 회귀 디코더(Autoregressive Decoder)** : **especially**, *시퀀스 생성 작업*에서 사용되는 모델 구조 중 하나입니다. <br>
	- 이 구조는 주어진 **입력 시퀀스**를 기반으로 **출력 시퀀스를 한 단계씩 생성**합니다. <br>
	- 각 단계에서 모델은 지금까지 생성된 시퀀스를 사용하여 **다음 토큰을 예측**합니다.<br>
<br>

>All of the existing RNN based network uses **Autoregressive Decoding**<br>
>**For Example**,  Neural machine translation condition each output word on previously generated outputs.<br>
>![](https://miro.medium.com/v2/resize:fit:700/1*8-GyLPpF4yBiuZ73g7xn8w.png)<br>

<br>

- **DALL-E의 문제점**:
    
1. **Full-Sequence Self-Attention** : **DALL-E**는 전체 시퀀스에 대한 *Self-Attention Mechanism*을 사용합니다. <br>
- 이는 계산과 **메모리 측면에서 $O(N^2)$의 비용을 발생**시킵니다. <br>
- 여기서 N은 시퀀스의 길이를 나타냅니다. <br>
-   이미지나 오디오와 같은 큰 시퀀스를 처리할 때 이는 큰 문제가 될 수 있습니다.<br>
<br>

2. **discrete domain에서의 작동**: **traditional autoregressive**방식은 discrete domain에서 작동합니다. <br>
- 이미지는 **양자화 오토인코더**를 사용하여 discrete token 시퀀스로 인코딩됩니다. <br>
- **Discrete domain**: 이산 도메인은 값들이 연속적이지 않고 분리된 상태로 존재하는 도메인을 의미합니다.<br> 예를 들어, 정수 집합 {1, 2, 3, ...}는 이산 도메인에 속합니다.<br>
- **Continuous domain**: 연속 도메인은 값들이 연속적으로 존재하는 도메인을 의미합니다.<br> 예를 들어, 실수 집합은 연속 도메인에 속합니다.<br>
- DALL-E는 이 token 시퀀스를 모델링합니다. <br>
- 이 방식은 이미지의 표현력 측면에서는 강점이지만, 이 토큰을 **실제 이미지 픽셀 값으로 다시 변환하는 디코더가 필요**하게 됩니다. <br>
- DALL-E에서 사용된 VQVAE 디코더는 대부분의 샘플에서 **흐릿하고 일관성 없는 결과**를 보이는 주요 원인으로 여겨집니다.<br>

> **양자화 오토인코더(Quantizing Autoencoder)**<br>
> ![Understanding VQ-VAE (DALL-E Explained Pt. 1)](https://images.velog.io/images/p2yeong/post/c7d4bab1-875c-49d2-9fa1-e59e2410e888/Screen_Shot_2020-06-28_at_4.26.40_PM.png)<br>

##### **DDPMs (Denoising Diffusion Probabilistic Models)**

기존의 generative model의 문제점인 **Mean-Seeking Behavior** 문제, **Model-Collapse** 문제를 극복하고 *일관되고 다양한 이미지*를 생성하려고 노력한 모델입니다.<br>

- **Mean-Seeking Behavior** : 모델은 **데이터의 평균** 또는 **중앙값에 가까운 출력만을 생성**하려는 경향을 말합니다.<br>
	- **In other words**, 모델은 데이터의 *다양성을 충분히 포착하지 못하고, 대신 데이터의 평균적인 특성만을 반영하는 출력을 생성*<br>
	- **For example**, 얼굴 이미지를 생성하는 모델이 있다고 가정해보겠습니다. <br>
		- 만약 이 모델이 Mean-Seeking Behavior를 보인다면, **생성된 얼굴 이미지는 모두 비슷하게 보일 것**입니다. <br>
		- 이는 모델이 훈련 데이터의 다양한 얼굴 특성을 충분히 학습하지 못하고, 대신 평균적인 얼굴 특성만을 반영하여 이미지를 생성하기 때문입니다.<br>
<br>

- **Model-Collapse** : *GAN*의 학습 과정에서 발생할 수 있는 문제로, 생성기가 항상 **매우 유사한 출력**만을 생성하는 현상을 의미합니다.<br>
	- **In other words** : 생성기는 데이터의 **다양한 모드(특성)를 포착하지 못하고**, 대신 한정된 몇 가지 패턴만을 반복적으로 생성하게 됩니다.<br>
	- **For example**, 다양한 종류의 동물 이미지를 생성하는 GANs 모델이 있다고 가정해보겠습니다. <br>
		- 만약 이 모델이 Mode-Collapse 문제에 직면한다면, 생성기는 **오직 개나 고양이와 같은 특정 동물의 이미지만을 생성**하게 될 것입니다.<br>

DDPMs는 low-quality guidance signals를 사용하여 그 signals가 파생된 **고차원 공간(highdimensional space)을 재구성**하는 데 매우 효과적인 특징이 있습니다.<br>
여기서 "**low-quality guidance signals**"란, 원본 데이터의 일부만을 포함하거나 *손상된 데이터*를 의미합니다. <br>

DDPMs는 이러한 부정확하거나 불완전한 정보를 바탕으로 원본 데이터의 정확한 분포를 재구성할 수 있습니다.<br>
다르게 표현하면, DDPMs는 **초고해상도(super-resolution) 작업**에 매우 뛰어나다고 할 수 있습니다.<br>

이 모델의 핵심 아이디어는 **원본 데이터의 분포**를 시뮬레이션하는 데에 **노이즈를 점진적으로 추가하고 제거**하는 과정을 사용하는 것입니다.<br>

![논문 리뷰 Denoising Diffusion Probabilistic Model(2020)](https://velog.velcdn.com/images/dongdori/post/a7c18bf8-7d30-4853-a1ea-da7dde57b1c6/image.png)<br>

**DDPMs의 주요 제한사항**:<br>

1. **고정된 출력 형태**: 전통적인 DDPMs 접근법은 sampling이 시작되기 전에 알려진 **고정된 출력 형태**에 의존합니다. <br>
	- 이 논문과 관련된 구체적인 예로, DDPMs는 텍스트와 오디오 사이의 암시적인 정렬 문제를 해결할 수 없기 때문에 **텍스트를 오디오 신호로 변환하는 방법**을 학습할 수 없습니다.<br>
	- **For Example**, "*안녕하세요*"라는 짧은 문장과 "*오늘은 좋은 날씨입니다*"라는 긴 문장이 있을 때, 이 두 문장에 해당하는 *오디오의 길이는 서로 다를 것*입니다. <br>
	- DDPMs는 이러한 텍스트와 오디오의 길이나 순서를 자동으로 맞추는 능력이 없습니다.<br>
<br>

2. **다중 반복 샘플링**: DDPMs는 **여러 번의 반복(multiple iterations)** 을 통해 sampling되어야 합니다. <br>
	- **Because**, 이는 DDPMs의 특성상 원본 데이터의 분포를 **점진적으로 모방**하기 위해 **노이즈를 추가**하고 **제거하는 과정**을 여러 차례 반복(*iterations*)하기 때문입니다.<br>
	- 이 샘플링 과정은 많은 계산을 필요로 하며, DDPM에서 샘플링을 할 때 항상 상당한 지연 시간 비용이 발생한다는 것을 의미합니다.<br>
	- **Finally**, DDPMs를 사용하여 데이터를 생성할 때는 "**지연 시간(latency)**"이라는 *비용이 발생*합니다. <br>

##### **Re-ranking**

**DALL-E**는 자동 회귀 모델(autoregressive models)의 출력을 "**재순위 지정(re-ranking)**"하는 과정을 도입했습니다. <br>
이 과정은 자동 회귀 모델에서 무작위로 샘플을 추출하고, 그 중에서 **가장 품질이 높은 출력을 선택**하여 **후속 작업에 사용**합니다.<br>
	이러한 절차는 **강력한 판별기(discriminator)를 필요**로 합니다. <br>
	판별기란, 좋은 텍스트/이미지 조합과 나쁜 조합을 구별할 수 있는 모델을 의미합니다.<br>

DALL-E는 **[CLIP](https://github.com/openai/CLIP)** 모델을 이용했습니다. <br>
CLIP은 텍스트와 이미지 조합을 **대조적(contrastive)으로 학습하는 목표를 가진 모델**입니다. <br>
이 모델은 텍스트와 이미지 사이의 관계를 학습하여, 텍스트 설명이 주어졌을 때 해당 설명에 가장 잘 맞는 이미지를 선택하거나, 반대로 이미지가 주어졌을 때 그 이미지를 가장 잘 설명하는 텍스트를 선택하는 데 사용됩니다.<br>

![](https://blog.kakaocdn.net/dn/cv1qbn/btrJEK6rY69/FfGWU0mMExD9VYjkHvLiuk/img.png)<br>

