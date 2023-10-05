---
title: Transformer (Attention Is All You Need)
layout: post
description: Paper review
use_math: true
post-image: https://miro.medium.com/v2/resize:fit:1030/1*tb9TT-mwFn1WPzkkbjoMCQ.png
category: nlp, paper review
tags:
  - Data
  - Science
  - nlp
  - 자연어처리
  - transformer
  - llm
  - deeplearning
  - paper
---

# Transformer (Attention Is All You Need)
## INTRODUCTION
![Transformer (Attention Is All You Need) 구현하기 (1/3)](https://paul-hyun.github.io/assets/2019-12-19/transformer-model-architecture.png)<br>

**Transformer** 모델로부터 현재 많은 **모델**들이 생겨나고 있습니다.<br>
GPT4, LLaMa와 같은 모델 모두 Transformer의 architecture를 조금 더 발전시켜서 만들어진 모델 이므로 Transformer가 얼마나 대단한 모델인지 알 수 있습니다.<br>
해당 [paper](https://arxiv.org/abs/1706.03762)는 2017년에 발표된 논문입니다.<br>
Transformer는 온전히 **attention mechanism**에만 기반한 구조입니다.<br> (recurrence 나 convolution은 사용하지 않음)<br> 
⇢ 더 **parallelizable**하고, 훨씬 적은 학습 시간이 걸린 특징이 있습니다.<br>
### Why Transformer got SOTA
1. **Recurrent model(RNN 류)** :<br>
   parallelization이 불가능해 longer sequence length에서 **치명적이라는 단점**이 있었습니다.<br>
   최근 연구에서 factorization trick과 conditional computation을 통해 계산 효율성을 많이 개선되고 특히 conditional computation은 모델 성능도 동시에 개선되었다고 하지만,<br>
   여전히 근본적인 **sequential computation의 문제**는 남아있었습니다.<br>
2. **Attention mechanism** :<br>
    다양한 분야의 sequence modeling과 transduction model에서 주요하게 다뤄졌습니다.<br>
    Attention mechanism은 input과 output sequence간 길이를 신경쓰지 않아도 되는 mechanism이지만<br>
    여전히 recurrent network와 함께 사용되었습니다.<br> 
3. **Transformer** :<br>
   input과 output간 **global dependency**를 뽑아내기 위해 recurrence를 사용하지 않고, attention mechanism만을 사용했습니다.<br>
   **parallelization이 가능해** 적은 시간으로 translation quality에서 **SOTA**를 달성할 수 있게 되었습니다.

---
## Model of Transformer
![image](https://github.com/sparkerhoney/NLP-Paper-Implementation/assets/108461006/fdf424d3-405f-423e-bb69-e73b7d9c5cf3)<br>

**Transformer**는 **input 문장**을 넣어서 **output 문장**을 생성해내는 모델입니다.<br>
- input과 동일한 문장을 만들수도, input의 역방향 문장을 만들수도, 번역을 할 수도 있습니다.(다른 언어로 변환)<br>
    - 모델의 학습과정에서 정해짐으로, **labelling**을 어떤 방식으로 하냐에 따라 모델이 달라질 수 있습니다.<br>
- finally, Transformer는 문장형태의 input이 들어가 문장형태의 output이 나오는 모델입니다.<br>
  
> 예를들어서, "안녕하세요"라는 문장이 들어갔을 때 "안녕하세요"도 나올 수 있고, "요세하녕안"이라는 말도 나올 수 있고, "HELLO"도 나올 수 있습니다.<br>

### Full Model Architecture
![image](https://github.com/sparkerhoney/NLP-Paper-Implementation/assets/108461006/ceb61d5f-0c66-4f03-aa99-00f8bfd16c6a)<br>

- Transformer는 Decoder와 Encoder로 구분
- Encoder는 좌측, Decoder는 우측

### Encoder of Transformer
![image](https://github.com/sparkerhoney/NLP-Paper-Implementation/assets/108461006/eff623b7-01f4-469e-aae2-0723b1ad3eac)<br>

**Encoder**는 문장을 input으로 받아 **context(문맥) vector를 생성**해내는 함수입니다.(이러한 과정을 Encoding이라 함)<br>
Encoder는 context를 제대로 생성(문장의 정보를 빠뜨리지 않고 압축)해내는 것을 목표로 학습합니다.<br>

### Decoder of Transformer
![image](https://github.com/sparkerhoney/NLP-Paper-Implementation/assets/108461006/d1baed6a-d8e6-46c1-8c32-ef8f2a0cafb4)<br>

Decoder는 Encoder와 반대 방향입니다.(context를 input으로 받아서 sentence 생성 <- Decoding)<br>
이 때, Decoder는 Decoder에서 output으로 생성하는 문장을 **Right Shift**한 문장으로 input 됩니다.<br>

- **Right Shift**<br>
  >  "*Right shift*"는 자연어 처리와 관련된 용어로서, 특히 Transformer 모델에서 사용되는 용어입니다.<br> Transformer 모델의 Decoder에서 "right shift"는 주로 학습 및 예측 과정에서 사용됩니다. <br>
    1. **Right Shift의 의미**:<br>
       - "Right shift"는 시퀀스의 모든 요소를 오른쪽으로 한 칸씩 이동시키는 것을 의미합니다.<br>
       - 예를 들어, 시퀀스 `[a, b, c, d]`가 있을 때, 이를 right shift하면 `[PAD, a, b, c]`와 같이 변환됩니다.<br>
       - 여기서 `PAD`는 **패딩 토큰**을 의미합니다.<br>

    2. **Transformer Decoder에서의 사용**:<br>
       - Transformer 모델의 학습 과정에서, Decoder는 이전 타임 스텝의 출력을 현재 타임 스텝의 입력으로 사용합니다.<br>
       - 그러나 실제 학습 데이터에서는 **이전 타임 스텝**의 출력을 알 수 없으므로, 실제 타깃 시퀀스를 right shift하여 **Decoder의 입력으로 사용**합니다.<br>
       - 이렇게 함으로써, Decoder는 각 타임 스텝에서 **다음 토큰을 예측**하는 방법을 학습하게 됩니다.<br>
       - 예를 들어, "안녕하세요"라는 문장을 학습할 때, 실제 타깃 시퀀스는 `[안, 녕, 하, 세, 요]`일 것입니다.<br>
       - 이를 right shift하면 `[PAD, 안, 녕, 하, 세]`가 되며, 이 시퀀스를 Decoder의 입력으로 사용하여 **각 타임 스텝에서 다음 토큰을 예측**하게 됩니다.<br>

    3. **왜 Right Shift를 사용하는가?**:<br>
       - Right shift를 사용함으로써, 모델은 각 타임 스텝에서 이전 타임 스텝의 출력을 기반으로 다음 토큰을 예측하는 방법을 학습하게 됩니다.<br>
       - 이는 실제 번역, 텍스트 생성 등의 작업에서 중요한 역할을 합니다.

### Encoder & Decoder in Transformer
![image](https://github.com/sparkerhoney/NLP-Paper-Implementation/assets/108461006/2eaa30bb-eeab-4cc7-9a30-da089fb183a8)<br>

- 대략적인 코드 세팅
```python
class Tramsformer(nn.Module):
    def __init__(self, encoder, decoder):
        super(Tramsformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x):
        out = self.encoder(x)
        return out

    def decode(self, c, z):
        out = self.decode(c, z)
        return out

    def forward(self, x, z):
        c = self.encode(x)
        y = self.decode(c, z)
        return y
```
# Encoder

전반적인 model의 architecture에 대해서 파악해보았습니다.<br>
이제, **Encoder**가 어떤 알고리즘을 진행이 되는지 조금 더 자세히 살펴볼겁니다.<br>

![image](https://github.com/sparkerhoney/NLP-Paper-Implementation/assets/108461006/6f1e8768-134c-40b0-b3fd-50098041c9ad)<br>

## Encoder Block

Encoder는 **Encoder Block**이 $N$개 쌓여진 형태로 구성이 되어 있는데, 논문에서는 $N = 6$을 사용하였습니다.<br>
Encoder Block에서 어떤 matrix를 input으로 받는다면 **동일한 shape**의 output을 생성해냅니다.<br>
첫번째 Encoder Block의 input은 전체 Encoder의 input으로 들어오는 문장을 embedding 시켜줍니다.<br>

첫번째 Block에서부터 두번째, 세번째 Block으로 계속해서 input이 들어가고 output이 생성되고 또 다시 input이 들어가는 형태로써 **sequential**하게 연결이 되어있습니다.<br>
마지막 Block에서 output으로 나오는 전제 Encoder의 output은 이전에 설명했듯 **context**가 됩니다.<br>
이 때, Encoder의 output 또한 input으로 들어간 matrix와 동일한 shape로 구성되어야한다는 점을 주의해야합니다.<br>

### Why are there "N" Encoder Blocks?

각 Encoder Block은 input으로 들어오는 vector에 대해서 더 높은 차원(넓은 관점)에서의 context(즉, 더욱 추상적인 정보)를 담기 때문입니다.<br>
Encoder **Block은** 내부적으로 어떤 Mechanism을 활용해서 context를 담게 되는데, 이때 겹겹이 쌓아 input의 context, context의 context, ... 로서 더 높은 차원의 context가 됩니다.(처음 Encoder Block에서 나오는 context vector는 문장에 대한 이해도가 많이 떨어지겠지만!)<br>

### Encoder의 대략적 code

```python
class Encoder(nn.Module):
    def __init__(self, encoder_bock, n_layer): #n_layer : Encoder Block의 개수
        super(Encoder, self).__init__()
        self.layers = []
        for i in range(n_layer):
            self.layers.append(copy.deepcopy(encoder_block))

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out
```

`forward()`를 주목해보면, Encoder Block들을 순차적으로 실행하면서, 이전 block의 output을 이후 block의 input으로 넣습니다.<br>
첫 `block`의 input은 `x`가 된다. 이후, 가장 마지막 block의 output은 context로서 `return`됩니다.<br> 

### Encoder Block의 대략적 code

![image](https://github.com/sparkerhoney/sparkerhoney.github.io/assets/108461006/440fd24d-64cc-44b7-b242-eca788f7ceb8)<br> 

Encoder Block의 구성은 크게 **Attention Layer**,** Position-wise Feed-Forward Layer**로 구성됩니다.<br> 

```python
class EncoderBlock(nn.Module):

    def __init__(self, self_attention, position_ff):
        super(EncoderBlock, self).__init__()
        self.self_attention = self_attention
        self.position_ff = position_ff

    def forward(self, x):
        out = x
        out = self.self_attention(out)
        out = self.position_ff(out)
        return(out)
```
#### What is Attention?
**Multi-Head Attention**은 **Scaled Dot Product Attention**을 *병렬적*으로 여러개 수행하는 layer입니다.<br>
**Attention**이란 주의, 집중이라는 뜻이고 우리가 recurrent한 모델링을 진행했을 때 문제점으로 꼽을 수 있는 **parallelization** 이 안되는 부분을 해결해주는 중요한 개념입니다.<br>



