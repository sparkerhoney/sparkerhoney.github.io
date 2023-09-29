---
title: Efficient deep neural networks에 대하여
layout: post
description: EfficientNet
post-image: https://user-images.githubusercontent.com/79881119/235352897-15e29b3f-ad6b-4063-898e-0a6bb468347a.png
category: paper review
use_math: true
tags:
- AI
- deep learning
- CNN
- efficient
---

# Efficient Network

딥러닝 네트워크를 구성하는데 있어서 고려해야할 점은, 학습에 사용될 dataset의 크기나 특성 그리고 궁극적으로 해결하고자 하는 task일 것이다. 그리고 이러한 것들을 모두 신경쓰고도 최종적으로 확인해야할 부분은, 내가 가진 resource(GPU와 같은 장비)를 사용해서 학습이 가능한가?라고 볼 수 있다.   
만약 누군가 우리에게 외주를 맡겼다고 가정해보자. 학습 가능한 GPU 성능도 있고, 굳이 저용량의 모바일 기기나 임베딩에 런칭하고 싶은 DNN이 아니라면 더 좋은 성능을 내기 위해 network 구조를 scaling up할 필요가 있을 것이다.   
바로 이러한 관점에서 **'효율적으로'** 네트워크의 크기를 키우자!며 등장했던 것이 efficient network의 개념이며, 해당 연구에서는 CNN의 storage, computational complexity에 대해 다음과 같은 세 가지의 변인을 정의했다.
1. Resolution : 이미지의 차원 수, feature map size를 의미한다.
2. Width : 네트워크의 채널 수
3. Depth : 네트워크의 레이어 수
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209087790-826f9967-3a81-42c9-9bd3-a05f5b75485b.png" width="800"/>
</p>

---

# EfficientNet v1
[EfficientNet v1](https://arxiv.org/abs/1905.11946v1?fbclid=IwAR15HgcBlYsePX34qTK2aHti_GiucEYpQHjben-8wsTf7O83YPhrJQgXEJ0)에서 고려한 Baseline architecture의 경우 NAS라는 AutoML(간단히 설명하자면, 좋은 성능을 보이는 deep learning network를 찾는 과정이나 hyper-parameter를 세부 조정하는 것을 자동화하는 알고리즘)을 통해 새로운 baseline network를 디자인하고 이를 scaling up하는 방식을 사용한다.   
이때, MobileNet-v2에서 사용되는 MBConv(mobile inverted bottleneck convolution) Block을 기본으로 한다.
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209087817-8f617ea2-fbd8-4a6a-8709-cce98db3dd76.png" width="400"/>
</p>
위의 그림 참고, lightweight model에 대해서도 포스팅한 글이 있으므로 해당 글을 참고해주면 감사하겠다. 암튼 그래서 baseline network 구조는 다음과 같이 잡고 시작한다.
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209087794-466ce338-0831-43ef-a859-b1288afae880.png" width="400"/>
</p>

---

# Depth scaling($d$)
앞서 scaling factor 변인이 총 세가지 제시가 되었다고 했는데, 그 중 가장 단순하게 네트워크의 깊이에 대한 변인으로 생각할 수 있는 depth에 대한 scaling이다.   
물론 ResNet의 등장 이후, 네트워크가 깊어지면 깊어질수록 receptive field의 크기도 커지고 서로 다른 여러 image feature간 상관관계를 잡아낼 확률도 높아진다는 사실은 어느 정도 당연하게 받아들이고 있을 것이다.
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209087798-705fc27b-304e-4d98-b1a6-900e430e26b2.png" width="400"/>
</p>
그러나 ResNet에서나 VGG Network에서 보였던 경향성과 같이, 네트워크가 지나치게 깊어지면 accuract 향상에 큰 도움이 안된다. 이는 학습 과정에서 파라미터 수가 많아지더라도 이 모든 representation을 최적화할 정도로 많은 데이터가 없을 수도 있고, 그게 아니라 단순히 네트워크가 깊어지면 input과 output 간의 거리가 멀어지면서 네트워크가 학습해야할 optimal solution까지 도달하기 어려워질 수 있다는 것이다.   
일련의 예로 ResNet-1000은 ResNet-101과 유사한 accuracy를 가지는 것을 보면 알 수 있다. 또한 위의 그래프를 보아도 $d = 8.0$ 부터는 accuracy가 수렴하는 것을 확인할 수 있다. 아, 그래프에 대해 설명을 안했는데 가로축은 연산 수랑 비례하는 FLOPS이고 세로축은 ImageNet Top-1 Accuracy를 의미한다.

---

# Width scaling($w$)
그 다음으로 생각해볼 수 있는 것은 network의 깊이를 유지한 채로 wide하게 만드는 것이다. Wider network는 spatial dimension으로 커지는게 아니라, channel 수를 증가시키는 개념으로 보면 된다.   

채널 수가 많은 네트워크일수록 fine-grained(디테일하다고 보면 됨)한 특징들을 더 잘 잡아낼 수 있고, 네트워크 깊이가 얕기 때문에 excessive depth에 의한 학습 저하 효과가 덜하다. 
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209087799-93edd523-edaf-47d2-9a74-193d8b71c5df.png" width="400"/>
</p>
그럼에도 불구하고 문제가 될 수 있는 것은 너무 얕으면서(depth $d$ $\downarrow$) 지나치게 넓은(width $w$ $\uparrow$) 네트워크는 쉽게 saturate된다는 것이다. 그래프에서 볼 수 있듯 대략 $w=5$ 정도에서 성능이 수렴한다고 한다.

---

# Resolution scaling($r$)
Spatial dimension을 더 크게 가져가는 resolution factor에 대한 부분에서, 앞선 내용과 유사하게 high resolution image는 fine grained feature를 얻는 데에 유리하다. 그러나 resolution이 증가할수록 FLOPS도 증가할 뿐더러 대략 $r = 2.5$ 정도가 되면 성능이 수렴한다고 한다.
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209087802-30e5a146-1035-46bd-bc39-333fd54be84d.png" width="400"/>
</p>

---

# Observations
이런 저런 실험을 통해 두 가지의 결론에 다다를 수 있었다. 그것은 바로
1. 네트워크의 깊이/너비/차원 등 모든 factor에 대해 키우면 키울수록 성능은 증가했지만, <U>증가폭은 점차 감소하더라</U>.
2. FLOPS나 네트워크의 용량 등 효율적인 측면을 고려했을 때 네트워크의 <U>모든 dimension에 대한 밸런스를 맞추는 것</U>이 중요하다.

따라서 이러한 조건 속에서 다음과 같은 objective를 가지게 된다.
\[
    \begin{aligned}
        &\max_{d, w, r} Accuracy(\mathcal{N}(d, w, r)) \newline
        \text{s.t. } &\mathcal{N}(d, w, r) = \bigodot_{i=1 \cdots s} \hat{F_i}^{d \cdot \hat{L_i}}(X_{(r \cdot \hat{H_i}, r \cdot \hat{W_i}, w \cdot \hat{C_i})}) \newline
        &\text{Memory}(\mathcal{N}) \le \text{target_memory} \newline
        &\text{FLOPS}(\mathcal{N}) \le \text{target_flops}
    \end{aligned}    
\]
식을 간단히 해석하자면 $\mathcal{N}$이라 표시된 것이 바로 비교하고자 하는 CNN 모델을 모아놓은 함수의 집합이라 생각하면 된다. 이상한 기호로 표시된 애는(두번째 줄에 s.t.하고 써있는 부분) ($d, w, r$)로 스케일링 된 CNN의 모든 레이어를 함수로 보고, 일련의 output을 input에 대한 합성 함수의 연산으로 해석한 것이다. 따라서 $\mathcal{N}(d, w, r)$은 input에 대한 output, 그리고 우리는 해당 output을 토대로 supervision이 주어지는 어떠한 task에 대해 최대의 정확도를 얻고 싶은 것.   
그러는 와중에 threshold memory나 flops를 넘지 않는 선에서 네트워크를 서칭하고자 하는 것이다.   
근데 이렇게 보면 수학적인 모델링이 아니라 그냥 "그런 모델을 만들고 싶다"니까, 실질적으로 수치화할 필요가 있다. 즉, $d, w, r$과 같은 weight factor가 memory나 flops에 어떤 영향을 미칠 것이고, 우리가 가용하고 싶은 resource의 개수를 어떤 방식으로 searching 하느냐이다.
\[
    \begin{aligned}
        \text{depth: }&d = \alpha^\phi \newline
        \text{width: }&w = \beta^\phi \newline
        \text{resolution: }&r = \gamma^\phi \newline
        \text{s.t. }&\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2 \newline
        &\alpha \ge 1, \beta \ge 1, \gamma \ge 1
    \end{aligned}    
\]
우리는 애초에 스케일을 키우면서 서칭할 것이기 때문에(첫번째 observation에 의한 결정) 모든 인자는 1보다 크거나 같아야 하고, $\phi$는 compound coefficient로서 얼마나 많은 resource를 사용할 지 지수로 결정한다. 그리고 FLOPS는 각각 network의 depth $d$에 대해서는 linear하게, 그리고 width와 resolution $w, r$에 대해서는 거의 quadratic하게 비례하므로 전체 FLOPS가 $2^\phi$가 넘지 않도록 하는 것이다. 결국 결정되어야 할 파라미터는 총 4개 $(\alpha,~\beta,~\gamma,~\phi)$이다.

---

# Implementation

실제로 어떤 식으로 구현하였는지 확인해보자. 앞서 baseline 모델은 MBConv(MobileNet v2에서 사용한 구조)를 이용했다고 했고, 이를 B0라고 하자. 구조는 NAS를 통해 구했다(효율적인 네트워크 시작점을 찾는 알고리즘이라고 생각하면 된다).
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209087794-466ce338-0831-43ef-a859-b1288afae880.png" width="600"/>
</p>

그 다음으로는 $\phi = 1$로 고정하고, 두 배의 resource가 더 가용된다고 가정하고 $\alpha,~\beta$ 그리고 $\gamma$에 대해 grid search를 진행한다. 앞서 constraints에 $2^\phi$로 설명했던 부분.   
B0에 대해 optimal value는 $\alpha = 1.2$, $\beta = 1.1$, 그리고 $\gamma = 1.15$가 가장 optimal value로 결정되었다.   
다음에는 $\alpha,~\beta$ 그리고 $\gamma$를 상수로 고정시킨 채(위에서 얻은 값들로) 서로 다른 $\phi$에 대해 테스트해본다.
바로 이 $\phi$를 변화시켜 얻은 것이 EfficientNets B1 부터 B7이다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209087804-57e25abc-ef8c-4854-92b8-1e23253f0074.png" width="600"/>
</p>

처음에 $\alpha,~\beta,~\gamma$를 잘 정하고 시작했기 때문에 parameter 수를 많이 증가시키지 않고도 기존 네트워크들과 비교하여 쉽게 우위를 점할 수 있는 것을 볼 수 있다. ~~역시 무지성으로 네트워크 구조 짜는게 답은 아니라는 것을 알 수 있다.~~

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209087808-02559f28-75a9-4a8d-bda5-763c06ff224b.png" width="800"/>
</p>

각 클래스 별로 네트워크가 어디를 중점적으로 보는지 시각화하는 CAM을 사용해서 본 결과다. **compound scaling**은 보다 디테일한 부분을 잡아내는 걸 확인할 수 있다.

---

# EfficientNet v2
아직 안끝났다. EfficientNet v1이 있다는 것은 v2도 있다는 것이다. 어벤져스 1편이 있으면 2편도 있고 그런거다(이게 뭔 소리지). 아무튼 [EfficientNet v2](https://arxiv.org/abs/2104.00298)는 비교적 최근에 나왔다(2021). 과거의 EfficientNet v1이 영광을 누리던 시절은 ViT같은 새로운 구조의 computer vision network도 없었고 그랬는데 암튼 시대가 많이 변했다. 발전이 필요해서 더 좋은 네트워크를 고안하기 시작한 것이다.   
EfficientNet v2에서 네트워크 구조를 서칭하는 과정은 정확히 EfficientNet v1과 동일하다. 다만 EfficientNet v2에서는 architecture(baseline)이 바뀐 것이 첫번째 contribution이고, **progressive learning**을 활용했다는 것이 두번째 contribution이다. 즉, image size에 대한 정규화를 진행했다는 것이다([PGGAN](https://arxiv.org/abs/1710.10196) 참고).   
이러한 방법을 토대로  EfficientNet v1에 대해 $11\times$ 빠른 학습 속도와 $6.8\times$ 나은 parameter efficiency를 보여주었다.

---

# Fused MBConv
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209087812-b386bf1e-9a5d-4b68-9d78-25f66889915a.png" width="600"/>
</p>
사용한 구조에서 fused MBConv를 간단히 설명하면, depthwise $3\times 3$ convolution과 expansion $1 \times 1$ convolution이 기존 MBConv를 구성하는데, 이걸 그냥 $3 \times 3$ convolution으로 바꾼 것이다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209087817-8f617ea2-fbd8-4a6a-8709-cce98db3dd76.png" width="600"/>
</p>

그래서 모든 레이어를 다 Fused MBConv로 한 건 아니고, 표에서 보는 것과 같이 기존 Efficient v1의 구조에서 일부분(표에서 stage로 표시된 부분이 바뀐 부분)을 바꿨을 때의 효과를 보여준다.   
그러나 저자들은 정확히 <U>어떤 부분에서 fused MBConv를 써야할 지</U> 확실한 답을 얻을 수 없었기에 이를 NAS에 맡긴다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209087818-e12926fd-96cf-4155-a4ca-e59d98f293ac.png" width="800"/>
</p>

그래서 얻은 baseline 구조(EfficientNetV2-S라 명명)에 대한 결론은,
1. 초반 layer에서 fused MBConv를 많이 사용하자.
2. EfficientNet v2($3 \times 3$)는 v1($5 \times 5$)에 비해 작은 kernel size가 좋았다.
3. 마지막 stride-1 stage를 없애버림

이다. 그냥 복잡하니까 NAS가 알아서 제일 좋은거 갖다줬다고 이해하면 될듯.  

---

# Progressive learning
점진적 학습의 가장 메인 아이디어는 작은 image size부터 차근차근 학습시키자는 것이다. 이를테면 총 10 epoch 동안 학습하는 과정에서 초반 6 epoch는 $224 \times 224$의 이미지 크기로 학습을 하고 남은 4 epoch은 좀 더 큰 $256 \times 256$에 대해 finetune하는 느낌.   
그러나 단순히 smaller image로부터 larger image로 점차 키우는 건 accuracy drop을 불러일으켰다고 한다.   
성능 더 좋게 할라고 한건데 왜...? ㅠㅠ 라고 하며 저자들이 생각했던 것은 image size가 달라지면, regularization strength도 그에 따라 변화시켜줘야 한다는 것이다. 즉 regularization을 무지성으로 똑같이 적용하다보니 unbalance 문제가 발생했다는 것.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209087820-d759593e-41bd-4e95-b59e-f2f7fa6811dc.png" width="800"/>
</p>

그래서 개선 방법으로 떠올렸던 것이 이미지 크기도 점차 증가시키고 그에 따른 정규화도 점차 키우는 것이다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209087823-69796a5f-d41d-4292-9cf4-8a2dfc2d91ad.png" width="800"/>
</p>

알고리즘을 보면 $R_i$가 regularization strength고 $S_i$가 이미지 크기를 의미한다. 되게 간단해서 딱히 설명 안해도 괜츈할 거 같다. iteration이 M번 증가하면서 이미지 크기는 초기값 $S_0$에서 linearly 증가시키고 마찬가지로 regularization strength도 초기값에서 linearly 증가시킨다.   
슬슬 여기서 들 수 있는 의문은 "그래서 대체 그 정규화가 어떤 정규화인데?" 싶을거다. 논문에서는 이러한 학습법이 존재하는 대부분의 정규화 작업에 적용 가능하며, 총 세 개의 정규화에 대해 실험하였다.

- **Dropout** - 네트워크 레벨에서의 정규화, 일부 채널을 학습 과정에서 dropping 한다. 즉 노드를 일부 끊어버린다고 생각하면 된다. Dropout rate $\gamma$가 각 노드가 버려질 확률을 의미하는데, 이를 조절하였다.
- **RandAugment** - 이미지별 data augmentation 방법에 해당된다. 말 그대로 랜덤으로 augmentation을 하는건데, 여기서 magnitude $\epsilon$을 조절했다.
- **Mixup** - 이미지 간의 data augmentation에 해당된다. 예컨데 개 이미지 $(x_i, y_i)$ 와 고양이 이미지 $(x_j, y_j)$ 가 있으면 이걸 0부터 1 사이의 mixup ratio인 $\lambda$를 기준으로 개냥이 이미지 $\tilde{x_i} = \lambda x_j + (1-\lambda)x_i$ 그리고 그에 해당되는 라벨 $\tilde{y_i} = \lambda y_j + (1-\lambda)y_i$를 만들어내는 정규화 작업. 여기서 mixup ratio $\lambda$를 조절했다고 한다.   

암튼 요로코롬 조로코롬 잘 조절해서 실험하니, 다음과 같은 결과를 얻었다고 함.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209087825-cff21495-b8b3-4916-bde0-6be8418d9efe.png" width="800"/>
</p>
