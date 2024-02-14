---
title: Semi-supervised Domain Adaptation via Minimax Entropy
layout: post
description: paper review
use_math: true
post-image: https://github.com/sparkerhoney/sparkerhoney.github.io/assets/108461006/1298b827-5920-4b16-95f3-87d45c8c37bd
category: paper review
tags:
- Data Science
- machine learning
- NLP
- paper review
---
# Semi-supervised Domain Adaptation via Minimax Entropy
## Introduction

해당 논문은 Domain Adation Methods에서 **Semi-Supervised Domain Adaption Methods(이하 SSDA)**에서 Traget Domain에 소수의 Label이 제공될때 효과적인 Domain Adaption을 위한 방법론을 제시합니다.<br>
위 문제를 Few-Shot Model을 Adversarial Optimize하는 **Minimax Entropy(MME)**를 이용하여 Target Domain의 소수의 labeled Data를 이용하여 Unlabeled Data에 대한 Model의 예측성능을 개선하는데 중점을 둡니다.<br>
<br>

> **Semi-Supervised Domain Adaption(SSDA)** <br>
  -  Model이 하나의 Domain(Source Domain)에서 학습된 지식을 다소 다른 Feature Distribution을 가지는 다른 Domain(Target Domain)으로 전이하는 방법론입니다. <br>
  -  SSDA의 핵심은 Source Data의 Labled Data와 함께 Target Domain의 소수의 Labled Data 및 많은 양의 Unlabled Data를 모두 활용하는 것입니다.<br>
<br>

## Related Work
### 1. Domain Adaptation
Source Domain에서 학습된 Model을 Target Domain에 적용할 때 발생하는 성능 저하 문제를 해결하기 위한 연구 분야입니다. <br>
기존의 Domain Adaptation 연구들은 주로 Unsupervised Method에 초점을 맞추어 왔으며, Source와 Target Domain 간의 **Feature Distribution 차이를 최소화**하는 다양한 기술을 개발해왔습니다. <br>
이러한 연구들은 Domain Classifier를 사용하여 Source와 Target Domain을 구별하려고 시도하며, Model이 Domain 간 차이를 최소화하도록 학습합니다.<br>
<br>

### 2. Semi-Supervised Learning
Labled 소수의 Data와 Unlabled 많은 양의 Data를 모두 활용하여 Model을 학습시키는 Method입니다. <br>
이 분야의 연구는 주로 Unlabled Data에서 유용한 정보를 Extraction하여 Model의 Performance를 향상시키는 기법에 초점을 맞춥니다. <br>
조건부 Entropy 최소화(Conditional Entropy Minimization)와 같은 기존 SSL Methods는 Unlabled Data를 활용하여 Model의 **Generalization 능력을 향상**시키려고 하지만, Domain 간의 차이가 클 경우 성능이 제한될 수 있습니다.<br>
<br>

### 3. Few-Shot Learning
매우 소수의 Labled 예시로부터 Class를 학습하는 문제에 초점을 맞춘 연구 분야입니다. <br>
이 Method는 Target Domain에 대한 소수의 Labled 예시에서도 Model이 잘 작동할 수 있도록 설계되었습니다. <br>
저자들은 기존의 Few-Shot Learning Methods가 Target Domain의 Unlabled Data를 충분히 활용하지 못한다는 한계를 지적하며, MME 방법을 통해 이러한 Data를 효과적으로 활용하여 Domain Adaption 문제를 해결하고자 합니다.<br>
<br>

## Minimax Entropy(MME)

**SSDA**에서는 세 가지 Data Set가 주어집니다. <br>

$D_s$ (Source Domain Labeled Data)<br>
- $\(D_s = \{(x_s^i, y_s^i)\}_{i=1}^{m_s}\)$
- \(D_s\)는 소스 도메인의 레이블이 있는 데이터 세트입니다.
- 여기서 \(x_s^i\)는 소스 도메인에서 \(i\)번째 이미지를 나타내고, \(y_s^i\)는 해당 이미지의 레이블(정답)입니다.
- \(m_s\)는 소스 도메인에서 레이블이 있는 이미지의 총 개수를 나타냅니다.
- 모델 학습 초기 단계에서 주로 사용되며, 모델이 소스 도메인의 지식을 학습하는 데 기초 자료로 활용됩니다.

### \(D_t\) (Target Domain Labeled Data)
- \(D_t = \{(x_t^i, y_t^i)\}_{i=1}^{m_t}\)
- \(D_t\)는 타겟 도메인의 레이블이 있는 데이터 세트입니다.
- 여기서 \(x_t^i\)는 타겟 도메인에서 \(i\)번째 이미지를 나타내고, \(y_t^i\)는 해당 이미지의 레이블입니다.
- \(m_t\)는 타겟 도메인에서 레이블이 있는 이미지의 총 개수를 나타냅니다.
- 타겟 도메인에 대한 지식을 모델에 전달하는 데 사용되며, 레이블이 있는 데이터의 양이 제한적입니다.

### \(D_u\) (Target Domain Unlabeled Data)
- \(D_u = \{x_u^i\}_{i=1}^{m_u}\)
- \(D_u\)는 타겟 도메인의 레이블이 없는 데이터 세트입니다.
- 여기서 \(x_u^i\)는 타겟 도메인에서 \(i\)번째 이미지를 나타내지만, 레이블 정보는 제공되지 않습니다.
- \(m_u\)는 타겟 도메인에서 레이블이 없는 이미지의 총 개수를 나타냅니다.
- 모델이 타겟 도메인에 더 잘 적응하도록 돕기 위해, 추가적인 정보를 제공하는 데 사용됩니다.

이러한 데이터 세트의 구분은 모델이 소스 도메인에서 학습한 지식을 타겟 도메인에 적용하고, 타겟 도메인의 구조와 분포를 더 잘 이해하는 데 필수적입니다. 레이블이 있는 데이터(\(D_s\)와 \(D_t\))는 모델이 정확한 예측을 학습하는 데 도움을 주고, 레이블이 없는 데이터(\(D_u\))는 타겟 도메인의 다양성을 모델에게 제시하여 일반화 능력을 향상시킵니다.

<img width="510" alt="Figure 1 Top" src="https://github.com/sparkerhoney/sparkerhoney.github.io/assets/108461006/f6da7e21-1780-4845-934a-89bbf5a85def">
<br>

기존 **Domain Classifier Based Methods**가 Source와 Target Domain의 Distribution을 **Align**하려는 시도를 보여줍니다.<br>
이 Methods는 Source와 Target Domain 간의 Distribution를 일치시키려고 하지만, Target Domain에서 구별 가능한 Class 경계를 학습하는 데 실패할 수 있습니다.<br>
이는 Task Decision Boundary 근처에서 **모호한 특성을 생성**할 수 있음을 시사합니다.<br>
<br>

<img width="505" alt="Figure 1 Bottom" src="https://github.com/sparkerhoney/sparkerhoney.github.io/assets/108461006/3400b036-625d-408f-b4c3-67d9129eb13a">
<br>

저자들이 제안하는 **Minimax Entropy (MME)** Method를 보여줍니다.<br>
이 Method는 각 Class에 대한 **Representative Point(Prototype)**을 추정하고, 새로운 **Minimax Entropy** Method를 사용하여 구별 가능한 Feature를 Extraction합니다.<br> 
해당 Approach는 Target Domain의 Unlabled Data에 대해 Entropy를 최대화하고, Feature Extractor를 Update하여 이러한 Data를 Prototype 주변에 더 잘 **Clustering**하도록 합니다.<br>
<br>

> **Entropy** <br>
> - Information Theory에서 **불확실성의 척도**로 사용되는 개념입니다. <br>
> - 높은 Entropy는 더 많은 불확실성을 의미하며, 낮은 Entropy는 더 많은 정보를 의미합니다. <br>
> - Model의 예측이 불확실할 때 Entropy는 높아지며, 예측이 확실할 때 Entropy는 낮아집니다. <br>
> - 따라서, Model의 예측이 불확실한 Data에 대한 Entropy를 최대화하면 Model이 더 많은 정보를 얻을 수 있습니다. <br>
> - 이러한 이유로, MME Method는 Target Domain의 Unlabled Data에 대한 Entropy를 최대화하여 Model의 Generalization 능력을 향상시키려고 합니다. <br>
> - 이러한 방법을 통해 Model은 Target Domain의 Unlabled Data를 더 잘 학습할 수 있으며, 이는 Domain Adaptation 문제를 해결하는데 도움이 될 수 있습니다. <br>
> - 또한, MME Method는 Target Domain의 Unlabled Data를 활용하여 **Feature Extractor를 Update**하여 Target Domain의 **Prototype 주변에 더 잘 Clustering**하도록 합니다. <br>
<br>

<img width="660" alt="Figure 2 Top" src="https://github.com/sparkerhoney/sparkerhoney.github.io/assets/108461006/6addcae4-3dbf-4f05-9dcc-d858a4ba948b">
<br>

위는 기존의 **few-shot** 학습 Method를 보여줍니다.<br>
기존의 **few-shot** 학습 Methods는 소수의 Labeld Data를 사용하여 각 Class에 대한 Prototype(대표적인 특성을 가진 Weight Vector)을 학습합니다.<br> 
이 Prototype은 새로운 Data가 어느 Class에 속하는지를 판단하는 데 사용됩니다.<br>
이 Method는 Class Prototype을 Weight Vector로 추정하지만, Target Domain의 Unlabeled Data를 직접적으로는 고려하지 않습니다.<br>
즉, 이러한 Methods는 Unlabeled Data로부터 추가적인 정보를 Extraction하거나, 이를 활용하여 Model의 Generalization 능력을 향상시키는 데 한계가 있습니다.<br>
<br>

<img width="673" alt="Figure 2 Bottom" src="https://github.com/sparkerhoney/sparkerhoney.github.io/assets/108461006/6a337d00-bb27-4efe-b61a-bcb3eb503561">
<br>

MME Method의 구체적인 작동 과정을 나타냅니다.<br>

`Step 1`에서는 Classifier 내의 추정된 Prototype을 Target Domain의 Unlabeled Data의 **Entropy를 최대화**하도록 업데이트합니다.<br>
- Entropy를 최대화한다는 것은 Model이 Unlabeled Data에 대해 더 불확실한 예측을 하도록 만들어, Data가 여러 **Class에 고르게 속할 가능성을 탐색**하게 만듭니다.<br>
- 이 과정을 통해, Model은 Target Domain의 다양한 Data Feature를 더 잘 파악하게 됩니다.

`Step 2`에서는 **Feature Classifier를 업데이트**하여 이러한 Data가 Prototype 주변에 **Clustering** 되도록 합니다.<br>
- 즉, 각 Data Point가 가장 가까운 Prototype(Class)에 할당되도록 Model을 조정합니다.<br> 
- 이 과정은 Model이 Target Domain의 **Data 구조를 더 잘 이해하고**, 구별 가능한 Feature를 학습하게 만듭니다.<br>
<br>

<img width="647" alt="Figure 3" src="https://github.com/sparkerhoney/sparkerhoney.github.io/assets/108461006/00a1fb96-503f-4b5e-a9b9-5422870335be">
<br>

논문에서 제안하는 **MME Method Architecture**의 전체적인 구조를 나타냅니다.<br>

### Model 구성 요소(Model Configuration):
**특성 추출기(Feature Extractor, $F$)**: Input Data로부터 유용한 Feature를 Extraction하는 역할을 합니다.<br> 이 구성 요소는 Model이 Data의 **중요한 정보를 식별**하고 이를 효율적으로 처리할 수 있도록 돕습니다.<br>
**분류기(Classifier, $C$)**: Extracted Feature를 바탕으로 각 Data Point가 **어떤 Class에 속하는지를 판단**합니다.<br> 이 구성 요소에는 **Weight Vector($W$)**와 **Temperatue Parameter($T$)**가 포함되어 있으며, 각 **Class에 대한 예측 확률을 계산**합니다.<br>

### 입력 Data(Input Data):
**레이블이 있는 Source Data**: 학습 초기 단계에서 Model의 기본 지식을 형성하는 데 사용됩니다.
**레이블이 있는 Target Data**: Target Domain에 대한 지식을 제공하며, 이 Data는 소수만 제공됩니다.
**레이블이 없는 Target Data**: Model이 Target Domain에 더 잘 적응할 수 있도록 추가 정보를 제공합니다.

### 학습 과정(Training Process):
**Entropy 최대화(Step 1)**: Classifier $C$는 Unlabled Target Data에 대한 Entropy를 최대화하도록 학습됩니다.<br> 이 단계는 Model이 Data Point에 대해 더 불확실한 예측을 하게 만들어, 다양한 Class 간의 경계를 더 잘 탐색하도록 돕습니다.<br>
**Entropy 최소화(Step 2)**: Feature Extractor $F$는 Entropy를 최소화하도록 학습됩니다.<br> 이는 Unlabled Target Data가 가장 가까운 프로토타입 주변에 잘 클러스터링되도록 만들어, Target Domain에서의 예측 정확도를 높입니다.<br>

### 적대적 학습(Adversarial Learning):
Model은 Entropy 손실에 대해 Adversarial Learning Method를 적용합니다.<br> 이 과정에서 Unlabled Target Data에 대한 Entropy 손실의 Gradient 부호를 반전시키는 **그래디언트 반전 계층(Gradient Reversal Layer)을 사용**합니다.<br> 이를 통해, **Classifier $C$**와 **Feature Extractor $F$**가 **상반된 목표**를 가지고 학습하면서도 전체적으로는 Target Domain에 대한 Model의 성능을 개선하게 됩니다.<br>