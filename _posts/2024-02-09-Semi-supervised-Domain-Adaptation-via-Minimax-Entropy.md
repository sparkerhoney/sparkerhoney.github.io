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
위 문제를 Few-Shot Model을 Adversarial Optimize하는 **Minimax Entropy(MME)**를 이용하여 Target Domain의 소수의 labeled Data를 이용하여 Unlabeled Data에 대한 모델의 예측성능을 개선하는데 중점을 둡니다.<br>
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
조건부 엔트로피 최소화(Conditional Entropy Minimization)와 같은 기존 SSL Methods는 Unlabled Data를 활용하여 Model의 **Generalization 능력을 향상**시키려고 하지만, Domain 간의 차이가 클 경우 성능이 제한될 수 있습니다.<br>
<br>

### 3. Few-Shot Learning
매우 소수의 Labled 예시로부터 Class를 학습하는 문제에 초점을 맞춘 연구 분야입니다. <br>
이 Method는 Target Domain에 대한 소수의 Labled 예시에서도 Model이 잘 작동할 수 있도록 설계되었습니다. <br>
저자들은 기존의 Few-Shot Learning Methods가 Target Domain의 Unlabled Data를 충분히 활용하지 못한다는 한계를 지적하며, MME 방법을 통해 이러한 Data를 효과적으로 활용하여 Domain Adaption 문제를 해결하고자 합니다.<br>
<br>

## Minimax Entropy(MME)

<img width="510" alt="Figure 1 Top" src="https://github.com/sparkerhoney/sparkerhoney.github.io/assets/108461006/f6da7e21-1780-4845-934a-89bbf5a85def">
<br>

기존 **Domain Classifier Based Methods**가 Source와 Target Domain의 Distribution을 **Align**하려는 시도를 보여줍니다.<br>
이 Methods는 Source와 Target Domain 간의 Distribution를 일치시키려고 하지만, Target Domain에서 구별 가능한 Class 경계를 학습하는 데 실패할 수 있습니다.<br>
이는 Task Decision Boundary 근처에서 **모호한 특성을 생성**할 수 있음을 시사합니다.<br>
<br>

<img width="505" alt="Figure 1 Bottom" src="https://github.com/sparkerhoney/sparkerhoney.github.io/assets/108461006/3400b036-625d-408f-b4c3-67d9129eb13a">
<br>

저자들이 제안하는 **Minimax Entropy (MME)** Method를 보여줍니다.<br>
이 Method는 각 Class에 대한 **Representative Point(Prototype)**을 추정하고, 새로운 **Minimax Entropy** Mrthod를 사용하여 구별 가능한 Feature를 Extraction합니다.<br> 
해당 Approach는 Target Domain의 Unlabled Data에 대해 Entropy를 최대화하고, Feature Extractor를 Update하여 이러한 Data를 Prototype 주변에 더 잘 **Clustering**하도록 합니다.<br>
<br>

> **Entropy**
> 
