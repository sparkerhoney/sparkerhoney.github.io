---
title: Feature Extraction 2nd(Non-monotonicity)
layout: post
description: Lecture summary
use_math: true
post-image: https://github.com/sparkerhoney/sparkerhoney.github.io/blob/master/_images/machine%20learning.png?raw=true
category: machine learning
tags:
- Data Science
- machine learning
---

# Features
## Handling Nonlinearity with Linear Methods
### Example Task: Predicting Health
예시를 한 번 들어보겠습니다.<br>
일반적으로 우리는 의료 진단용으로 가능성이 있는 모든 feature들을 추출합니다.<br>
- ex) height, weight, body temperature, blood pressure, etc...<br>

---

### Issues for Linear Predictors
선형 예측 모델에서, 특징(feature)이 어떻게 추가되는지가 중요합니다.<br>
문제를 일으킬 수 있는 비선형성(nonlinearity)의 세 가지 유형은 다음과 같습니다:<br>

1. **단조성이 아님(non-monotonicity)**
2. **포화(saturation)**
3. **특징 간 상호작용(interactions between features)**<br>

이러한 비선형성(nonlinearities)이 있을 경우, 모델이 예측을 위해 사용하는 특징(feature)들의 중요도나 가중치를 잘못 판단할 수 있습니다.<br>
이는 모델의 정확성과 일반화 능력을 떨어뜨릴 수 있습니다.<br>

---

>"단조성이 아님(non-monotonicity)", "포화(saturation)", "특징 간 상호작용(interactions between features)"은 모두 비선형(nonlinear)적인 관계를 나타내는 특징입니다.<br>
"단조성이 아님(non-monotonicity)": 두 변수 사이의 관계가 단조적(monotonic)이지 않은 경우를 의미합니다. 즉, 한 변수가 증가할 때 다른 변수의 값이 단조적으로 증가하거나 감소하지 않는 경우입니다.<br>
"포화(saturation)": 입력값이 일정 수준 이상이 되면 출력값이 더 이상 증가하지 않고 일정 수준을 유지하는 경우를 말합니다. 이는 종종 물리적인 한계나 한계점을 나타내는 경우가 많습니다.<br>
"특징 간 상호작용(interactions between features)": 하나의 특징이 다른 특징에 영향을 미치는 경우를 의미합니다. 이러한 상호작용은 종종 두 특징을 더해주거나 곱해주는 등의 방식으로 모델에 반영됩니다.<br>

---

### Non-monotonicity: The Issue
- **단조성이 아닌(non-monotonic)** 특징은 특정 값의 범위에서는 결과에 부정적인 영향을 끼칠 수 있지만 다른 값의 범위에서는 결과에 긍정적인 영향을 끼칠 수 있습니다.<br>
단조적(monotonic)이라는 용어는 기울기가 항상 증가하거나 항상 감소하는 선형 관계를 의미하는 것은 아닙니다.<br>
단조적이라는 말은 단지 변수 간의 관계가 항상 같은 방향으로 움직인다는 것을 의미합니다.<br> 예를 들어, x가 증가할 때 y도 항상 증가하면 x와 y 사이의 관계는 단조적입니다.<br> 하지만 x가 증가할 때 y가 먼저 감소하다가 다시 증가한다면 이는 비단조적인 관계입니다.<br>
따라서, Non-monotonicity 문제는 변수 간의 관계가 비선형이거나 불규칙한 경우에도 발생할 수 있습니다.<br>
예를 들어, 당신이 주식 투자를 한다고 가정해 봅시다.<br>
당신은 어떤 회사의 주식 가격이 상승할 것이라는 예측을 하고 그 주식을 매수합니다.<br> 그러나 나중에 그 회사에서 부정한 일이 드러나면, 그 주식 가격은 급격히 하락할 것입니다.<br> 이 때, 당신의 예측이 잘못되어서 이 주식을 매수한 것이므로, 이 문제는 Non-monotonicity 문제입니다.<br> 
이는 변수인 "예측"과 "실제 주식 가격" 간의 비선형적이고 불규칙한 관계 때문에 발생한 것입니다.<br>
이 경우 모델은 이러한 특징의 중요도를 적절하게 평가하기 어려울 수 있습니다.<br>
Non-monotonicity가 발생하는 경우에는 변수 간에 선형 관계가 아닌, 비선형 관계가 존재하거나, 우리가 알지 못하는 다른 변수가 영향을 미치는 경우가 있을 수 있습니다.<br> 이런 경우에는 적절한 feature engineering이나 다른 기술적인 방법을 사용하여 모델을 개선해야 합니다.<br>

---

- Feature Map: $$φ(x) = [1,temperature(x)]$$<br>
- Action: Predict health score $$y ∈ R$$ (positive is good)<br>
- Hypothesis Space $$F = {affine\ functions\ of\ temperature}$$<br>

이 예시에서는 온도(temperature) 값을 이용하여 건강 점수$(y)$를 예측하는 모델을 고려합니다.<br>
모델은 온도 값의 선형 함수로 구성되며, 이를 수식으로 나타내면 $$φ(x) = [1,temperature(x)]$$로 표현됩니다.<br>

여기서 $$φ(x)$$는 입력 $$x$$의 특징 벡터(feature vector)를 나타내며, $$[1,temperature(x)]$$는 상수 1과 입력 x의 온도 값으로 이루어져 있습니다.<br>
그러나 건강 점수는 온도의 선형 함수가 아닙니다.<br>
따라서 온도와 건강 점수 사이에는 선형 관계가 존재하지 않습니다.<br>
이 모델에서 선형 함수를 사용하면, 높은 온도와 낮은 온도 모두 건강에 좋지 않은 영향을 미친다는 문제가 발생합니다.<br>
선형 함수는 높은 온도가 나쁘다고 하면, 낮은 온도는 좋다고 말할 수밖에 없기 때문입니다.<br>

하지만 이 경우에는 낮은 온도와 높은 온도 모두 건강에 나쁜 영향을 미치는 것입니다.<br>
따라서 이러한 문제를 해결하려면 선형 함수 대신에 비선형 함수를 사용해야 합니다.<br>
이 예시에서는 온도와 건강 점수 사이에 비선형 관계가 존재하므로, 선형 함수로는 이를 모델링할 수 없습니다.<br>

---

### Non-monotonicity: Solution 1
- Transform the input: $$φ(x) = [1,{temperature(x)-37}^2]$$ <br>

위의 예시에서는 체온(temperature)이라는 feature로부터 건강 점수(health score)를 예측하는 문제를 다루고 있습니다.<br>
그러나 건강 점수는 체온의 선형 함수로는 정확하게 예측할 수 없는 경우였습니다.<br> 따라서 이 문제를 해결하기 위해 체온의 변형을 feature로 사용하는 것이 제안되었습니다.<br>

이를 위해 입력 $$x$$를 변형하여 $$φ(x) = [1, (temperature(x) - 37)^2]$$ 로 만들었습니다.<br> 이제 건강 점수 $$y$$는 $$φ(x)$$를 사용하여 예측됩니다.<br>
그러나 이러한 입력 변형은 전문 지식이 필요한 경우가 있습니다.<br>

예를 들어, 위의 예시에서는 정상 체온이 37℃인 것을 알아야 합니다.<br>
따라서 이러한 변형은 도메인 전문가의 도움이 필요합니다.<br>
이러한 변형을 사용하지 않고도 모델을 학습할 수 있으나, 때로는 입력에 대한 도메인 지식이 모델의 예측 능력을 향상시키는 데 도움이 될 수 있습니다.<br>

---

### Non-monotonicity: Solution 2

- $$φ(x) = [1,temperature(x) ,(temperature(x))^2]$$<br>

기존에 가지고 있는 feature들을 최대한 활용해서 solution1보다 표현력이 뛰어나게끔 만들어준 mapping 함수입니다.<br>
이 경우, 예시로 들어진 feature인 온도(temperature)를 그대로 사용하지 않고 온도의 제곱도 함께 feature로 추가하는 것입니다.<br>
이렇게 하면 모델이 **더 복잡한 패턴도 학습할 수 있게 됩니다.** <br>

일반적으로 좋은 feature는 간단하면서도 중요한 정보를 포함하는 것입니다.<br>
이러한 feature들을 조합하여 모델을 만들면 더 복잡한 패턴을 학습할 수 있습니다.<br>
따라서 feature는 단순하고 각 *feature가 독립적으로 중요한 정보*를 제공하는 것이 좋습니다.<br>

---

> 비선형성이 있는 경우에도 **feature extraction이 적용** 이 되는지?<br>
  비선형성이 있는 경우에도 feature extraction은 적용됩니다. 사실, 비선형성이 있는 경우에 feature extraction이 더욱 중요해집니다.<br>
  비선형성이 있으면, 입력과 출력 간의 관계가 복잡하게 되어서, 이를 표현하기 위해 더 많은 feature가 필요합니다.<br>
  이를 위해, 보다 복잡한 feature extraction 기법이 필요할 수 있습니다.<br>
  예를 들어, 비선형 관계가 있는 경우, 다항식 특성(polynomial features)을 추가할 수 있습니다.<br>
  이를 통해 비선형성을 나타내는 새로운 feature를 만들어 내어 모델의 표현력을 높일 수 있습니다.<br>
  또한, 신경망 같은 비선형 모델에서는 더 복잡한 feature extraction 기법이 사용될 수 있습니다.<br>

---

[*출처 : FOUNDATIONS OF MACHINE LEARNING by Bloomberg ML EDU*](https://bloomberg.github.io/foml/#home).
