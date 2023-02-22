---
title:  "Feature Extraction"
excerpt: "Feature Extraction"

categories:
  - ml
tags:
  - [ml, data science]

use_math: true
toc: true
toc_sticky: true
 
date: 2023-02-22
last_modified_at: 2023-02-22
---
# Features
## Feature Extraction
### The Input Space $X$ 
일반 적으로 input space $X$에 대해서는 가정이 없습니다.<br>
그러나 우리가 개발한 구체적인 방법들에 대해서는 input space $X=R^d$ 차원의 실수 공간입니다.<br>
- ex) Ridge regression, Lasso regression, Linear SVM<br>

기본적으로 실수차원의 input을 사용하지 않는 경우가 많습니다.<br>
- ex)Text documents, Image files, Sound recordings, DNA sequences<br>

컴퓨터 안 모든 것은 숫자의 연속입니다.<br>
따라서 각 시퀀스의 $i$번째 항목은 동일한 의미를 가져야합니다.(colum별로 동일한 의미)<br>
그리고 모든 시퀀스의 길이가 같아합니다.<br>

### Feature Extraction
$X$에서 $R^d$차원의 벡터로 매핑시키는 것을 feature extraction or featurization 라고 부릅니다.<br>
Raw data에서 feature를 뽑아내서 컴퓨터가 해당 data에서의 feature를 읽을 수 있게끔 하기 위해 vector화 시켜줍니다.<br>
- ex) 이미지에서 특징을 추출해 특징들을 정의할 수 있습니다.<br>

![image](https://user-images.githubusercontent.com/108461006/220524902-f9958b06-8644-41cf-ba0f-7d54a59ef67d.png)

##	Feature Templates
###	Example: Detecting Email Addresses

![image](https://user-images.githubusercontent.com/108461006/220525484-f5a86efe-0afc-446b-acf2-bf9896b9bc80.png)

String(문자열)이 email 주소인지 여부 예측<br>
위 그림에서 보면 도메인 지식을 사용해서 오른쪽을 나타낼 수 있습니다.<br>
여기서 우리가 놓친 것이 있을 것입니다 더 체계적일 수 있을까요?<br>

- 정의 : feature template은 유사한 방법으로 계산된 feature의 그룹, Feature extraction에서 사용되는 일종의 패턴이나 규칙 세트<br>

- ex) input이 abc@gmail.com 이라면 feature template은

'Last three characters equal ___' 처럼 '마지막 3 단어는 ___ 입니다'로 일종의 패턴으로 특징을 추출할 수 있는 겁니다.<br>
- ex) 특징 템플릿을 사용하여 텍스트 데이터에서 단어의 특성을 추출하는 방법은 다음과 같습니다.<br>
1. N-gram 템플릿: 이 방법은 연속된 단어들을 추출하는 방법입니다. 예를 들어, "I love cats"와 같은 문장에서 2-gram 템플릿을 사용하면 "I love", "love cats"와 같은 연속된 단어 쌍을 추출할 수 있습니다.<br>
2. 품사 템플릿: 이 방법은 단어의 품사를 추출하는 방법입니다. 예를 들어, "I like playing soccer"와 같은 문장에서 명사(noun)와 동사(verb)를 추출하여 문장의 의미를 분석할 수 있습니다.
3. 주요 단어 템플릿: 이 방법은 문서에서 자주 등장하는 단어를 추출하는 방법입니다. 이를 통해 문서의 핵심 단어를 파악할 수 있습니다.
4.	TF-IDF 템플릿: 이 방법은 특정 단어가 문서에서 중요한 역할을 하는지를 나타내는 지표입니다. 이 방법은 문서에서 자주 등장하지만, 다른 문서에서는 그렇지 않은 단어를 추출합니다.<br>

이러한 특징 템플릿은 텍스트 분류, 감성 분석, 정보 검색 등의 자연어 처리 작업에서 매우 유용하게 사용될 수 있습니다.<br>

### Feature Template: Last Three Characters Equal ___

![image](https://user-images.githubusercontent.com/108461006/220525650-46059477-4cba-46e4-83b4-e61fcf756af8.png)

### Feature Template: One-Hot Encoding

![image](https://user-images.githubusercontent.com/108461006/220525783-c7150a02-a731-4b37-aea8-55052b688224.png)

- 정의 : 원핫 인코딩(One-hot encoding)은 항상 하나의 비트만 1인 이진수 벡터를 사용하여 범주형(categorical) 데이터를 나타내는 방법입니다.<br>
 이때, 범주형 데이터를 특징(feature)으로 표현할 때, 해당 특징에 대응하는 비트만 1이고, 나머지 비트는 모두 0입니다.<br>
 이러한 방식으로, 범주형 데이터를 이진수 벡터로 표현하여 기계학습 알고리즘이 이해할 수 있도록 합니다.<br>

따라서, 위에서 언급한 문장에서는 원핫 인코딩이라는 방식이 항상 하나의 비트만 1을 가지는 이진수 벡터를 사용하여 범주형 데이터를 나타내는 방법이라는 내용을 설명하고 있습니다. <br>이러한 원핫 인코딩은 분류(classification) 작업에서 매우 유용하게 사용됩니다.<br>

###	Feature Vector Representations(표현 방법의 이야기)

![image](https://user-images.githubusercontent.com/108461006/220525930-3fc17bb4-d529-4c85-a7b6-ff5034daf653.png)

위에 제시된 두 가지 방법은 특징(feature)을 나타내는 방식에 있어서 차이가 있습니다.<br>

1. 배열(array) 표현 방식:
- 밀집된(dense) 특징에 대해서는 배열(array) 표현 방식이 유용합니다.<br>
-	배열 표현 방식은 각 특징의 값을 배열의 인덱스에 대응하여 저장합니다.<br>
-	위의 예시에서는 0.85, 1번째 특징의 값인 'fracOfAlpha'에 대응하는 인덱스에 저장하고, 나머지 특징의 값은 0으로 초기화되어 저장됩니다.<br>
- 배열 방식은 특징들이 고정된 순서(order)를 가지고 있다고 가정합니다.<br>
- 만약 특징들 중 대부분이 0이 아닌 값(nonzero elements)을 가진다면, 배열 방식은 매우 효율적인 방법입니다.<br>
- 이러한 "밀집된(dense) 특징 벡터(dense feature vectors)"는 공간과 계산 시간 측면에서 매우 효율적입니다. 특히, GPU를 사용하여 연산을 가속화할 수 있습니다.
2.	맵(map) 표현 방식:
-	희소(sparse)한 특징에 대해서는 맵(map) 표현 방식이 유용합니다.<br>
-	맵 표현 방식은 각 특징의 이름(name)과 값을 매핑하여 저장합니다.<br>
-	위의 예시에서는 'fracOfAlpha'라는 이름과 0.85라는 값을 매핑하여 저장하고, 'contains_@'라는 이름과 1이라는 값을 매핑하여 저장합니다.<br>
- 맵 방식은 대부분의 값이 0인 "희소한(sparse) 특징 벡터(sparse feature vectors)"를 표현하기에 좋습니다.<br>
- 맵 방식은 특징 이름(name)과 값(value)을 매핑하여 저장합니다. 만약 맵에 없는 특징이 주어지면, 해당 특징은 기본값(default value)인 0으로 가정합니다.
- 예를 들어, "example string"이라는 문자열이 주어졌을 때, "ends with last 3 characters"라는 특징을 맵 방식으로 표현하면 {"endsWith_ing": 1}과 같습니다.<br>
이때, 'endsWith_'와 문자열의 마지막 세 글자를 합쳐서 이름으로 사용하고, 값을 1로 설정합니다.
- 맵 방식은 배열 방식과 비교하여 공간적인 오버헤드(overhead)가 있지만, 희소한 데이터를 다룰 때에는 맵 방식이 더 효율적입니다.<br>
따라서, 특징 벡터가 얼마나 희소한지에 따라 적절한 방법을 선택하여 사용해야 합니다.<br>

두 방식 모두 특징을 저장하고 표현하는 방식이 다르지만, 각각의 장단점이 있습니다.<br> 
밀집된 특징은 배열 방식으로 저장하면 메모리 사용이 효율적이며, 빠른 계산이 가능합니다.<br> 
반면에 희소한 특징은 맵 방식으로 저장하면 메모리 사용이 효율적이며, 데이터 전처리 과정에서 데이터의 크기가 크게 줄어듭니다.<br>

이와 같은 특징의 표현 방식은 머신러닝 알고리즘에서 매우 중요합니다.<br>
특히, 데이터가 매우 희소한 경우(대부분의 값이 0인 경우)에는 맵 표현 방식을 사용하는 것이 좋습니다.<br>

## Handling Nonlinearity with Linear Methods
### Example Task: Predicting Health
예시를 한 번 들어보겠습니다.<br>
일반적으로 우리는 의료 진단용으로 가능성이 있는 모든 feature들을 추출합니다.<br>
- ex) height, weight, body temperature, blood pressure, etc...

### Issues for Linear Predictors
선형 예측 모델에서, 특징(feature)이 어떻게 추가되는지가 중요합니다.<br>
문제를 일으킬 수 있는 비선형성(nonlinearity)의 세 가지 유형은 다음과 같습니다:<br>

1. **단조성이 아님(non-monotonicity)**
2. **포화(saturation)**
3. **특징 간 상호작용(interactions between features)**

이러한 비선형성(nonlinearities)이 있을 경우, 모델이 예측을 위해 사용하는 특징(feature)들의 중요도나 가중치를 잘못 판단할 수 있습니다.<br>
이는 모델의 정확성과 일반화 능력을 떨어뜨릴 수 있습니다.<br>
1. **단조성이 아닌(non-monotonic)** 특징은 특정 값의 범위에서는 결과에 부정적인 영향을 끼칠 수 있지만 다른 값의 범위에서는 결과에 긍정적인 영향을 끼칠 수 있습니다.<br>
단조적(monotonic)이라는 용어는 기울기가 항상 증가하거나 항상 감소하는 선형 관계를 의미하는 것은 아닙니다.<br>
단조적이라는 말은 단지 변수 간의 관계가 항상 같은 방향으로 움직인다는 것을 의미합니다.<br> 예를 들어, x가 증가할 때 y도 항상 증가하면 x와 y 사이의 관계는 단조적입니다.<br> 하지만 x가 증가할 때 y가 먼저 감소하다가 다시 증가한다면 이는 비단조적인 관계입니다.
따라서, Non-monotonicity 문제는 변수 간의 관계가 비선형이거나 불규칙한 경우에도 발생할 수 있습니다.<br>
예를 들어, 당신이 주식 투자를 한다고 가정해 봅시다.<br>
당신은 어떤 회사의 주식 가격이 상승할 것이라는 예측을 하고 그 주식을 매수합니다.<br> 그러나 나중에 그 회사에서 부정한 일이 드러나면, 그 주식 가격은 급격히 하락할 것입니다.<br> 이 때, 당신의 예측이 잘못되어서 이 주식을 매수한 것이므로, 이 문제는 Non-monotonicity 문제입니다.<br> 
이는 변수인 "예측"과 "실제 주식 가격" 간의 비선형적이고 불규칙한 관계 때문에 발생한 것입니다.<br>
이 경우 모델은 이러한 특징의 중요도를 적절하게 평가하기 어려울 수 있습니다.<br>
Non-monotonicity가 발생하는 경우에는 변수 간에 선형 관계가 아닌, 비선형 관계가 존재하거나, 우리가 알지 못하는 다른 변수가 영향을 미치는 경우가 있을 수 있습니다.<br> 이런 경우에는 적절한 feature engineering이나 다른 기술적인 방법을 사용하여 모델을 개선해야 합니다.

### Non-monotonicity: The Issue
- Feature Map: $φ(x) = [1,temperature(x)]$
- Action: Predict health score $y ∈ R$ (positive is good)
- Hypothesis Space $F = {affine\ functions\ of\ temperature}$

이 예시에서는 온도(temperature) 값을 이용하여 건강 점수$(y)$를 예측하는 모델을 고려합니다.<br>
모델은 온도 값의 선형 함수로 구성되며, 이를 수식으로 나타내면 $φ(x) = [1,temperature(x)]$로 표현됩니다.<br>

여기서 $φ(x)$는 입력 $x$의 특징 벡터(feature vector)를 나타내며, $[1,temperature(x)]$는 상수 1과 입력 x의 온도 값으로 이루어져 있습니다.<br>
그러나 건강 점수는 온도의 선형 함수가 아닙니다.<br>
따라서 온도와 건강 점수 사이에는 선형 관계가 존재하지 않습니다.<br>
이 모델에서 선형 함수를 사용하면, 높은 온도와 낮은 온도 모두 건강에 좋지 않은 영향을 미친다는 문제가 발생합니다.<br>
선형 함수는 높은 온도가 나쁘다고 하면, 낮은 온도는 좋다고 말할 수밖에 없기 때문입니다.<br>

하지만 이 경우에는 낮은 온도와 높은 온도 모두 건강에 나쁜 영향을 미치는 것입니다.<br>
따라서 이러한 문제를 해결하려면 선형 함수 대신에 비선형 함수를 사용해야 합니다.<br>
이 예시에서는 온도와 건강 점수 사이에 비선형 관계가 존재하므로, 선형 함수로는 이를 모델링할 수 없습니다.<br>

### Non-monotonicity: Solution 1
- Transform the input: $φ(x) = [1,{temperature(x)-37}^2]$ <br>

위의 예시에서는 체온(temperature)이라는 feature로부터 건강 점수(health score)를 예측하는 문제를 다루고 있습니다.<br>
그러나 건강 점수는 체온의 선형 함수로는 정확하게 예측할 수 없는 경우였습니다.<br> 따라서 이 문제를 해결하기 위해 체온의 변형을 feature로 사용하는 것이 제안되었습니다.<br>

이를 위해 입력 $x$를 변형하여 $φ(x) = [1, (temperature(x) - 37)^2]$ 로 만들었습니다.<br> 이제 건강 점수 $y$는 $φ(x)$를 사용하여 예측됩니다.<br>
그러나 이러한 입력 변형은 전문 지식이 필요한 경우가 있습니다.<br>

예를 들어, 위의 예시에서는 정상 체온이 37℃인 것을 알아야 합니다.<br>
따라서 이러한 변형은 도메인 전문가의 도움이 필요합니다.<br>
이러한 변형을 사용하지 않고도 모델을 학습할 수 있으나, 때로는 입력에 대한 도메인 지식이 모델의 예측 능력을 향상시키는 데 도움이 될 수 있습니다.<br>

### Non-monotonicity: Solution 2

- $φ(x) = [1,temperature(x) ,(temperature(x))^2]$

기존에 가지고 있는 feature들을 최대한 활용해서 solution1보다 표현력이 뛰어나게끔 만들어준 mapping 함수입니다.<br>
이 경우, 예시로 들어진 feature인 온도(temperature)를 그대로 사용하지 않고 온도의 제곱도 함께 feature로 추가하는 것입니다.<br>
이렇게 하면 모델이 더 복잡한 패턴도 학습할 수 있게 됩니다.<br>

일반적으로 좋은 feature는 간단하면서도 중요한 정보를 포함하는 것입니다.<br>
이러한 feature들을 조합하여 모델을 만들면 더 복잡한 패턴을 학습할 수 있습니다.<br>
따라서 feature는 단순하고 각 feature가 독립적으로 중요한 정보를 제공하는 것이 좋습니다.<br>

> 비선형성이 있는 경우에도 feature extraction이 적용이 되는지?<br>
  비선형성이 있는 경우에도 feature extraction은 적용됩니다. 사실, 비선형성이 있는 경우에 feature extraction이 더욱 중요해집니다.<br>
  비선형성이 있으면, 입력과 출력 간의 관계가 복잡하게 되어서, 이를 표현하기 위해 더 많은 feature가 필요합니다.<br>
  이를 위해, 보다 복잡한 feature extraction 기법이 필요할 수 있습니다.<br>
  예를 들어, 비선형 관계가 있는 경우, 다항식 특성(polynomial features)을 추가할 수 있습니다.<br>
  이를 통해 비선형성을 나타내는 새로운 feature를 만들어 내어 모델의 표현력을 높일 수 있습니다.<br>
  또한, 신경망 같은 비선형 모델에서는 더 복잡한 feature extraction 기법이 사용될 수 있습니다.<br
  
2. **포화(saturation)** 가 발생하는 특징은 값이 일정 범위를 넘어설 경우, 결과에 더 이상 긍정적인 영향을 끼치지 않는 경향이 있습니다.<br>
Saturation은 입력값이 일정 수준 이상이 되면 출력값이 더 이상 증가하지 않는 현상을 말합니다.<br>
 예를 들어, 어떤 자전거 브레이크의 제동력이 브레이크 패드와 바닥 사이의 마찰 계수에 비례한다고 가정해보겠습니다.<br>
 그렇다면, 마찰 계수가 일정 수준 이상이 되면 브레이크 제동력이 더 이상 증가하지 않을 것입니다.<br>
 이러한 경우에는 선형 모델이나 로지스틱 회귀 모델 같은 단순한 모델을 사용하면 제대로된 결과를 얻을 수 없습니다.<br>
 이런 경우에는 적절한 모델링 기법이나 feature engineering 기술을 사용하여 모델을 개선해야 합니다.<br>
이 경우 모델은 특정 값 이상의 특징을 무시하거나, 잘못된 가중치를 부여할 수 있습니다.<br>

### Saturation: The Issue
Setting: 사용자의 검색어와 관련된 제품을 찾습니다.<br>
Input: Product $X$<br>
Action: 


3. **특징 간 상호작용(interactions between features)** 이 있을 경우, 두 개 이상의 특징이 함께 사용될 때 결과가 예상과 다를 수 있습니다.<br> 
Interactions between features는 두 개 이상의 feature가 함께 사용될 때 예측값에 영향을 미치는 경우를 의미합니다.<br>
예를 들어, 성별과 키 두 가지 feature가 있다고 가정해보겠습니다.<br>
만약 성별에 따라 키에 대한 평균값이 다르다면, 성별과 키 두 가지 feature가 함께 사용될 때 예측값에 영향을 미치게 됩니다.<br>
이러한 경우, 두 feature를 모두 고려해야 더 정확한 예측값을 얻을 수 있습니다.<br>
하지만 때로는 두 feature가 함께 사용될 때 예측값에 영향을 미치지 않는 경우도 있습니다.<br>
예를 들어, 온도와 강수량이라는 두 feature가 있다고 가정해보겠습니다.<br> 온도가 높아지면서 강수량이 많아진다고 해도, 강수량이 높은 경우 온도가 높아질 수도 있고 낮아질 수도 있기 때문에 두 feature가 함께 사용될 때 예측값에 큰 영향을 미치지 않을 수 있습니다.<br>
Interactions between features는 feature engineering 과정에서 매우 중요합니다.<br> 모델이 두 feature 간의 interaction을 고려하지 않는다면, 모델이 예측할 수 있는 범위가 한정될 수 있습니다.<br>
따라서 feature engineering 과정에서는 모든 가능한 interaction을 고려하여 예측값에 미치는 영향을 파악하고, 이를 모델에 반영하여 보다 정확한 예측을 할 수 있도록 해야 합니다.<br>
교호작용은 하나의 feature가 반영하는 정보가 다른 feature들에 따라서 영향을 받는 경우를 의미합니다.<br>
예를 들어, 날씨와 교통량이라는 두 feature가 있을 때, 날씨가 좋을 때는 교통량이 많아지는 경향이 있고, 날씨가 나쁠 때는 교통량이 감소하는 경향이 있다면, 날씨와 교통량은 교호작용을 가지고 있다고 할 수 있습니다.<br>
교호작용은 모델의 성능을 높이기 위해서 중요한 정보이며, feature engineering 단계에서 교호작용을 고려하여 새로운 feature를 만들어내는 것이 일반적입니다.<br>
이 경우 모델은 이러한 상호작용을 적절하게 고려하지 않을 경우 예측 결과가 부정확해질 수 있습니다.<br>








[*[출처] : FOUNDATIONS OF MACHINE LEARNING by Bloomberg ML EDU*](https://bloomberg.github.io/foml/#home).