---
title: Feature Extraction 1st
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
## Feature Extraction
### The Input Space $$X$$ 
일반 적으로 input space $$X$$에 대해서는 가정이 없습니다.<br>
그러나 우리가 개발한 구체적인 방법들에 대해서는 input space $$X=R^d$$ 차원의 실수 공간입니다.<br>
- ex) Ridge regression, Lasso regression, Linear SVM<br>

기본적으로 실수차원의 input을 사용하지 않는 경우가 많습니다.<br>
- ex)Text documents, Image files, Sound recordings, DNA sequences<br>

컴퓨터 안 모든 것은 숫자의 연속입니다.<br>
따라서 각 시퀀스의 $$i$$번째 항목은 동일한 의미를 가져야합니다.(colum별로 동일한 의미)<br>
그리고 모든 시퀀스의 길이가 같아합니다.<br>

### Feature Extraction
$$X$$에서 $$R^d$$차원의 벡터로 매핑시키는 것을 feature extraction or featurization 라고 부릅니다.<br>
Raw data에서 feature를 뽑아내서 컴퓨터가 해당 data에서의 feature를 읽을 수 있게끔 하기 위해 vector화 시켜줍니다.<br>
- ex) 이미지에서 특징을 추출해 특징들을 정의할 수 있습니다.<br>

![image](https://user-images.githubusercontent.com/108461006/220524902-f9958b06-8644-41cf-ba0f-7d54a59ef67d.png)

## Feature Templates
### Example: Detecting Email Addresses

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

[*[출처] : FOUNDATIONS OF MACHINE LEARNING by Bloomberg ML EDU*](https://bloomberg.github.io/foml/#home).
