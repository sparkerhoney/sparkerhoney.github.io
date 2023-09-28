---
layout: post
title: Kernel Methods 3rd
description: Kernel Methods 3rd
summary: 
tags: machine learning
minute: 1
---

# Kernel Methods
## Representer Theorem
Representer Theorem은 함수 공간에서 최적화 문제의 해가 필연적으로 명시적인 형태로 표현될 수 있음을 보여주는 이론입니다.<br> 이 이론은 기계학습 분야에서 자주 사용되는 커널 기법과 관련이 있습니다.<br>

주어진 입력과 출력 데이터 쌍 $(\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), \dots, (\mathbf{x}_n, y_n)$이 있을 때, 목표는 일반화 성능을 최대화하는 함수 $f: \mathcal{X} \rightarrow \mathcal{Y}$를 찾는 것입니다.<br> 이때, 일반적으로 $f$를 다음과 같이 표현합니다.<br>


​

여기서 $k(\mathbf{x}_i, \mathbf{x})$는 커널 함수이고, $\alpha_i$는 $i$번째 훈련 데이터의 가중치입니다. Representer theorem은 이러한 함수 $f$가 다음과 같은 형태로 표현될 수 있음을 보여줍니다.


여기서 $\beta_i$는 $k(\mathbf{x}_i, \mathbf{x})$와 $y_i$의 선형 결합으로 나타낼 수 있습니다.

이러한 형태의 함수 $f$를 구하는 것은 단순 선형 회귀와 같은 선형 대수 문제로 풀 수 있으며, 이를 통해 최적의 $\beta_i$를 찾을 수 있습니다. 이러한 해법은 명시적이고 효율적이며, Representer theorem은 이러한 해법이 최적화 문제의 일반적인 해법이 될 수 있음을 보여줍니다.








[*[출처] : FOUNDATIONS OF MACHINE LEARNING by Bloomberg ML EDU*](https://bloomberg.github.io/foml/#home).