---
title:  "[Optimization] 02 Lagrange Dual Problem"
excerpt: "Optimization 02 Lagrange Dual Problem. "

categories:
  - op
tags:
  - [op, data science]

use_math: true
toc: true
toc_sticky: true
 
date: 2023-02-17
last_modified_at: 2023-02-17

--- 

# Lagrange Dual Problem
## Primal Problem
Primal 문제는 주어진 함수 f(x)에 대해 최소 또는 최대값을 구하는 문제로, 아래와 같이 수식으로 표현할 수 있습니다.</br>
"원래" 해결하고자 했던 문제라는 뜻에서 이를 **primal problem**이라 합니다.

$minimize\ or\ maximize\quad f(x)$</br>
$subject\ to\quad gᵢ(x) ≤ 0, i=1,...,m$</br>
$\qquad\quad\quad\quad\ hⱼ(x) = 0, j=1,...,p$

위 식에서 $g(x)$와 $h(x)$는 각각 부등식과 등식 제약조건입니다.</br> 
이러한 문제에서 라그랑주 함수 $L(x,λ,μ)$는 다음과 같이 정의됩니다.

$L(x,λ,μ) = f(x) + \sum_{i=1}^mλᵢgᵢ(x) + ∑_{j=1}^rμⱼhⱼ(x)$</br>
$where\quad \lambda\in\mathbb{R}^m,\mu\in\mathbb{R}^r$

여기서 $λ$와 $μ$는 각각 부등식 제약조건과 등식 제약조건에 대한 라그랑주 승수(Lagrange multipliers)입니다.</br>
이 때, primal 문제의 최적해를 찾기 위해서는 라그랑주 함수 $L(x,λ,μ)$를 미분하여 최적화 문제를 해결할 수 있습니다.</br> 
즉, $L(x,λ,μ)$를 $x, λ, μ$에 대해 각각 미분하여 0으로 만드는 해를 찾으면 됩니다. 이렇게 찾은 해를 primal 문제의 최적해라고 합니다.
