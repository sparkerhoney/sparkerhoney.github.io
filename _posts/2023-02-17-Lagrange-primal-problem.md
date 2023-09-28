---
layout: post
title: Lagrange Multiplier Method 2nd
description: Lagrange Multiplier Method 2nd
summary: 
tags: statistics
minute: 1
---

# Lagrange primal Problem
## Primal Problem
Primal 문제는 주어진 함수 f(x)에 대해 최소 또는 최대값을 구하는 문제로, 아래와 같이 수식으로 표현할 수 있습니다.
"원래" 해결하고자 했던 문제라는 뜻에서 이를 **primal problem**이라 합니다.

$minimize\ or\ maximize\quad f(x)$

$subject\ to\quad gᵢ(x) = 0, i=1,...,m$

$\qquad\quad\quad\quad\ hⱼ(x) ≤ 0, j=1,...,r$

위 식에서 $g(x)$와 $h(x)$는 각각 부등식과 등식 제약조건입니다.<br>
이러한 문제에서 라그랑주 함수 $L(x,λ,μ)$는 다음과 같이 정의됩니다.<br>

$L(x,λ,μ) = f(x) + \sum_{i=1}^mλᵢgᵢ(x) + ∑_{j=1}^rμⱼhⱼ(x)$

$where\quad \lambda\in\mathbb{R}^m,\mu\in\mathbb{R}^r,μ_i>0\ for\ all\ i$

### Why always μ>0?
>라그랑주 승수법에서는 일반적으로 부등식 제약조건이 $h_j(x) \le 0$ 형태로 주어지는 경우가 많습니다.<br> 
이 경우 라그랑주 승수를 도입할 때 $h_j(x)$ 앞에 음수 부호를 붙여서 $-h_j(x) \ge 0$ 형태로 바꾸어 주는 것이 일반적입니다.<br> 따라서 라그랑주 함수는 다음과 같이 정의됩니다.<br>

>$L(x,\lambda,\mu) = f(x) + \sum_{i=1}^m \lambda_i g_i(x) + \sum_{j=1}^r \mu_j (-h_j(x))$

>여기서 $\mu_j$는 $-h_j(x)$ 앞에 라그랑주 승수를 붙인 것으로, 부등식 제약조건에서는 보통 $h_j(x) \le 0$이므로 $\mu_j$는 양수일 수 있습니다.<br> 
그러나 때로는 부등식 제약조건이 $h_j(x) \ge 0$ 형태로 주어지는 경우도 있습니다.<br> 
이 경우에는 $h_j(x)$ 앞에 음수 부호를 붙여서 $-h_j(x) \le 0$ 형태로 바꾸어 주어야 합니다.<br> 
따라서 라그랑주 함수는 다음과 같이 정의됩니다.<br>

>$L(x,\lambda,\mu) = f(x) + \sum_{i=1}^m \lambda_i g_i(x) + \sum_{j=1}^r \mu_j (-h_j(x))$

>여기서 $\mu_j$는 $-h_j(x)$ 앞에 라그랑주 승수를 붙인 것으로, 부등식 제약조건에서는 보통 $h_j(x) \ge 0$이므로 $\mu_j$는 양수가 아닌 $0$ 이상의 값이어야 합니다.<br> 
이는 라그랑주 승수법을 적용하기 위해 사용되는 KKT(Karush-Kuhn-Tucker) 조건 중 하나입니다.<br> 
따라서 $u_i > 0$인 이유는 부등식 제약조건에서 음수 부호를 붙이기 위해 $\mu_j$가 양수이거나 $0$ 이상이어야 하기 때문입니다.<br>

여기서 $λ$와 $μ$는 각각 부등식 제약조건과 등식 제약조건에 대한 라그랑주 승수(Lagrange multipliers)입니다.</br>
이 때, primal 문제의 최적해를 찾기 위해서는 라그랑주 함수 $L(x,λ,μ)$를 미분하여 최적화 문제를 해결할 수 있습니다.<br>

즉, $L(x,λ,μ)$를 $x, λ, μ$에 대해 각각 미분하여 0으로 만드는 해를 찾으면 됩니다. 이렇게 찾은 해를 primal 문제의 최적해라고 합니다.

## Lagrange multiplier의 성질
위에서 설명한 함수 $L(x,λ,μ)$는 다음과 같은 중요한 성질이 존재합니다.

$f(x)\ge L(x,λ,μ),\quad at\ each\ feasible\ x$

이 때 우리가 최적화 하려고 하는 목적함수$f$가 어떤 input data $x$에서 **feasible**(실현가능)하다는 것은 "제약조건(constraint)을 만족한다는 뜻".<br>
즉, 다시 말해서 *"최적화 문제의 해가 될 수 있는"*$x$라는 의미입니다.

primal problem의 제약조건을 만족하는 모든 $x$에 대해 위 식이 성립한다는 의미입니다.<br>
공식 상에서는 해당 $x$가 최소, 최대가 될 가능성이 있는 지점 즉 임계점이라고 설명하고 있습니다.<br>
여기서 $f(x)$는 최적화 문제에서 최소화하거나 최대화하고자 하는 함수입니다.

### why always feasible?
>이때, $f(x) \ge L(x,\lambda,\mu)$라는 부등식은 모든 가능한 $x$에 대해 성립해야 합니다.<br>
즉, $g_i(x)$와 $h_j(x)$를 만족시키는 모든 $x$에 대해서 $f(x)$는 $L(x,\lambda,\mu)$보다 크거나 같아야 합니다.<br>
따라서 이 부등식을 만족하는 $x$를 찾는 것이 제약조건이 있는 최적화 문제에서 라그랑주 승수법을 이용하여 최적해를 구하는 핵심적인 과정 중 하나입니다.<br>

위 식이 성립하는 이유는 아래의 식으로 해결이 가능합니다.<br>
feasible한 $x$는 다음을 만족합니다.<br>

$gᵢ(x) = 0, i=1,...,m$

$hⱼ(x) \le 0, j=1,...,r$

이를 $L(x,u,v)$의 식에 대입하면, 위에서 말한 부등식을 얻을 수 있습니다.<br>
위에서 정의한 $L(x,u,v)$에서 $u$가 모든 $i$에 대해서 $u_i>0$을 만족하독한 이유가 바로 아래의 부등식을 만족하게 하기 위함입니다.<br>

$L(x,u,v)=f(x) + \sum_{i=1}^mλᵢgᵢ(x) + ∑_{j=1}^rμⱼhⱼ(x)\le f(x)\quad at\ each\ feasible\ x$

이때 feasible한 $x$를 모두 모은 집합을 $C$라고 하고, primal problem의 optimal value를 $f^*$라고 하면 아래의 등식을 만족하게 된다.<br>

$f^*=min_{x\in C}f(x)$

다음과 같이 L(x,u,v)를 $x$에 대해서 최소화하면, $f$의 optimal value $f^*$에 대한 다음과 같은 부등식을 얻을 수 있다.<br>

또한,우리는 이를 통해 $u_i>0\ for\ all\ i$를 만족하는 $u,v$에 대해서 정의된 함수$g(u,v)$를 정의한다.<br>

$f^*=min_{x\in C}f(x)\ge min_{x\in C}L(x,u,v)\ge min_{x}L(x,u,v):=g(u,v)$

이와 같이 정의된 함수$g(u,v)$를 Lagrange dual function이라고 한다.<br>

$g(u,v)=min_{x}L(x,u,v)$

$where u\in \mathbb{R^m},v\in \mathbb{R^r},u_i>0\ for\ all\ i$