---
title: Lagrange dual Problem
layout: post
description: Lecture summary
use_math: true
post-image: https://github.com/sparkerhoney/sparkerhoney.github.io/blob/master/_images/optimization.png?raw=true
category: optimization
tags:
- Data Science
- optimization
---

# Lagrange dual Problem
---

## Primal과 Dual의 관계
---

최적화(optimization) 문제에서, 주어진 조건(제약) 하에서 목적함수(objective function)를 최소화 또는 최대화하는 것을 목표로 합니다.<br>
이때, 원래 문제를 **primal problem** 이라고 하며, primal problem에서 파생되는 보조 문제를 **dual problem** 이라고 합니다.<br>
Primal problem과 dual problem은 서로 다른 형태의 최적화 문제를 나타내며, 각 문제의 목표는 다릅니다.<br>
Primal problem에서는 목적함수를 최소화 또는 최대화하는 것이 목표이며, dual problem에서는 원래 문제의 제약 조건(constraints)을 이용하여 새로운 목적함수를 정의하고, 이를 최소화 또는 최대화하는 것이 목표입니다.<br>

또한, primal problem과 dual problem은 서로 관련이 있습니다.<br>
예를 들어, primal problem에서 최소화해야 하는 목적함수 값은 dual problem에서 최대화해야 하는 목적함수 값의 하한(lower bound)이 됩니다.<br>

Primal problem에서 최소화해야 하는 목적함수 값을 $p^`$라 하고, dual problem에서 최대화해야 하는 목적함수 값을 $d^`$라 하면, 다음과 같은 부등식이 성립합니다.<br>

$$p^` \ge d^`$$<br>

이 부등식은 "primal problem에서 최소화해야 하는 목적함수 값은 dual problem에서 최대화해야 하는 목적함수 값의 하한(lower bound)이 된다"는 의미를 가집니다.<br>
이 부등식은 strong duality condition이 성립할 때에만 성립하며, 일반적으로는 성립하지 않을 수 있습니다.<br>
strong duality condition이란, primal problem과 dual problem이 모두 최적해를 가질 때 성립하는 조건을 말합니다.<br>

Dual problem에서 최대화해야 하는 목적함수 값은 primal problem에서 최소화해야 하는 목적함수 값의 상한(upper bound)이 됩니다. 이러한 상하한은 각 문제에서 도출된 결과의 유효성을 판단하는 데 유용하게 사용됩니다.<br>

최적화 문제를 푸는 데 있어서, primal problem과 dual problem은 모두 중요한 개념이며, 이를 이해하고 활용하는 것이 최적화 문제 해결의 핵심입니다.<br>

---

[*출처 : FOUNDATIONS OF MACHINE LEARNING by Bloomberg ML EDU*](https://bloomberg.github.io/foml/#home).
