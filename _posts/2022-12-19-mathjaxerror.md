---
title: Mathjax 수식 입력 시 렌더링이 안될 경우
description: Error, Github blog
post-image: https://user-images.githubusercontent.com/79881119/235353418-c053793a-554e-41b8-a147-7310650173cb.png
category: github blog
use_math: true
tags:
- blog
- mathjax
- error
- rendering
---

# 수식 입력 시 오류 발생

예전에 작성했던 것과 같이 현재 블로그에는 Mathjax를 기반으로 수식을 입력하고 있다. 인라인 수식 입력 같은 경우에는 달러 싸인을 두 개 사용하고, 수학 공식 블록의 경우에는 역슬래쉬 + 대괄호를 사용하고 있다. 예를 들어, 다음과 같이 작성하면 잘 나오는 것을 볼 수 있다.

```
$\bar x = \underset{x}\arg \min \sum_{i=1}^{n} \left( x - x_i \right)$
\[
    \bar x = \underset{x}\arg \min \sum_{i=1}^{n} \left( x - x_i \right)

\]
```
---

$\bar x = \underset{x}\arg \min \sum_{i=1}^{n} \left( x - x_i \right)$
\[
    \bar x = \underset{x}\arg \min \sum_{i=1}^{n} \left( x - x_i \right)
\]

---

그런데 가끔 수식 렌더링이 안되는 경우가 생겼다. 사실 그전에는 어느 부분에서 오류가 생기는지 몰라서 계속 수식 notation 일부를 고치거나 그랬는데, 다음과 같은 경우 이슈가 발생하는 걸 확인할 수 있었다.

```
$\mathcal{L}_{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y_i})^2$
\[
    \mathcal{L}_{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y_i})^2
\]
```
---

$\mathcal{L}_{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y_i})^2$
\[
    \mathcal{L}_{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y_i})^2
\]

---
이는 보통 mathcal과 같이 폰트 변경이나 시그마 다음에 나오는 <U>아래첨자 '_'가 이테릭체가 아니라 아래 첨자임을</U> 확실하게 알지 못하기 때문에 렌더링에 실패하는 것으로 확인되었다. 따라서 모든 아래 첨자에 ```\_``` 를 사용하게 되면 아무 문제 없이 출력되는 것을 알 수 있다.

```
$\mathcal{L}\_{MSE} = \frac{1}{N} \sum \_{i=1}^N (y\_i - \hat{y\_ i})^2$
\[
    \mathcal{L}\_{MSE} = \frac{1}{N} \sum \_{i=1}^N (y\_i - \hat{y\_ i})^2
\]
```

---

$\mathcal{L}\_{MSE} = \frac{1}{N} \sum \_{i=1}^N (y\_i - \hat{y\_ i})^2$
\[
    \mathcal{L}\_{MSE} = \frac{1}{N} \sum \_{i=1}^N (y\_i - \hat{y\_ i})^2
\]

---