---
title: LaTex 자주 쓰는 수식 및 심볼 정리
layout: post
description: LaTex 문법 정리노트
use_math: true
post-image: https://user-images.githubusercontent.com/79881119/228403706-3045f7c3-a935-4a96-b3ad-1936b137daf4.png
category: github blog
tags:
- LeTex
- Markdown
- github blog
---

깃허브 블로그나 레이텍에서 수식을 작성하다보니 모르는 심볼이나 문법들이 많아서 자주 사용할 것들만 모아놓는 용도..

# Font types

```latex
ABCDEF
\mathcal{ABCDEF}
\mathbb{ABCDEF}
\mathbf{ABCDEF}
\mathfrak{ABCDEF}
\mathrm{ABCDEF}
\mathit{ABCDEF}
\mathsf{ABCDEF}
\mathtt{ABCDEF}
```

\[
    \begin{aligned}
        \text{\ABCDEF} & : ABCDEF \newline
        \text{\mathcal{ABCDEF}} & : \mathcal{ABCDEF} \newline
        \text{\mathbb{ABCDEF}} & : \mathbb{ABCDEF} \newline
        \text{\mathbf{ABCDEF}} & : \mathbf{ABCDEF} \newline
        \text{\mathfrak{ABCDEF}} & : \mathfrak{ABCDEF} \newline
        \text{\mathrm{ABCDEF}} & : \mathrm{ABCDEF} \newline
        \text{\mathit{ABCDEF}} & : \mathit{ABCDEF} \newline
        \text{\mathsf{ABCDEF}} & : \mathsf{ABCDEF} \newline
        \text{\mathtt{ABCDEF}} & : \mathtt{ABCDEF} \newline
    \end{aligned}
\]

---

# 연산자 및 관계성 기호(집합 포함)

|LaTex|결과|LaTex|결과|LaTex|결과|LaTex|결과|
|---:|:---|---:|:---|---:|:---|---:|:---|
|```\leq```|$\leq$|```\geq```|$\geq$|```<```|$<$|```>```|$>$|
|```=```|$=$|```\neq```|$\neq$|```\nleq```|$\nleq$|```\ngeq```|$\ngeq$|
|```\cong```|$\cong$|```\equiv```|$\equiv$|```\sim```|$\sim$|```\approx```|$\approx$|
|```\doteqdot```|$\doteqdot$|```\times```|$\times$|```+```|$+$|```-```|$-$|
|```\div```|$\div$|```\cdot```|$\cdot$|```\ast```|$\ast$|```\pm```|$\pm$|
|```\mp```|$\mp$|```\circ```|$\circ$|```\oplus```|$\oplus$|```\otimes```|$\otimes$|
|```\odot```|$\odot$|```\bigcirc```|$\bigcirc$|```\bigoplus```|$\bigoplus$|```\bigotimes```|$\bigotimes$|
|```\bigodot```|$\bigodot$|```\propto```|$\propto$|```\cdots```|$\cdots$|```\dots```|$\dots$|
|```\because```|$\because$|```\therefore```|$\therefore$|```\forall```|$\forall$|```\exists```|$\exists$|
|```\in```|$\in$|```\subset```|$\subset$|```\subseteq```|$\subseteq$|```\notin```|$\notin$|
|```\supset```|$\supset$|```\supseteq```|$\supseteq$|```\subsetneq```|$\subsetneq$|```\supsetneq```|$\supsetneq$|
|```\not\subset```|$\not\subset	$|```\not\supset```|$\not\supset$|```\not\subseteq```|$\not\subseteq$|```\not\supseteq```|$\not\supseteq$|
|```\emptyset```|$\emptyset$|```\varnothing```|$\varnothing$|```\oslash```|$\oslash$|```\cap```|$\cap$|
|```\cup```|$\cup$|```\vert```|$\vert$|```\parallel```|$\parallel$|```\bot```|$\bot$|
|```\top```|$\top$|```\vdots```|$\vdots$|```\ddots```|$\ddots$|```\circ```|$\circ$|
|```\bullet```|$\bullet$|```\neq```|$\neq$|```\wedge```|$\wedge$|```\vee```|$\vee$|
|```\leftarrow```|$\leftarrow$|```\rightarrow```|$\rightarrow$|```\leftrightarrow```|$\leftrightarrow$|```\mapsto```|$\mapsto$|
|```\Leftarrow```|$\Leftarrow$|```\Rightarrow```|$\Rightarrow$|```\Leftrightarrow```|$\Leftrightarrow$|```\leftrightarrows```|$\leftrightarrows$|
|```\prod```|$\prod$|```\sum```|$\sum$|```\int```|$\int$|```\oint_{C}```|$\oint_{C}$|

---

# 글자꼴 기호(소문자 및 대문자)

|LaTex|결과|LaTex|결과|LaTex|결과|LaTex|결과|
|---:|:---|---:|:---|---:|:---|---:|:---|
|```\alpha```|$\alpha$|```\beta```|$\beta$|```\gamma```|$\gamma$|```\delta```|$\delta$|
|```\epsilon```|$\epsilon$|```\zeta```|$\zeta$|```\eta```|$\eta$|```\theta```|$\theta$|
|```\iota```|$\iota$|```\kappa```|$\kappa$|```\lambda```|$\lambda$|```\mu```|$\mu$|
|```\nu```|$\nu$|```\xi```|$\xi$|```\omicron```|$\omicron$|```\pi```|$\pi$|
|```\rho```|$\rho$|```\sigma```|$\sigma$|```\tau```|$\tau$|```\upsilon```|$\upsilon$|
|```\phi```|$\phi$|```\chi```|$\chi$|```\psi```|$\psi$|```\omega```|$\omega$|
|```\Gamma```|$\Gamma$|```\Delta```|$\Delta$|```\Theta```|$\Theta$|```\Lambda```|$\Lambda$|
|```\Xi```|$\Xi$|```\Pi```|$\Pi$|```\Sigma```|$\Sigma$|```\Upsilon```|$\Upsilon$|
|```\Phi```|$\Phi$|```\Psi```|$\Psi$|```\Omega```|$\Omega$|||

---

# 괄호

소괄호, 중괄호 및 대괄호를 사용할 때 주의해야할 점은 중괄호의 경우 escape가 필요하다는 것이다. 예를 들어 소괄호를 사용하고자 한다면 아래와 같이 사용할 수 있다. ```\left```와 ```\right```는 항상 짝을 이루어서 사용해야하며, 그냥 사용할 때와 다르게 내부에 들어가는 수식의 크기에 맞게 조정된다.

```latex
(x, y)
\left( x, y \right)
(\frac{1}{x})
\left( \frac{1}{x} \right)
```

\[  
    \begin{aligned}
    (x, y) \newline
    \left( x, y \right) \newline 
    (\frac{1}{x}) \newline
    \left( \frac{1}{x} \right)
    \end{aligned}
\]

그리고 깃허브 블로그에서는 ```\{```, ```\}```를 사용하면 어쩐 일인지 LaTex escape가 되지 않는 문제가 발생한다. 이럴 때는 ```\\{```, ```\\}```를 사용하면 된다. 앞서 소괄호에서 사용했던 것과 같이 ```\left``` 및 ```\right```를 사용할 수 없기 때문에 이럴 때는 크기에 따라

- ```\bigl```, ```\bigr```
- ```\Bigl```, ```\Bigr```
- ```\biggl```, ```\biggr```
- ```\Biggl```, ```\Biggr```

을 사용할 수 있다. 예컨데 아래와 같이 중괄호에 적용할 수 있다.

```latex
\{x, y\}
\{ \frac{1}{x} \}
\Bigl\{ \frac{1}{x} \Bigr\}
```

\[  
    \begin{aligned}
    \\{x, y \\} \newline
    \\{ \frac{1}{x} \\} \newline
    \Bigl\\{ \frac{1}{x} \Bigr\\}
    \end{aligned}
\]

대괄호도 앞서 사용한 방식과 유사하게 쓸 수 있다.

```latex
[x, y]
\left[ x, y \right]
[\frac{1}{x}]
\left[ \frac{1}{x} \right]
```
<p align="center">
    $[x, y]$
</p>
<p align="center">
    $\left[ x, y \right]$
</p>
<p align="center">
    $[\frac{1}{x}]$
</p>

이외에 이런저런 괄호들을 사용할 수 있다. Latex와 markdown에서 문법이 일부 달라지는 경우도 많은 것으로 보인다.

|LaTex|결과|LaTex|결과|LaTex|결과|LaTex|결과|
|---:|:---|---:|:---|---:|:---|---:|:---|
|```()```|$ () $|```\{ \}```|$ \\{ \\} $|```[]```|$ [] $|```\langle \rangle```|$\langle \rangle$|
|```\| \|```|$\| \|$|```\lvert \rvert```|$\lvert \rvert$|```\bigl\vert \bigr\vert```|$\bigl\vert \bigr\vert$|||

---

# 다양한 함수

|LaTex|결과|LaTex|결과|LaTex|결과|LaTex|결과|
|---:|:---|---:|:---|---:|:---|---:|:---|
|```\sin```|$\sin$|```\cos```|$\cos$|```\tan```|$\tan$|```\exp```|$\exp$|
|```\arcsin```|$\arcsin$|```\arccos```|$\arccos$|```\arctan```|$\arctan$|```\log```|$\log$|
|```\ln```|$\ln$|```\sec```|$\sec$|```\csc```|$\csc$|```\cot```|$\cot$|
|```\sinh```|$\sinh$|```\cosh```|$\cosh$|```\tanh```|$\tanh$|```\coth```|$\coth$|
|```\sgn```|$\sgn$|```\min```|$\min$|```\max```|$\max$|```\arg```|$\arg$|
|```\inf```|$\inf$|```\sup```|$\sup$|```\det```|$\det$|```\lim```|$\lim$|
|```\deg```|$\deg$|```\sqrt{x}```|$\sqrt{x}$|```\sqrt[3]{x}```|$\sqrt[3]{x}$|```\ker```|$\ker$|

만약 LaTex 문법 상 정의되지 않은 함수라면 다음과 같이 써볼 수 있다.

```latex
    \operatorname{Sigmoid}\, (x)
```

\[
    \operatorname{Sigmoid}\, (x)
\]

그리고 만약 ```\arg\max```와 같이 특정 인자에 대한 지칭이 필요할 때는 ```underset```을 사용하여 다음과 같이 쓸 수 있다.

```latex
\hat{z} = \underset{z}{\arg\min}
```

\[
    \hat{z} = \underset{z}{\arg\min}    
\]

```overset```으로는 위에 올릴 수도 있다.

```latex
f(x) \overset{C}{=} g(x)
```

\[
    f(x) \overset{C}{=} g(x)    
\]

---

# 모자 씌우기

|LaTex|결과|LaTex|결과|LaTex|결과|LaTex|결과|
|---:|:---|---:|:---|---:|:---|---:|:---|
|```\dot{x}```|$\dot{x}$|```\ddot{x}```|$\ddot{x}$|```\acute{x}```|$\acute{x}$|```\grave{x}```|$\grave{x}$|
|```\check{x}```|$\check{x}$|```\breve{x}```|$\breve{x}$|```\tilde{x}```|$\tilde{x}$|```\bar{x}```|$\bar{x}$|
|```\hat{x}```|$\hat{x}$|```\widehat{x}```|$\widehat{x}$|```\vec{x}```|$\vec{x}$|``` ```|$ $|

---

# 분수, 이항계수, 행렬, 경우 나누기(case), 여러 줄 쓰기

### 분수

```latex
1 \over x
\frac{1}{x}
```

\[ 
    \frac{1}{x}
\]

### 이항계수

```latex
{n \choose k} = _{n}\mathrm{C}_{k}
```

\[
    {n \choose k} =~ \_{n}\mathrm{C}_{k}
\]

### 행렬

```latex
\begin{matrix}
    x & y \newline z & v
\end{matrix}

\begin{vmatrix}
    x & y \newline z & v
\end{vmatrix}

\begin{pmatrix}
    x & y \newline z & v
\end{pmatrix}

\begin{bmatrix}
    x & y \newline z & v
\end{bmatrix}

```
\[
    \begin{matrix}
        x & y \newline z & v
    \end{matrix} 
\]

\[
    \begin{vmatrix}
        x & y \newline z & v
    \end{vmatrix}
\]

\[
    \begin{pmatrix}
        x & y \newline z & v
    \end{pmatrix}    
\]

\[
    \begin{bmatrix}
        x & y \newline z & v
    \end{bmatrix}    
\]

### 경우 나누기

```latex
\begin{cases}
    x, & x > 0 \newline
    0, & \text{Otherwise}
\end{cases}
```

\[
    \begin{cases}
        x, & x > 0 \newline
        0, & \text{Otherwise}
    \end{cases}
\]

### 여러 줄의 방정식 쓰기

여러 줄의 방정식을 쓸 때 정렬을 맞추고 싶은 부분에 ```&```를 각 줄에 넣어준다.

```latex
\begin{aligned}
    y =& x^2 + x + 2x + 3 -2 \newline
    =& x^2 + 3x +1
\end{aligned}
```

\[
    \begin{aligned}
        y =& x^2 + x + 2x + 3 -2 \newline
        =& x^2 + 3x +1
    \end{aligned}    
\]