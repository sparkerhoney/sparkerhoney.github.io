---
title: Sentence Piece
layout: post
description: paper review
use_math: true
post-image: https://machinelearnit.files.wordpress.com/2019/10/sentencepiece.jpg
category: paper review
tags:
- Data Science
- machine learning
- NLP
- paper review
---
# SentencePiece

안녕하세요!<br>
**SentencePiece**에 대한 내용을 다뤄보려고 합니다.<br>
특히, **SentencePiece**와 다른 토크나이저와의 차이점, 그리고 한국어 토크나이징에 대해 깊게 알아보겠습니다.<br>
<br>
<br>

또한 *코드로 구현한 과정*은 다음과 같습니다.<br>
<br>
<br>

- [Sentence Piece 구현 깃허브](https://github.com/sparkerhoney/Paper-Implementation/tree/main/Sentence%20Piece)

---

## 1. 토크나이저란?

토크나이저는 텍스트 데이터를 작은 단위로 나누는 도구입니다.<br>
이 작은 단위를 '토큰'이라고 부릅니다.<br>
토크나이징은 자연어 처리에서 매우 중요한 전처리 과정 중 하나입니다.<br>

---

## 2. SentencePiece와 다른 토크나이저의 차이

### SentencePiece

- **언어 독립적**: SentencePiece는 언어에 종속적인 로직이 없기 때문에 모든 언어를 지원합니다.<br>
- **알고리즘**: 바이트 페어 인코딩 (BPE) 및 유니그램 언어 모델을 포함합니다.<br>
- **라이선스**: Apache 2.0 라이선스로 제공됩니다.<br>

### 다른 토크나이저

- 대부분의 토크나이저는 특정 언어나 도메인에 최적화된 로직을 포함하고 있습니다.<br>
- 예를 들면, WordPiece, FastText, CharCNN 등 다양한 토크나이저가 있으며, 각각의 특징과 사용법이 다릅니다.<br>

---

## 3. 센텐스피스(SentencePiece)의 특징과 사용법

센텐스피스는 BPE 알고리즘과 Unigram Language Model Tokenizer를 구현한 도구로, 구글에서 제공하고 있습니다.<br>
특히, 사전 토큰화 작업 없이 원시 텍스트 데이터에 바로 서브워드 토크나이징을 수행할 수 있어 언어에 종속되지 않는 큰 장점이 있습니다.<br>

센텐스피스를 사용하기 위해서는 먼저 pip를 통해 설치해야 합니다.<br>

```python
pip install sentencepiece
```

설치 후, 센텐스피스를 사용하여 토크나이징을 수행할 수 있습니다.<br>
예를 들어, IMDB 리뷰 데이터나 네이버 영화 리뷰 데이터를 토큰화하는 과정은 다음과 같습니다.<br>

```python
import sentencepiece as spm

# 모델 학습
spm.SentencePieceTrainer.train('--input=data.txt --model_prefix=m --vocab_size=2000 --model_type=bpe')

# 모델 로드
sp = spm.SentencePieceProcessor()
sp.load('m.model')

# 토크나이징
tokens = sp.encode_as_pieces("안녕하세요. SentencePiece 테스트 중입니다.")
print(tokens)
```
---

## 4. SentencePiece의 깊은 이해

SentencePiece tokenizer는 언어에 무관하게 동작하며, 띄어쓰기 유무에 영향을 받지 않습니다.<br>
또한, 매우 빠르게 동작하며, 기존의 vocab_size를 벗어난 경우 발생하던 [UNK] 토큰의 발생을 크게 줄여줍니다.<br>
이는 "안녕하세요"를 "안녕, 하세요"와 같이 어절 안쪽을 쪼개서 tokenize하기 때문입니다.<br>
그렇다면 이러한 원리는 어떻게 동작하는 것일까요?<br>

### Unigram Language Model

글은 순서를 갖는 sequence이기 때문에 어떠한 문장이든 앞에 나온 단어에 기반하여 뒤에 나올 단어를 유추할 수 있습니다.<br>
예를 들어, "오늘 마라탕 먹어야 하는데 같이 갈 사람 구함"이라는 문장에서 "갈" 뒤에 "사람"이 나올 확률은 "앵무새"보다 높을 것입니다.<br>
이전 몇 개의 토큰을 바탕으로 예측하는지에 따라 N-gram에서 N이 바뀝니다.<br>
이를 N-gram model이라고 합니다.<br>
그러나 unigram model은 맥락을 전혀 신경쓰지 않습니다.<br>
어떠한 문장이 등장할 확률은 그저 전체 말뭉치(Corpus)에서 각 토큰이 등장할 확률을 곱한 것에 불과하며, 이는 토큰의 순서를 고려하지 않습니다.<br>

### Byte Pair Encoding (BPE)

BPE는 말뭉치에서 자주 등장하는 연이은 토큰이 있다면, 그것을 하나의 토큰으로 합쳐버리는 과정입니다.<br>
예를 들어, "house home hostile"로 BPE를 해본 결과, 초기에는 각 글자가 분해되어 있으나, merge 과정을 거치면서 'ho'나 'e_'와 같은 토큰이 합쳐지게 됩니다.<br>

# Training SentencePiece

SentencePiece의 훈련 과정은 Variational inference의 일종입니다.<br>

<!-- 주어진 관측 데이터(Evidence)와 모델 파라미터($θ$)를 바탕으로 가설(Hypothesis)에 대한 분포 ($P$)를 variational parameter를 도입하여 ($Q$)로 근사합니다.<br>

$P(H|E,θ) \approx Q(H|E,λ) = \prod_{i=1}^{|H|} q_i(H_i|λ_i)$<br>

여기서 근사의 목적은 Evidence Lower Bound \( L(λ,θ) \)를 극대화하는 것입니다.<br>

$L(λ,θ) = \sum_H [Q(H|E,λ) \ln P(H,E|θ) - Q(H|E,λ) \ln Q(H|E,λ)]$<br>

각 ($λ_i$)에 대해 차례로 최적화하면, ($j$)번째 variational parameter ($λ_j$)에 대한 극대화 식은 다음과 같습니다:<br>

$L(λ_j) = \sum_H \prod_{i=1}^{|H|} q_i(H_i|E,λ_i) \{ \ln P(H|E,θ) - \sum_{k=1}^{|H|} \ln q_k(H_k|E,λ_k) \}$<br>

이를 정리하면:<br>

$\sum_{H_j} -KL(q_j(H_j|E,λ_j) || \tilde{P}(H,E|θ)) + C'$<br>

여기서 ($\tilde{P}(H,E|θ)$)는 ($E_{q_i \neq j} [\ln P(H,E|θ)] + C$)입니다.<br>

$q_j^*(H_j|E,λ_j) = \tilde{P}(H,E|θ)$<br>
$\ln q_j^*(H_j|E,λ_j) = E_{q_i \neq j} [\ln P(H,E|θ)] + Const.$<br>

---

## SentencePiece의 evidence log-likelihood

숨겨진 변수 ($π$)를 도입하고, *Dirichlet Distribution*을 이용합니다.<br>

$p(π|α) = Dir(π,α_K) = \prod_{k=1}^K π_k^{α_K - 1}$<br>
$p(x|π) = \prod_{n=1}^N \prod_{k=1}^K π_k^{x_{nk}}$<br> -->

<!-- 주어진 관측 데이터(Evidence)와 모델 파라미터($θ$)를 바탕으로 가설(Hypothesis)에 대한 분포 ($P$)를 variational parameter를 도입하여 ($Q$)로 근사합니다.<br>

$$ P(H \mid E, \theta) \approx Q(H \mid E, \lambda) = \prod_{i=1}^{|H|} q_i(H_i \mid \lambda_i) $$<br>

여기서 근사의 목적은 Evidence Lower Bound $(L(₩lambda, ₩theta))$를 극대화하는 것입니다.<br>
<br>
<br>

$$ L(\lambda, \theta) = \sum_H [Q(H \mid E, \lambda) \ln P(H, E \mid \theta) - Q(H \mid E, \lambda) \ln Q(H \mid E, \lambda)] $$

<br>
<br>


각 ($λ_i$)에 대해 차례로 최적화하면, ($j$)번째 variational parameter ($λ_j$)에 대한 극대화 식은 다음과 같습니다:<br>
<br>
<br>

$$ L(\lambda_j) = \sum_H \prod_{i=1}^{|H|} q_i(H_i \mid E, \lambda_i) \left\{ \ln P(H \mid E, \theta) - \sum_{k=1}^{|H|} \ln q_k(H_k \mid E, \lambda_k) \right\} $$

<br>
<br>

이를 정리하면:

<br>
<br>

$$ \sum_{H_j} -KL(q_j(H_j \mid E, \lambda_j) \mid\mid \tilde{P}(H, E \mid \theta)) + C' $$

<br>
<br>

여기서 \( \tilde{P}(H, E \mid \theta) \)는 \( E_{q_{i \neq j}} [\ln P(H, E \mid \theta)] + C \)입니다.<br>

$$ q_j^*(H_j \mid E, \lambda_j) = \tilde{P}(H, E \mid \theta) $$<br>

<br>
<br>

$$ \ln q_j^*(H_j \mid E, \lambda_j) = E_{q_{i \neq j}} [\ln P(H, E \mid \theta)] + \text{Const.} $$<br>

## SentencePiece의 evidence log-likelihood <br>

숨겨진 변수 \( \pi \)를 도입하고, *Dirichlet Distribution*을 이용합니다. <br>
<br>
<br>

$$ p(\pi \mid \alpha) = \text{Dir}(\pi, \alpha_K) = \prod_{k=1}^K \pi_k^{\alpha_K - 1} $$ <br>
$$ p(x \mid \pi) = \prod_{n=1}^N \prod_{k=1}^K \pi_k^{x_{nk}} $$ <br>

<br>
<br>
여기서 ($x_{nk}$)는 sequence에서 ($n$)번째 토큰이 ($k$)번째 unigram인 경우 1, 아니면 0입니다.<br> -->


주어진 관측 데이터(Evidence)와 모델 파라미터 \( \theta \)를 바탕으로 가설(Hypothesis)에 대한 분포 \( P \)를 variational parameter를 도입하여 \( Q \)로 근사합니다.

\[ P(H \mid E, \theta) \approx Q(H \mid E, \lambda) = \prod_{i=1}^{|H|} q_i(H_i \mid \lambda_i) \]

여기서 근사의 목적은 Evidence Lower Bound \( L(\lambda, \theta) \)를 극대화하는 것입니다.

\[ L(\lambda, \theta) = \sum_H [Q(H \mid E, \lambda) \ln P(H, E \mid \theta) - Q(H \mid E, \lambda) \ln Q(H \mid E, \lambda)] \]

각 \( \lambda_i \)에 대해 차례로 최적화하면, \( j \)번째 variational parameter \( \lambda_j \)에 대한 극대화 식은 다음과 같습니다:

\[ L(\lambda_j) = \sum_H \prod_{i=1}^{|H|} q_i(H_i \mid E, \lambda_i) \left\{ \ln P(H \mid E, \theta) - \sum_{k=1}^{|H|} \ln q_k(H_k \mid E, \lambda_k) \right\} \]

이를 정리하면:

\[ \sum_{H_j} -KL(q_j(H_j \mid E, \lambda_j) \mid\mid \tilde{P}(H, E \mid \theta)) + C' \]

여기서 \( \tilde{P}(H, E \mid \theta) \)는 \( E_{q_{i \neq j}} [\ln P(H, E \mid \theta)] + C \)입니다.

\[ q_j^*(H_j \mid E, \lambda_j) = \tilde{P}(H, E \mid \theta) \]
\[ \ln q_j^*(H_j \mid E, \lambda_j) = E_{q_{i \neq j}} [\ln P(H, E \mid \theta)] + \text{Const.} \]

## SentencePiece의 evidence log-likelihood

숨겨진 변수 \( \pi \)를 도입하고, *Dirichlet Distribution*을 이용합니다.

\[ p(\pi \mid \alpha) = \text{Dir}(\pi, \alpha_K) = \prod_{k=1}^K \pi_k^{\alpha_K - 1} \]
\[ p(x \mid \pi) = \prod_{n=1}^N \prod_{k=1}^K \pi_k^{x_{nk}} \]

---



- [센텐스피스 논문](https://arxiv.org/pdf/1808.06226.pdf)<br>
- [센텐스피스 깃허브](https://github.com/google/sentencepiece)<br>
