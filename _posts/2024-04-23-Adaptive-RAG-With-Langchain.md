---
title: Command R+을 사용한 Adaptive RAG을 Langchain을 활용해서 구현하기
layout: post
description: anti-hallucination와 relevant answers를 위한 Adaptive RAG을 구현하기 위해 Command R+을 사용하고, Langchain을 활용하여 구현하는 방법을 소개합니다.
use_math: true
post-image: https://media.licdn.com/dms/image/D5622AQFUuGWD2M6bEw/feedshare-shrink_2048_1536/0/1712254525802?e=1717027200&v=beta&t=o3w2wg2A_F45hW0fxlIDh6qivP1kypijipsCAXRmur4
category: nlp
tags:
- NLP
- LLM
- RAG
---

# 0. Overview
## What is Command R+?
[**Command R+**](https://huggingface.co/CohereForAI/c4ai-command-r-plus) 모델은 Cohere가 개발한 대규모 언어 모델로서, 특히 대화형 상호작용과 긴 문맥 작업에 최적화되어 있습니다.<br> 
이 모델은 검색 보강 생성(Retrieval-Augmented Generation, RAG)에서 높은 정밀도를 자랑합니다​​.<br> 
<br> 
Command R+은 특히 RAG 작업을 최적화하여 개발되었으며, 다양한 기업용 워크로드에 대응할 수 있는 높은 확장성과 강력한 정확성을 제공합니다.<br> 
이 모델은 복잡한 RAG 워크플로우에 적합하며, 다양한 언어를 지원하여 글로벌 시장에서의 응용 가능성을 넓힙니다.<br> 
<br>
Command R+ 모델은 영어, 프랑스어, 스페인어, 이탈리아어, 독일어, 브라질 포르투갈어, 일본어, 한국어, 중국어(간체), 아랍어 등 다양한 언어에서 높은 정확도로 작동하도록 훈련되었습니다.<br>
<br>

## What is Adaptive-RAG?
[**Adaptive-RAG 논문**](https://arxiv.org/abs/2403.14403)은 질문의 복잡성을 통해 검색 기능을 통합한 대규모 언어 모델을 적응시키는 방법을 배우는 기술에 대해 다루고 있습니다.<br> 
RAG(Retrieval-Augmented Generation) 모델은 기존의 언어 모델에 정보 검색 기능을 결합하여 특정 질문에 대한 보다 정확하고 상세한 답변을 생성할 수 있도록 합니다.<br>
<br>
"Adaptive-RAG" 논문에서 제안하는 아키텍처는 기본적으로 다음과 같이 구성됩니다:<br>

1. 질문 복잡성 분석: 먼저, 시스템은 입력된 질문의 복잡성을 평가합니다.<br> 
이는 문장의 길이, 사용된 어휘, 구문의 복잡도, 주제의 특수성 등 여러 요소를 분석하여 결정될 수 있습니다.<br> 
이 단계에서 질문이 간단한지, 아니면 더 깊은 정보 검색을 요구하는지를 판단합니다.<br>
2. 적응형 검색 모듈: 복잡성 평가에 따라 시스템은 검색 모듈을 동적으로 조정합니다.<br> 
간단한 질문의 경우, 시스템은 빠르고 효율적인 검색을 수행할 수 있는 간단한 데이터베이스나 캐시된 정보를 사용할 수 있습니다.<br> 
반면에 더 복잡한 질문에는 더 광범위하고 상세한 정보가 필요할 수 있으므로, 시스템은 더 크고 다양한 데이터 소스에 접근할 수 있도록 조정됩니다.<br>
3. RAG (Retrieval-Augmented Generation) 구조: 기본적으로 RAG 모델은 질문에 대한 답을 생성하기 전에 관련 정보를 검색하는 두 가지 주요 구성요소로 이루어져 있습니다.<br> 
첫 번째는 검색기(retriever)로, 관련 정보를 데이터베이스에서 찾아내는 역할을 합니다.<br> 
두 번째는 생성기(generator)로, 검색된 정보를 바탕으로 사용자 질문에 대한 답변을 생성합니다.<br>
4. 학습과 적응: 모델은 다양한 질문과 그에 따른 정보 요구 사항에 맞게 학습하여 자신의 검색 전략을 지속적으로 개선하고 최적화합니다.<br> 
이 과정에서 기계 학습 알고리즘, 특히 강화 학습이나 메타 학습 기법이 사용될 수 있습니다.<br>

이러한 아키텍처를 통해, Adaptive-RAG는 다양한 복잡성을 가진 질문에 유연하게 대응할 수 있으며, 사용자에게 더 정확하고 만족스러운 답변을 제공할 가능성을 높입니다.<br>

이 논문에서는 특히, 질문의 복잡성을 판단하고 그에 맞추어 모델이 스스로 검색 방법을 조정하도록 하는 '적응형' 접근 방식을 제안합니다.<br> 예를 들어, 간단한 질문은 기본적인 정보만을 검색하여 처리할 수 있는 반면, 더 복잡하고 깊이 있는 질문은 더 넓은 범위의 데이터베이스나 문헌에서 정보를 검색해야 할 수도 있습니다.<br>














