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
# What is Command R?
Command R 모델은 Cohere가 개발한 대규모 언어 모델로서, 특히 대화형 상호작용과 긴 문맥 작업에 최적화되어 있습니다.<br> 
이 모델은 검색 보강 생성(Retrieval-Augmented Generation, RAG)에서 높은 정밀도를 자랑합니다​​.<br> 
<br> 
Command R은 특히 RAG 작업을 최적화하여 개발되었으며, 다양한 기업용 워크로드에 대응할 수 있는 높은 확장성과 강력한 정확성을 제공합니다.<br> 
이 모델은 복잡한 RAG 워크플로우에 적합하며, 다양한 언어를 지원하여 글로벌 시장에서의 응용 가능성을 넓힙니다.<br> 
<br>

