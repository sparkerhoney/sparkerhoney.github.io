---
title: AWS SageMaker Custom Model Serving(Deployment)
layout: post
description: MLOps
use_math: true
post-image: https://logohistory.net/wp-content/uploads/2023/06/AWS-Emblem.png
category: paper review
tags:
- Data Science
- machine learning
- MLOps
---
# Objective
ML Model을 빠르게 구축, 훈련, 최적화, 배포 할 수 있도록 하는 **완전 관리형** ML Servive.<br>
기존에 Local 환경에서 훈련된 Custom Model을 **AWS SageMaker**에서 배포하는 방법을 작성하려한다.<br>

# 어떤 파일들이 준비 돼 있어야 하나요?
우선, 우리가 하고 싶은 Object에 대해서 생각해보면 명확해지는 부분이다.<br>
우리는 사이즈가 큰 Pretrained ML Model을 우리의 서비스에 배포하려고 한다.<br>
<br>
그럼 우리는 다음과 같은 모델의 구성요소 파일들이 준비 돼 있어야 한다.<br>
<br>

```bash
.
├── config.json
├── generation_config.json
├── special_tokens_map.json
├── tokenizer.json
├── pytorch_model.bin
└── tokenizer_config.json
```
<br>

위와 같은 파일들은 기본적으로 Pretrained Model을 저장할 때 생성되는 파일들이다.<br>
우리는 이러한 파일들을 **Docker Image**로 만들어서 **AWS SageMaker**에 배포할 것이다.<br>
<br>

# Docker Image를 Handling 하는 방법
## Docker Image가 뭔데?
Docker Image는 **Container**를 만들기 위한 Template이다.<br>
그럼 Container는 뭘까? Container는 **OS의 가상화**를 통해 **Application을 실행**하는 방법이다.<br>
즉, Docker Image는 우리가 실행하고자 하는 Application을 실행하기 위한 **모든 설정과 파일**을 포함하고 있는 Template이라고 할 수 있다.<br>
<br>

## Docker Image를 만드는 방법
Docker Image를 만드는 방법 중 가장 대표적인 방법은 **Dockerfile**을 이용하는 방법이다.<br>
Dockerfile은 Docker Image를 만들기 위한 **명령어들의 집합**이라고 할 수 있다.<br>
<br>

```Dockerfile
# Python 3.10 이상 버전 사용
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    nginx \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# PyTorch 2.1.2 및 관련 라이브러리 설치 (CUDA 12.1 지원)
# 이 명령은 종속성이 변경될 가능성이 적기 때문에 여기에 위치합니다.
RUN pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --extra-index-url https://download.pytorch.org/whl/cu121

# Python 의존성 파일 복사
# requirements.txt 파일은 변경될 가능성이 높기 때문에 나중에 복사합니다.
COPY requirements.txt .

# requirements.txt에서 나머지 Python 의존성 설치
# torch, torchvision, torchaudio는 위에서 설치했으므로 requirements.txt에서 제외해야 함
RUN pip install --no-cache-dir -r requirements.txt

# 나머지 필요한 파일 복사
# 이 파일들은 변경될 가능성이 높으므로 마지막에 복사합니다.
COPY inference.py .
COPY config.json .
COPY special_tokens_map.json .
COPY tokenizer_config.json .
COPY tokenizer.json .
COPY pytorch_model.bin .
COPY generation_config.json .
COPY serve.py .

# SageMaker에서 추론 스크립트를 실행하기 위한 환경 변수 설정
ENV SAGEMAKER_PROGRAM inference.py

# 애플리케이션 실행
ENTRYPOINT ["python", "serve.py"]
```
<br>

위 Dockerfile의 구성요소들을 간단히 설명하면 다음과 같다.<br>
<br>

- **FROM**: Base Image를 설정하는 명령어이다. 여기서는 Python 3.10-slim을 Base Image로 사용한다.
- **WORKDIR**: 작업 디렉토리를 설정하는 명령어이다. 여기서는 /app으로 설정한다.
- **RUN**: 명령어를 실행하는 명령어이다. 여기서는 apt-get update, apt-get install 등을 실행한다.
- **COPY**: 파일을 복사하는 명령어이다. 여기서는 requirements.txt, inference.py, config.json 등을 복사한다.
- **ENV**: 환경 변수를 설정하는 명령어이다. 여기서는 SAGEMAKER_PROGRAM을 inference.py로 설정한다.
- **ENTRYPOINT**: Container가 시작될 때 실행할 명령어를 설정하는 명령어이다. 여기서는 python serve.py를 실행한다.
<br>

이때, 위(tree.)에서 못보던 파일들이 있을텐데 이게 뭔지 차근차근이 이야기 해보겠다<br>
<br>

### requirements.txt
```txt
transformers==4.12.0
torch==1.9.1
```
<br>

`reqirements.txt`는 우리가 사용하는 Python 라이브러리들을 명시하는 파일이다.<br>
여기서는 예시로서 **transformers**와 **torch**를 사용하고 있으므로 이 두 라이브러리를 명시하였다.<br>
<br>

### inference.py
```python
from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

class ModelHandler:
    def __init__(self):
        # 모델 및 토크나이저 초기화
        self.model = None
        self.tokenizer = None
        self.device = None  # 사용할 디바이스 (CPU 또는 CUDA)
        self.model_dir = None  # 모델 디렉토리 경로
        self.generation_config = None  # 생성 설정

    def initialize(self, model_dir='.'):
        # 모델 및 토크나이저 로드
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        quantization = True  # 양자화 사용 여부
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16 if quantization else torch.float32,
            pad_token_id=50256,  # GPT-2 의 pad 토큰 ID
            low_cpu_mem_usage=True
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model.eval()  # 모델을 평가 모드로 설정

    def preprocess(self, request_body):
        # 입력 데이터 전처리
        data = json.loads(request_body)
        input_text = data['text']
        inputs = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
        return inputs

    def inference(self, inputs):
        # 모델 추론
        outputs = self.model.generate(inputs, max_length=50, num_return_sequences=3)
        return outputs

    def postprocess(self, outputs):
        # 출력 데이터 후처리
        return [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

class InputData(BaseModel):
    text: str  # 입력 데이터 모델

model_handler = ModelHandler()

@app.on_event("startup")
async def initialize_model():
    # 모델 초기화 이벤트
    model_dir = '.'  # 모델 디렉토리 경로 지정
    model_handler.initialize(model_dir=model_dir)

def input_fn(request_body, request_content_type='application/json'):
    # 입력 데이터 처리 함수
    if request_content_type == 'application/json':
        return request_body
    else:
        raise ValueError("Unsupported request content type: {}".format(request_content_type))

def predict_fn(input_data, model_handler):
    # 예측 수행 함수
    inputs = model_handler.preprocess(input_data)
    prediction_outputs = model_handler.inference(inputs)
    return model_handler.postprocess(prediction_outputs)

def output_fn(prediction, content_type='application/json'):
    # 출력 데이터 포맷팅 함수
    return json.dumps({"predictions": prediction}), content_type

@app.get("/ping")
def ping():
    # 서비스 상태 확인
    return {"message": "Service is online"}

@app.post("/invocations")
async def invocations(request: Request):
    # 모델 추론 요청 처리
    request_body = await request.body()
    input_data = input_fn(request_body, 'application/json')
    data = InputData(**json.loads(input_data))
    prediction = predict_fn(json.dumps(data.dict()), model_handler)
    output = output_fn(prediction, 'application/json')
    return json.loads(output[0])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
```
<br>

`inference.py`는 우리가 추론을 위한 Python 파일이다.<br>
여기서는 예시로서 **transformers** 라이브러리를 사용하여 **GPT2LMHeadModel**을 사용하여 추론을 수행하는 코드를 작성하였다.<br>
이때, FastAPI를 사용하여 추론을 수행하였으며, **uvicorn**을 사용하여 서버를 실행하였다.<br>
FastAPI와 uvicorn에 대한 자세한 내용은 [여기](https://fastapi.tiangolo.com/)를 참고하면 된다.<br>
<br>

#### FastAPI
FastAPI는 Python을 사용하여 빠르게 API를 개발할 수 있도록 도와주는 라이브러리이다.<br>
이 라이브러리는 **Starlette**와 **Pydantic**을 기반으로 하여 빠르고 안정적인 API를 개발할 수 있도록 도와준다.
<br>
<br>

#### Starlette
Starlette는 Python을 사용하여 빠르게 서버를 실행할 수 있도록 도와주는 라이브러리이다.<br>
이 라이브러리는 **ASGI**를 기반으로 하여 빠르고 안정적인 서버를 실행할 수 있도록 도와준다.<br>
<br>

#### Pydantic
Pydantic은 Python을 사용하여 빠르게 데이터를 검증할 수 있도록 도와주는 라이브러리이다.<br>
이 라이브러리는 **Data Validation**을 기반으로 하여 빠르고 안정적인 데이터 검증을 수행할 수 있도록 도와준다.<br>
<br>

#### uvicorn
uvicorn은 Python을 사용하여 빠르게 서버를 실행할 수 있도록 도와주는 라이브러리이다.<br>
이 라이브러리는 **ASGI**를 기반으로 하여 빠르고 안정적인 서버를 실행할 수 있도록 도와준다.<br>

#### ASGI
ASGI는 Python을 사용하여 빠르게 서버를 실행할 수 있도록 도와주는 프로토콜이다.<br>
이 프로토콜은 **Asynchronous Server Gateway Interface**를 기반으로 하여 빠르고 안정적인 서버를 실행할 수 있도록 도와준다.<br>
<br>

### serve.py
```python
import uvicorn

if __name__ == "__main__":
    uvicorn.run("inference:app", host="0.0.0.0", port=8080)
```
<br>

`serve.py`는 추론을 위한 서버를 실행하는 Python 파일이다.<br>
여기서는 **uvicorn**을 사용하여 **inference.py**의 **app**을 실행하였다.<br>
<br>

이렇게 Docker Image를 만들기 위한 파일들을 작성하였다면, 이제 Docker Image를 만들어야 한다.<br>
<br>

## Docker Image를 만드는 방법
Docker Image를 만드는 방법은 다음과 같다.<br>
반드시 Docker Demon이 실행 중이어야 한다.<br>
<br>

```bash
docker build -t <image-name>:<tag> .
```
<br>

위 명령어를 실행하면 Docker Image를 만들 수 있다.<br>
이때, **-t** 옵션을 사용하여 **<image-name>:<tag>**로 Docker Image의 이름과 태그를 설정할 수 있다.<br>
<br>

예를 들어, **<image-name>**을 **my-image**로, **<tag>**을 **latest**로 설정하고 싶다면 다음과 같이 명령어를 실행하면 된다.<br>
<br>

```bash
docker build -t my-image:latest .
```
<br>

이렇게 Docker Image를 만들었다면, 이제 AWS SageMaker에 배포할 수 있다.<br>
<br>







