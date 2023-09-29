---
title: (LLama2) GPU에서 돌아가는 나만의 디스코드 챗봇 만들기
layout: post
description: Discord bot, Chatting bot
use_math: true
post-image: https://github.com/junia3/junia3.github.io/assets/79881119/97ca9bcd-4822-4782-8ab6-4ddac8068abe
category: development
tags:
- Automatic system
- Chatting
- LLM
- NLP
---

# GPU에서 동작하는 챗봇 구현하기
이전 글에서 Llama-2-cpp를 사용하여 CPU로도 동작하는 챗봇을 구현했었다. 이번에는 리소스가 있다는 가정 하에, 보다 빠르게 입력된 prompt에 대한 답변을 처리할 수 있는 GPU를 활용한 챗봇을 만들어보기로 하였다. 마찬가지로 Llama-2를 사용하였으며, Llama-2의 경우에는 meta에 신청서만 제출하면 네트워크를 다운받을 수 있는 링크가 주어진다. 우선 [Llama-2 github](https://github.com/facebookresearch/llama)에 들어간다.

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/75d65b67-a6b9-4f1d-ab53-b41de523a97d" width="800">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/77f26735-339b-479d-8326-f1dc5288ca48" width="800">
</p>

다운받고자 한다면 accept를 받아야한다는 소리가 나온다. 경험상 양식을 채우고 이메일 수신을 기다렸을때 빠르면 5분 안에 바로 승인이 났다. 그리고 링크를 받게되면 이메일로 오게 되는데, 이때의 링크를 잘 저장해두자. 참고로 링크의 유효기간은 24시간이기 때문에, 승인 받고나서 '나중에 다운받아야지'하면 안된다. 당일에 모델을 다운받지 않으면 바로 해당 링크는 무용지물.

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/ff901970-2c54-4a5e-8560-a357d20bd190" width="800">
</p>

빨갛게 가려놓은 부분이 바로 링크다. 만약 본인이 이메일로 해당 내용을 회신받지 못했다면 넉넉하게 기다렸다가 다시 승인 요청을 보내보는 것을 추천한다. 아무튼 다운받는 법은 정말로 간단하다. 우선 Llama 깃허브를 클론한 뒤에 다운로드 스크립트를 켠다.

```bash
git clone https://github.com/facebookresearch/llama.git
cd llama
pip install -e .
bash download.sh
```
제대로 진행했다면 다음과 같은 문구가 뜨는데,
<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/17e9b043-2491-497a-9714-5f149b05f041" width="500">
</p>
바로 여기에 아까 받은 이메일의 링크를 복붙해서 넣으면 된다. 이후 다운로드 받는 과정은 output을 보면서 차근차근 따라하면 된다. 본인은 가능한 용량인 7B, 13B만 다운받았다.

---

# Llama-2 챗봇 모델과 discord bot 코드 연결하기
우선 알아야할 점은 Llama-cpp와 Llama-2 GPU 버전은 모델이 동작하는 형태가 다르다. Llama-2 챗봇은 다음과 같은 input prompt 구조를 가져야한다. 예컨데 1명이 챗봇을 사용하는 상황을 가정하겠다.

```python
prompt_for_llama2: List[Dialog] = [
    [{"role" : "system", "content" : "챗봇이 어떠한 방식으로 대답했으면 좋겠는가 작성"},
    {"role" : "user", "content" : "Query #1"},
    ...]
]
```

엄밀하게 따지면 prompt는 ```List[Dialog]``` 형태이지만 간단하게는 리스트 안에 리스트가 내장된 구조를 생각하면 편하다. 가장 외곽의 리스트는 $n$개의 병렬적인 대화를 수용한다고 생각하면 되고, 우리는 실제로 단일 dialog에 하나의 유저에 대한 대화를 지속적으로 이어나갈 것이기 때문에 위와 같이 내부의 리스트에만 계속 query를 쌓아가는 구조가 된다. 간단한 코드 구조는 다음과 같다.

```python
from llama import Llama, Dialog
from typing import List
import fire

def chat_example():
    # Initialize Llama for text generation
    generator = Llama.build(
        ckpt_dir="llama-2-7b-chat/",
        tokenizer_path="tokenizer.model",
        max_seq_len=1024,
        max_batch_size=4,
    )

    dialog_prompt: List[Dialog] = [
        [
            {"role": "system", "content": "You are a kitty deep learning researcher named 'DEV' and 10-year old. Reply with English with Emoji."},
            {"role": "user", "content": "Hello?"}
        ]
    ]

    # Inference on Llama2 Model
    results = generator.chat_completion(dialog_prompt,
                                        max_gen_len=None,
                                        temperature=0.6,
                                        top_p=0.9,
                                        )

    # Extract only answer(Bot reply)
    answer = results[-1]["generation"]["content"]

if __name__ == "__main__":
    fire.Fire(chat_example)
```

---

# 대화 쌓아가기

기존에 했던 방식과 마찬가지로(이전 글인 '[CPU에서 돌아가는 나만의 디스코드 챗봇 만들기](https://junia3.github.io/blog/chatbot)' 참고) 챗봇은 이전 대화 내용을 어느 정도 고려하여 맥락을 맞출 필요가 있다. 이를 위해 대화를 쌓는 방식을 다음과 같이 지정해주었다.

```python

# Initialize global variables
dialog_prompt: Dialog = [{"role": "system", "content": "You are a kitty deep learning researcher named 'DEV' and 10-year old. Reply with English with Emoji."}]
dialogs_logs: Dialog = []

# Define a function to make user dialog
def make_usr_dialog(query):
    dialog = [{"role": "user", "content": query}]
    return dialog

# Define a function to make AI dialog
def make_ai_dialog(answer):
    dialog = [{"role": "assistant", "content": answer}]
    return dialog

def answer_for_chat(query):
    global dialogs_logs
    try:
        dialog = make_usr_dialog(query)
        dialogs_logs += dialog
        dialog_temp: List[Dialog] = [dialog_prompt + dialogs_logs]
        results = generator.chat_completion(
            dialog_temp,
            max_gen_len=None,
            temperature=0.6,
            top_p=0.9,
        )
        answer = results[-1]["generation"]["content"]
        dialog = make_ai_dialog(answer)
        dialogs_logs += dialog

        if len(dialogs_logs) > max_chat_logs:
            dialogs_logs = dialogs_logs[2:]
    except Exception as e:
        dialogs_logs = []
        answer = "I could not generate message 🥲 ..."

    return answer
```

쌓는 방식을 요약하면 다음과 같다. 챗봇이 대답해야하는 형태 (본인의 챗봇의 컨셉은 고양이이므로, 이를 system이라는 역할로 알려줌)를 prompt로 고정해둔다. 그리고 유저가 질문하는 내용을 llama에서 요구하는 프롬프트 형태로 바꿔주는 함수와, assistant의 대답을 llama에서 요구하는 프롬프트 형태로 바꿔주는 함수 ```make_usr_dialog```와 ```make_ai_dialog```를 각각 설정해준다. 쿼리가 들어오게 되면 이를 유저의 질문 형태로 바꿔 기존 로그에 추가하고, 모델에 들어가는 input에는 챗봇의 컨셉 + 질의 응답 로그룰 ```List```로 wrapping하는 절차를 거친다.

또한 ```max_chat_logs```라는 integer value를 통해 대화 로그의 메모리 관리를 하게 되는데, 대화 내용이 너무 길어지게 되면 모델 inference 시간이 증가하므로 이를 방지하기 위함이다. CPU에서의 방식과 다른 점이 있다면, input으로 들어가는 질의 응답의 경우 list의 요소가 유저/AI가 반복되어 들어가기 때문에 짝수 갯수만큼을 지워줘야하고, 이를 단순히 리스트의 indexing(windowing)으로 구현하였다. 대화가 반복될수록 이전 대화는 지워지고, 새로운 대화 내용이 로그에 남아 input에 사용된다.

---

# Distributed parallel for ChatBot

GPU 연산의 경우 랩 사용량이 꽤 되므로 이를 보조하기 위한 GPU 병렬 처리 시스템이 기본이다. Llama-2에서는 이를 자동으로 구현하였으며, 다음과 같이 본인의 컴퓨터 스펙에 따라 GPU 사용을 결정해주면 된다.
```bash
>>> torchrun --nproc_per_node 1 chatbot.py # GPU 1개 사용
>>> torchrun --nproc_per_node 2 chatbot.py # GPU 2개 사용
>>> torchrun --nproc_per_node 3 chatbot.py # GPU 3개 사용
>>> torchrun --nproc_per_node 4 chatbot.py # GPU 4개 사용
```

참고로 기본 셋팅의 경우 GPU는 아이디 오름차순으로 쓰이게 된다.

---

# 디스코드 챗봇 시스템 코드와 연결하기
기존 디스코드 챗봇 시스템은 병렬 GPU에 대한 고려가 없었기 때문에 이를 무시할 수 있었지만, Llama 코드가 들어간 이상 병렬 처리가 가능하게끔 코드를 일부 손봐야한다. 구현 과정은 다음과 같다.

```python
# Discord bot command to chat with Llama
@bot.command()
async def chat(ctx, *, query):
    answer = fire.Fire(answer_for_chat(query))
    await ctx.send(answer)
```

만약 커맨드 상에서 chatting을 요청하고 이에 대한 query를 전송하면, fire 함수를 통해 ```answer_for_chat``` 함수를 불러오게 된다. 해당 함수는 위에서 소개한 함수 코드와 완전히 동일하며, query에 대한 Llama 모델의 인퍼런스를 담당한다. 모델을 불러오는 것은 코드 최초 실행 시 단 '한번만' 수행한다. 최초 실행 시에 광역변수 추가 및 초기화 과정에 대해 다음 코드를 추가하였다.

```python
# Initialize Llama for text generation
generator = Llama.build(
    ckpt_dir="llama-2-7b-chat/",
    tokenizer_path="tokenizer.model",
    max_seq_len=1024,
    max_batch_size=4,
)

# Initialize global variables
dialog_prompt: Dialog = [{"role": "system", "content": "You are a kitty deep learning researcher named 'DEV' and 10-year old. Reply with English with Emoji."}]
dialogs_logs: Dialog = []
max_chat_logs: int = 10
```

---

# 결과 확인해보기

본인은 A6000 서버를 사용하여 테스트하였고, 각 GPU의 RAM 용량은 49기가바이트이다. 단일 GPU를 사용하면 다음과 같이 돌아가게 된다.

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/d7da1cb6-ceab-4d50-981b-b0f28f025361" width="800">
</p>

대략 15~16기가 정도의 용량만 있다면 7B 모델을 넉넉하게 수용할 수 있는 것을 확인하였다. CPU와 비교했을때 긴 대답을 요구하는 질의에 대해서도 확연히 올라간 채팅 성능을 보여준다.

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/91578953-0e09-4e85-b239-7d128f0845d0" width="800">
</p>

13B 모델 또한 수용이 가능하여 확인해보았는데, 13B 모델의 경우에는 무슨 일인지 단일 GPU로는 돌아가지 않고 무조건 2개 이상의 GPU로 돌려야했다. 참고하면 좋을 것 같다.