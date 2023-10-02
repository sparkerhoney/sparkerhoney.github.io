---
title: OPEN AI Function Calling
layout: post
description: Prompt Engineering
use_math: true
post-image: https://cdn.startupn.kr/news/photo/202302/32023_33904_230.jpg
category: Prompt Engineering
tags:
- Data Science
- machine learning
- NLP
- ChatGPT
- Prompt Engineering
---

# **OpenAI의 Function Calling 기능 소개**

2023년 6월, OpenAI는 "Function Calling"이라는 새로운 기능을 발표했습니다.

이 기능은 OpenAI API 사용 시, 사용자가 "functions"이라는 파라미터로 사용 가능한 기능들의 목록을 제공하면, GPT의 자의적인 판단 하에 해당 기능을 사용하는 것입니다.

---

## **주요 업데이트 내용 요약**

1. **모델 지원 연장**: gpt-3.5-turbo-0301, gpt-4-0314 및 gpt-4-32k-0314 모델의 지원이 2024년 6월 13일까지 연장되었습니다.
2. **새로운 모델 발표**: gpt-4-0613 및 gpt-3.5-turbo-0613 모델이 발표되었으며, 이 모델들은 함수 호출 기능을 지원합니다.
3. **Function Calling**: 개발자는 이제 GPT 모델에 함수를 설명하고, 모델은 해당 함수를 호출하기 위한 JSON 객체를 출력할 수 있습니다. 이 기능을 통해 외부 도구와 API와 GPT의 기능을 더 신뢰성 있게 연결할 수 있습니다.
4. **가격 인하**: OpenAI는 가장 인기 있는 임베딩 모델인 text-embedding-ada-002의 가격을 75% 인하했으며, gpt-3.5-turbo의 입력 토큰 가격도 25% 인하되었습니다.
5. **모델 폐기 계획**: gpt-3.5-turbo-0301 및 gpt-4-0314 모델의 폐기 일정이 발표되었습니다. 이전 모델들은 2024년 6월 13일까지 사용 가능하며, 그 이후에는 해당 모델 이름으로의 요청이 실패하게 됩니다.

OpenAI의 최신 업데이트는 개발자들에게 더 많은 기능과 향상된 성능을 제공하며, 가격 인하를 통해 더 많은 사용자들이 이러한 기능을 경험할 수 있게 되었습니다. OpenAI는 계속해서 사용자의 피드백을 바탕으로 플랫폼을 발전시켜 나갈 계획입니다.

  

## **주요 특징 및 주의사항**

- 이 기능은 위험성을 내포하고 있어, 이메일 전송, 웹에 글 게시, 구매 결정 등의 동작을 구성할 때 신중한 설계가 필요합니다.
- JSON 형식을 사용하므로, Python에서 JSON 개념과 json 모듈의 사용법을 알고 있으면 도움이 됩니다.
- 현재 gpt-3.5-turbo-0613 및 gpt-4-0613 두 모델에서만 지원됩니다.
- 개발자는 function 파라미터에 JSON Schema를 통해 모델에 함수를 설명하면, 선택적으로 특정 함수를 호출하도록 할 수 있습니다.

---

## **Function Calling의 구성**

1. **model**: 사용할 모델 명을 명시합니다. (gpt-3.5-turbo-0613 또는 gpt-4-0613)
2. **functions**: function 리스트입니다. 각 function은 다음 key를 가질 수 있습니다.
    - **name**: function 이름입니다. (필수)
    - **description**: function의 설명입니다. (선택사항)
    - **parameters**: 함수가 허용하는 매개 변수입니다. (선택사항)
3. **function_call**: 모델(GPT)이 Function calling에 응답하는 방식을 설정합니다.

## **Function Calling의 활용 예시**

- **외부 API 호출**: ChatGPT 플러그인과 같은 외부 API를 호출하여 질문에 답변하는 챗봇을 구축할 수 있습니다.
- **자연어를 API 호출로 변환**: "내 최고 고객은 누구입니까?"와 같은 자연어 질문을 API 호출로 변환할 수 있습니다.
- **텍스트에서 구조화된 데이터 추출**: 텍스트에서 필요한 정보를 추출하는 함수를 정의하여 사용할 수 있습니다.

---

## **실제 사용 예시**

**(해당 [깃허브](https://github.com/sparkerhoney/function_calling)에 상세하게 정리해놓았습니다.)**

1. **Step 1**: 더미 함수를 정의합니다. 실제 환경에서는 이 부분에 backend API나 외부 API를 연결할 수 있습니다.
2. **Step 2**: message와 function을 정의하고 API 호출 시 functions 파라미터를 추가합니다.
3. **Step 3**: GPT의 응답에서 function_call 부분을 확인하고 해당 함수를 호출합니다.
4. **Step 4**: function call의 응답을 참고하여 메시지를 추가합니다.
5. **Step 5**: function call로 응답된 결과를 참고하여 최종적인 날씨 정보를 확인합니다.


```

import openai
import json

# Step 1: 더미 함수 정의
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    weather_info = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info)

# Step 2: message와 function 정의
messages = [{"role": "user", "content": "What's the weather like in Boston?"}]
functions = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    }
]

# Step 3: GPT의 응답에서 function_call 확인
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=messages,
    functions=functions,
    function_call="auto",
)
response_message = response["choices"][0]["message"]

# Step 4: function call의 응답 참고하여 메시지 추가
if response_message.get("function_call"):
    available_functions = {
        "get_current_weather": get_current_weather,
    }
    function_name = response_message["function_call"]["name"]
    function_args = json.loads(response_message["function_call"]["arguments"])
    function_response = available_functions[function_name](
        location=function_args.get("location"),
        unit=function_args.get("unit"),
    )
    messages.append(response_message)
    messages.append(
        {
            "role": "function",
            "name": function_name,
            "content": function_response,
        }
    )

# Step 5: function call로 응답된 결과 참고하여 최종 정보 확인
second_response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=messages,
)
print(second_response.choices[0].message.content)


```

---

# **결론**

Function Calling은 OpenAI의 강력한 기능 중 하나로, 개발자들에게 다양한 활용 방안을 제공합니다. 하지만 이 기능을 사용할 때는 주의사항을 잘 숙지하고 안전한 사용을 위해 신중한 설계가 필요합니다.

더 자세한 내용은 [OpenAI 공식 블로그](https://openai.com/blog/function-calling-and-other-api-updates)에서 확인하실 수 있습니다.