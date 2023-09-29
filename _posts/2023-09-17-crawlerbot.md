---
title: 특정 홈페이지에서 원하는 정보 크롤링하는 디스코드 봇 만들기
layout: post
description: Discord bot, Crawling bot
use_math: true
post-image: https://github.com/junia3/junia3.github.io/assets/79881119/edc8163a-9787-4cbd-b282-14b4c67bef9e
category: development
tags:
- Automatic system
- Crawling
---

# 특정 홈페이지에서 원하는 정보 크롤링하는 디스코드 봇

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/05cbedd9-aa9b-43de-98ac-9900a9520258" width="300">
</p>

내가 원하는 정보를 얻고자 할 때 그런 사소한 것들을 자동화를 해보겠다는 생각을 하지 않은 것은 아니었으나, 사실 그동안 실천할 의지가 없었음을 뒤늦게 깨달았다. 뭐 연구자로서의 자질이라던지 책임감 이런 고상한 내용은 아니지만, 간단한 부분이라도 내가 무언갈 할 수 있겠다라는 생각이 들면 일단 실천을 해야겠다는 결심이 섰다. 서론이 길었는데, 내가 하고자 했던 간단한 구현은 다음과 같다. 본인은 긱뉴스([https://news.hada.io/](https://news.hada.io/))라는 커뮤니티에서 상당히 괜찮은 양질의 정보를 얻을 수 있었고, 이는 곧 Hacker News([https://news.ycombinator.com/](https://news.ycombinator.com/)) 사이트의 코리안 버전이라고 할 수 있다. 이러한 커뮤니티에 공유되는 정보들을 꾸준히 탐색했던 이유는 다음과 같다.
첫번째로 현직에 나와있지 않은 현재의 내 상태로 개발자들에게 공유되는 좋은 최신 뉴스들을 알기가 어려웠다는 점이고, 두번째는 인공지능 대학원을 다니면서도 꾸준히 다른 개발 분야에 대한 지식 습득은 이후의 커리어에도 큰 도움이 될 것이라고 판단하였기 때문이다. 그렇기에 해당 사이트에서 많은 정보들을 얻는게 일상처럼 굳어졌는데 사실 어느 순간부터 꼭 얻어야하는 정보와 얻지 않아도 될 것 같은 정보를 구분하기 애매해졌고, 그러면서 해당 사이트에서의 정보를 간단하게 접할 수 있는 방법이 좋지 않을까 생각되기 시작했다. 살면서 크롤링이란 걸 한번도 해본 적은 없지만 블로그를 제작 및 운영하면서 얻은 html이나 학부 때의 통신 지식을 어렴풋하게나마 기억해냈고, 정말 다행이게도 discord 관련 API가 이미 모듈로 구현이 되어있어서 간단하게 디스코드 봇 형태로 제어할 수 있었다.

---

# Discord bot 생성

가장 먼저, **디스코드 봇을 생성**하고 이를 내가 가지고 있는 **디스코드 채널에 추가하는 방법**이다. 디스코드에서는 [개발자 모드](https://discord.com/login?redirect_to=%2Fdevelopers)에서 직접 어플리케이션을 개발 및 이를 적용할 수 있는데, 간단하게 bot을 생성 및 토큰을 부여하고 이를 통제하는 파이썬 코드를 통해 제어하는 방식을 취할 것이다. 본인 계정이 우선 개발자 모드가 활성화되어있는지 확인하기 위해 discord 어플리케이션의 설정 화면을 들어가보도록 한다.

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/70772f41-5604-4ebf-be99-6089de4bfc16" width="400">
</p>

위와 같이 본인 프로필 우측에 보면 톱니바퀴의 사용자 설정이 있고, 이를 클릭하면 화면이 뜨는데

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/6332a11a-e03a-4bec-bbdc-abf6d9522d0c" width="700">
</p>

이렇게 고급 탭에 들어가서 개발자 모드 토글이 활성화가 된 상태라면 그대로 놔두고, 만약 활성화가 되어있지 않다면 켜준다. 그런 뒤 [‘discord developer portal’](https://discord.com/developers/docs/game-sdk/applications)로 접속하여 로그인한 뒤, 어플리케이션을 만들어주도록 하자.

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/8d6d2164-c821-4202-9106-8c729ad1e55a" width="700">
</p>

본인은 이미 만들어둔 어플리케이션이 있긴 하지만 처음부터 하는 경우를 가정하여 application을 새로 파도록 하겠다. 어플리케이션 이름은 실제 봇 이름이 아니므로 그냥 아무렇게나 정해도 된다. 

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/ff9b160b-d429-4c40-9d01-3ff0b0b39525" width="700">
</p>

새로 탄생할 봇을 기념하기 위해 **YouCanBeAGoodRobot**이라는 이름으로 어플리케이션을 생성해주었다. 사실 기깔난 이름을 지어볼려고 했는데 실패함. Create 버튼을 누르면 아래와 같이 어플리케이션이 생성된다.

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/64e057da-ba11-42aa-94ab-25470148c3ff" width="800">
</p>

하지만 어플리케이션은 일종의 컨테이너 역할을 할 뿐, 우리가 하고자 하는 것은 이 어플리케이션 내에서 크롤링 봇을 만드는 것이다. 봇을 새로 만들기 위해서는 좌측 메뉴에서 ‘Bot’을 선택해준다.

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/f5fe499d-3c4b-4ceb-948e-d905e38de11a" width="700">
</p>

봇을 눌러보게 되면 어플리케이션과 동일한 네이밍의 봇이 이미 생성되어있는 것을 볼 수 있다. 봇의 이미지를 바꿔줄 수도 있고 이름을 바꿔줄 수도 있고 암튼 이건 자기 마음.

이 봇을 파이썬으로 제어할려면 봇에 접근해야하고, 물론 봇에 아무나 막 접근해서 이상하게 써먹으면 안되기 때문에 우리는 암호화된 “토큰”이 필요하다. 해당 토큰을 얻는 방법은 ‘Reset Token’ 버튼을 눌러주면 된다.

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/08d1e85d-99b9-49e9-a47f-d277dc9b1045" width="600">
</p>

토큰을 리셋하면 내가 코드에서 제어하고 있는 토큰을 모두 바꿔줘야한다는 말. 그냥 일반적인 경고 메시지일 뿐이니 무시하고 Yes, do it!을 클릭해주자.

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/6ae1bb97-170d-4c5c-9067-c792f4d39d37" width="700">
</p>

그러면 위 그림에 빨갛게 지워진 부분에 봇의 토큰이 뜨게 되고, 이후에 사용하기 위해 이걸 copy해서 어딘가에 복붙해놓는다. 이정도까지 했다면 봇을 제어할 준비는 완료됐는데, 이제 우리가 실제로 이 봇을 초대해서 써먹기 위해서는 봇의 url이 필요하다. URL을 생성함과 동시에 디스코드 봇의 authorization도 함께 진행한다.

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/3b1deb3b-212e-455f-a4e7-170c10a44184" width="700">
</p>

좌측에서 ‘OAuth2’를 클릭, 그리고 URL Generator를 누르게 되면 다음과 같은 화면이 뜬다. 우리는 디스코드 봇이 특정 홈페이지에서 원하는 정보를 가져와서 그냥 복붙하는 시스템을 만들 것이기 때문에 Scopes 탭에서는 ‘Bot’을 선택해주고,

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/f19f9207-dff2-4851-a916-4d1c1b739247" width="700">
</p>

Bot permissions 탭에서는 send messages를 선택해준다.

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/b3378723-a04a-4580-bf16-b94d84b865ad" width="700">
</p>

선택하고 나면 자동으로 URL이 아래에 생성되고, 이 URL도 어딘가에 잘 복붙해놓도록 하자

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/5b84441e-92ce-4572-bee7-cbe612208b43" width="700">
</p>

---

# 디스코드 봇을 디스코드 채널에 불러오기

디스코드 봇을 만들었으니, 이 친구를 이제 특정 채널에 초대해보도록 하자. 마찬가지로 본인은 이미 채널이 있기는 하지만 처음 해보는 걸 가정하기 때문에 처음부터 설명하자면, 우선 좌측에서 새로운 서버를 연다는 가정을 해보도록 하겠다. TEST라는 서버를 임시로 만들 생각이다.

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/63ffbbba-2d92-4a6d-abdb-2c9aa4ed62e0" width="400">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/83801dd5-abfc-4979-8549-4e73cd4dd03f" width="400">
</p>

그런 뒤 크롬이나 본인이 사용하는 브라우저를 연 뒤에 아까 어딘가에 복붙해둔 OAuth2에서 획득한 URL에 접속한다. 접속하면 아래와 같이 해당 봇을 내 discount 서버에 넣고자 하는 화면이 뜬다. 추가하고자 하는 서버인 TEST를 선택하고 Continue를 클릭해보자.

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/58f39641-b374-4e2e-8217-b773c40490d0" width="400">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/5edcb440-4313-4f81-b79f-a23975996fb5" width="400">
</p>

내가 URL을 만드는 과정에서 메시지 전송에 대한 권한을 선택했기 때문에, 이를 디스코드 계정 주인에게 하여금 확인받는 작업이다. 내가 만든 봇이기 때문에 권한을 허용해주도록 하겠다.

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/76656f52-dfbc-4bb2-b739-e2d03bfb17ec" width="800">
</p>

그러면 서버에 우리가 만든 봇이 잘 추가된 것을 확인할 수 있다.

---

# Python code

이제 대망의 코딩 시간이 다가왔다. 파이썬으로 디스코드 봇을 제어하기 위해서는 몇가지 모듈이 필요한데, 아마도 깔려있지 않은 경우가 많기 때문에 모두 깔아주도록 한다.

```bash
python -m pip install --upgrade pip
pip install requests
pip install discord
pip install beautifulsoup4
```

본인이 구현하고자 했던 구조는 특정 홈페이지의 뉴스의 헤드라인 타이틀, 원본 뉴스의 링크 그리고 홈페이지에서 짧게 요약한 뉴스의 링크를 가져오는 구조였다. [Geeknews](https://news.hada.io/)를 들어가보면,

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/2a26fd6b-f956-460e-b8e3-52ee0d892587" width="700">
</p>

이런 식으로 최신글부터 순서대로 정렬되어있다. Python코드는 다음과 같이 동작할 것이다.
<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/dc9a1c1b-507a-4536-ba37-c83cf60a86e7" width="1000">
</p>

방법은 정말 간단하게도 웹페이지로부터 얻어올 수 있는 정보만 가져오는 과정을 특정 시간 간격으로 반복하게 하고, 만약 가져온 정보가 새로운 정보라면 (최신 뉴스라면) 이걸 디스코드로 보내서 올리는 방식이다. Python full code는 다음과 같다.

```python
import discord
from discord.ext import commands, tasks
import requests
from bs4 import BeautifulSoup

# Store the last scraped news
last_scraped_title = ""

# Set default intent for discord client
intents = discord.Intents.default()
intents.typing = False
intents.presences = False

# Bot TOKEN, SERVER ID, CHANNEL ID
TOKEN = 'YOUR BOT TOKEN NUMBER'
GUILD_ID = YOUR SERVER ID
CHANNEL_ID = YOUR CHANNEL ID
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')
    guild = discord.utils.get(bot.guilds, id=GUILD_ID)
    channel = discord.utils.get(guild.text_channels, id=CHANNEL_ID)
    # Start the news scraping task
    news_sender.start(channel)

# 6초마다 새로운 성보 가져오도록 루프 돌리는 함수
@tasks.loop(minutes=0.1)
async def news_sender(channel):
    global last_scraped_title
    new_scraped_news = scrape_news()

    # new_scraped news로부터 제목 text 정보만 가져오기
    new_scraped_title = new_scraped_news["topictitle"].find('h1').text
    if new_scraped_title != last_scraped_title:
        print(f"[New!] {new_scraped_title}")
        await send_news(channel, new_scraped_news, new_scraped_title)
        last_scraped_title = new_scraped_title
    
    # 만약 기존에 가져왔던 뉴스랑 겹치면 디스코드로 보내지 않고 다음 뉴스를 기다림
    else:
        print("Keep waiting for another news ...")

# 뉴스 스크랩하는 함수
def scrape_news():
    url = 'https://news.hada.io/new'
    baseurl = 'https://news.hada.io/'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    news_container = soup.find(class_='topic_row')
    topictitle = news_container.find(class_='topictitle')
    topicdesc = news_container.find(class_='topicdesc')
    return {"baseurl": baseurl, "topictitle": topictitle, "topicdesc": topicdesc}

# 디스코드로 뉴스를 보내는 함수
async def send_news(channel, news, title):
    # Extract the link within the 'a' tag
    link = news["topictitle"].find('a')['href'].strip()
    link_desc = news["topicdesc"].find('a')['href'].strip()
    news_info = f'{"# "+title}\n- 원본 링크 : {link}\n- 긱뉴스 링크 : {news["baseurl"]+link_desc}'
    await channel.send(news_info)

bot.run(TOKEN)
```

코드는 각자 스크랩 및 크롤링을 원하는 홈페이지에서 f12(개발자 도구)를 열어서 각 요소를 확인하는 과정이 필요하다. 예컨데 geeknews에서는

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/e7f810ff-aa11-414b-833b-154c9bb02ac9" width="900">
</p>

본인이 가져오고자 했던 정보는 여러 topic 중에서 가장 최상단에 위치한 topic_row라는 클래스 이름의 div였고,

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/6d7f240e-0874-4345-b779-6352725ccaf8" width="600">
</p>

그 안에서도 topictitle, topicdesc라는 div 클래스에 속한 url을 가져오는 것이 목표였기 때문에 코드를 위와 같이 작성하였다. 또한 디스코드에 메시지를 보낼 때 마크다운 형식을 함께 붙여주었는데,

```python
news_info = f'{"# "+title}\n- 원본 링크 : {link}\n- 긱뉴스 링크 : {news["baseurl"]+link_desc}'
```

이렇게 해주면 제목은 크게 나타나며, 원본 링크와 긱뉴스 링크는 각각 아이템으로 명시되는 것을 볼 수 있다. 각자 원하는 형식에 맞춰서 디스코드로 보낼 메시지를 정해주면 될 것 같다.

```python
# Bot TOKEN, SERVER ID, CHANNEL ID
TOKEN = 'YOUR BOT TOKEN NUMBER'
GUILD_ID = YOUR SERVER ID
CHANNEL_ID = YOUR CHANNEL ID
```

그리고 각자의 봇 토큰, 서버 아이디 그리고 채널 아이디를 입력하는 부분이 있는데, 봇 토큰은 앞서 우리가 봇을 만들면서 어딘가에 잘 복붙해두었던 봇의 토큰을 입력해주면(string 형태로 따옴표 안에 넣어주기) 되고,

GUILD_ID에는 서버 ID, CHANNEL_ID에는 채널 ID를 각자 디스코드에서 복사해서 가져와서 넣어주면 된다(integer 형태로 그냥 넣어주기). 복사하는 방법은 아래 그림 참고.

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/7f10d6c6-d6e9-49a9-88d9-4f29d44a55ea" width="300">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/0f9d3490-fddb-497a-aa44-5470935aea2c" width="300">
</p>

복사가 완료되었다면 파이썬 코드를 돌리면 된다. 돌렸을 때 결과가 다음처럼 나오면 성공

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/af27e0b3-66dd-4901-bd63-c7fa2221c29f" width="600">
</p>
<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/212fb91c-3bb7-44d9-ab1e-3a9e828079e2" width="700">
</p>