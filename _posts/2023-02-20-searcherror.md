---
title: 깃허브 블로그를 구글 검색 엔진에 등록했지만 검색이 안된다면?
layout: post
description: 블로그 검색, 구글 검색 오류
use_math: true
post-image: https://user-images.githubusercontent.com/79881119/219984439-657bf38a-7dc9-4e16-94de-c06dee0fae9e.gif
category: github blog
tags:
- web designer
- blog
- github
---

# 깃허브 블로그가 검색이 왜 안됨?

은 사실 본인 이야기다. 예전에 [구글 서치 콘솔](https://search.google.com/search-console/about)에 본인의 블로그를 검색 가능하게끔 만드는 과정을 포스팅했었는데, 분명 sitemap 제출, robot.txt 추가 등등 사람들이 올려놓은 방법들을 모두 깃허브 블로그에 적용하였고, 심지어 <U>사이트맵 제출 상태가 성공임에도</U> 불구하고 검색이 안되는 문제가 발생하였다. 사이트맵 제출 상태가 '성공'이 아니라면 굳이 이 글을 읽을 필요가 없고, 본인 글은 사이트맵 제출 상태가 '성공'임에도 구글 검색을 통해 노출이 되지 않는 분들을 위한 작은 일기장이랄까..

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/219984721-bdae306e-9036-4fa5-b366-c84e49e89ed5.png" width="600"/>
</p>

우선 본인은 블로그가 검색이 되지 않는다는 사실을 굉장히 늦게 깨달았고, 기존 sitemap 코드랑 이것저것 일부 수정한 뒤 다시 사이트맵을 제출하였다. 사실상 거의 그대로 다시 낸 것과 다름이 없는데, 원인은 의외로 사이트맵이 아니라 다른 곳에서 발견되었다.

---

# 색인 생성 확인

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/219984858-4fcd9cc9-7c56-4c87-8372-bd69c67e5946.png" width="400"/>
</p>

사이드바 메뉴 중에서 'Sitemaps' 위에 보면 '페이지'라고 나와있는 부분을 클릭해보자. 본인 사이트맵이 처음 승인이 되었을 때가 크리스마스 이브 쯤이었는데, 그때 검색이 가능하도록 하고 싶었던 블로그 하위 페이지 갯수가 총 $26$개 정도 되었던 것 같다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/219985027-24a1685e-f7c5-406b-b975-3758d29a75ac.png" width="600"/>
</p>

그러자 어찌된 영문인지 모르겠지만 $26$개의 페이지에 대해서 색인이 생성되지 않았다고 하고, 원인은 그냥 '<U>알 수 없음</U>'으로 나오는 것이었다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/219985124-53478237-e1d2-486e-a51b-ebe60bf42ee3.png" width="400"/>
</p>

상당히 당황스러웠지만 예전에 봤던 글 중에서 <U>'색인'이 생성되어야</U> 실제로 구글 상에서 검색이 가능한데, 이게 사이트맵만 등록한다고 해서 다 되는건 아니고 <U>직접 해줘야할 수도 있다</U>고 했던 것이 기억이 났다. 그래서 이번에는 그냥 막연하게 기다리지 않고 직접 본인 포스트 주소 하나하나 <U>색인 등록 우선 순위에 올리기</U>로 결정하였다.

---

# URL 검사
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/219985501-3cc1e7fb-46ab-4874-8dd4-ffa10e41d444.png" width="400"/>
</p>

우선 본인의 블로그 주소부터 URL 검사를 돌려보는 것을 추천한다. URL 검사를 누르면 검사를 원하는 페이지 주소를 입력하게 하는데, 여기에 <U>본인 깃허브 블로그 주소를 입력</U>하면 된다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/219985673-9f0e464d-d715-44ff-a01d-79570d731ad9.png" width="600"/>
</p>

만약 해당 페이지에 <U>색인이 생성이 되어있다면</U> 다음과 같은 화면이 나온다. 그리고 경험상 색인이 생성된 페이지가 <U>실제 구글에 노출이 되기 시작하자</U> 해당 페이지를 등록한 이메일로 알림 메일이 발송되었다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/219985739-d03d2fbd-ce67-441a-ac29-79902e4b1a75.png" width="600"/>
</p>

그런데 만약 색인이 생성되어있지 않다고 나온다면 색인 생성을 요청하면 된다. 하지만 하루에 <U>색인 생성을 요청할 수 있는 페이지 수가 한정</U>되어있기 때문에 블로그 초창기에 인덱싱하고 싶은 블로그의 모든 페이지를 색인 등록을 해놓은 뒤, 이후 포스팅이 올라갈 때마다 하나씩 추가해주는 형태로 하면 괜찮을 것 같다고 생각되었다. 본인도 아직 등록이 안된 페이지가 있기 때문에 등록을 해보도록 하겠다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/219986074-2f7cd79a-5b75-4317-8b76-0e642cd0d857.png" width="600"/>
</p>

등록이 되어있지 않다고 나온다. <U>색인 생성 요청</U>을 눌러본다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/219986141-cc096af2-141b-46c2-b824-b8a768162052.png" width="600"/>
</p>

색인 생성 요청을 기다리다보면 요청이 되었다고 나오고, 이 상태에서 하루나 이틀 기다리면 해당 페이지가 구글 검색이 가능하게 된다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/219986337-830644a0-f446-4f0a-b80e-2e35c32d84c7.png" width="600"/>
</p>

만약 다음과 같이 <U>할당량 초과 문구</U>가 떴다면 하루 뒤에 다시 시도하면 된다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/219986576-05b1493e-8c36-42ff-9e82-380aaddfefda.png" width="600"/>
</p>

실제로 기다리면 요청한 사이트에 대해서 색인이 차례대로 승인되는데, 승인된 경우 구글에서 검색이 가능하다. <U>BLIP 논문 리뷰 게시글</U>을 검색해보자.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/219987294-4d1022ae-a7e7-417e-ad42-3b69295300e4.png" width="600"/>
</p>

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/219987344-00651b25-3dee-4808-9b6b-d3e98974a287.png" width="600"/>
</p>

<U>잘 검색되는 것</U>을 볼 수 있다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/219987404-150fc1c7-45bf-4928-859a-43829f033911.png" width="600"/>
</p>

아직 승인되지 않은 사이트들도 있고 모바일 환경에서는 검색이 되는데 반대로 웹에서는 검색이 안되는 사이트들도 여럿 있는 것 같다. 아직 이 부분은 공부를 해봐야할 것 같아서 차차 <U>개선사항이 생기면</U> 업데이트하도록 하겠다.