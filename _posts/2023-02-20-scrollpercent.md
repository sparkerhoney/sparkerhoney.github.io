---
title: 스크롤 내릴 때 페이지 상단에 가로축으로 내려온 정도(퍼센트) 표시하기
layout: post
description: 블로그 스크롤 퍼센트 표시
use_math: true
post-image: https://user-images.githubusercontent.com/79881119/220035111-68ae1a27-80c3-41d5-bfb5-bce659d97284.gif
category: github blog
tags:
- web designer
- blog
- github
---

물론 스크롤바가 오른쪽에 있기 때문에 <U>굳이 표시해주지 않더라도</U> 포스트를 읽는 과정에서 본인이 어디쯤까지 읽었는지 확인할 수 있지만, 포스팅 상단의 가로축을 통해 스크롤 정도를 표시해주면 훨씬 가독성이 좋은 것 같아보였다. 해당 내용을 구현하기 위해 참고한 블로그는 다음과 같다([참고 링크](https://juni-official.tistory.com/108)). 기존에 블로그 테마 작업했던 수준에 비하면 그리 어려운 일은 아닌 것 같아보였다. 해야할 일은 세 가지로 나눌 수 있다.

1. HTML에 스크롤 상태바를 위한 ```span``` 태그 삽입해주기
2. 삽입한 span의 스타일을 css로 지정해주기
3. span을 자바스크립트로 제어해주기

# HTML 태그 삽입
<U>포스팅을 제외한 레이아웃</U>에는 굳이 스크롤 기능이 들어가지 않아도 상관없기 때문에 본인 블로그 테마 기준으로는 ```_layouts/post.html``` 파일에 해당 내용을 추가하였다. 파일을 보게 되면 ```body``` 태그의 가장 상단부에 navigation bar가 있는데, 해당 ```include```문 다음에 바로 다음과 같이 추가해주었다.

```html
<span class="bar"></span>
```

클래스는 딱히 겹치는 이름이 없었기 때문에 "bar"로 지정하였고, 다른 이름으로 하고 싶다면 다른 이름으로 설정해도 상관없다.

---

# CSS에 스타일 지정해주기
스타일은 percentage bar의 모양이나 색을 잡아주는 코드가 된다. 우선 결과부터 보면 본인은 dark theme과 light theme이 있기 때문에 각각 테마에 맞는 색깔대로 그라데이션을 넣어주기로 했다. 코드를 보기 전에 결과물을 먼저 보게되면,
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/220036701-275b1fb0-db68-4b4f-bf74-b1913ed4a76d.png" width="1200"/>
</p>
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/220036789-c5c34c6c-d633-4f4d-981d-eba7a3afcde3.png" width="1200"/>
</p>

위와 같다. 각자의 테마의 상태에 따라 다르겠지만 본인은 dark theme이 가져오는 ```main_dark.css``` 파일에는 다음과 같이 작성해주고,

```css
.bar{
	position: fixed;
	top: 0px; left: 0px;
	width: 0%; height: 5px;
	background: linear-gradient(to right, black, coral, salmon, pink) !important;
	z-index: 9999999;
}
```

기본 theme이 가져오는 ```main.css``` 파일에는 다음과 같이 작성해주었다.

```css
.bar{
	position: fixed;
	top: 0px; left: 0px;
	width: 0%; height: 5px;
	background: linear-gradient(to right, black, rgb(0, 76, 148), rgb(9, 158, 250), #009999);
	z-index: 9999999;
}
```

```!important```의 역할은 <U>기본 테마를 기준</U>으로 <U>다른 테마의 컬러가 무시</U>되는 경우가 생기기 때문에 우선순위를 지정해준 것이라고 보면 된다.

---

# 자바 스크립트
자바 스크립트 내용은 scroll bar가 들어간 ```_layouts/post.html``` 파일 맨 하단에 추가해주면 된다. 각자 span 태그가 들어간 layout 파일 하단에 스크립트 태그를 붙여서 넣어주면 된다.

```html
<script>
    $(window).scroll(function() {
            var scrollY = ($(window).scrollTop() / ($(document).height() - $(window).height()) * 100).toFixed(3);
            $(".bar").css({"width" : scrollY + "%"});
    });
</script>	
```
코드를 보면 JQuery 문법으로 보는게 더 맞는 듯하다. 해당 코드는 앞서 말했던 것처럼 [이 블로그](https://juni-official.tistory.com/108)에서 참고했는데, 코드 상에서 ```toFixed```를 통해 소숫점 자리를 맞춰준 이유는 ```floor```를 통해 정수 단위의 퍼센트만 사용했을 때 모바일 환경에서 끊기는 현상이 발생했기 때문이라고 한다. 아무래도 가로축이 세로축보다 짧은 기기 특성상 어쩔 수 없는 부분인 듯하다.  