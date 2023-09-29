---
title: 스크롤 내릴 때 컨텐츠 fade out(페이드 아웃) 구현하기
layout: post
description: Blog theme, 서버 관리
post-image: https://user-images.githubusercontent.com/79881119/235353288-c8a73883-e9c3-48a7-bfe2-c2e265a5b574.gif
category: github blog
use_math: true
tags:
- web designer
- blog
- github
---

# 스크롤에 따른 페이드 아웃 기능 구현하기
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209129598-faa17942-0ee2-4a2e-bf0f-080f7d9e6a33.gif"/>
</p>

본인 블로그의 경우 여러 개의 글이 하나의 포스팅 페이지에 세로로 나열되어있다. 그렇다보니 스크롤할 때 위로 올라간 포스트 카드에 대해서는 살짝 페이드 아웃처럼 효과를 주고 싶었다. 위와 같이 구현하고자 했던 것이다. 해당 기능을 구현하기 위해, 일단 코드에서 어떤 방식으로 포스트 카드를 가져와서 보여주는지 과정을 알 필요가 있었다.

---

# 포스팅 카드 웹 상에서 차례로 보여주기
본인의 블로그는 blog.html 그리고 카테고리별 html 파일에서 관련된 컨텐츠를 보여준다. 그렇기 때문에 각각의 html을 확인해보면,
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209129610-8e901b1b-ceaf-4d10-822d-be588022ee45.png"/>
    <img src="https://user-images.githubusercontent.com/79881119/209129612-eb64e5f6-432a-4466-af77-6718a895d996.png"/>
</p>
이처럼 site에 있는 모든 _post 내부 파일들을 불러와서 보여준다. 카테고리별 html에서는 해당 카테고리만 따로 띄워서 보여주게끔 한다. 아무튼 이런 방식으로 include를 하기 때문에, blog-card.html의 형식을 가지는 모든 컨텐츠들을 불러와주기 위해, 다음과 같이 이름을 부여해주었다.

---

# blog-card.html 가장 상위 div에 name 부여하기
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209129614-9120c132-f119-4ab9-a24e-ebb1a7e2bcb6.png"/>
</p>

name = "blogcards"를 따로 해준 이유는 바로 뒤에서 설명할 것인데, 이는 자바스크립트에서 name = 'blogcards'인 컨텐츠들을 배열 형태로 받은 뒤에 이를 foreach문을 돌면서 하나하나 조건에 대해 style option을 주기 위함이다.

---

# getelementbyid와 getelementsbyname의 차이?
위에서 설명을 안하고 내려왔는데, 자바 스크립트 상에서 그냥 <U>getElementById</U>를 했을 경우에 문제가 있다. 아까 위쪽에서 <U>site에 있는 모든 _post 내부 파일들을 불러와서 보여준다</U>라고 했었는데, 이 모든 파일들이 blog-card.html 형식으로 통일되기 때문에 Id로는 구분이 되지 않는다. 만약 하나의 페이지에 $n$개의 똑같은 Id를 가진 컨텐츠들이 순서대로 보여지면, 자바스크립트에서의 <U>getElementById</U> 함수는 같은 Id를 가지는 요소들 중 첫번째만을 반환한다. 즉,

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209129617-ba3e1045-3ec8-4a12-b4e0-f989fa40b2ca.png"/>
</p>

이런 식으로 여러 줄의 카드뉴스들이 있는데, 이 친구들은 모두 동일한 Id 값을 소유하기 때문에 이 중에서 <U>가장 윗글인 지금 포스트의 카드뉴스만</U> 불러오게 되는 것이다.   
따라서 해당 방식으로는 스크립트를 적용하여도 맨 윗쪽 카드뉴스만 적용되고, 그 아래에 있는 나머지 카드뉴스에는 적용되지 않게 된다.   
이를 해결하는 방법으로는 Id로 요소를 받는게 아니라 Name으로 요소의 리스트를 받은 뒤에, 해당 리스트에 있는 요소들 전체에 대해 스크립트를 적용하는 것이다. 그렇기 때문에 name을 부여해주었다.

---

# 자바스크립트 적용하기
자바스크립트는 blog.html, category별 html 모두에 대해 동일하게 적용해주었다.
```javascript
<script>
    let cards = document.getElementsByName("blogcards");
    window.addEventListener("scroll", (event)=> {
        cards.forEach((card) => {
            let cardmiddle = (card.getBoundingClientRect().top + card.getBoundingClientRect().bottom)/2.0;
            
            if(cardmiddle <= 120){
                card.style.opacity = 0.15;
            }
            else{
                card.style.opacity = 1.0;
        }
        });
    });
</script>
```
코드의 내용을 간단히 풀어쓰자면 cards는 위에서 언급했던 바와 같이 name이 우리가 부여해준 <U>"blogcards"</U>와 같은 요소들을 모두 리스트 형태로 반환하게 되고, window.addEventListener 함수를 통해 스크롤이 될 때마다 다음과 같은 함수를 호출한다.   
함수 내부에서는 cards 리스트 내에 있는 모든 card 요소의 상대 위치 중 중간을 반환한다. 여기서 절대 위치가 아니라 상대 위치를 쓰는 이유는 <U>스크롤이 내려간 상태의 화면을 기준</U>으로 페이드 아웃을 적용하기 위해서다.   
페이드 아웃의 opacity는 숫자를 linear하게도 감소시킬수도 있고, 내가 했던 것처럼 상수값으로도 표현할 수 있다. Linear하게 감소시키고 싶다면 다음과 같이 작성해도 될 것 같다.
```javascript
<script>
    let cards = document.getElementsByName("blogcards");
    window.addEventListener("scroll", (event)=> {
        cards.forEach((card) => {
            let cardmiddle = (card.getBoundingClientRect().top + card.getBoundingClientRect().bottom)/2.0;
            if(cardmiddle <= 120){
                card.style.opacity = 0.5-0.35*(cardmiddle/120.0);
            }
            else{
                card.style.opacity = 1.0;
        }
        });
    });
</script>
```
잘 될지는 모르겠는데, 그냥 위는 예시로 보여준 것이고 원하는 방향성대로 작성해주면 될 듯하다.   
이런 식으로 스크롤 작업마다 모든 카드뉴스의 상태를 확인하고 스크립트를 적용하게 되면 적용이 끝나게 된다.

---

# 아래쪽에도 fadeout 적용하기
글을 쓰다보니 아래쪽 포스트에도 비슷하게 적용하는게 좋을 것 같아서 좀 수정했다.
```javascript
<script>
    let cards = document.getElementsByName("blogcards");
    window.addEventListener("scroll", (event)=> {
        cards.forEach((card) => {
            let cardmiddle = (card.getBoundingClientRect().top + card.getBoundingClientRect().bottom)/2.0;
            
            if(cardmiddle <= 120 || cardmiddle >= window.innerHeight-120){
                card.style.opacity = 0.15;
            }
            else{
                card.style.opacity = 1.0;
        }
        });
    });
</script>
```