---
title: github.io 댓글 다크모드 적용 방법
layout: post
description: Blog theme, 서버 관리
post-image: https://user-images.githubusercontent.com/79881119/235353112-5b9bfd8a-5ce2-4bb8-a287-66698da67dda.png
category: github blog
tags:
- web designer
- blog
- github
- comment
---

# 지난번엔,,,

저번에는 다크모드를 지금 내가 가지고 있는 테마에 적용했었는데, 그 후로 몇가지 바뀐 점이 있다. 우선 첫번째로는 게시물의 태그랑 하이퍼링크 색에 대해서 아예 테마를 맞춰버렸다.
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209127880-4f8acac3-7ae4-4127-a61c-298f1f6c4a0e.png" width="700"/>
    <img src="https://user-images.githubusercontent.com/79881119/209127890-4e75e1ce-0a03-436e-850f-58fa4d152290.png" width="700"/>
</p>
근데도 바꾸지 못했던 것이 있는데, utterance(코멘트 기능)였다. 어떻게 할까 엄청 고민하다가 결국 style에 hide 옵션이라는 걸 사용해서 해결할 수 있었다.

# Div로 두 개의 utterance를 분리

```html
<div class="utterance-light" id="comment_light">
    <script src="https://utteranc.es/client.js"
        repo="junia3/comments"
        issue-term="pathname"
        theme="github-light"
        crossorigin="anonymous"
        async>
    </script>
</div>

<div class="utterance-dark" id="comment_dark">
    <script src="https://utteranc.es/client.js"
        repo="junia3/comments"
        issue-term="pathname"
        theme="github-dark"
        crossorigin="anonymous"
        async>
    </script>
</div>
```
위와 같이 두 개의 서로 다른 "theme"을 가지게 될 utterance 코드를 임베딩하였다. 각각 class 이름이랑 Id를 부여해줬다.   
이후 작업은 생각보다 심플하다.

# style.scss와 style_dark.scss 수정하기
style.scss가 먼저 적용이된 뒤에 style_dark.scss가 적용이 된다.
이걸 감안하면 다음과 같은 flow로 코드를 작성하면 된다.

```css
@import "main.scss";

#comment_light{
    display: block;
}

#comment_dark{
    display: none;
}
```

위와 같이 style.scss에 먼저 작성해준다. 이제 default(다크 모드가 켜져있지 않은 상태)에서는 **"Id가 comment_light"인 애만 display**되고, **"Id가 comment_dark"인 애는 display하지 않는다**.   
이를 코드로 쓰게 되면 위와 같이 된다. 마찬가지로 style_dark.scss에는,

```css
@import "main_dark.scss";
#comment_light{
    display: none !important;
}

#comment_dark{
    display: block !important;
}
```

위와 같이 작성해주면 되는데, 여기서 !important를 빼먹으면 안되는 이유는 아까도 말했듯이 default로 style.scss의 스타일이 먼저 적용되게끔 해놨기 때문에 변환 시에 스타일 적용을 해줘야한다는 의미로 넣어준다. 이렇게 되면 결과는 다음과 같이 잘 나온다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/209127892-c1de94cb-6290-4a8f-bfb5-caa041b94812.gif" width="700"/>
</p>

처음엔 다크모드 토글에 적용했던 것처럼 스크립트 제어를 통해 테마만 attribute로 넣어줄라했는데 그런 방법들은 잘 안돼서 포기하고 오히려 이 방법이 더 깔끔하고 코드도 몇줄 안바꿔서 좋은 것 같다.