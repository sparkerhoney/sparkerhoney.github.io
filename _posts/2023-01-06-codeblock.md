---
title: Github blog 코드 블럭 하이라이트 적용해보기
layout: post
description: Code theme, CSS
post-image: https://user-images.githubusercontent.com/79881119/210911206-a918bb17-9774-41f2-80c2-c371706de58d.png
category: github blog
use_math: true
tags:
- web designer
- blog
- github
- Code highlighting
---

코드 하이라이트를 적용해보기로 하였다. 깃허브 블로그 테마마다 조금씩 다르겠지만, 내가 사용하고 있는 테마에서는 코드 블럭 문구를 사용해서 코드를 삽입해도 키워드가 하이라이팅이 안되는 문제가 있었다. 키워드가 하이라이팅된다는 것은 다음과 같다. 흔히 우리가 마크다운에 파이썬 코드를 넣고 싶다면 \`를 3개 붙인 뒤 python을 넣게 되고, 결과물은 다음과 같다. 만약 <U>코드 하이라이팅</U>이 적용되지 않는다면, 결과는 그냥 다음과 같이 나올 것이다.

```

import numpy as np
import torch
import torch.nn as nn

# This is not a code
class Sample(nn.Module):
    def __init__(self):
        super().__init__()

```

하지만 이러면 코드마다 중요한 키워드들을 하이라이팅하지 않으므로 <U>가독성이 떨어진다</U>.

```python

import numpy as np
import torch
import torch.nn as nn

# This is not a code
class Sample(nn.Module):
    def __init__(self):
        super().__init__()

```

만약 이런 식으로 class나 def 등등 키워드가 되는 부분에 하이라이팅을 준다면, 코드를 보는 사람 입장에도 주석과 코드를 구분하거나 작성된 코드를 이해하는데 큰 도움이 될 것이다. 오늘은 이 작업을 수행하는 방법에 대해서 작성해보도록 하겠다.

---

# Install rouge

코드 하이라이팅을 위한 설치가 된다. 아마 지킬 블로그 테마를 적용해본 사람이라면 모두 설치가 되어있을 것인데, 혹시라도 없을 수 있으니까 설치해주자.

```bash
gem install kramdown rouge
```

그리고 <U>_config.yml</U> 파일에 다음과 같이 하이라이터를 명시해준다.

```yaml
# Code highlighter
highlighter: rouge
```
---

# Apply code highlighting theme

그리고 가능한 테마를 확인하기 위해 다음과 같이 코드를 작성한다. 되도록이면 해당 테마를 다운받을 폴더 상에서 작성해주면 된다. 본인은 하이라이팅 관련 <U>css 파일 내용</U>을 복사해서 본인 테마와 관련된 <U>style_dark.scss</U>와 <U>style.scss</U>에 복붙을 해주기 위함이었기에, 그냥 깃허브 블로그 작업하는 디렉토리에서 실행하였다.

```bash
rougify help style
```

가능한 스타일을 모두 보여달라는 것이고, 사용 가능한 테마는 다음과 같다.

```bash
available themes:
  base16, base16.dark, base16.light, base16.monokai, base16.monokai.dark, base16.monokai.light, base16.solarized, base16.solarized.dark, base16.solarized.light, bw, colorful, github, gruvbox, gruvbox.dark, gruvbox.light, igorpro, magritte, molokai, monokai, monokai.sublime, pastie, thankful_eyes, tulip
```

이 중 본인이 원하는 테마 하나를 고르고 다음과 같이 작성하면 된다. 테마를 직접 눈으로 볼 수 없는건 아쉽긴 한데, 직접 하나씩 적용해보기만 하면 되니까 큰 상관은 없다.

```bash
rougify style monokai > /assets/css/temp.css
```

이런 식으로 하위 디렉토리를 명시해서 특정 위치에 css 파일이 만들어지게 해도 되고, 그냥 아무렇게나 뽑은 다음에 내용만 복붙해도 된다. 본인은 복붙 용도였기 때문에 <U>temp.css</U>로 대충 이름을 지어주었다.

만들어진 파일 내부에 가보면 다음과 같이 되어있을 것이다.

```css
.highlight table td { padding: 5px; }
.highlight table pre { margin: 0; }
.highlight .c, .highlight .ch, .highlight .cd, .highlight .cpf {
  color: #75715e;
  font-style: italic;
}
.highlight .cm {
  color: #75715e;
  font-style: italic;
}
.highlight .c1 {
  color: #75715e;
  font-style: italic;
}
...
```

이걸 각자의 블로그의 asset이 되는 style css에 그대로 복붙해서 쓰거나, 만약 커스터마이징을 하고 싶다면 하이라이트 색을 조금씩 바꾸면 된다. 내가 사용하고 있는 블로그 테마의 경우 dark mode에서는 다른 테마를 사용해야했는데, important로 덮어씌우기를 해야했기 때문에 본인은 두 테마에서 서로 겹치지 않는 부분에 대해서는 색을 직접 지정해서 쓰는 방법을 썼다.