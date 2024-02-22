# 为什么要开发这个包？
[![PyPI version](https://img.shields.io/pypi/v/zhipuai_pydantic_v1.svg)](https://pypi.org/project/zhipuai_pydantic_v1/)

为了在 langchain 中可以方便使用，做了如下额外工作：

- 问题1: 智谱AI的官方包使用了 pydantic v2，这与 langchain（尤其是langserve）不兼容
- 问题2: langchain_community 的国内包更新不及时，无法在 langchain 的 LCEL 语法中使用

## 已支持全部 langchain 接口

1. invoke
2. ainvoke
3. batch
4. abatch
5. stream
6. astream
7. astream_events
8. asteram_log

## 已支持模型能力

- 模型名称："glm-3-turbo", "glm-4"
- 逻辑推理和对话生成
- 支持工具回调

## 使用举例

```python
from zhipuai_pydantic_v1 import ChatZhipuAI
llm = ChatZhipuAI()

# invoke
llm.invoke("hi")

# stream
for s in llm.stream("hi"):
  print(s)

# astream
async for s in llm.astream("hi"):
  print(s)
```

------------------------------------------
**以下是 zhipuai v2.0.1 的原有文档**
[![PyPI version](https://img.shields.io/pypi/v/zhipuai.svg)](https://pypi.org/project/zhipuai/)

智谱[开放平台](https://open.bigmodel.cn/dev/api)大模型接口 Python SDK（Big Model API SDK in Python），让开发者更便捷的调用智谱开放API

# 智谱大模型开放接口SDK

## 简介
- 对所有接口进行了类型封装。
- 初始化client并调用成员函数，无需关注http调用过程的各种细节，所见即所得。
- 默认缓存token。

## 安装


- 运行环境： [**Python>=3.7**](https://www.python.org/)

- 使用 pip 安装 `zhipuai` 软件包及其依赖

```sh
pip install zhipuai
```

## 使用

- 调用流程：
    1. 使用 APISecretKey 创建 Client
    2. 调用 Client 对应的成员方法
- 开放平台[接口文档](https://open.bigmodel.cn/dev/api)以及[使用指南](https://open.bigmodel.cn/dev/howuse/)中有更多的 demo 示例，请在 demo 中使用自己的 ApiKey 进行测试。

### 创建Client

```python
from zhipuai import ZhipuAI

client = ZhipuAI(
    api_key="", # 填写您的 APIKey
) 
```

### 同步调用

```python
from zhipuai import ZhipuAI
client = ZhipuAI(api_key="") # 填写您自己的APIKey
response = client.chat.completions.create(
    model="",  # 填写需要调用的模型名称
    messages=[
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "我是人工智能助手"},
        {"role": "user", "content": "你叫什么名字"},
        {"role": "assistant", "content": "我叫chatGLM"},
        {"role": "user", "content": "你都可以做些什么事"}
    ],
)
print(response.choices[0].message)
```

### SSE 调用

```python
from zhipuai import ZhipuAI
client = ZhipuAI(api_key="") # 请填写您自己的APIKey
response = client.chat.completions.create(
    model="",  # 填写需要调用的模型名称
    messages=[
        {"role": "system", "content": "你是一个人工智能助手，你叫叫chatGLM"},
        {"role": "user", "content": "你好！你叫什么名字"},
    ],
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].delta)
```

### 异常处理

When the library is unable to connect to the API (for example, due to network connection problems or a timeout), a subclass of `openai.APIConnectionError` is raised.

When the API returns a non-success status code (that is, 4xx or 5xx
response), a subclass of `openai.APIStatusError` is raised, containing `status_code` and `response` properties.

All errors inherit from `openai.APIError`.

```python
import openai
from openai import OpenAI

client = OpenAI()

try:
    client.fine_tunes.create(
        training_file="file-XGinujblHPwGLSztz8cPS8XY",
    )
except openai.APIConnectionError as e:
    print("The server could not be reached")
    print(e.__cause__)  # an underlying Exception, likely raised within httpx.
except openai.RateLimitError as e:
    print("A 429 status code was received; we should back off a bit.")
except openai.APIStatusError as e:
    print("Another non-200-range status code was received")
    print(e.status_code)
    print(e.response)
```

Error codes are as followed:

| Status Code | Error Type                 |
| ----------- | -------------------------- |
| 400         | `BadRequestError`          |
| 401         | `AuthenticationError`      |
| 403         | `PermissionDeniedError`    |
| 404         | `NotFoundError`            |
| 422         | `UnprocessableEntityError` |
| 429         | `RateLimitError`           |
| >=500       | `InternalServerError`      |
| N/A         | `APIConnectionError`       |

