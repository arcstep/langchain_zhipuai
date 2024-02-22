# 为什么要开发这个包？
[![PyPI version](https://img.shields.io/pypi/v/langchain_zhipuai.svg)](https://pypi.org/project/langchain_zhipuai/)

为了方便在 langchain 中使用，在官方SDK基础上做了如下额外工作：

- 问题1: 智谱AI的官方SDK使用了 pydantic v2，这与 langchain（尤其是langserve）不兼容
- 问题2: langchain.community 的国内包更新不及时，无法在 langchain 的 LCEL 语法中使用

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

**接口指南** 智谱[开放平台](https://open.bigmodel.cn/dev/api)大模型接口 Python SDK（Big Model API SDK in Python），让开发者更便捷的调用智谱开放API

**官方SDK** [![PyPI version](https://img.shields.io/pypi/v/zhipuai.svg)](https://pypi.org/project/zhipuai/)

**官方SDK能力简介**
- 对所有接口进行了类型封装。
- 初始化client并调用成员函数，无需关注http调用过程的各种细节，所见即所得。
- 默认缓存token。


