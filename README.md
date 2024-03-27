# 为什么要开发这个包？
[![PyPI version](https://img.shields.io/pypi/v/langchain_zhipu.svg)](https://pypi.org/project/langchain_zhipu/)

为了方便在 langchain 中使用，在官方SDK基础上做了如下额外工作：

- 问题1: 智谱AI的官方SDK使用了 pydantic v2，这与 langchain（尤其是langserve）不兼容
- 问题2: langchain.community 的国内包更新不及时，无法在 langchain 的 LCEL 语法中使用

# 已支持全部 langchain 接口

1. invoke
2. ainvoke
3. batch
4. abatch
5. stream
6. astream
7. astream_events
8. asteram_log

# 已支持模型能力

- 已支持生成模型："glm-3-turbo", "glm-4", "glm-4v"
- 已支持向量模型："embedding-2"
- 逻辑推理和对话生成
- 支持工具回调
- 支持智能体
- 支持RAG

# 使用

## 简单的例子

```python
from langchain_zhipu import ChatZhipuAI
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

## retrieval 工具

```python
from langchain_zhipu import convert_to_retrieval_tool
llm.bind(tools=[convert_to_retrieval_tool(knowledge_id="1772979648448397312")]).invoke("你知道马冬梅住哪里吗？")
```

## web_search 工具

```python
from langchain_zhipu import convert_to_web_search_tool
llm.bind(tools=[convert_to_web_search_tool(search_query="周星驰电影")]).invoke("哪部电影好看？")
```

## function 工具

```python
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain.tools import tool

@tool
def search(query: str) -> str:
    """查询 langchan 资料; args: query 类型为字符串，描述用户的问题."""
    return "langchain_chinese 是一个为中国大模型优化的langchain模块"

llm.bind(tools=[convert_to_openai_tool(search)]).invoke("langchain_chinese是啥？请查询本地资料回答。")
```

## 使用glm-4v

```python
from langchain_zhipu import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate

llm4v = ChatZhipuAI(model="glm-4v")

prompt = ChatPromptTemplate.from_messages([
    ("human", [
          {
            "type": "text",
            "text": "图里有什么"
          },
          {
            "type": "image_url",
            "image_url": {
                "url" : "https://img1.baidu.com/it/u=1369931113,3388870256&fm=253&app=138&size=w931&n=0&f=JPEG&fmt=auto?sec=1703696400&t=f3028c7a1dca43a080aeb8239f09cc2f"
            }
          }
        ]),
])

(prompt|llm4v).invoke({})
```

## 智能体

请查看 [agent.ipynb](https://github.com/arcstep/langchain_zhipuai/blob/main/agent.ipynb)

# 更多用法

请参考 [langchain_chinese](https://github.com/arcstep/langchain_chinese)

------------------------------------------

**官方接口指南** 智谱[开放平台](https://open.bigmodel.cn/dev/api)大模型接口 Python SDK（Big Model API SDK in Python），让开发者更便捷的调用智谱开放API

**官方SDK** [![PyPI version](https://img.shields.io/pypi/v/zhipuai.svg)](https://pypi.org/project/zhipuai/)
