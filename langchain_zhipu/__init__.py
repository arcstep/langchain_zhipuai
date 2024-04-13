
""" 
各模块已经全部修改为直接访问 HTTP 接口，并升级为 4.1.x 版本，可以按如下方式使用：
```
from langchain_zhipu import (
    ChatZhipuAI,               # 用于V4版本接口的通用大模型对话
    ZhipuAIEmbeddings,         # 用户V4版本接口的向量模型
    KnowledgeChatZhipuAI,      # 使用对话模式直接访问基于知识库的大模型应用
    KnowledgeManagerZhipuAI,   # 用于新接口的知识库管理的REST接口
    convert_to_retrieval_tool, # 声明通用大模型中自带的知识库检索的工具
    convert_to_web_search_tool # 声明通用大模型中自带的Web搜索的工具
)
```

```
如果要使用 4.0.x 版本的 ChatZhipuAI 和 ZhipuAIEmbeddings 可以按如下方式导出：
from langchain_zhipu.chat import ChatZhipuAI
from langchain_zhipu.embeddings import ZhipuAIEmbeddings
```
"""
from ._client import ZhipuAI
from .http.manager import KnowledgeManagerZhipuAI
from .http.chat import ChatZhipuAI, KnowledgeChatZhipuAI
from .http.embeddings import ZhipuAIEmbeddings
from .utils import convert_to_retrieval_tool, convert_to_web_search_tool

from .core._errors import (
    ZhipuAIError,
    APIStatusError,
    APIRequestFailedError,
    APIAuthenticationError,
    APIReachLimitError,
    APIInternalError,
    APIServerFlowExceedError,
    APIResponseError,
    APIResponseValidationError,
    APITimeoutError,
)

from .__version__ import __version__
