
""" 
各模块已经全部修改为直接访问 HTTP 接口，并升级为 4.1.x 版本，可以按如下方式使用：
```
from langchain_zhipu import (
    ChatZhipuAI,               # 用于V4版本接口的通用大模型对话
    ZhipuAIEmbeddings,         # 用户V4版本接口的向量模型
    KnowledgeChatZhipuAI,      # 使用对话模式直接访问基于知识库的大模型应用（实际上访问了V3大模型接口）
    KnowledgeManagerZhipuAI,   # 用于新接口的知识库管理的REST接口
    convert_to_retrieval_tool, # 声明通用大模型中自带的知识库检索的工具
    convert_to_web_search_tool # 声明通用大模型中自带的Web搜索的工具
)
```
"""
from .http.manager import KnowledgeManagerZhipuAI
from .http.chat import ChatZhipuAI, KnowledgeChatZhipuAI
from .http.embeddings import ZhipuAIEmbeddings
from .utils import convert_to_retrieval_tool, convert_to_web_search_tool
from .__version__ import __version__
