# common types
from typing import Type, Any, Mapping, Dict, Iterator, List, Optional, cast

from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import (
    BaseModel,
    Field,
    root_validator,
)
from langchain_core.utils import (
    convert_to_secret_str,
    get_from_dict_or_env,
    get_pydantic_field_names,
)

# async
import asyncio
from typing import AsyncIterator

from ..http import RestAPI

import os

DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/llm-application/open"

class ZhipuAIKnowledge(BaseModel):
    """支持最新的智谱API向量模型"""

    client: Any = None
    """访问智谱AI的客户端"""
    
    base_url: str = DEFAULT_BASE_URL
    """访问智谱AI的服务器地址"""
    
    api_key: str = os.environ.get('ZHIPUAI_API_KEY') 
    """API KEY"""
    
    model: str = Field(default="embedding-2")
    """所要调用的模型编码"""

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        values["client"] =  RestAPI(base_url=values["base_url"], api_key=values["api_key"])
        return values

    def list_models(self):
        """用于获取当前支持的向量模型列表。"""
        response = self.client.action_get("embedding")
        return response
