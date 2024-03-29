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

from .._client import ZhipuAI

class ZhipuAIKnowledge(BaseModel):
    """支持最新的智谱API向量模型"""

    client: Any = None
    """访问智谱AI的客户端"""
    
    api_key: str = None
    
    model: str = Field(default="embedding-2")
    """所要调用的模型编码"""

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        base_url: str = "https://open.bigmodel.cn/api/llm-application/open"
        
        if values["api_key"] is not None:
            values["client"] =  ZhipuAI(api_key=values["api_key"], base_url=base_url)
        else:
            values["client"] =  ZhipuAI(base_url=base_url)
        return values
    
    def list_models(self):
        """Get all embedding models."""
        response = self.client.knowledge_embeddings.list_models()
        return response
