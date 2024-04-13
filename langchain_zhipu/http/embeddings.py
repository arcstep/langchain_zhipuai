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

from .api import RestAPI

class ZhipuAIEmbeddings(BaseModel, Embeddings):
    """支持最新的智谱API向量模型"""

    client: Any = None
    """访问智谱AI的客户端"""
    
    base_url: str = None
    """访问智谱AI的服务器地址"""

    api_key: str = None
    """访问智谱AI的ZHIPU_API_KEY"""
    
    model: str = Field(default="embedding-2")
    """所要调用的模型编码"""

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        values["client"] =  RestAPI(base_url=values["base_url"], api_key=values["api_key"])
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        return [self._get_embedding(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self._get_embedding(text)
        
    def _get_embedding(self, text: str) -> List[float]:
        response = self.client.action_post(
            request="api/paas/v4/embeddings", 
            model=self.model,
            input=text,
        )
        if 'data' in response:
            results = response['data']
            if len(results) == 1:
                return results[0]['embedding']

        raise BaseException(response)
