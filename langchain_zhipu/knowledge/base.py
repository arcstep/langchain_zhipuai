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

from ..http import RestAPI
from .types import KnowledgeBase

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

    ############ 模型列表 ###########
    def models_list(self):
        """
        用于获取当前支持的向量模型列表。
        """
        response = self.client.action_get(request="embedding")
        return response

    ############ 知识库管理 ###########
    def knowledge_create(self, knowledge: KnowledgeBase):
        """
        创建个人知识库。

        返回生成的知识库ID。
        """
        response = self.client.action_post(request="knowledge", data=knowledge)
        return response
    
    def knowledge_update(self, id: str, knowledge: KnowledgeBase):
        """
        修改知识库。

        Args:
            id (str): 要修改的知识库ID
            knowledge (KnowledgeBase): 修改内容
        """
        response = self.client.action_put(request=f"knowledge/{id}", data=knowledge)
        return response

    def knowledge_list(self, page: int = 1, size: int = 10):
        """
        列举知识库清单。
        """
        params = {"page": page, "size": size}
        response = self.client.action_get(request="knowledge", **params)
        return response

    def knowledge_detail(self, id: str):
        """
        获取个人知识库详情。
        """
        response = self.client.action_get(request=f"knowledge/{id}")
        return response
    
    def knowledge_remove(self, id: str):
        """
        获取个人知识库详情。
        """
        response = self.client.action_delete(request=f"knowledge/{id}")
        return response
    
    def knowledge_capacity(self):
        """
        知识库使用量详情
        """
        response = self.client.action_get(request="knowledge/capacity")
        return response
    
    ############ 知识文档管理 ###########
    