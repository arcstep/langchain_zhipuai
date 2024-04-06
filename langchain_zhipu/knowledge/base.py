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
from .types import KnowledgeUrlsMeta

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
    def knowledge_create(self, **kwargs):
        """
        创建个人知识库。

        返回生成的知识库ID。
        """
        response = self.client.action_post(request="knowledge", **kwargs)
        return response
    
    def knowledge_update(self, knowledge_id: str, **kwargs):
        """
        修改知识库。

        Args:
            id (str): 要修改的知识库ID
            knowledge (KnowledgeLibraryKnowledgeLibraryKnowledgeLibrary): 修改内容
        """
        response = self.client.action_put(request=f"knowledge/{knowledge_id}", **kwargs)
        return response

    def knowledge_list(self, **kwargs):
        """
        列举知识库清单。
        """
        response = self.client.action_get(request="knowledge", **kwargs)
        return response

    def knowledge_detail(self, knowledge_id: str):
        """
        获取个人知识库详情。
        """
        response = self.client.action_get(request=f"knowledge/{knowledge_id}")
        return response
    
    def knowledge_remove(self, knowledge_id: str):
        """
        删除个人知识库详情。
        """
        response = self.client.action_delete(request=f"knowledge/{knowledge_id}")
        return response
    
    def knowledge_capacity(self):
        """
        知识库使用量详情
        """
        response = self.client.action_get(request="knowledge/capacity")
        return response
    
    ############ 知识文档管理 ###########
    def document_upload_url(self, knowledge_id: str, urls: List[KnowledgeUrlsMeta]):
        """
        按照知识库ID和URL，下载文件或读取网页并创建为知识

        Args:
            knowledge_id (str): 知识库ID
            url (List[KnowledgeUrlsMeta]): 要下载的文件或网页描述
        """
        params = {"knowledge_id": knowledge_id, "upload_detail": urls}
        response = self.client.action_post(request=f"document/upload_url/", **params)

        return response    

    def document_upload_files(self, knowledge_id: str, file_paths: List[str], **kwargs):
        """
        按照知识库ID，上传知识文档

        Args:
            knowledge_id (str): 知识库ID
            file_paths (List[str]): 知识库文档
            **kwargs: 文档元数据描述
        """
        # 创建文件列表
        files = [('files', (os.path.basename(path), open(path, 'rb'))) for path in file_paths]
        
        try:
            response = self.client.action_post(request=f"document/upload_document/{knowledge_id}", files=files, **kwargs)
        finally:
            # 关闭所有打开的文件
            for _, (_, file) in files:
                file.close()

        return response

    def document_update(self, document_id: str, **kwargs):
        """
        根据文档ID，修改知识文档元数据

        Args:
            document_id (str): 知识文档ID
            **kwargs: 文档元数据描述
        """
        # 使用字典推导式创建一个文件名到文件数据的映射
        response = self.client.action_put(request=f"document/{document_id}", **kwargs)        
        return response

    def document_list(self, knowledge_id: str, **kwargs):
        """
        获取个人知识库文档清单。

        Args:
            knowledge_id (str): 知识库ID
        """
        response = self.client.action_get(request=f"document/{knowledge_id}", **kwargs)
        return response
    
    def document_detail(self, document_id: str):
        """
        获取某个知识库文档的详情。

        Args:
            document_id (str): 知识文档ID
        """
        # 使用字典推导式创建一个文件名到文件数据的映射
        response = self.client.action_get(request=f"document/{document_id}")        
        return response

    def document_remove(self, document_id: str):
        """
        删除个人知识库详情。
        """
        response = self.client.action_delete(request=f"document/{document_id}")
        return response

    def document_retry_embedding(self, document_id: str):
        """
        根据文档ID，重新向量化文档

        Args:
            document_id (str): 知识文档ID
        """
        # 使用字典推导式创建一个文件名到文件数据的映射
        response = self.client.action_post(request=f"document/embedding/{document_id}")        
        return response

