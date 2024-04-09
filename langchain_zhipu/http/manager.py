# common types
from typing import Type, Any, Mapping, Dict, Iterator, List, Optional, cast

from langchain_core.language_models.chat_models import (
    BaseChatModel,
    generate_from_stream,
)

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

from langchain_community.adapters.openai import (
    convert_message_to_dict,
    convert_dict_to_message,
)

from .api import RestAPI
from .types import KnowledgeUrlsMeta

import os

DEFAULT_BASE_URL = "https://open.bigmodel.cn"

class KnowledgeManagerZhipuAI(BaseModel):
    """
    支持V4版本智谱AI的云服务中知识库能力。
    
    **知识库管理** 有以下函数可用：
    - knowledge_create 用来创建知识库
    - knowledge_update 用来更新知识库元数据
    - knowledge_list 用来列举有哪些知识库
    - knowledge_detail 用来查看知识库详情
    - knowledge_remove 用来删除已经建好的知识库
    - knowledge_capacity 用来查看已使用的知识库容量

    **知识文档管理** 有以下函数可用：
    - document_upload_url
    - document_upload_files
    - document_update
    - document_list
    - document_detail
    - document_remove
    - document_retry_embedding

    **应用管理** 有以下函数可用：
    - application_create
    - application_update
    - application_list
    - application_detail
    - application_remove

    """

    client: Any = None
    """访问智谱AI的客户端"""
    
    base_url: str = DEFAULT_BASE_URL
    """访问智谱AI的服务器地址"""
    
    api_key: str = os.environ.get('ZHIPUAI_API_KEY') 
    """API KEY"""
    
    model: str = Field(default="embedding-2")
    """所要调用的模型编码"""

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"zhipuai_api_key": "ZHIPUAI_API_KEY"}

    @property
    def _llm_type(self) -> str:
        """Return the type of chat model."""
        return "zhipuai-knowledge"

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        values["client"] =  RestAPI(base_url=values["base_url"], api_key=values["api_key"])
        return values

    ############ 模型列表 ###########
    def models_list(self):
        """
        用于获取当前支持的向量模型列表。
        """
        response = self.client.action_get(request="api/llm-application/open/embedding")
        return response

    ############ 知识库管理 ###########
    def knowledge_create(self, **kwargs):
        """
        创建知识库。

        返回生成的知识库ID。
        """
        response = self.client.action_post(request="api/llm-application/open/knowledge", **kwargs)
        return response
    
    def knowledge_update(self, knowledge_id: str, **kwargs):
        """
        修改知识库。

        Args:
            id (str): 要修改的知识库ID
            knowledge (KnowledgeLibraryKnowledgeLibraryKnowledgeLibrary): 修改内容
        """
        response = self.client.action_put(request=f"api/llm-application/open/knowledge/{knowledge_id}", **kwargs)
        return response

    def knowledge_list(self, **kwargs):
        """
        列举知识库清单。
        """
        response = self.client.action_get(request="api/llm-application/open/knowledge", **kwargs)
        return response

    def knowledge_detail(self, knowledge_id: str):
        """
        获取知识库详情。
        """
        response = self.client.action_get(request=f"api/llm-application/open/knowledge/{knowledge_id}")
        return response
    
    def knowledge_remove(self, knowledge_id: str):
        """
        删除知识库。
        """
        response = self.client.action_delete(request=f"api/llm-application/open/knowledge/{knowledge_id}")
        return response
    
    def knowledge_capacity(self):
        """
        获取整体的知识库使用量详情。
        """
        response = self.client.action_get(request="api/llm-application/open/knowledge/capacity")
        return response
    
    ############ 知识文档管理 ###########
    def document_upload_url(self, knowledge_id: str, urls: List[KnowledgeUrlsMeta]):
        """
        按照知识库ID和URL，下载文件或读取网页并创建为知识。

        Args:
            knowledge_id (str): 知识库ID
            url (List[KnowledgeUrlsMeta]): 要下载的文件或网页描述
        """
        params = {"knowledge_id": knowledge_id, "upload_detail": urls}
        response = self.client.action_post(request=f"api/llm-application/open/document/upload_url/", **params)

        return response    

    def document_upload_files(self, knowledge_id: str, file_paths: List[str], **kwargs):
        """
        按照知识库ID，上传知识文档。

        Args:
            knowledge_id (str): 知识库ID
            file_paths (List[str]): 知识库文档
            **kwargs: 文档元数据描述
        """
        # 创建文件列表
        files = [('files', (os.path.basename(path), open(path, 'rb'))) for path in file_paths]
        
        try:
            response = self.client.action_post(request=f"api/llm-application/open/document/upload_document/{knowledge_id}", files=files, **kwargs)
        finally:
            # 关闭所有打开的文件
            for _, (_, file) in files:
                file.close()

        return response

    def document_update(self, document_id: str, **kwargs):
        """
        根据文档ID，修改知识文档元数据。

        Args:
            document_id (str): 知识文档ID
            **kwargs: 文档元数据描述
        """
        response = self.client.action_put(request=f"api/llm-application/open/document/{document_id}", **kwargs)        
        return response

    def document_list(self, knowledge_id: str, **kwargs):
        """
        获取知识库文档清单。

        Args:
            knowledge_id (str): 知识库ID
        """
        response = self.client.action_get(request=f"api/llm-application/open/document/{knowledge_id}", **kwargs)
        return response
    
    def document_detail(self, document_id: str):
        """
        获取某个知识库文档的详情。

        Args:
            document_id (str): 知识文档ID
        """
        response = self.client.action_get(request=f"api/llm-application/open/document/{document_id}")        
        return response

    def document_remove(self, document_id: str):
        """
        删除知识库文档。
        """
        response = self.client.action_delete(request=f"api/llm-application/open/document/{document_id}")
        return response

    def document_retry_embedding(self, document_id: str):
        """
        根据文档ID，重新向量化文档。

        Args:
            document_id (str): 知识文档ID
        """
        response = self.client.action_post(request=f"api/llm-application/open/document/embedding/{document_id}")        
        return response

    ############ 应用管理 ###########
    def application_create(self, name: str, desc: str, knowledge_ids: List[str], **kwargs):
        """
        创建基于知识库的应用。

        Args:
            name (str): 应用名称
            desc (str): 应用描述
            knowledge_ids (List[str]): 知识库列表
        """
        params = {"name": name, "desc": desc, "knowledge_ids": knowledge_ids, **kwargs}
        response = self.client.action_post(request=f"api/llm-application/open/application", **params)

        return response 

    def application_update(self, application_id: str, **kwargs):
        """
        修改知识库应用。

        KWArgs:
            name (str): 应用名称
            desc (str): 应用描述
            knowledge_ids (List[str]): 知识库列表
            ...
        """
        response = self.client.action_put(request=f"api/llm-application/open/application/{application_id}", **kwargs)

        return response 

    def application_list(self, **kwargs):
        """
        获取知识库应用清单。

        Args:
        """
        response = self.client.action_get(request=f"api/llm-application/open/application", **kwargs)
        return response

    def application_detail(self, application_id: str):
        """
        获取某个知识库应用的详情。

        Args:
            application_id (str): 应用ID
        """
        response = self.client.action_get(request=f"api/llm-application/open/application/{application_id}")        
        return response

    def application_remove(self, application_id: str):
        """
        删除知识库应用。
        """
        response = self.client.action_delete(request=f"api/llm-application/open/application/{application_id}")
        return response
