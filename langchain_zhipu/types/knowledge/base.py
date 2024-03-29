from typing import Optional, List

from langchain_core.pydantic_v1 import BaseModel

__all__ = ["KnowledgeData"]

class KnowledgeData(BaseModel):
    id: Optional[str] = None
    embedding_id: Optional[int] = None
    name: Optional[str] = None
    description: Optional[str] = None
    background: Optional[str] = None
    icon: Optional[str] = None
    word_num: Optional[int] = None
    length: Optional[int] = None
    document_size: Optional[int] = None
