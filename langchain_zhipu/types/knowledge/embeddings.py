from typing import Optional, List

from langchain_core.pydantic_v1 import BaseModel

__all__ = ["EmbeddingModel", "ModelsList"]

class EmbeddingModel(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None

class ModelsList(BaseModel):
    list: Optional[EmbeddingModel] = None
    total: Optional[int] = None
