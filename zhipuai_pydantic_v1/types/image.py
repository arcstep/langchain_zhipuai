from __future__ import annotations

from typing import Optional, List

from zhipuai_pydantic_v1.pydantic_v1 import BaseModel

__all__ = ["GeneratedImage", "ImagesResponded"]


class GeneratedImage(BaseModel):
    b64_json: Optional[str] = None
    url: Optional[str] = None
    revised_prompt: Optional[str] = None


class ImagesResponded(BaseModel):
    created: int
    data: List[GeneratedImage]