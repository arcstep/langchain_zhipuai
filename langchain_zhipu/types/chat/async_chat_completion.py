from typing import List, Optional

from langchain_core.pydantic_v1 import BaseModel

from .chat_completion import CompletionChoice, CompletionUsage

__all__ = ["AsyncTaskStatus"]


class AsyncTaskStatus(BaseModel):
    id: Optional[str] = None
    request_id: Optional[str] = None
    model: Optional[str] = None
    task_status: Optional[str] = None


class AsyncCompletion(BaseModel):
    id: Optional[str] = None
    request_id: Optional[str] = None
    model: Optional[str] = None
    task_status: str
    choices: List[CompletionChoice]
    usage: CompletionUsage