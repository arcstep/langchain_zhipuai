from __future__ import annotations

from typing import Union, List, Optional, TYPE_CHECKING

import httpx

from ...core._base_api import BaseAPI
from ...core._base_type import NotGiven, NOT_GIVEN, Headers
from ...core._http_client import make_user_request_input
from ...types.knowledge.embeddings import ModelsList

if TYPE_CHECKING:
    from ..._client import ZhipuAI


class Embeddings(BaseAPI):
    def __init__(self, client: "ZhipuAI") -> None:
        super().__init__(client)
        
    def list_models(
            self,
    ):
        _cast_type = object
        return self._get(
            "/embedding",
            cast_type=ModelsList,
        )
