from typing import overload, Union, Literal, List, Optional, Dict, TYPE_CHECKING, Type, Any

import httpx

from zhipuai.core._base_api import BaseAPI
from zhipuai.core._base_type import NotGiven, NOT_GIVEN, Headers, ResponseT
from zhipuai.core._http_client import make_user_request_input
from zhipuai.core._sse_client import StreamResponse
from zhipuai.core._utils import to_json_response_wrapper
from zhipuai.types.chat.chat_completion import Completion
from zhipuai.types.chat.chat_completion_chunk import ChatCompletionChunk
from zhipuai.types.chat import chat_completions_create_param

if TYPE_CHECKING:
    from zhipuai._client import ZhipuAI


class Completions(BaseAPI):
    def __init__(self, client: "ZhipuAI") -> None:
        super().__init__(client)

        # self.with_raw_response = CompletionsWithRawResponse(self)

    def create(
            self,
            *,
            model: str,
            request_id: Optional[str] | NotGiven = NOT_GIVEN,
            do_sample: Optional[Literal[False]] | Literal[True] | NotGiven = NOT_GIVEN,
            stream: Optional[Literal[False]] | Literal[True] | NotGiven = NOT_GIVEN,
            temperature: Optional[float] | NotGiven = NOT_GIVEN,
            top_p: Optional[float] | NotGiven = NOT_GIVEN,
            max_tokens: int | NotGiven = NOT_GIVEN,
            seed: int | NotGiven = NOT_GIVEN,
            ref: Optional[chat_completions_create_param.Reference] | NotGiven = NOT_GIVEN,
            messages: Union[str, List[str], List[int], List[List[int]], None],
            stop: Optional[Union[str, List[str], None]] | NotGiven = NOT_GIVEN,
            tools: Optional[str] | NotGiven = NOT_GIVEN,
            tool_choice: str | NotGiven = NOT_GIVEN,
            # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
            # The extra values given here take precedence over values defined on the client or passed to this method.
            extra_headers: Headers | None = None,
            return_json: Optional[bool] | None = None,
            # extra_query: Query | None = None,
            # extra_body: Body | None = None,
            timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> Completion | StreamResponse[ChatCompletionChunk]:
        _cast_type = Completion
        _stream_cls = StreamResponse[ChatCompletionChunk]
        if return_json:
            _cast_type = object
            _stream_cls = StreamResponse[object]
        return self._post(
            "/chat/completions",
            body={
                "model": model,
                "request_id": request_id,
                "temperature": temperature,
                "top_p": top_p,
                "ref": ref,
                "do_sample": do_sample,
                "max_tokens": max_tokens,
                "seed": seed,
                "messages": messages,
                "stop": stop,
                "stream": stream,
                "tools": tools,
                "tool_choice": tool_choice,
            },
            options=make_user_request_input(
                extra_headers=extra_headers,
            ),
            cast_type=_cast_type,
            enbale_stream=stream or False,
            stream_cls=_stream_cls,
        )




