from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)

from langchain_core.language_models.chat_models import (
    BaseChatModel,
    generate_from_stream,
)

from langchain_community.adapters.openai import (
    convert_message_to_dict,
    convert_dict_to_message,    
)

from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator

# common types
from typing import (
    Type, Any, Mapping, Dict, Iterator, List, Optional, cast,
    AsyncIterator, Union, Literal, AbstractSet, Collection
)

from abc import ABC, abstractmethod

import asyncio
import tiktoken
import inspect

# all message types
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    ToolMessage,
    ToolMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ChatMessage,
    ChatMessageChunk,
)

from .api import RestAPI

def get_all_attributes(obj):
    attrs = {}
    for cls in inspect.getmro(obj.__class__):
        attrs.update(vars(cls))
    attrs.update(vars(obj))  # 实例属性覆盖类属性
    return attrs

def convert_delta_to_message_chunk(
    _dict: Mapping[str, Any], default_class: Type[BaseMessageChunk]
) -> BaseMessageChunk:
    id_ = _dict.get("id")
    role = cast(str, _dict.get("role"))
    content = cast(str, _dict.get("content") or "")
    additional_kwargs: Dict = {}
    if _dict.get("function_call"):
        function_call = dict(_dict["function_call"])
        if "name" in function_call and function_call["name"] is None:
            function_call["name"] = ""
        additional_kwargs["function_call"] = function_call
    if _dict.get("tool_calls"):
        additional_kwargs["tool_calls"] = _dict["tool_calls"]

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content, id=id_)
    elif role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(
            content=content, additional_kwargs=additional_kwargs, id=id_
        )
    elif role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content, id=id_)
    elif role == "function" or default_class == FunctionMessageChunk:
        return FunctionMessageChunk(content=content, name=_dict["name"], id=id_)
    elif role == "tool" or default_class == ToolMessageChunk:
        return ToolMessageChunk(
            content=content, tool_call_id=_dict["tool_call_id"], id=id_
        )
    elif role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role, id=id_)
    else:
        return default_class(content=content, id=id_)  # type: ignore


class BaseChatZhipuAI(BaseChatModel, ABC):
    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"zhipuai_api_key": "ZHIPUAI_API_KEY"}

    @property
    def _llm_type(self) -> str:
        """Return the type of chat model."""
        return "zhipuai"

    @property
    def lc_attributes(self) -> Dict[str, Any]:
        attributes: Dict[str, Any] = {}

        if self.model:
            attributes["model"] = self.model

        if self.streaming:
            attributes["streaming"] = self.streaming

        if self.return_type:
            attributes["return_type"] = self.return_type

        return attributes

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "chat_models", "ZhipuAI"]
    
    client: Any = None
    """访问智谱AI的客户端"""

    model: str = Field(default="glm-4")
    """所要调用的模型编码"""

    base_url: str = None
    """访问智谱AI的服务器地址"""

    api_key: str = None
    """访问智谱AI的ZHIPU_API_KEY"""
    
    allowed_special: Union[Literal["all"], AbstractSet[str]] = set()
    """Set of special tokens that are allowed。"""

    disallowed_special: Union[Literal["all"], Collection[str]] = "all"
    """Set of special tokens that are not allowed。"""

    @root_validator()
    def base_validate_environment(cls, values: Dict) -> Dict:
        values["client"] =  RestAPI(base_url=values["base_url"], api_key=values["api_key"])
        return values
    
    @classmethod
    def filter_model_kwargs(cls):
        """过滤可以在调用时使用的参数"""
        return {}

    @abstractmethod
    def _ask_remote(self, prompt: Any, stop: Optional[List[str]] = None, **kwargs):
        """同步调用"""

    @abstractmethod
    def _ask_remote_sse(self, prompt: Any, stop: Optional[List[str]] = None, **kwargs):
        """同步的SSE调用"""

    async def _ask_aremote_sse(self, prompt: Any, stop: Optional[List[str]] = None, **kwargs):
        """异步的SSE调用"""
        def _func():
            return list(self._ask_remote_sse(prompt, stop, **kwargs))

        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, _func)
        for obj in results:
            yield obj

    # 获得模型调用参数
    def get_model_kwargs(self):
        params = {}
        attrs = get_all_attributes(self)
        for attr, value in attrs.items():
            if attr in self.__class__.filter_model_kwargs() and value is not None:
                params[attr] = value
        return params

    # 实现 invoke 调用方法
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """实现 ZhiputAI 的同步调用"""

        # 支持根据 stream 或 streaming 生成流
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        prompt = [convert_message_to_dict(message) for message in messages]
        response = self._ask_remote(prompt, stop, **kwargs)

        generations = []
        id = response.get("id")
        if not isinstance(response, dict):
            response = response.dict()
        for res in response["choices"]:
            message_dict = res["message"]
            message = convert_dict_to_message(message_dict)
            generation_info = dict(finish_reason=res.get("finish_reason"))
            gen = ChatGeneration(
                message=message,
                generation_info=generation_info,
            )
            generations.append(gen)
        token_usage = response.get("usage", {})
        llm_output = {
            "id": id,
            "created": response.get("created"),
            "token_usage": token_usage,
            "model_name": self.model,
        }
        return ChatResult(generations=generations, llm_output=llm_output)

    # 实现 stream 调用方法
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """实现 ZhiputAI 的事件流调用"""
        prompt = [convert_message_to_dict(message) for message in messages]

        default_chunk_class = AIMessageChunk
        try:
            for response in self._ask_remote_sse(prompt, stop, **kwargs): 
                # print(response)
                if not isinstance(response, dict):
                    response = response.dict()
                if "choices" not in response or len(response["choices"]) == 0:
                    print("no choices")
                    continue
                choice = response["choices"][0]
                delta = choice["delta"]
                delta.update({"id": response["id"]})
                message_chunk = convert_delta_to_message_chunk(
                    delta, default_chunk_class
                )
                generation_info = {}
                if finish_reason := choice.get("finish_reason"):
                    generation_info["finish_reason"] = finish_reason
                if usage := choice.get("usage"):
                    generation_info["usage"] = usage
                default_chunk_class = message_chunk.__class__
                chunk = ChatGenerationChunk(
                    message=message_chunk, generation_info=generation_info or None
                )
                # print(chunk)
                if run_manager is not None:
                    run_manager.on_llm_new_token(chunk.text, chunk=chunk)
                yield chunk
        except Exception as e:
            # 处理错误
            print(f"Error: {e}")
            
    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """实现 ZhiputAI 的事件流调用"""
        prompt = [convert_message_to_dict(message) for message in messages]

        default_chunk_class = AIMessageChunk
        async for response in self._ask_aremote_sse(prompt, stop, **kwargs):
            if not isinstance(response, dict):
                response = response.dict()
            if len(response["choices"]) == 0:
                continue
            choice = response["choices"][0]
            delta = choice["delta"]
            delta.update({"id": response["id"]})
            chunk = convert_delta_to_message_chunk(
                delta, default_chunk_class
            )
            generation_info = {}
            if finish_reason := choice.get("finish_reason"):
                generation_info["finish_reason"] = finish_reason
            default_chunk_class = chunk.__class__
            chunk = ChatGenerationChunk(
                message=chunk, generation_info=generation_info or None
            )
            if run_manager:
                await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk
            
    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
            overall_token_usage: dict = {}
            system_fingerprint = None
            for output in llm_outputs:
                if output is None:
                    # Happens in streaming
                    continue
                token_usage = output["token_usage"]
                if token_usage is not None:
                    for k, v in token_usage.items():
                        if k in overall_token_usage:
                            overall_token_usage[k] += v
                        else:
                            overall_token_usage[k] = v
                if system_fingerprint is None:
                    system_fingerprint = output.get("system_fingerprint")
            combined = {"token_usage": overall_token_usage, "model_name": self.model}
            if system_fingerprint:
                combined["system_fingerprint"] = system_fingerprint
            return combined

    def get_token_ids(self, text: str) -> List[int]:
        """Get the token IDs using the tiktoken package."""

        encoding_model = tiktoken.get_encoding("cl100k_base")
        return encoding_model.encode(
            text,
            allowed_special=self.allowed_special,
            disallowed_special=self.disallowed_special,
        )
