from langchain_core.pydantic_v1 import BaseModel, Field, root_validator

# common types
from typing import (
    Type, Any, Mapping, Dict, Iterator, List, Optional, cast,
    AsyncIterator, Union, Literal, AbstractSet, Collection
)

from .base import BaseChatZhipuAI

class ChatZhipuAI(BaseChatZhipuAI):
    """支持最新的智谱API"""

    model: str = Field(default="glm-4")
    """所要调用的模型编码"""

    request_id: Optional[str] = None
    """
    由用户端传参，需保证唯一性；用于区分每次请求的唯一标识，用户端不传时平台会默认生成。
    """

    do_sample: Optional[bool] = None
    """
    do_sample 为 true 时启用采样策略;
    do_sample 为 false 时采样策略 temperature、top_p 将不生效
    """

    temperature: Optional[float] = None
    """
    采样温度，控制输出的随机性，必须为正数；
    取值范围是：
      - (0.0,1.0]，不能等于 0，默认值为 0.95,值越大，会使输出更随机，更具创造性；
      - 值越小，输出会更加稳定或确定；

    建议您根据应用场景调整 top_p 或 temperature 参数，但不要同时调整两个参数。
    """

    top_p: Optional[float] = None
    """
    用温度取样的另一种方法，称为核取样：
    取值范围是：(0.0, 1.0) 开区间，不能等于 0 或 1，默认值为 0.7。
    模型考虑具有 top_p 概率质量tokens的结果。

    例如：0.1 意味着模型解码器只考虑从前 10% 的概率的候选集中取tokens
    建议您根据应用场景调整 top_p 或 temperature 参数，但不要同时调整两个参数。
    """

    max_tokens: Optional[int] = None
    """模型输出最大tokens"""

    stop: Optional[List[str]] = None
    """
    模型在遇到stop所制定的字符时将停止生成，目前仅支持单个停止词，格式为["stop_word1"]    
    """

    tools: List[Any] = None
    """
    可供模型调用的工具列表,tools字段会计算 tokens ，同样受到tokens长度的限制。
    """

    tool_choice: Optional[str] = "auto"
    """
    用于控制模型是如何选择要调用的函数，仅当工具类型为function时补充。默认为auto，当前仅支持auto。
    """

    streaming: Optional[bool] = False
    """
    流式输出。
    """

    @classmethod
    def filter_model_kwargs(cls):
        """
        ZhipuAI在调用时只接受这些参数。
        """
        return [
            "model",
            "request_id",
            "do_sample",
            "temperature",
            "top_p",
            "max_tokens",
            "stop",
            "tools",
            "tool_choice",
        ]

    def _ask_remote(self, prompt: Any, stop: Optional[List[str]] = None, **kwargs):
        # 构造参数序列
        params = self.get_model_kwargs()
        params.update(kwargs)
        params.update({"stream": False})
        if stop is not None:
            params.update({"stop": stop})
    
        # 调用模型
        return self.client.chat.completions.create(
            messages=prompt,
            **params
        )

    def _ask_remote_sse(self, prompt: Any, stop: Optional[List[str]] = None, **kwargs):
        # 构造参数序列
        params = self.get_model_kwargs()
        params.update(kwargs)
        params.update({"stream": True})
        if stop is not None:
            params.update({"stop": stop})
    
        # 调用模型
        return self.client.chat.completions.create(
            messages=prompt,
            **params
        )

class KnowledgeChatZhipuAI(BaseChatZhipuAI):
    """
    支持V4版本智谱AI的云服务中知识库能力的对话应用。
    """

    application_id: str = None
    """基于在线知识库的大模型应用ID"""
    
    model: str = Field(default="glm-4")
    """所要调用的模型编码"""

    def _ask_remote(self, prompt: Any, stop: Optional[List[str]] = None, **kwargs):
        params = {"prompt": prompt, "stop": stop, **kwargs}
        response = self.client.action_post(request=f"/api/llm-application/open/model-api/{self.application_id}/invoke", **params)
        
        return [response]

    def _ask_remote_sse(self, prompt: Any, stop: Optional[List[str]] = None, **kwargs):
        params = {"prompt": prompt, "stop": stop, "incremental": incremental, **kwargs}
        response = self.client.action_sse_post(request=f"/api/llm-application/open/model-api/{self.application_id}/sse-invoke", **params)

        return [response]
