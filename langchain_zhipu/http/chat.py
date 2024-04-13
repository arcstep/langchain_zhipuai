from langchain_core.pydantic_v1 import BaseModel, Field, root_validator

# common types
from typing import (
    Type, Any, Mapping, Dict, Iterator, List, Optional, cast,
    AsyncIterator, Union, Literal, AbstractSet, Collection
)

from .base import BaseChatZhipuAI

import time
import json

class ChatZhipuAI(BaseChatZhipuAI):
    """支持最新的智谱API"""

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
        params = self.get_model_kwargs()
        params.update({"messages": prompt, **kwargs})
        params.update({"stream": False})
        if stop is not None:
            params.update({"stop": stop})
    
        reply = self.client.action_post(request=f"api/paas/v4/chat/completions", **params)
        
        return reply

    def _ask_remote_sse(self, prompt: Any, stop: Optional[List[str]] = None, **kwargs):
        """
        这是V4版本的SSE接口解析，与V3差异较大。
        """
        params = self.get_model_kwargs()
        params.update({"messages": prompt, **kwargs})
        params.update({"stream": True})
        if stop is not None:
            params.update({"stop": stop})
    
        replies = self.client.action_sse_post(request=f"api/paas/v4/chat/completions", **params)
        
        for line in replies.iter_lines():
            if line:  # 过滤掉心跳信号（即空行）
                line_utf8 = line.decode('utf-8')

                # 如果这一行是数据就返回结果，否则忽略
                if line_utf8.startswith("data: {"):
                    text = line_utf8[6:]
                    if text is not None:
                        yield json.loads(text)

class KnowledgeChatZhipuAI(ChatZhipuAI):
    """
    支持V4版本智谱AI的云服务中知识库能力的对话应用。
    """

    application_id: str = None
    """基于在线知识库的大模型应用ID"""
    
    incremental = True
    """incremental为True时，流式输出时是否按增量输出，否则按全量输出"""
    
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
            "incremental",
        ]

    @root_validator()
    def raise_invalid_params(cls, values: Dict) -> Dict:
        if values["application_id"] is None:
            raise Exception("MUST supply application_id")
        return values

    def _ask_remote(self, prompt: Any, stop: Optional[List[str]] = None, **kwargs):
        params = self.get_model_kwargs()
        params.update({"prompt": prompt, **kwargs})
        if stop is not None:
            params.update({"stop": stop})

        reply = self.client.action_post(request=f"api/llm-application/open/model-api/{self.application_id}/invoke", **params)
        
        return ({
            "id": reply["data"]["requestId"],
            "created": int(reply["timestamp"] / 1000),
            "model": self.model,
            "usage": {},
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": reply["data"]["content"],
                },
            }],
        })

    def _ask_remote_sse(self, prompt: Any, stop: Optional[List[str]] = None, **kwargs):
        """
        这是V3版本的SSE接口解析。
        """
        params = self.get_model_kwargs()
        params.update({"prompt": prompt, **kwargs})
        if stop is not None:
            params.update({"stop": stop})

        replies = self.client.action_sse_post(request=f"api/llm-application/open/model-api/{self.application_id}/sse-invoke", **params)

        # 使用 iter_lines 方法处理 SSE 响应
        current_id = None
        for line in replies.iter_lines():
            if line:  # 过滤掉心跳信号（即空行）
                line_utf8 = line.decode('utf-8')

                # 如果这一行是一个事件，忽略它
                if line_utf8.startswith("event:"):
                    continue

                # 如果这一行是一个 ID，更新当前的 ID
                elif line_utf8.startswith("id:"):
                    current_id = line[3:]

                # 如果这一行是数据，立即返回结果
                elif line_utf8.startswith("data:"):
                    id = current_id
                    text = line_utf8[5:]
                    if id is not None and text is not None:
                        yield {
                            "id": id,
                            "created": int(time.time()),
                            "model": self.model,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "content": text,
                                },
                            }],
                        }
        yield {
            "id": current_id,
            "created": int(time.time()),
            "model": self.model,
            "usage": {},
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "delta": {
                    "role": "assistant",
                    "content": "",
                },
            }],
        }
