import requests
from requests.exceptions import SSLError
import json
import time
import jwt
import os
from typing import Any, Dict
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator

from .env import init_api_key, init_base_url

class RestAPI(BaseModel):
    """
    访问大模型的REST API

    初始化:
        - base_url 访问大模型的地址
        - apikey 默认从环境变量 ZHIPUAI_API_KEY 读取
    """

    base_url: str = None
    """
    访问API的地址
    """

    api_key: str = None
    """
    访问智谱官方的 ZHIPU_API_KEY
    """

    exp_seconds: int = 600
    """
    构造令牌的默认过期时间是600秒
    """
    
    @root_validator()
    def base_validate_environment(cls, values: Dict) -> Dict:
        values["base_url"] = init_base_url(values["base_url"])
        values["api_key"] = init_api_key(values["api_key"])
        values["session"] = requests.Session()
        return values

    def action_get(self, request: str, **kwargs):
        """GET"""
        
        url = f'{self.base_url}/{request}'
        headers = self._generate_headers()
        response = self.session.get(url, headers=headers, params=kwargs)
        
        obj = json.loads(response.text)
        if obj["code"] == 200:
            return obj
        else:
            raise Exception(obj)

    def action_post(self, request: str, files=None, **kwargs):
        """POST"""
        
        url = f'{self.base_url}/{request}'
        headers = self._generate_headers(files)
        
        # 如果提供了 files 参数就需要将 kwargs 视作表单来处理
        data = kwargs if files else json.dumps(kwargs)
        
        response = self.session.post(url, headers=headers, data=data, files=files)
        print(url)
        print(headers)
        print(data)
        print(response)

        obj = json.loads(response.text)
        if obj["code"] == 200:
            return obj
        else:
            raise Exception(obj)
    
    def action_sse_post(self, request: str, **kwargs):
        """POST for SSE"""
        
        url = f'{self.base_url}/{request}'
        headers = self._generate_headers()
        data = json.dumps(kwargs)

        # 尝试发送请求，如果发生 SSLError 异常，重试请求
        print(url)
        print(headers)
        print(data)
        
        for _ in range(3):
            try:
                response = self.session.post(url, headers=headers, data=data, stream=True)
                break
            except SSLError:
                continue
        else:
            raise Exception("Max retries exceeded with SSLError")

        # 检查响应的状态码
        print(response)
        
        if response.status_code != 200:
            raise Exception(f"Request failed with status {response.status_code}")
        
        # 使用 iter_lines 方法处理 SSE 响应
        current_id = None
        for line in response.iter_lines():
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
                    yield ({
                        "id": "",
                        "choices": [
                            {
                                "delta": {
                                    "text": line_utf8[5:],
                                    "finish_reason": "sse-chunk",
                                    "request_id": current_id
                                }
                            }
                        ]
                    })

    def action_put(self, request: str, **kwargs):
        """PUT"""
        
        url = f'{self.base_url}/{request}'
        headers = self._generate_headers()
        response = self.session.put(url, headers=headers, data=json.dumps(kwargs))
        
        obj = json.loads(response.text)
        if obj["code"] == 200:
            return obj
        else:
            raise Exception(obj)

    def action_delete(self, request: str):
        """DELETE"""
        
        url = f'{self.base_url}/{request}'
        headers = self._generate_headers()
        response = self.session.delete(url, headers=headers)
        
        obj = json.loads(response.text)
        if obj["code"] == 200:
            return obj
        else:
            raise Exception(obj)
        
    def _generate_headers(self, files = None) -> dict:
        headers = {
            'Authorization': f'Bearer {self._generate_token()}',
            'Content-Type': 'application/json' if not files else None,
        }
        return headers

    def _generate_token(self) -> str:
        try:
            id, secret = self.api_key.split(".")
        except Exception as e:
            raise Exception("invalid apikey", e)

        payload = {
            "api_key": id,
            "exp": int(round(time.time() * 1000)) + self.exp_seconds * 1000,
            "timestamp": int(round(time.time() * 1000)),
        }

        return jwt.encode(
            payload,
            secret,
            algorithm="HS256",
            headers={"alg": "HS256", "sign_type": "SIGN"},
        )