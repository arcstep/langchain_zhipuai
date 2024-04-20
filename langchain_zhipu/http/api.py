import cachetools.func
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

    def __hash__(self):
        return hash((self.base_url, self.api_key, self.exp_seconds))

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

        response = self._retry_request(self.session.get, url, headers=headers, params=kwargs)

        obj = json.loads(response.text)
        if obj["code"] == 200:
            return obj
        else:
            raise BaseException(obj)        

    def action_post(self, request: str, files=None, **kwargs):
        """POST"""
        
        url = f'{self.base_url}/{request}'
        headers = self._generate_headers(files)
        # 如果提供了 files 参数就需要将 kwargs 视作表单来处理
        data = kwargs if files else json.dumps(kwargs)

        response = self._retry_request(self.session.post, url, headers=headers, data=data, files=files)

        if response.text:
            resp = response.json()
            if "code" in resp and resp["code"] != 200:
                raise BaseException(resp)
            else:
                return resp
        else:
            return {}
    
    def action_sse_post(self, request: str, **kwargs):
        """POST for SSE"""
        
        url = f'{self.base_url}/{request}'
        headers = self._generate_headers()
        data = json.dumps(kwargs)

        response =  self._retry_request(self.session.post, url, headers=headers, data=data, stream=True)
        return response

    def action_put(self, request: str, **kwargs):
        """PUT"""
        
        url = f'{self.base_url}/{request}'
        headers = self._generate_headers()
        data = json.dumps(kwargs)
        
        response = self._retry_request(self.session.put, url, headers=headers, data=data)
        return response.json()

    def action_delete(self, request: str):
        """DELETE"""
        
        url = f'{self.base_url}/{request}'
        headers = self._generate_headers()
        response = self.session.delete(url, headers=headers)
        
        response = self._retry_request(self.session.delete, url, headers=headers)
        return response.json()
        
    def _generate_headers(self, files = None) -> dict:
        headers = {
            'Authorization': f'Bearer {self._generate_token()}',
            'Content-Type': 'application/json' if not files else None,
        }
        return headers

    # 优先使用60秒内调用过的缓存
    @cachetools.func.ttl_cache(maxsize=10, ttl=60)
    def _generate_token(self) -> str:
        try:
            id, secret = self.api_key.split(".")
        except Exception as e:
            raise BaseException("Invalid ZHIPUAI_API_KEY", e)

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
        
    def _retry_request(self, func, *args, **kwargs):
        last_exception = None
        for _ in range(3):
            try:
                response = func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                print("Retry for HTTP Error ...")
                continue
            else:
                # 如果响应状态码不是 200，也引发一个异常
                if response.status_code != 200:
                    raise Exception({
                        "status_code": response.status_code,
                        "headers": response.headers,
                        "text": response.text,
                    })
                return response
        else:
            raise last_exception