import requests
import json
import time
import jwt

class RestAPI():
    """
    访问大模型的REST API

    初始化:
        - base_url 访问大模型的地址
        - apikey 默认从环境变量 ZHIPUAI_API_KEY 读取

    """
    
    base_url: str = None
    api_key: str = None
    exp_seconds: int = 100
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key

    def action_get(self, request: str):
        """GET"""
        
        url = f'{self.base_url}/{request}'
        headers = {
            'accept': "*/*",
            'Authorization': f'Bearer {self.generate_token()}',
            'Content-Type': 'application/json',
        }

        response = requests.get(url, headers=headers)
        
        return json.loads(response.text)

    def action_post(rself, equest: str, data):
        """POST"""
        
        url = f'{self.base_url}/{request}'
        headers = {
            'Authorization': f'Bearer {self.generate_token()}',
            'Content-Type': 'application/json',
        }

        response = requests.post(url, headers=headers, data=json.dumps(data))
        
        return json.loads(response.text)

    def action_put(self, request: str, id: str, data):
        """PUT"""
        
        url = f'{self.base_url}/{request}/{id}'
        headers = {
            'Authorization': f'Bearer {self.generate_token()}',
            'Content-Type': 'application/json',
        }

        response = requests.put(url, headers=headers, data=json.dumps(data))
        
        return json.loads(response.text)

    def generate_token(self) -> str:
        """
        api_key : 从智谱AI申请的 ZHIPUAI_API_KEY
        exp_seconds : JWT的过期时间，单位为秒数，调用时应当每次重新生成，并在exp_seconds规定时间内完成调用
        """

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