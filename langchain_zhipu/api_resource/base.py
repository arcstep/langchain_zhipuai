import time
import jwt
import os

def generate_token(apikey: str = None, exp_seconds: int = 100) -> str:
    """
    apikey : 从智谱AI申请的 ZHIPUAI_API_KEY
    exp_seconds : JWT的过期时间，单位为秒数，调用时应当每次重新生成，并在exp_seconds规定时间内完成调用
    
    需要将生成的鉴权 token 放入 HTTP 的 Authorization header 头中：
        Authorization: 鉴权token
        Example：curl请求中的token参数示例

    curl --location 'https://open.bigmodel.cn/api/paas/v4/chat/completions' \
    --header 'Authorization: Bearer <你的token>' \
    --header 'Content-Type: application/json' \
    --data '{
        "model": "glm-4",
        "messages": [
            {
                "role": "user",
                "content": "你好"
            }
        ]
    }'
    """

    if apikey is None:
	    apikey = os.environ.get('ZHIPUAI_API_KEY') 
    
    try:
        id, secret = apikey.split(".")
    except Exception as e:
        raise Exception("invalid apikey", e)

    payload = {
        "api_key": id,
        "exp": int(round(time.time() * 1000)) + exp_seconds * 1000,
        "timestamp": int(round(time.time() * 1000)),
    }

    return jwt.encode(
        payload,
        secret,
        algorithm="HS256",
        headers={"alg": "HS256", "sign_type": "SIGN"},
    )