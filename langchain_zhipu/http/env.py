import os
DEFAULT_BASE_URL="https://open.bigmodel.cn"

def init_base_url(base_url: str):
    if base_url:
        return base_url

    data_env = os.getenv("ZHIPUAI_BASE_URL")

    if data_env is None:
        return DEFAULT_BASE_URL
    else:
        return data_env

def init_api_key(api_key: str):
  if api_key:
    return api_key
  else:
    return os.getenv("ZHIPUAI_API_KEY")
