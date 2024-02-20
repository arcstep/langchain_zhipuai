import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

from zhipuai import ZhipuAI

client = ZhipuAI() 

response = client.chat.completions.create(
    model="glm-4",
    messages=[
        {"role": "user", "content": "你好"},
    ],
)
print(response.choices[0].message)