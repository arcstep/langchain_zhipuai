from base import generate_token, DEFAULT_BASE_URL
from ..types.knowledge import (
    KnowledgeBase,
    UploadDetail,
    FailInfo,
    DocumentData,
    ApplicationData,
    Status,
    ErrorCode,
    DocumentType,
)

import requests
import json

def action_post(request: str):
    url = f'{DEFAULT_BASE_URL}/completions'
    headers = {
        'Authorization': f'Bearer {generate_token()}',
        'Content-Type': 'application/json',
    }
    data = {
        "model": "glm-4",
        "messages": [
            {
                "role": "user",
                "content": "你好"
            }
        ]
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))