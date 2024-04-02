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

def models_list():
    return action_get("embedding")

def embedding_create(
    embedding_id: id,  # 知识库绑定的向量化模型
    name: str,
    description: str = None,
    background: str = "blue", # 背景颜色（给枚举） *blue*, *red*, *orange*, *purple*, "sky"
    icon: str = "question", # 知识库图标（给枚举）, question :问号, book :书籍, seal :印章, wrench :扳手, tag :标签,horn :喇叭, house :房子    
):
    return action_post(
        "knowledge",
        data={
            "embedding_id": embedding_id,
            "name": name,
            "description": description,
            "background": background,
            "icon": icon,
        }
    )

