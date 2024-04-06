from typing import Optional, List
from langchain_core.pydantic_v1 import BaseModel

class KnowledgeLibrary(BaseModel):
    id: Optional[str]               # 知识库唯一 id
    embedding_id: Optional[int]     # 知识库绑定的向量化模型
    name: Optional[str]             # 知识库名称
    description: Optional[str]      # 知识库描述
    background: Optional[str]       # 背景颜色
    icon: Optional[str]             # 知识库图标
    word_num: Optional[int]         # 知识库总字数
    length: Optional[int]           # 知识库总大小（字节）
    document_size: Optional[int]    # 知识文件数量

class KnowledgeFilesMeta(BaseModel):
    knowledge_type: Optional[int]   # 知识库文档的类型：
                                    # 1 文章知识：pdf、url、docx
                                    # 2 问答知识-文档：支持pdf、url、docx
                                    # 3 问答知识-表格：支持xlsx
                                    # 4 商品库-表格：支持xlsx
                                    # 5 自定义：支持pdf、url、docx
    custom_seperator: Optional[List[str]] # 当知识类型为自定义时的分割字符串，默认\n
    custom_sentence_size: Optional[List[int]] # 当知识类型为自定义时的切片大小，20-2000，默认为300

class KnowledgeUrlsMeta(BaseModel):
    url: Optional[str]              # 网页或文件下载地址
    knowledge_type: Optional[int]   # 知识库文档的类型：
                                    # 1 文章知识：pdf、url、docx
                                    # 2 问答知识-文档：支持pdf、url、docx
                                    # 3 问答知识-表格：支持xlsx
                                    # 4 商品库-表格：支持xlsx
                                    # 5 自定义：支持pdf、url、docx
    custom_seperator: Optional[List[str]] # 当知识类型为自定义时的分割字符串，默认\n
    custom_sentence_size: Optional[List[int]] # 当知识类型为自定义时的切片大小，20-2000，默认为300
    

class UploadDetail(BaseModel):
    key: Optional[str]              # 上传文档接口返回的key
    file_name: Optional[str]        # 文档名称

class FailInfo(BaseModel):
    embedding_code: Optional[int]   # 失败码
    embedding_msg: Optional[str]    # 失败原因

class DocumentData(BaseModel):
    id: Optional[str]               # 知识唯一 id
    custom_separator: Optional[str] # 切片规则
    sentence_size: Optional[str]    # 切片大小
    length: Optional[int]           # 文件大小（字节）
    word_num: Optional[int]         # 文件字数
    name: Optional[str]             # 文件名
    url: Optional[str]              # 文件下载链接
    embedding_stat: Optional[int]   # 0:向量化中 1:向量化完成 2:向量化失败
    failInfo: Optional[FailInfo]    # 失败原因 向量化失败embedding_stat=2的时候 会有此值

class ApplicationData(BaseModel):
    id: Optional[str]               # 应用唯一 id
    name: Optional[str]             # 应用名称
    desc: Optional[str]             # 应用描述
    prompt: Optional[str]           # 模版,格式为“ {{知识}}xxx{{用户}} ”,必须包含知识和用户 ，且占位 符 {{知识}} 有且只有一个 ， 占位符 {{用户}} 有且只有一个
    top_p: Optional[str]            # 默认0.7 ， 用温度取样的另一种方法 ，称为核取样
    temperature: Optional[str]      # 默认0.95 ，采样温度 ，控制输出的随机性 ，必须为正数
    knowledge_ids: Optional[List[int]]  # 知识库id列表
    slice_count: Optional[int]      # 分片数量, 默认是1
    model: Optional[str]            # 取值范围：chatglm_pro,chatglm_std, chatglm_lite,chatglm_lite_32k
    icon_color: Optional[str]       # 取值范围：blue ，red ，orange ，purple ，sky
    icon_type: Optional[str]        # 取值范围： robot_assist、robot_navigate、robot_guardian、robot_service、robot_messenger、robot_chat、robot_engineer
