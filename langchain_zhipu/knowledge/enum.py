from enum import Enum

class Status(Enum):
  PROCESSING = 0                    # 向量化过程中
  SUCCESS = 1                       # 向量化成功
  FAILED = 2                        # 向量化失败

class ErrorCode(Enum):
  LIMIT_REACHED = 10001             # 知识不可用 ，知识库空间已达上限
  WORD_LIMIT_REACHED = 10002        # 知识不可用 ，知识库空间已达上限(字数超出限制)

class DocumentType(Enum):
  TXT = 1
  PDF = 3
  URL = 4
  DOCX = 5
  XLSX = 6