from typing import Dict, Any

def convert_to_retrieval_tool(
    knowledge_id: str, prompt_template: str=None,
) -> Dict[str, Any]:
    '''Convert a retrieval tool for ZhipuAI.
    
    Args:
        knowledge_id: 当涉及到知识库ID时，请前往开放平台的知识库模块进行创建或获取。
        prompt_template: 请求模型时的知识库模板，默认模板：
        
        从文档
        """
        {{knowledge}}
        """
        中找问题
        """
        {{question}}
        """
        的答案，找到答案就仅使用文档语句回答问题，找不到答案就用自身知识回答并且告诉用户该信息不是来自文档。
        不要复述问题，直接开始回答

        注意：用户自定义模板时，知识库内容占位符 和用户侧问题占位符必是 {{knowledge}} 和{{question}}，
        其他模板内容用户可根据实际场景定义
    '''
    return ({
        "type": "retrieval",
        "retrieval": {
            "knowledge_id": knowledge_id,
            "prompt_template": prompt_template
        }
    })

def convert_to_web_search_tool(
    search_query: str=None, enable: bool=True,
) -> Dict[str, Any]:
    '''Convert a web_search tool for ZhipuAI.
    
    Args:
        knowledge_id: 当涉及到知识库ID时，请前往开放平台的知识库模块进行创建或获取。
        search_query: 强制搜索自定义关键内容，此时模型会根据自定义搜索关键内容返回的结果作为背景知识来回答用户发起的对话。
    '''
    return ({
        "type": "web_search",
        "web_search": {
            "enable": enable,
            "search_query": search_query
        }
    })
