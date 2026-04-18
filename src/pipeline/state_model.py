from typing import TypedDict, Optional, Dict, Any, List
import operator
from typing import Annotated, Sequence
from langgraph.graph.message import add_messages

class MessageState(TypedDict):
    """状态字典，定义工作流中传递的状态数据结构"""

    messages: Annotated[List, add_messages]
    next_node: str
    #用户信息
    user_id: str
    user_query: str
    depth_demand: str = None
    need_memory_manage: bool = False
    is_lookup_outline: bool = False
    user_document: Optional[List[str]]=None
    user_document_meta: Dict[str, Any]=None

    #每步的结果状态
    status: str = "success"
    error: Optional[str] = None

    # 任务分解相关
    subtasks: List[Dict[str, Any]]

    # 搜索结果相关
    search_results: List[Dict[str, Any]]
    search_status: str
    total_results: int

    # 搜索简化相关
    search_summary: Optional[Dict[str, Any]]=None
    #大纲
    outline_content: Optional[Dict[str, Any]]=None
    outline_feedback: Optional[str]=None

    #章节内容
    chapter_content: Annotated[List[Dict[str, Any]],operator.add]=[]
