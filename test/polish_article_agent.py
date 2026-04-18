"""
图片生成智能体
使用 LangGraph 框架，根据 controller_agent 传来的消息并行生成图片
"""

import json
import os
import asyncio
import re
from typing import TypedDict, Dict, Any, Optional, List,Annotated
from pathlib import Path

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage,AIMessage
from dotenv import load_dotenv

from src.utils.logs import logger
from langchain_core.messages import BaseMessage

class PolishState(TypedDict):
    messages: Annotated[List[BaseMessage], "用户消息列表"]
    user_id: Optional[str]


def polish_article_node(state: PolishState) -> PolishState:
    # 核心修复：messages 必须是【列表】+【BaseMessage对象】
    reply = f"用户ID：123456的文章已经润色完成"
    
    # 返回完整状态：保留原有字段 + 正确格式的messages
    return {
        # 继承原有状态（不丢失user_id）
        **state,
        # 必须是列表！必须是AIMessage/SystemMessage等消息对象
        "messages": [AIMessage(content=reply)]
    }

def create_polish_article_graph():
    """
    创建文章润色工作流图

    Returns:
        编译后的工作流图
    """
    logger.info("注册文章润色智能体")

    workflow = StateGraph(PolishState)

    workflow.add_node("polish_article", polish_article_node)

    workflow.set_entry_point("polish_article")

    workflow.add_edge("polish_article", END)

    return workflow.compile()

