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

class UserState(TypedDict):
    messages: Annotated[List[BaseMessage], "用户消息列表"]
    user_id: Optional[str]


def search_user_node(state: UserState) -> UserState:
        # 从状态中获取 user_id（你的子智能体核心参数）
    user_id = "12345"
    state["user_id"]="12345"
    # 核心修复：messages 必须是【列表】+【BaseMessage对象】
    reply = f"用户ID：{user_id}，男士，已婚，每月薪资3000"
    
    # 返回完整状态：保留原有字段 + 正确格式的messages
    return {
        # 继承原有状态（不丢失user_id）
        **state,
        # 必须是列表！必须是AIMessage/SystemMessage等消息对象
        "messages": [AIMessage(content=reply)]
    }

def create_search_user_graph():
    """
    创建图片生成工作流图

    Returns:
        编译后的工作流图
    """
    logger.info("注册图片生成智能体")

    workflow = StateGraph(UserState)

    workflow.add_node("search", search_user_node)

    workflow.set_entry_point("search")

    workflow.add_edge("search", END)

    return workflow.compile()
