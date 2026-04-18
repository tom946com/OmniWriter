import json
import os
from typing import Dict, Any, Optional, List

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.types import interrupt, Command
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from src.pipeline.state_model import MessageState
from src.core.model_client import model_client
from src.prompts.load_prompt import load_prompt
from src.utils.logs import logger

load_dotenv()


MEMORY_MANAGE_TRIGGER_RATIO = float(os.getenv("MEMORY_MANAGE_TRIGGER_RATIO", "0.8"))
DEFAULT_CONTEXT_WINDOW = int(os.getenv("DEFAULT_LLM_CONTEXT_WINDOW", "128000"))
MODEL_CONTEXT_WINDOWS = {
    "GLM-5": 128000,
    "DeepSeek-V3.2": 64000,
    "DeepSeek-R1-0528": 64000,
}


def _normalize_content(content: Any) -> str:
    """Normalize message content to plain text for token estimation."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _to_langchain_messages(messages: List[Any]) -> List[Any]:
    """Convert raw state messages to langchain message objects."""
    converted: List[Any] = []
    for msg in messages:
        if isinstance(msg, (SystemMessage, HumanMessage, AIMessage)):
            converted.append(msg)
            continue

        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = _normalize_content(msg.get("content", ""))
            if role in ("assistant", "ai"):
                converted.append(AIMessage(content=content))
            elif role in ("system",):
                converted.append(SystemMessage(content=content))
            else:
                converted.append(HumanMessage(content=content))
            continue

        role = getattr(msg, "role", "user")
        content = _normalize_content(getattr(msg, "content", ""))
        if role in ("assistant", "ai"):
            converted.append(AIMessage(content=content))
        elif role in ("system",):
            converted.append(SystemMessage(content=content))
        else:
            converted.append(HumanMessage(content=content))

    return converted


def _estimate_message_tokens(messages: List[Any]) -> int:
    """
    Estimate token usage for state messages.
    Prefer model tokenizer when available; fallback to char-based estimate.
    """
    llm = model_client.llm
    lc_messages = _to_langchain_messages(messages)

    if llm is not None and hasattr(llm, "get_num_tokens_from_messages"):
        try:
            return int(llm.get_num_tokens_from_messages(lc_messages))
        except Exception as e:
            logger.warning(f"token counting via model failed, fallback to char estimate: {e}")

    total_chars = sum(len(_normalize_content(getattr(m, "content", ""))) for m in lc_messages)
    return max(total_chars, 1)


def _resolve_context_window() -> int:
    """Resolve model context window size from env or known model mapping."""
    env_context_window = os.getenv("DEFAULT_LLM_CONTEXT_WINDOW")
    if env_context_window:
        try:
            return int(env_context_window)
        except ValueError:
            logger.warning(f"invalid DEFAULT_LLM_CONTEXT_WINDOW value: {env_context_window}")

    model_name = os.getenv("DEFAULT_LLM_NAME", "")
    if model_name in MODEL_CONTEXT_WINDOWS:
        return MODEL_CONTEXT_WINDOWS[model_name]

    llm = model_client.llm
    runtime_model_name = str(
        getattr(llm, "model_name", "") or getattr(llm, "model", "")
    )
    for known_model, context_window in MODEL_CONTEXT_WINDOWS.items():
        if known_model.lower() in runtime_model_name.lower():
            return context_window

    return DEFAULT_CONTEXT_WINDOW


def _should_manage_memory(messages: List[Any]) -> bool:
    """Decide whether to trigger memory compression."""
    if not messages:
        return False

    context_window = _resolve_context_window()
    token_usage = _estimate_message_tokens(messages)
    usage_ratio = token_usage / max(context_window, 1)

    logger.info(
        f"context usage check: tokens={token_usage}, context_window={context_window}, usage_ratio={usage_ratio:.2%}"
    )
    return usage_ratio >= MEMORY_MANAGE_TRIGGER_RATIO


class RouteResult(BaseModel):
    route_to: str = Field(description="路由目标节点，只能是 title_decomposer 或 deepagents")
    user_query: str = Field(description="核心写作任务")
    depth_demand: str = Field(description="额外要求")
    reason: str = Field(description="路由原因说明")


class HeadAgent:
    """
    路由头智能体
    
    识别用户是要从零生成文章还是修改已有文章，并提取可写入状态的关键字段。
    根据用户输入判断路由到 title_decomposer 或 deepagents 节点。
    """
    
    PROMPT_NAME = "head_prompt"
    
    def __init__(self):
        self._prompt_template: str = load_prompt(self.PROMPT_NAME)
    
    def _build_messages(self, user_query: str, depth_demand: Optional[str] = None) -> list:
        """构建 LLM 消息列表"""
        prompt = self._prompt_template.format(
            user_query=user_query,
            depth_demand=depth_demand or "无"
        )
        
        return [
            SystemMessage(content=prompt),
            HumanMessage(content=user_query)
        ]
    
    async def route(self, user_query: str, depth_demand: Optional[str] = None) -> Dict[str, Any]:
        """
        执行路由判断和状态提取
        
        Args:
            user_query: 用户输入的查询内容
            depth_demand: 用户额外需求
            
        Returns:
            包含路由结果和提取状态的字典
        """
        messages = self._build_messages(user_query, depth_demand)
        
        try:
            llm = model_client.llm
            if llm is None:
                raise RuntimeError("LLM is not initialized")
            
            structured_llm = llm.with_structured_output(RouteResult)
            response: RouteResult = await structured_llm.ainvoke(messages)
            
            result = {
                "next_node": response.route_to,
                "user_query": response.user_query,
                "depth_demand": response.depth_demand,
                "reason": response.reason
            }
            
            logger.info(f"路由结果: {result}")
            return result
            
        except Exception as e:
            logger.error(f"路由判断失败: {e}")
            raise


async def head_agent_node(state: MessageState) -> Dict[str, Any]:
    """
    路由头智能体节点函数
    
    处理系统入口的路由逻辑：
    1. 从状态中获取用户消息
    2. 调用 LLM 判断路由到哪个节点
    3. 提取并标准化状态字段
    
    Args:
        state: 当前工作流状态
        
    Returns:
        更新后的状态字典，包含 next_node、user_query、depth_demand 等字段
    """
    agent = HeadAgent()
    
    messages = state.get("messages", [])
    if not messages:
        raise ValueError("状态中没有找到消息")
    
    user_query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or (isinstance(msg, dict) and msg.get("role") == "user"):
            user_query = msg.content if hasattr(msg, 'content') else msg.get("content", "")
            break
    
    if not user_query:
        raise ValueError("无法从消息中提取用户查询")
    
    depth_demand = state.get("depth_demand")
    need_memory_manage = _should_manage_memory(list(messages))
    
    result = await agent.route(user_query, depth_demand)
    
    return {
        "next_node": result["next_node"],
        "user_query": result["user_query"],
        "depth_demand": result["depth_demand"],
        "need_memory_manage": need_memory_manage
    }
