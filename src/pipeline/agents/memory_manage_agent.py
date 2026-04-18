import json
import os
from typing import Dict, Any, Optional, List
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages.utils import trim_messages
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from src.pipeline.state_model import MessageState
from src.core.model_client import model_client
from src.prompts.load_prompt import load_prompt
from src.utils.logs import logger

load_dotenv()


class PersonalizationData(BaseModel):
    writing_style: Optional[str] = Field(default=None, description="写作风格偏好")
    tone_preference: Optional[str] = Field(default=None, description="语气偏好")
    topic_interests: List[str] = Field(default_factory=list, description="主题兴趣列表")
    length_preference: Optional[str] = Field(default=None, description="篇幅偏好")
    structure_preference: Optional[str] = Field(default=None, description="结构偏好")
    audience_preference: Optional[str] = Field(default=None, description="读者对象偏好")
    other_preferences: Optional[str] = Field(default=None, description="其他特殊要求")


class MemoryManageResult(BaseModel):
    summary: str = Field(description="历史消息总结")
    personalization: PersonalizationData = Field(description="用户个性化需求数据")


class MemoryManageAgent:
    """
    记忆管理智能体
    
    负责两件事：
    1. 总结历史对话消息，防止超出模型上下文限制
    2. 提取用户的个性化需求（写作偏好、风格喜好等）
    """
    
    PROMPT_NAME = "memory_manage_prompt"
    KEEP_LATEST_MESSAGES = 5
    
    def __init__(self):
        self._prompt_template: str = load_prompt(self.PROMPT_NAME)
    
    def _format_messages_for_summary(self, messages: List) -> str:
        """将消息列表格式化为文本，供 LLM 总结"""
        formatted_parts = []
        for msg in messages:
            role = getattr(msg, 'role', msg.get('role', 'unknown'))
            content = getattr(msg, 'content', msg.get('content', ''))
            if role == 'human' or role == 'user':
                formatted_parts.append(f"[用户]: {content}")
            elif role == 'ai' or role == 'assistant':
                formatted_parts.append(f"[助手]: {content}")
            elif role == 'system':
                formatted_parts.append(f"[系统]: {content}")
        
        return "\n".join(formatted_parts)
    
    def _build_messages(self, history_messages: List) -> List:
        """构建 LLM 消息列表"""
        history_text = self._format_messages_for_summary(history_messages)
        prompt = self._prompt_template.format(
            history_messages=history_text
        )
        
        return [
            SystemMessage(content=prompt)
        ]
    
    async def manage_memory(self, messages: List, user_id: str) -> Dict[str, Any]:
        """
        执行记忆管理和个性化提取
        
        Args:
            messages: 完整的消息列表
            user_id: 用户ID
            
        Returns:
            包含更新后消息和个性化数据的字典
        """
        if len(messages) <= self.KEEP_LATEST_MESSAGES:
            logger.info("消息数量未超过阈值，无需总结")
            return {
                "messages": messages,
                "personalization": None,
                "summarized": False
            }
        
        latest_messages = messages[-self.KEEP_LATEST_MESSAGES:]
        older_messages = messages[:-self.KEEP_LATEST_MESSAGES]
        
        try:
            llm_messages = self._build_messages(older_messages)
            
            llm = model_client.llm
            if llm is None:
                raise RuntimeError("LLM is not initialized")
            structured_llm = llm.with_structured_output(MemoryManageResult)
            response: MemoryManageResult = await structured_llm.ainvoke(llm_messages)
            
            summary_message = HumanMessage(content=f"[历史对话总结]\n{response.summary}")
            updated_messages = [summary_message] + list(latest_messages)
            
            personalization_data = response.personalization.model_dump()
            
            self._save_personalization(user_id, personalization_data)
            
            logger.info(f"记忆管理完成：总结了 {len(older_messages)} 条历史消息")
            
            return {
                "messages": updated_messages,
                "personalization": personalization_data,
                "summarized": True
            }
            
        except Exception as e:
            logger.error(f"记忆管理失败: {e}")
            raise
    
    def _save_personalization(self, user_id: str, personalization_data: Dict[str, Any]):
        """
        保存用户个性化数据到文件
        
        Args:
            user_id: 用户ID
            personalization_data: 个性化数据字典
        """
        try:
            base_dir = Path("data") / user_id
            base_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = base_dir / "user_personalization.txt"
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(personalization_data, ensure_ascii=False, indent=2))
            
            logger.info(f"用户个性化数据已保存到: {file_path}")
            
        except Exception as e:
            logger.error(f"保存个性化数据失败: {e}")


async def memory_manage_agent_node(state: MessageState) -> Dict[str, Any]:
    """
    记忆管理智能体节点函数
    
    处理逻辑：
    1. 从状态中获取 messages 列表
    2. 保留最新的 5 条消息
    3. 对其余历史消息进行总结
    4. 提取用户个性化需求并保存到文件
    5. 将总结后的消息插入到最新消息之前
    
    Args:
        state: 当前工作流状态
        
    Returns:
        更新后的状态字典，包含更新后的 messages
    """
    agent = MemoryManageAgent()
    
    messages = state.get("messages", [])
    user_id = state.get("user_id", "default_user")
    
    result = await agent.manage_memory(list(messages), user_id)
    
    return {
        "messages": result["messages"]
    }
