"""
排版智能体
负责文章组装和排版优化
"""

import re
import os
import json
from typing import List, Dict, Any, Optional, TypedDict
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

from src.pipeline.state_model import MessageState
from src.core.model_client import model_client
from src.prompts.load_prompt import load_prompt
from src.utils.logs import logger

load_dotenv()


class LayoutState(TypedDict):
    """排版智能体状态"""
    messages: List
    article_content: str
    layout_demand: str
    user_id: str
    formatted_article: str


class LayoutAgent:
    """排版智能体"""

    PROMPT_NAME = "layout_prompt"
    DATA_DIR = Path("D:\\study_notebook\\DeepStudy_jupyter\\SGG\\OmniWriter\\data\\")

    def __init__(self):
        """初始化排版智能体"""
        self._prompt_template: str = load_prompt(self.PROMPT_NAME)

    def _read_file(self, file_path: Path) -> Optional[str]:
        """
        读取文件内容

        Args:
            file_path: 文件路径

        Returns:
            文件内容，失败返回 None
        """
        try:
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
        except Exception as e:
            logger.warning(f"读取文件失败 {file_path}: {str(e)}")
        return None

    def _save_file(self, file_path: Path, content: str) -> bool:
        """
        保存文件内容

        Args:
            file_path: 文件路径
            content: 文件内容

        Returns:
            是否保存成功
        """
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"文件已保存到: {file_path}")
            return True
        except Exception as e:
            logger.error(f"保存文件失败 {file_path}: {str(e)}")
            return False

    def _assemble_article(self, user_id: str) -> Optional[str]:
        """
        组装文章：处理图片和代码占位符

        Args:
            user_id: 用户 ID

        Returns:
            组装后的文章内容
        """
        user_dir = self.DATA_DIR / user_id
        article_path = user_dir / "article.txt"

        article_content = self._read_file(article_path)
        if not article_content:
            logger.error(f"文章内容为空或读取失败: {article_path}")
            return None

        assembled_content = article_content

        image_pattern = r'\{image_(\d+):([^}]+)\}'
        image_matches = re.findall(image_pattern, assembled_content)
        
        for image_id, image_info in image_matches:
            image_dir = user_dir / "images"
            if not image_dir.exists():
                logger.warning(f"图片目录不存在: {image_dir}")
                continue

            image_files = list(image_dir.glob(f"*{image_id}*"))
            if not image_files:
                image_files = list(image_dir.glob("*"))
            
            if image_files:
                image_path = image_files[0]
                image_tag = f'<div align="center"><img src="{image_path}" alt="图片{image_id}"></div>'
                placeholder = f"{{image_{image_id}:{image_info}}}"
                assembled_content = assembled_content.replace(placeholder, image_tag)
            else:
                logger.warning(f"未找到图片文件: {image_id}")

        code_pattern = r'\{code_(\d+):([^}]+)\}'
        code_matches = re.findall(code_pattern, assembled_content)
        
        for code_id, code_info in code_matches:
            code_dir = user_dir / "codes"
            if not code_dir.exists():
                logger.warning(f"代码目录不存在: {code_dir}")
                continue

            code_files = list(code_dir.glob(f"*{code_id}*"))
            if not code_files:
                code_files = list(code_dir.glob("*"))
            
            if code_files:
                code_path = code_files[0]
                code_content = self._read_file(code_path)
                if code_content:
                    code_tag = f"```\n{code_content}\n```"
                    placeholder = f"{{code_{code_id}:{code_info}}}"
                    assembled_content = assembled_content.replace(placeholder, code_tag)
            else:
                logger.warning(f"未找到代码文件: {code_id}")

        return assembled_content

    def _parse_user_id(self, messages: List) -> Optional[str]:
        """
        从 messages 中解析 user_id

        Args:
            messages: 消息列表

        Returns:
            user_id
        """
        if not messages:
            return None

        for message in reversed(messages):
            if hasattr(message, 'content'):
                content = message.content
                
                try:
                    msg_dict = json.loads(content)
                    if isinstance(msg_dict, dict) and 'user_id' in msg_dict:
                        return msg_dict['user_id']
                except (json.JSONDecodeError, TypeError):
                    pass
                
                user_id_pattern = r'user_id[：:]\s*([a-zA-Z0-9_-]+)'
                match = re.search(user_id_pattern, content)
                if match:
                    return match.group(1)

        return "thread_id_123"

    def _parse_layout_demand(self, messages: List) -> str:
        """
        从 messages 中解析排版需求

        Args:
            messages: 消息列表

        Returns:
            排版需求
        """
        if not messages:
            return ""

        for message in reversed(messages):
            if hasattr(message, 'content'):
                content = message.content
                
                try:
                    msg_dict = json.loads(content)
                    if isinstance(msg_dict, dict) and 'layout_demand' in msg_dict:
                        return msg_dict['layout_demand']
                except (json.JSONDecodeError, TypeError):
                    pass
                
                demand_pattern = r'排版[需求要求]*[：:]\s*([^\n]+)'
                match = re.search(demand_pattern, content)
                if match:
                    return match.group(1).strip()

        return ""

    def _load_article_node(self, state: LayoutState) -> Dict[str, Any]:
        """
        加载文章内容节点

        Args:
            state: 当前状态

        Returns:
            更新后的状态
        """
        messages = state.get("messages", [])
        if not messages:
            logger.error("消息列表为空，无法解析 user_id 和 layout_demand")
            return {"article_content": "", "user_id": "", "layout_demand": ""}

        user_id = self._parse_user_id(messages)
        layout_demand = self._parse_layout_demand(messages)
                
        article_content = self._assemble_article(user_id)
        
        if not article_content:
            logger.error("文章内容加载失败")
            return {"article_content": "", "user_id": user_id, "layout_demand": layout_demand}
        else:
            return {
                "article_content": article_content,
                "user_id": user_id,
                "layout_demand": layout_demand
            }

    async def _format_article_node(self, state: LayoutState) -> Dict[str, Any]:
        """
        排版文章节点

        Args:
            state: 当前状态

        Returns:
            更新后的状态
        """
        article_content = state.get("article_content", "")
        layout_demand = state.get("layout_demand", "")

        if not article_content:
            logger.error("文章内容为空，无法排版")
            return {"formatted_article": ""}

        try:
            user_message_content = f"""## 文章内容
{article_content}

## 排版需求
{layout_demand if layout_demand else "请按照默认的排版规范进行排版"}

请根据以上内容进行排版优化。"""

            messages = [
                SystemMessage(content=self._prompt_template),
                HumanMessage(content=user_message_content)
            ]

            response = await model_client.call_llm(messages)
            
            if response:
                formatted_content = response.content
                return {"formatted_article": formatted_content}
            else:
                logger.error("LLM 调用失败，使用原始内容")
                return {"formatted_article": article_content}

        except Exception as e:
            logger.error(f"排版过程出错: {str(e)}")
            return {"formatted_article": article_content}

    def _save_article_node(self, state: LayoutState) -> Dict[str, Any]:
        """
        保存文章节点

        Args:
            state: 当前状态

        Returns:
            更新后的状态
        """
        user_id = state.get("user_id", "thread_id_123")
        formatted_article = state.get("formatted_article", "")

        if not formatted_article:
            logger.error("排版后的文章内容为空，无法保存")
            return {}

        user_dir = self.DATA_DIR / user_id
        article_path = user_dir / "article.md"

        success = self._save_file(article_path, formatted_article)
        
        if success:
            logger.info(f"排版后的文章已保存到: {article_path}")
            result_message = AIMessage(content=json.dumps({
                "role": "assistant",
                "type": "layout_result",
                "user_id": user_id,
                "content": f"排版后的文章已保存到: {article_path}"
            }, ensure_ascii=False))
            return {"messages": [result_message]}
        else:
            logger.error(f"文章保存失败: {article_path}")
            return {}

    def create_layout_graph(self) -> StateGraph:
        """
        构建排版智能体的 LangGraph 工作流

        Returns:
            StateGraph 实例
        """
        workflow = StateGraph(LayoutState)

        workflow.add_node("load_article", self._load_article_node)
        workflow.add_node("format_article", self._format_article_node)
        workflow.add_node("save_article", self._save_article_node)

        workflow.set_entry_point("load_article")
        workflow.add_edge("load_article", "format_article")
        workflow.add_edge("format_article", "save_article")
        workflow.add_edge("save_article", END)

        return workflow.compile()

if __name__ == "__main__":
    import asyncio
    async def test():
        message = {
            "user_id": "thread_id_123",
            "layout_demand": "请按照默认的排版规范进行排版"
        }
        test_messages = [
            HumanMessage(content=json.dumps(message, ensure_ascii=False))
        ]
        state = LayoutState(
            messages=test_messages,
            article_content="",
            layout_demand="",
            user_id="",
            formatted_article=""
        )

        layout_graph = LayoutAgent().create_layout_graph()
        final_state = await layout_graph.ainvoke(state)
        print("最终状态:")
        if final_state.get("messages"):
            last_msg = final_state["messages"][-1]
            content = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
            print("\n结果消息:")
            print(json.dumps(json.loads(content), ensure_ascii=False, indent=2))

    asyncio.run(test())