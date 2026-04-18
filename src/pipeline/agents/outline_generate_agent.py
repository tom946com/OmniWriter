import json
import re
import os
from typing import List, Dict, Any, Optional
from enum import Enum

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.exceptions import OutputParserException
from langgraph.types import interrupt, Command
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, retry_if_exception_type, wait_exponential, before_sleep_log
from src.pipeline.state_model import MessageState
from src.core.model_client import model_client
from src.prompts.load_prompt import load_prompt
from src.utils.logs import logger
import asyncio

load_dotenv()


class SectionRole(str, Enum):
    HOOK = "hook"
    PAIN_POINT = "pain_point"
    ANALYSIS = "analysis"
    SOLUTION = "solution"
    CASE_STUDY = "case_study"
    COUNTER_ARGUMENT = "counter_argument"
    CONCLUSION = "conclusion"


class DisplayType(str, Enum):
    MAIN_HEADING = "main_heading"
    SUB_HEADING = "sub_heading"
    TRANSITION = "transition"


class MicroOutlineItem(BaseModel):
    paragraph_intent: str = Field(description="段落意图，说明该段落要表达的核心内容")
    format_hint: str = Field(description="格式提示，如'纯文本'、'加粗强调'、'列表形式'等")


class OutlineSection(BaseModel):
    chapter_id: str = Field(description="章节唯一标识符，建议使用'chap_01'、'chap_02'等格式")
    theme_anchor: str = Field(description="章节主题锚点，用简短有力的语句概括本章核心")
    section_role: SectionRole = Field(description="章节在全文中的角色定位")
    display_type: DisplayType = Field(description="展示类型")
    custom_heading: Optional[str] = Field(default=None, description="自定义章节标题（可选），应简洁有力、吸引眼球")
    execution_directive: str = Field(description="执行指令，指导写作智能体如何撰写本章")
    word_budget: int = Field(description="字数预算，本章建议的字数范围")
    can_include_material: Optional[List[str]] = Field(default=None, description="可使用的素材关键词列表（可选），供写作时参考")
    micro_outline: List[MicroOutlineItem] = Field(description="微观大纲，定义章节内部的段落结构和节奏")


class GlobalConstraints(BaseModel):
    target_audience: str = Field(description="目标读者群体，决定语言风格和内容深度")
    tone: str = Field(description="文章语调，如'专业严谨'、'轻松幽默'、'犀利批判'等")


class ArticleOutline(BaseModel):
    global_constraints: GlobalConstraints = Field(description="全局约束")
    outline: List[OutlineSection] = Field(description="章节列表")


class OutlineGenerateAgent:
    """
    大纲生成智能体
    
    根据搜索简化后的结果生成完整的公众号文章写作大纲
    """
    
    PROMPT_NAME = "outling_generate_prompt"
    MAX_PARSE_RETRIES = 3
    
    def __init__(self):
        self._prompt_template: str = load_prompt(self.PROMPT_NAME)
    
    def _build_messages(self, search_summary: List[Dict[str, Any]], demand_depth: Optional[str] = None, outline_feedback: Optional[str] = None) -> List:
        """
        构建LLM消息列表
        
        Args:
            search_summary: 搜索简化后的结果列表
            demand_depth: 需求深度说明
            
        Returns:
            LLM消息列表
        """
        formatted_summary = self._format_search_summary(search_summary)
        
        user_prompt = f"""请根据以下内容片段，生成一份公众号文章大纲。
## 输入内容片段
{formatted_summary}"""
        if demand_depth:
            user_prompt += f"""
## 需求深度
{demand_depth}
## 大纲反馈
{outline_feedback}"""
        
        user_prompt += """

请按照要求的JSON格式输出大纲，确保大纲逻辑连贯、有吸引力。"""
        
        return [
            SystemMessage(content=self._prompt_template),
            HumanMessage(content=user_prompt)
        ]
    
    def _format_search_summary(self, search_summary: List[Dict[str, Any]]) -> str:
        """
        格式化搜索简化结果为用户提示词
        
        Args:
            search_summary: 搜索简化结果列表
            
        Returns:
            格式化后的文本
        """
        formatted_parts = []
        
        if not search_summary:
            return "无内容片段"
        
        for i, item in enumerate(search_summary, 1):
            task_id = item.get("task_id", "unknown")
            summary = item.get("summary", "")
            structure = item.get("structure", "")
            retrieval_keywords = item.get("retrieval_keywords", [])
            
            formatted_parts.append(f"### 内容片段 {i}")
            formatted_parts.append(f"**task_id**: {task_id}")
            formatted_parts.append(f"**核心摘要**: {summary}")
            formatted_parts.append(f"**逻辑结构**: {structure}")
            if retrieval_keywords:
                formatted_parts.append(f"**检索关键词**: {', '.join(retrieval_keywords)}")
            formatted_parts.append("")
        
        return "\n".join(formatted_parts)
    
    def _extract_json_from_response(self, response_text: str) -> Optional[str]:
        """
        从响应文本中提取 JSON
        
        Args:
            response_text: LLM 返回的文本
            
        Returns:
            提取的 JSON 字符串，未找到则返回 None
        """
        json_pattern = r'```json\s*([\s\S]*?)\s*```|```\s*([\s\S]*?)\s*```|(\{[\s\S]*"outline"[\s\S]*\})'
        match = re.search(json_pattern, response_text)
        if match:
            return match.group(1) or match.group(2) or match.group(3)
        return None
    
    async def _generate_with_structured_output(
        self, 
        messages: List,
        retry_count: int = 0,
        last_error: Optional[str] = None
    ) -> ArticleOutline:
        """
        使用结构化输出生成大纲，支持重试
        
        Args:
            messages: 消息列表
            retry_count: 当前重试次数
            last_error: 上一次的错误信息
            
        Returns:
            解析后的 ArticleOutline 对象
        """
        current_messages = messages.copy()
        
        if retry_count > 0 and last_error:
            error_message = f"""
之前的输出无法解析，错误信息：{last_error}

请重新生成符合要求的JSON格式大纲，确保：
1. 只输出JSON格式，不要添加额外的说明
2. 确保所有必需字段都存在
3. 确保字段类型正确
"""
            current_messages = current_messages + [HumanMessage(content=error_message)]
        
        try:
            response = await model_client.call_llm(current_messages)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            json_str = self._extract_json_from_response(response_text)
            if not json_str:
                raise OutputParserException("无法从响应中提取 JSON")
            
            parsed_data = json.loads(json_str)
            result = ArticleOutline(**parsed_data)
            return result
            
        except OutputParserException as e:
            logger.warning(f"结构化输出解析失败 (尝试 {retry_count + 1}/{self.MAX_PARSE_RETRIES}): {str(e)}")
            if retry_count < self.MAX_PARSE_RETRIES - 1:
                return await self._generate_with_structured_output(messages, retry_count + 1, str(e))
            else:
                raise
        except json.JSONDecodeError as e:
            logger.warning(f"JSON 解析失败 (尝试 {retry_count + 1}/{self.MAX_PARSE_RETRIES}): {str(e)}")
            if retry_count < self.MAX_PARSE_RETRIES - 1:
                return await self._generate_with_structured_output(messages, retry_count + 1, f"JSON 解析错误: {str(e)}")
            else:
                raise
        except Exception as e:
            logger.error(f"生成过程出错: {str(e)}", exc_info=True)
            raise
    
    async def generate_outline(self, search_summary: List[Dict[str, Any]], demand_depth: Optional[str] = None) -> Dict[str, Any]:
        """
        生成公众号文章大纲
        
        Args:
            search_summary: 搜索简化后的结果列表
            demand_depth: 需求深度说明
            
        Returns:
            包含大纲的字典
        """
        if not search_summary:
            return {
                "status": "failed",
                "outline": [],
                "error": "搜索简化结果为空"
            }
        
        try:
            messages = self._build_messages(search_summary, demand_depth)
            result = await self._generate_with_structured_output(messages)
            
            return {
                "status": "success",
                "outline": result.model_dump()
            }
            
        except Exception as e:
            logger.error(f"生成大纲失败: {str(e)}", exc_info=True)
            return {
                "status": "failed",
                "outline": {},
                "error": str(e)
            }
    
    def generate_outline_sync(self, search_summary: List[Dict[str, Any]], demand_depth: Optional[str] = None) -> Dict[str, Any]:
        """同步版本的生成大纲方法"""
        return asyncio.run(self.generate_outline(search_summary, demand_depth))


def save_outline_to_file(outline_content: Dict[str, Any], user_id: str) -> str:
    """
    将大纲保存到文件中
    
    Args:
        outline_content: 大纲内容字典
        user_id: 用户ID
        
    Returns:
        保存的文件路径
    """
    # 构建文件路径
    data_dir = r"D:\study_notebook\DeepStudy_jupyter\SGG\OmniWriter\data"
    user_dir = os.path.join(data_dir, user_id)
    
    # 创建目录（如果不存在）
    os.makedirs(user_dir, exist_ok=True)
    
    file_path = os.path.join(user_dir, "outline.txt")
    
    # 将大纲内容转换为格式化的JSON字符串并保存
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(outline_content, f, ensure_ascii=False, indent=2)
    
    logger.info(f"大纲已保存到: {file_path}")
    return file_path


async def outline_generate_node(state: MessageState) -> MessageState:
    """
    模块入口函数：生成公众号文章大纲
    
    Args:
        state: Graph状态字典，包含search_summary和demand_depth
        
    Returns:
        更新后的状态字典，包含outline_content
    """
    agent = OutlineGenerateAgent()
    
    search_summary = state.get("search_summary", [])
    demand_depth = state.get("depth_demand", "")
    
    result = await agent.generate_outline(search_summary, demand_depth)
    
    state["outline_content"] = result.get("outline", {})
    state["status"] = result.get("status")
    
    if result.get("error"):
        state["error"] = result.get("error")
    
    return state


async def outline_review_node(state: MessageState) -> MessageState:
    """
    大纲审查节点：保存大纲到文件并使用langgraph中断机制
    
    Args:
        state: Graph状态字典
        
    Returns:
        更新后的状态字典
    """
    
    user_id = state.get("user_id", "default_user")
    outline_content = state.get("outline_content", {})
    is_lookup_outline = state.get("is_lookup_outline", False)
    
    # 保存大纲到文件
    file_path = ""
    if outline_content:
        file_path = save_outline_to_file(outline_content, user_id)
        # state["outline_file_path"] = file_path
    
    # 如果需要查看大纲，则进行中断
    if is_lookup_outline:
        # 使用langgraph的中断机制
        user_response = interrupt({
            "message": f"大纲已生成，并保存在{file_path}中，可进行查看或修改。若有修改意见可进行反馈。",
        })
        
        # 处理用户响应
        if isinstance(user_response, dict):
            action = user_response.get("action", "")
            feedback = user_response.get("feedback", "")
            
            if action == "cancel":
                # 用户取消，保存反馈并返回大纲生成节点重新生成
                state["outline_feedback"] = feedback
                state["status"] = "needs_revision"
                return Command(goto="outline_generate_node")
            else:
                return state
        else:
            return state
    
    # 不需要查看大纲，直接继续
    state["status"] = "success"
    return state


if __name__ == "__main__":
  pass
    # test_search_summary = [
    #     {
    #         "task_id": "task_1",
    #         "summary": "2026年人工智能行业将迎来重大技术突破与发展机遇。中美在AI发展路径上呈现差异化竞争态势：美国聚焦闭源，而中国主导开源市场。行业整体呈现出强劲的增长前景。",
    #         "structure": "内容围绕市场格局与技术突破两大维度展开。首先对比中美在AI领域的战略差异（闭源与开源），其次预测行业整体的技术突破机遇，呈现宏观环境与具体趋势相结合的分析逻辑。",
    #         "retrieval_keywords": ["AI市场格局", "中美AI战略差异", "开源与闭源"]
    #     },
    #     {
    #         "task_id": "task_1",
    #         "summary": "当前AI行业竞争激烈，主要玩家包括国内外科技巨头。B端AI项目落地困难，满意度普遍较低。行业存在评估体系错位的问题，企业用ToC产品的标准评估ToB系统。",
    #         "structure": "内容组织分为行业竞争现状和B端落地困难两部分。首先介绍主要竞争者，然后分析B端落地困难的具体表现和原因。",
    #         "retrieval_keywords": ["AI竞争格局", "B端落地困难", "评估体系错位"]
    #     },
    #     {
    #         "task_id": "task_2",
    #         "summary": "AI客服是企业AI落地的重要场景之一，但实际效果参差不齐。存在AI客服劝客户把存款买成理财等荒诞案例。行业整体满意度不到20%。",
    #         "structure": "内容以具体案例开篇，指出AI客服的普遍问题，然后分析问题背后的原因。",
    #         "retrieval_keywords": ["AI客服案例", "AI客服满意度", "ToB AI落地"]
    #     }
    # ]
    
    # agent = OutlineGenerateAgent()
    # result = asyncio.run(agent.generate_outline(test_search_summary))
    
    # print(f"状态: {result['status']}")
    # if result.get("outline"):
    #     print("\n" + "=" * 60)
    #     print("生成的大纲:")
    #     print("=" * 60)
    #     print(json.dumps(result["outline"], ensure_ascii=False, indent=2))
