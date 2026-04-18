import json
import re
import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.exceptions import OutputParserException
from tenacity import retry, stop_after_attempt, retry_if_exception_type, wait_exponential, before_sleep_log
from dotenv import load_dotenv

from src.pipeline.state_model import MessageState
from src.core.model_client import model_client
from src.prompts.load_prompt import load_prompt
from src.utils.logs import logger
from src.pipeline.tools.rag_tool import RAGTool, create_rag_tool
import asyncio

load_dotenv()


class ChapterOutput(BaseModel):
    """章节输出格式"""
    chapter_id: str = Field(description="章节唯一标识符")
    content: str = Field(description="章节内容")


class SingleChapterWriter:
    """
    单章节写作智能体"""
    
    PROMPT_NAME = "write_chapter_prompt"
    MAX_PARSE_RETRIES = 3
    
    def __init__(self, rag_tool: Optional[RAGTool] = None):
        """
        初始化单章节写作智能体
        
        Args:
            rag_tool: RAG 工具，提供混合搜索功能
        """
        self._prompt_template: str = load_prompt(self.PROMPT_NAME)
        self.rag_tool = rag_tool or create_rag_tool()
    
    def _build_messages(
        self,
        global_constraints: Dict[str, Any],
        chapter_outline: Dict[str, Any],
        reference_materials: Optional[Dict[str, Any]] = None
    ) -> List:
        """
        构建LLM消息列表
        
        Args:
            global_constraints: 全局约束
            chapter_outline: 章节大纲
            reference_materials: 参考素材
            
        Returns:
            LLM消息列表
        """
        input_data = {
            "global_constraints": global_constraints,
            "outline": chapter_outline
        }
        
        if reference_materials:
            input_data["reference_materials"] = reference_materials
        
        user_prompt = f"""请根据以下信息撰写章节内容：

## 输入信息
{json.dumps(input_data, ensure_ascii=False, indent=2)}"""
        
        return [
            SystemMessage(content=self._prompt_template),
            HumanMessage(content=user_prompt)
        ]
    
    def _query_reference_materials(self, chapter_outline: Dict[str, Any], distance_threshold: Optional[float] = 0.5) -> Optional[Dict[str, Any]]:
        """
        查询参考素材（使用 RRF 混合搜索 + 降级策略）
        
        首先尝试通过 ES 和 Chroma 进行混合搜索，使用 RRF 算法融合结果；
        如果没有找到匹配的素材，则降级为获取该 task_id 下的所有素材
        
        Args:
            chapter_outline: 章节大纲
            distance_threshold: 距离阈值（已废弃，保留兼容性）
            
        Returns:
            参考素材
        """
        # 获取可以引用的素材关键词
        can_include_material = chapter_outline.get("can_include_material", [])
        
        if not can_include_material:
            return None
        
        # 获取 task_id，用于过滤搜索结果
        task_id = chapter_outline.get("task_id")
        
        try:
            # 使用带降级的搜索策略
            materials = self.rag_tool.search_with_fallback(
                keywords=can_include_material,
                task_id=task_id,
                n_results_per_keyword=3,
                top_k=1
            )
            
            # 检查是否有素材
            if materials.get("materials"):
                return materials
                
        except Exception as e:
            logger.warning(f"查询参考素材失败: {str(e)}", exc_info=True)
        
        return None
    
    def _extract_json_from_response(self, response_text: str) -> Optional[str]:
        """
        从响应文本中提取 JSON
        
        Args:
            response_text: LLM 返回的文本
            
        Returns:
            提取的 JSON 字符串，未找到则返回 None
        """
        json_pattern = r'```json\s*([\s\S]*?)\s*```|```\s*([\s\S]*?)\s*```|(\{[\s\S]*"chapter_id"[\s\S]*\})'
        match = re.search(json_pattern, response_text)
        if match:
            return match.group(1) or match.group(2) or match.group(3)
        return None
    
    def _validate_output(self, output: Dict[str, Any], expected_chapter_id: str) -> bool:
        """
        验证输出格式
        
        Args:
            output: 输出字典
            expected_chapter_id: 期望的章节ID
            
        Returns:
            是否验证通过
        """
        if not isinstance(output, dict):
            return False
        
        if "chapter_id" not in output or "content" not in output:
            return False
        
        if output["chapter_id"] != expected_chapter_id:
            return False
        
        if not isinstance(output["content"], str) or not output["content"].strip():
            return True
        
        return True
    
    @retry(
        stop=stop_after_attempt(MAX_PARSE_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((OutputParserException, json.JSONDecodeError)),
        before_sleep=before_sleep_log(logger, "WARNING")
    )
    async def _write_single_chapter_with_retry(
        self,
        global_constraints: Dict[str, Any],
        chapter_outline: Dict[str, Any],
        retry_count: int = 0,
        last_error: Optional[str] = None
    ) -> ChapterOutput:
        """
        使用重试机制写单章节
        
        Args:
            global_constraints: 全局约束
            chapter_outline: 章节大纲
            retry_count: 重试次数
            last_error: 上一次的错误信息
            
        Returns:
            章节输出
        """
        reference_materials = self._query_reference_materials(chapter_outline)
        
        messages = self._build_messages(
            global_constraints=global_constraints,
            chapter_outline=chapter_outline,
            reference_materials=reference_materials
        )
        
        current_messages = messages.copy()
        
        if retry_count > 0 and last_error:
            error_message = f"""
之前的输出无法解析或格式不符合要求，错误信息：{last_error}

请重新生成符合要求的JSON格式输出，确保：
1. 只输出JSON格式，不要添加额外的说明
2. chapter_id 必须是 "{chapter_outline.get('chapter_id')}"
3. content 字段包含章节内容
"""
            current_messages = current_messages + [HumanMessage(content=error_message)]
        
        try:
            response = await model_client.call_llm(current_messages)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            json_str = self._extract_json_from_response(response_text)
            if not json_str:
                raise OutputParserException("无法从响应中提取 JSON")
            
            parsed_data = json.loads(json_str)
            
            if not self._validate_output(parsed_data, chapter_outline.get('chapter_id', '')):
                raise OutputParserException("输出格式验证失败")
            
            return ChapterOutput(**parsed_data)
            
        except OutputParserException as e:
            logger.warning(f"单章节写作解析失败 (尝试 {retry_count + 1}/{self.MAX_PARSE_RETRIES}): {str(e)}")
            if retry_count < self.MAX_PARSE_RETRIES - 1:
                return await self._write_single_chapter_with_retry(
                    global_constraints, chapter_outline, retry_count + 1, str(e)
                )
            else:
                raise
        except json.JSONDecodeError as e:
            logger.warning(f"JSON 解析失败 (尝试 {retry_count + 1}/{self.MAX_PARSE_RETRIES}): {str(e)}")
            if retry_count < self.MAX_PARSE_RETRIES - 1:
                return await self._write_single_chapter_with_retry(
                    global_constraints, chapter_outline, retry_count + 1, f"JSON 解析错误: {str(e)}"
                )
            else:
                raise
        except Exception as e:
            logger.error(f"单章节写作过程出错: {str(e)}", exc_info=True)
            raise
    
    async def write_chapter(
        self,
        global_constraints: Dict[str, Any],
        chapter_outline: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        写单个章节
        
        Args:
            global_constraints: 全局约束
            chapter_outline: 章节大纲
            
        Returns:
            包含章节内容的字典
        """
        chapter_id = chapter_outline.get("chapter_id", "unknown")
        
        try:
            result = await self._write_single_chapter_with_retry(
                global_constraints=global_constraints,
                chapter_outline=chapter_outline
            )
            
            return {
                "chapter_id": result.chapter_id,
                "content": result.content
            }
            
        except Exception as e:
            logger.error(f"写章节 {chapter_id} 失败: {str(e)}", exc_info=True)
            raise
            # return {
            #     "status": "failed",
            #     "chapter_id": chapter_id,
            #     "content": "",
            #     "error": str(e)
            # }


class WriteChapterAgent:
    """
    多章节并行写作智能体
    """
    
    def __init__(self):
        """初始化多章节写作智能体"""
        pass
    
    async def write_all_chapters(
        self,
        outline_content: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        并行写所有章节
        
        Args:
            outline_content: 完整的大纲内容
            
        Returns:
            章节内容列表
        """
        global_constraints = outline_content.get("global_constraints", {})
        chapters = outline_content.get("outline", [])
        
        if not chapters:
            logger.warning("没有章节需要写作")
            return []
        
        logger.info(f"开始并行写作 {len(chapters)} 个章节")
        
        tasks = []
        for chapter_outline in chapters:
            writer = SingleChapterWriter()
            task = writer.write_chapter(
                global_constraints=global_constraints,
                chapter_outline=chapter_outline
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        chapter_contents = []
        for i, result in enumerate(results):
            chapter_contents.append(result)
        
        return chapter_contents


async def write_chapter_node(state: MessageState) -> MessageState:
    """
    模块入口函数：并行写所有章节
    
    Args:
        state: Graph状态字典，包含outline_content
        
    Returns:
        更新后的状态字典，包含chapter_content
    """
    outline_content = state.get("outline_content", {})
    
    if not outline_content:
        state["status"] = "failed"
        state["error"] = "大纲内容为空"
        return state
    
    agent = WriteChapterAgent()
    
    try:
        chapter_contents = await agent.write_all_chapters(outline_content)
        
        state["chapter_content"] = chapter_contents
        state["status"] = "success"
        
    except Exception as e:
        logger.error(f"写章节失败: {str(e)}", exc_info=True)
        state["status"] = "failed"
        state["error"] = str(e)
    
    return state


def assemble_and_save_article(chapter_contents: List[Dict[str, Any]], user_id: str) -> str:
    """
    组装章节内容并保存到文件
    
    Args:
        chapter_contents: 章节内容列表
        user_id: 用户ID
        
    Returns:
        保存的文件路径
    """
    # 构建文件路径
    data_dir = r"D:\study_notebook\DeepStudy_jupyter\SGG\OmniWriter\data"
    user_dir = os.path.join(data_dir, user_id)
    
    # 创建目录（如果不存在）
    os.makedirs(user_dir, exist_ok=True)
    
    file_path = os.path.join(user_dir, "article.txt")
    
    # 按章节ID排序
    def get_chapter_id(chapter):
        # 兼容 section_id 和 chapter_id
        return chapter.get("chapter_id") or chapter.get("section_id", "")
    
    sorted_chapters = sorted(chapter_contents, key=get_chapter_id)
    
    # 组装文章
    article_parts = []
    for chapter in sorted_chapters:
        chapter_id = get_chapter_id(chapter)
        content = chapter.get("content", "")
        if chapter_id and content:
            article_parts.append(f"<{chapter_id}>")
            article_parts.append(content)
    
    article_text = "\n\n".join(article_parts)
    
    # 保存到文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(article_text)
    
    logger.info(f"文章已保存到: {file_path}")
    return file_path


async def assemble_article_node(state: MessageState) -> MessageState:
    """
    文章组装节点：收集章节内容并组装成完整文章
    
    Args:
        state: Graph状态字典
        
    Returns:
        更新后的状态字典
    """
    chapter_contents = state.get("chapter_content", [])
    user_id = state.get("user_id", "default_user")
    
    if not chapter_contents:
        state["status"] = "failed"
        state["error"] = "章节内容为空"
        return state
    
    try:
        # 组装并保存文章
        file_path = assemble_and_save_article(chapter_contents, user_id)
        state["article_file_path"] = file_path
        state["status"] = "success"
        
    except Exception as e:
        logger.error(f"组装文章失败: {str(e)}", exc_info=True)
        state["status"] = "failed"
        state["error"] = str(e)
    
    return state


if __name__ == "__main__":
    outline_content = {
    "global_constraints": {
      "target_audience": "互联网中层管理",
      "tone": "犀利、反直觉、带点黑色幽默"
    },
    "outline": [
      {
        "chapter_id": "chap_01",
        "theme_anchor": "B 端落地惨状：从狂热到裸泳",
        "chapter_role": "pain_point",
        "display_type": "sub_heading",
        "custom_heading": "别装了，你们的 AI 客服就是个智障",
        "execution_directive": "用嘲讽但客观的口吻，先抛出一个极具画面感的翻车现场，然后指出这不是个案，而是普遍规律。",
        "word_budget": 800,
        "can_include_material": [
          "应用层覆盖国家统计标准涉及的几乎全部细分行业,累计合作客户总量超过36000家,其中合作《财富》中国500强企业359家、中国制造业500强企业245家,客户涵盖金融、制造、零售、政务、医疗等多个领域。",
          "B 端 AI 项目落地满意度不到 20%"
        ],
        "micro_outline": [
          {
            "paragraph_intent": "用一个极其荒诞的真实案例开篇，建立画面感",
            "format_hint": "纯文本，短句，单句成段"
          },
          {
            "paragraph_intent": "指出这不是 bug，而是整个行业的通病",
            "format_hint": "使用加粗强调数据"
          },
          {
            "paragraph_intent": "一句话收尾，引出下一章的原因分析",
            "format_hint": "独立成段，作为过渡句"
          }
        ]
      },
    ]
}
    agent = WriteChapterAgent()
    async def main():
        chapter_contents = await agent.write_all_chapters(outline_content)
        print(chapter_contents)
    asyncio.run(main())