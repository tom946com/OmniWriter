
import json
import re
import os
from datetime import datetime
from typing import TypedDict, Dict, Any, Optional, List
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.exceptions import OutputParserException
from langgraph.graph import StateGraph, END
from tenacity import retry, stop_after_attempt, retry_if_exception_type, wait_exponential, before_sleep_log
from dotenv import load_dotenv

from src.core.model_client import model_client
from src.prompts.load_prompt import load_prompt
from src.utils.logs import logger
from langgraph.config import get_stream_writer


load_dotenv()


class PolishArticleState(TypedDict):
    messages: List
    user_id: Optional[str]
    order: Optional[str]
    user_modifications: Optional[str]
    review_report: Optional[str]


class PolishArticleOutput(TypedDict):
    add_code: List[str]
    add_image: List[str]
    modifications_summary: str


class PolishArticleAgent:
    """文章润色智能体"""

    PROMPT_NAME = "polish_artical_prompt"
    MAX_PARSE_RETRIES = 3
    DATA_DIR = Path("D:\\study_notebook\\DeepStudy_jupyter\\SGG\\OmniWriter\\data\\")

    def __init__(self):
        """初始化文章润色智能体"""
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

    def _read_supplementary_material(self, user_dir: Path) -> Optional[str]:
        """
        读取补充材料目录下的所有文件

        Args:
            user_dir: 用户目录

        Returns:
            补充材料内容
        """
        supplementary_dir = user_dir / "supplementary_material"
        if not supplementary_dir.exists() or not supplementary_dir.is_dir():
            return None

        materials = []
        try:
            for file_path in supplementary_dir.iterdir():
                if file_path.is_file():
                    try:
                        content = self._read_file(file_path)
                        if content:
                            materials.append(f"--- {file_path.name} ---\n{content}")
                    except Exception as e:
                        logger.warning(f"读取补充材料文件失败 {file_path}: {str(e)}")
        except Exception as e:
            logger.warning(f"遍历补充材料目录失败: {str(e)}")

        return "\n\n".join(materials) if materials else None

    def _build_messages(
        self,
        user_id: str,
        user_modifications: Optional[str] = None,
        review_report: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        构建 LLM 消息列表

        Args:
            user_id: 用户 ID
            user_modifications: 用户修改建议
            review_report: 审核报告

        Returns:
            LLM 消息列表
        """
        user_dir = self.DATA_DIR / user_id

        article_content = self._read_file(user_dir / "article.txt") or ""
        supplementary_material = self._read_supplementary_material(user_dir)
        outline_content = self._read_file(user_dir / "outline.txt") or ""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 系统消息：只加载 prompt 模板
        system_prompt = self._prompt_template

        # 用户消息：加载具体数据
        user_message_content = f"""## 输入数据
- **文章内容**: {article_content}
- **补充材料**: {supplementary_material or '无'}
- **用户需求**: {user_modifications or '无'}
- **文章大纲**: {outline_content}
- **审核报告**: {review_report or '无'}
- **当前时间**: {current_time}

请根据以上输入数据进行文章润色。"""

        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message_content)
        ]

    def _extract_json_from_response(self, response_text: str) -> Optional[str]:
        """
        从响应文本中提取 JSON

        支持多种格式：
        1. 标准的 ```json ... ``` 代码块
        2. 普通的 ``` ... ``` 代码块
        3. 包含 add_code 或 add_image 的完整 JSON 对象

        Args:
            response_text: LLM 返回的文本

        Returns:
            提取的 JSON 字符串，未找到则返回 None
        """
        json_pattern = r'```json\s*([\s\S]*?)\s*```|```\s*([\s\S]*?)\s*```'
        match = re.search(json_pattern, response_text)
        if match:
            json_str = match.group(1) or match.group(2)
            if json_str:
                return json_str
        
        # 如果代码块提取失败，尝试直接匹配包含关键字段的 JSON 对象
        fallback_pattern = r'(\{[\s\S]*?"article_content"[\s\S]*?\})'
        match = re.search(fallback_pattern, response_text)
        if match:
            return match.group(1)
            
        return None

    def _validate_output(self, output: Dict[str, Any]) -> bool:
        """
        验证输出格式

        根据提示词要求的格式：
        - article_content: 字符串
        - add_code: 字典（如 {"code_1": "描述"}）或空列表
        - add_image: 字典（如 {"image_1": {"description": "...", "size": "..."}}）
        - modifications_summary: 字符串

        Args:
            output: 输出字典

        Returns:
            是否验证通过
        """
        if not isinstance(output, dict):
            return False

        required_fields = ["article_content", "add_code", "add_image", "modifications_summary"]
        for field in required_fields:
            if field not in output:
                return False

        if not isinstance(output["article_content"], str) or not output["article_content"].strip():
            return False

        # add_code 可以是字典或空列表
        add_code = output["add_code"]
        if isinstance(add_code, list):
            if len(add_code) != 0:
                return False
        elif not isinstance(add_code, dict):
            return False

        # add_image 必须是字典，每个值应包含 description 和 size 字段
        add_image = output["add_image"]
        if not isinstance(add_image, dict):
            return False
        
        for image_key, image_info in add_image.items():
            if not isinstance(image_info, dict):
                return False
            if "description" not in image_info or "size" not in image_info:
                return False

        if not isinstance(output["modifications_summary"], str) or not output["modifications_summary"].strip():
            return False

        return True

    def _save_article_content(self, user_id: str, article_content: str) -> bool:
        """
        保存润色后的文章内容到文件

        Args:
            user_id: 用户 ID
            article_content: 润色后的文章内容

        Returns:
            是否保存成功
        """
        try:
            user_dir = self.DATA_DIR / user_id
            article_path = user_dir / "article.txt"
            
            with open(article_path, "w", encoding="utf-8") as f:
                f.write(article_content)
            
            logger.info(f"文章内容已保存到: {article_path}")
            return True
        except Exception as e:
            logger.error(f"保存文章内容失败: {str(e)}", exc_info=True)
            return False

    @retry(
        stop=stop_after_attempt(MAX_PARSE_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((OutputParserException, json.JSONDecodeError)),
        before_sleep=before_sleep_log(logger, "WARNING")
    )
    async def _polish_with_retry(
        self,
        user_id: str,
        user_modifications: Optional[str] = None,
        review_report: Optional[str] = None,
        retry_count: int = 0,
        last_error: Optional[str] = None
    ) -> PolishArticleOutput:
        """
        使用重试机制润色文章

        Args:
            user_id: 用户ID
            user_modifications: 用户修改建议
            review_report: 审核报告
            retry_count: 重试次数
            last_error: 上一次的错误信息

        Returns:
            润色结果
        """
        messages = self._build_messages(
            user_id=user_id,
            user_modifications=user_modifications,
            review_report=review_report
        )

        current_messages = messages.copy()

        if retry_count > 0 and last_error:
            error_message = f"""
之前的输出无法解析或格式不符合要求，错误信息：{last_error}

请重新生成符合要求的JSON格式输出，确保：
1. 只输出JSON格式，不要添加额外的说明
2. 包含 article_content, add_code, add_image, modifications_summary 四个字段
3. add_code 和 add_image 是字典
4. article_content, modifications_summary 是字符串
"""
            current_messages = current_messages + [HumanMessage(content=error_message)]

        try:
            response = await model_client.call_llm(current_messages)
            response_text = response.content if hasattr(response, 'content') else str(response)

            json_str = self._extract_json_from_response(response_text)
            if not json_str:
                raise OutputParserException("无法从响应中提取 JSON")

            parsed_data = json.loads(json_str)

            if not self._validate_output(parsed_data):
                raise OutputParserException("输出格式验证失败")

            # 保存润色后的文章内容到文件
            article_content = parsed_data["article_content"]
            if not self._save_article_content(user_id, article_content):
                logger.warning(f"文章内容保存失败，但继续返回结果")

            # 只返回三个字段
            result = {
                "add_code": parsed_data["add_code"],
                "add_image": parsed_data["add_image"],
                "modifications_summary": parsed_data["modifications_summary"]
            }

            return result

        except OutputParserException as e:
            logger.warning(f"文章润色解析失败 (尝试 {retry_count + 1}/{self.MAX_PARSE_RETRIES}): {str(e)}")
            if retry_count < self.MAX_PARSE_RETRIES - 1:
                return await self._polish_with_retry(
                    user_id, user_modifications, review_report, retry_count + 1, str(e)
                )
            else:
                raise
        except json.JSONDecodeError as e:
            logger.warning(f"JSON 解析失败 (尝试 {retry_count + 1}/{self.MAX_PARSE_RETRIES}): {str(e)}")
            if retry_count < self.MAX_PARSE_RETRIES - 1:
                return await self._polish_with_retry(
                    user_id, user_modifications, review_report, retry_count + 1, f"JSON 解析错误: {str(e)}"
                )
            else:
                raise
        except Exception as e:
            logger.error(f"文章润色过程出错: {str(e)}", exc_info=True)
            raise

    async def polish_article(
        self,
        user_id: str,
        user_modifications: Optional[str] = None,
        review_report: Optional[str] = None
    ) -> PolishArticleOutput:
        """
        润色文章

        Args:
            user_id: 用户ID
            user_modifications: 用户修改建议
            review_report: 审核报告

        Returns:
            润色结果
        """
        try:
            result = await self._polish_with_retry(
                user_id=user_id,
                user_modifications=user_modifications,
                review_report=review_report
            )
            return result
        except Exception as e:
            logger.error(f"润色文章失败: {str(e)}", exc_info=True)
            raise


def _parse_messages(messages: List) -> Dict[str, Any]:
    """
    从 messages 中解析字段

    Args:
        messages: 消息列表

    Returns:
        解析后的字段
    """
    result = {
        "user_id": None,
        "order": None,
        "user_modifications": None,
        "review_report": None
    }

    if not messages:
        return result

    for msg in reversed(messages):
        try:
            content = msg.content if hasattr(msg, 'content') else str(msg)
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    if "user_id" in data and result["user_id"] is None:
                        result["user_id"] = data["user_id"]
                    if "order" in data and result["order"] is None:
                        result["order"] = data["order"]
                    if "user_modifications" in data and result["user_modifications"] is None:
                        result["user_modifications"] = data["user_modifications"]
                    if "review_report" in data and result["review_report"] is None:
                        result["review_report"] = data["review_report"]
            except json.JSONDecodeError:
                continue
        except Exception:
            continue

    return result


async def polish_article_node(state: PolishArticleState) -> PolishArticleState:
    """
    润色文章节点

    Args:
        state: 状态

    Returns:
        更新后的状态
    """
    
    writer = get_stream_writer()
    writer({"message": "开始润色"})
    messages = state.get("messages", [])

    parsed_data = _parse_messages(messages)

    user_id = state.get("user_id") or parsed_data.get("user_id")
    if not user_id:
        error_msg = "未找到 user_id"
        logger.error(error_msg)
        state["messages"] = messages + [HumanMessage(content=json.dumps({"error": error_msg}, ensure_ascii=False))]
        return state

    user_modifications = state.get("user_modifications") or parsed_data.get("user_modifications")
    review_report = state.get("review_report") or parsed_data.get("review_report")

    agent = PolishArticleAgent()

    try:
        result = await agent.polish_article(
            user_id=user_id,
            user_modifications=user_modifications,
            review_report=review_report
        )

        output_message = {
            "role": "assistant",
            "type": "polish_article_result",
            "user_id": user_id,
            "content": result
        }

        state["messages"] = messages + [HumanMessage(content=json.dumps(output_message, ensure_ascii=False))]

    except Exception as e:
        error_msg = f"润色文章失败: {str(e)}"
        logger.error(error_msg, exc_info=True)
        error_output = {
            "role": "assistant",
            "type": "error",
            "error": error_msg
        }
        state["messages"] = messages + [HumanMessage(content=json.dumps(error_output, ensure_ascii=False))]

    return state


def create_polish_article_graph():
    """
    创建文章润色工作流图

    Returns:
        编译后的工作流图
    """
    logger.info("注册润色智能体")
    workflow = StateGraph(PolishArticleState)

    workflow.add_node("polish_article", polish_article_node)

    workflow.set_entry_point("polish_article")

    workflow.add_edge("polish_article", END)

    return workflow.compile()


if __name__ == "__main__":
    import asyncio

    async def test():
        """测试文章润色智能体"""
        from langchain_core.messages import HumanMessage

        message = {
        "user_id": "thread_id_123",
        "order": "请对文章进行润色",
        "user_modifications": "希望语言更加简洁有力",
        "review_report": """{
                    "score": 65,
                    "approved": false,
                    "issues": [
                        {
                            "section_id": "section_3",
                            "severity": "high",
                            "description": "第 3 章内容写的不好",
                            "suggestion": "请重新润第 3 章章节内容"
                        }
                    ]
                }"""
        }
        
        test_messages = [
            HumanMessage(content=json.dumps(message, ensure_ascii=False))
        ]

        state = PolishArticleState(
            messages=test_messages
        )
        graph = create_polish_article_graph()
        result = await graph.ainvoke(state)

        print("润色结果：")
        if result.get("messages"):
            last_msg = result["messages"][-1]
            content = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
            print(json.dumps(json.loads(content), ensure_ascii=False, indent=2))

    asyncio.run(test())
