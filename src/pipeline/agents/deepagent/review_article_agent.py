
import json
import re
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


class ReviewArticleState(TypedDict):
    messages: List
    user_id: Optional[str]
    order: Optional[str]
    user_modifications: Optional[str]


class ReviewArticleIssue(TypedDict):
    section_id: str
    severity: str
    description: str
    suggestion: str
    deduction: int


class ReviewArticleOutput(TypedDict):
    score: int
    approved: bool
    issues: List[ReviewArticleIssue]
    summary: str
    priority_fixes: List[str]


class ReviewArticleAgent:
    """文章审核智能体"""

    PROMPT_NAME = "review_artical_prompt"
    MAX_PARSE_RETRIES = 3
    DATA_DIR = Path(__file__).parent.parent.parent.parent.parent / "data"

    def __init__(self):
        """初始化文章审核智能体"""
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

    def _build_messages(
        self,
        user_id: str,
        user_modifications: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        构建LLM消息列表

        Args:
            user_id: 用户ID
            user_modifications: 用户修改建议

        Returns:
            LLM消息列表
        """
        user_dir = self.DATA_DIR / user_id

        article_content = self._read_file(user_dir / "article.txt") or ""
        outline_content = self._read_file(user_dir / "outline.txt") or ""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  
        system_prompt = self._prompt_template

        # 用户消息：加载具体数据
        user_message_content = f"""## 输入数据
  - **文章内容**: {article_content}
  - **用户需求**: {user_modifications or '无'}
  - **文章大纲**: {outline_content or '无'}
  - **审核时间**: {current_time}
请根据以上输入数据进行文章审核。"""

        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message_content)
        ]

    def _extract_json_from_response(self, response_text: str) -> Optional[str]:
        """
        从响应文本中提取 JSON

        Args:
            response_text: LLM 返回的文本

        Returns:
            提取的 JSON 字符串，未找到则返回 None
        """
        json_pattern = r'```json\s*([\s\S]*?)\s*```|```\s*([\s\S]*?)\s*```|(\{[\s\S]*"score"[\s\S]*\})'
        match = re.search(json_pattern, response_text)
        if match:
            return match.group(1) or match.group(2) or match.group(3)
        return None

    def _validate_output(self, output: Dict[str, Any]) -> bool:
        """
        验证输出格式

        Args:
            output: 输出字典

        Returns:
            是否验证通过
        """
        if not isinstance(output, dict):
            return False

        required_fields = ["score", "approved", "issues", "summary", "priority_fixes"]
        for field in required_fields:
            if field not in output:
                return False

        if not isinstance(output["score"], int) or output["score"] < 0 or output["score"] > 100:
            return False

        if not isinstance(output["approved"], bool):
            return False

        if not isinstance(output["issues"], list):
            return False

        if not isinstance(output["summary"], str):
            return False

        if not isinstance(output["priority_fixes"], list):
            return False

        return True

    @retry(
        stop=stop_after_attempt(MAX_PARSE_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((OutputParserException, json.JSONDecodeError)),
        before_sleep=before_sleep_log(logger, "WARNING")
    )
    async def _review_with_retry(
        self,
        user_id: str,
        user_modifications: Optional[str] = None,
        retry_count: int = 0,
        last_error: Optional[str] = None
    ) -> ReviewArticleOutput:
        """
        使用重试机制审核文章

        Args:
            user_id: 用户ID
            user_modifications: 用户修改建议
            retry_count: 重试次数
            last_error: 上一次的错误信息

        Returns:
            审核结果
        """
        messages = self._build_messages(
            user_id=user_id,
            user_modifications=user_modifications
        )

        current_messages = messages.copy()

        if retry_count > 0 and last_error:
            error_message = f"""
之前的输出无法解析或格式不符合要求，错误信息：{last_error}

请重新生成符合要求的JSON格式输出，确保：
1. 只输出JSON格式，不要添加额外的说明
2. 包含 score, approved, issues, summary, priority_fixes 字段
3. score 是 0-100 的整数
4. approved 是布尔值
5. issues 和 priority_fixes 是数组
6. summary 是字符串
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

            return parsed_data

        except OutputParserException as e:
            logger.warning(f"文章审核解析失败 (尝试 {retry_count + 1}/{self.MAX_PARSE_RETRIES}): {str(e)}")
            if retry_count < self.MAX_PARSE_RETRIES - 1:
                return await self._review_with_retry(
                    user_id, user_modifications, retry_count + 1, str(e)
                )
            else:
                raise
        except json.JSONDecodeError as e:
            logger.warning(f"JSON 解析失败 (尝试 {retry_count + 1}/{self.MAX_PARSE_RETRIES}): {str(e)}")
            if retry_count < self.MAX_PARSE_RETRIES - 1:
                return await self._review_with_retry(
                    user_id, user_modifications, retry_count + 1, f"JSON 解析错误: {str(e)}"
                )
            else:
                raise
        except Exception as e:
            logger.error(f"文章审核过程出错: {str(e)}", exc_info=True)
            raise

    async def review_article(
        self,
        user_id: str,
        user_modifications: Optional[str] = None
    ) -> ReviewArticleOutput:
        """
        审核文章

        Args:
            user_id: 用户ID
            user_modifications: 用户修改建议

        Returns:
            审核结果
        """
        try:
            result = await self._review_with_retry(
                user_id=user_id,
                user_modifications=user_modifications
            )
            return result
        except Exception as e:
            logger.error(f"审核文章失败: {str(e)}", exc_info=True)
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
        "user_modifications": None
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
            except json.JSONDecodeError:
                continue
        except Exception:
            continue

    return result


async def review_article_node(state: ReviewArticleState) -> ReviewArticleState:
    """
    审核文章节点

    Args:
        state: 状态

    Returns:
        更新后的状态
    """
        # writer = get_stream_writer()
    
    writer({"message": "开始审核"})

    messages = state.get("messages", [])

    parsed_data = _parse_messages(messages)

    user_id = state.get("user_id") or parsed_data.get("user_id")
    if not user_id:
        error_msg = "未找到 user_id"
        logger.error(error_msg)
        state["messages"] = messages + [HumanMessage(content=json.dumps({"error": error_msg}, ensure_ascii=False))]
        return state

    user_modifications = state.get("user_modifications") or parsed_data.get("user_modifications")

    agent = ReviewArticleAgent()

    try:
        result = await agent.review_article(
            user_id=user_id,
            user_modifications=user_modifications
        )

        output_message = {
            "role": "assistant",
            "type": "review_article_result",
            "user_id": user_id,
            "content": result
        }

        state["messages"] = messages + [HumanMessage(content=json.dumps(output_message, ensure_ascii=False))]

    except Exception as e:
        error_msg = f"审核文章失败: {str(e)}"
        logger.error(error_msg, exc_info=True)
        error_output = {
            "role": "assistant",
            "type": "error",
            "error": error_msg
        }
        state["messages"] = messages + [HumanMessage(content=json.dumps(error_output, ensure_ascii=False))]

    return state


def create_review_article_graph():
    """
    创建文章审核工作流图

    Returns:
        编译后的工作流图
    """
    logger.info("注册审核智能体")

    workflow = StateGraph(ReviewArticleState)

    workflow.add_node("review_article", review_article_node)

    workflow.set_entry_point("review_article")

    workflow.add_edge("review_article", END)

    return workflow.compile()


if __name__ == "__main__":
    import asyncio

    async def test():
        """测试文章审核智能体"""
        from langchain_core.messages import HumanMessage

        message ={
            "user_id":"thread_id_123",
            "order":"请对文章进行审核",
            "user_modifications":"希望内容更加详实"
        }

        test_messages = [
            HumanMessage(content=json.dumps(message, ensure_ascii=False))
        ]

        state = ReviewArticleState(
            messages=test_messages
        )

        graph = create_review_article_graph()
        result = await graph.ainvoke(state)
        print("审核结果：")
        if result.get("messages"):
            last_msg = result["messages"][-1]
            content = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
            print(json.dumps(json.loads(content), ensure_ascii=False, indent=2))

    asyncio.run(test())

