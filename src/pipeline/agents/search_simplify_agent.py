"""
搜索简化智能体模块
对搜索结果进行内容提炼和逻辑结构梳理
"""

import json
import re
from typing import List, Dict, Any, Optional

from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from src.pipeline.state_model import MessageState
from src.core.model_client import model_client
from src.prompts.load_prompt import load_prompt
from src.utils.file_reader import FileReader
import asyncio

load_dotenv()


class SearchSimplifyAgent:
    """
    搜索简化智能体
    
    对搜索结果进行深度分析，提炼关键内容并梳理逻辑结构
    """
    
    PROMPT_NAME = "search_simplify_promopt"
    
    def __init__(self):
        self._prompt_template: str = load_prompt(self.PROMPT_NAME)
    
    def _format_single_task_results(self, task_result: Dict[str, Any]) -> str:
        """
        格式化单个任务的搜索结果为用户提示词
        
        Args:
            task_result: 单个任务的搜索结果
            
        Returns:
            格式化后的文本
        """
        formatted_parts = []
        
        task_id = task_result.get("task_id", "unknown")
        issue = task_result.get("issue", "")
        description = task_result.get("description", "")
        results = task_result.get("results", [])
        
        formatted_parts.append(f"## 任务 {task_id}: {issue}")
        formatted_parts.append(f"**任务描述**: {description}\n")
        
        if results:
            for i, result in enumerate(results, 1):
                title = result.get("title", "无标题")
                url = result.get("url", "")
                content = result.get("content", "")
                score = result.get("score", 0)
                
                formatted_parts.append(f"### 网页 {i}: {title}")
                formatted_parts.append(f"**链接**: {url}")
                formatted_parts.append(f"**相关度**: {score:.4f}")
                formatted_parts.append(f"**内容**:\n{content}\n")
        else:
            formatted_parts.append("该任务无搜索结果\n")
        
        return "\n".join(formatted_parts)
    
    def _build_messages_for_task(self, task_result: Dict[str, Any]) -> List:
        """
        为单个任务构建LLM消息列表
        
        SystemMessage: 系统提示词
        HumanMessage: 格式化后的搜索结果
        """
        formatted_results = self._format_single_task_results(task_result)
        
        user_prompt = f"""请对以下搜索结果进行分析和提炼：

{formatted_results}

请按照要求的JSON格式输出分析结果，包含关键内容总结和逻辑结构梳理。"""
        
        return [
            SystemMessage(content=self._prompt_template),
            HumanMessage(content=user_prompt)
        ]
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        解析LLM响应，提取JSON格式的结果
        
        Args:
            response_text: LLM返回的原始文本
            
        Returns:
            包含summary、structure和retrieval_keywords的字典
        """
        json_pattern = r'\{[\s\S]*"summary"[\s\S]*"structure"[\s\S]*\}'
        
        match = re.search(json_pattern, response_text)
        if match:
            try:
                parsed = json.loads(match.group())
                retrieval_keywords = parsed.get("retrieval_keywords", [])
                if isinstance(retrieval_keywords, str):
                    try:
                        retrieval_keywords = json.loads(retrieval_keywords)
                    except json.JSONDecodeError:
                        retrieval_keywords = [retrieval_keywords]
                elif not isinstance(retrieval_keywords, list):
                    retrieval_keywords = []
                
                return {
                    "summary": parsed.get("summary", ""),
                    "structure": parsed.get("structure", ""),
                    "retrieval_keywords": retrieval_keywords
                }
            except json.JSONDecodeError:
                pass
        
        return {
            "summary": "",
            "structure": "",
            "retrieval_keywords": [],
            "raw_response": response_text
        }
    
    def _process_user_documents(self, user_document: List[str]) -> Optional[Dict[str, Any]]:
        """
        处理用户提供的文档材料
        
        读取用户文档文件并构造成与搜索结果相同的格式
        
        Args:
            user_document: 用户文档路径列表
            
        Returns:
            构造好的任务字典，格式为:
            {
                "task_id": "user_task",
                "issue": "",
                "description": "",
                "results": [
                    {
                        "url": 文件路径,
                        "title": 文件名,
                        "content": 文件内容,
                        "score": 1.0
                    },
                    ...
                ]
            }
            如果读取失败或列表为空，返回 None
        """
        if not user_document:
            return None
        
        success_files, failed_files = FileReader.read_files(user_document)
        
        if not success_files:
            return None
        
        results = []
        for file_info in success_files:
            results.append({
                "url": file_info["filepath"],
                "title": file_info["filename"],
                "content": file_info["content"],
                "score": 1.0
            })
        
        return {
            "task_id": "user_task",
            "issue": "",
            "description": "",
            "results": results
        }
    
    async def _simplify_single_task(self, task_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        对单个任务的搜索结果进行简化处理
        
        Args:
            task_result: 单个任务的搜索结果
            
        Returns:
            包含task_id和简化结果的字典
        """
        task_id = task_result.get("task_id", "unknown")
        
        if not task_result.get("results"):
            return {
                "task_id": task_id,
                "status": "failed",
                "summary": "",
                "structure": "",
                "retrieval_keywords": [],
                "error": "该任务无搜索结果"
            }
        
        try:
            messages = self._build_messages_for_task(task_result)
            response = await model_client.llm.ainvoke(messages)
            response_text = response.content
            
            parsed = self._parse_response(response_text)
            
            return {
                "task_id": task_id,
                "status": "success",
                "summary": parsed.get("summary", ""),
                "structure": parsed.get("structure", ""),
                "retrieval_keywords": parsed.get("retrieval_keywords", [])
            }
            
        except Exception as e:
            return {
                "task_id": task_id,
                "status": "failed",
                "summary": "",
                "structure": "",
                "retrieval_keywords": [],
                "error": str(e)
            }
    
    async def simplify(self, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        对搜索结果进行并行简化处理
        
        不同task_id的任务并行处理，同一task_id的结果一起传入大模型
        
        Args:
            search_results: 搜索结果列表
            
        Returns:
            {
                "status": "success" | "partial_success" | "failed",
                "search_summary": [
                    {
                        "task_id": "task_1",
                        "summary": "关键内容总结",
                        "structure": "逻辑结构梳理"
                    },
                    ...
                ],
                "error": Optional[str]
            }
        """
        if not search_results:
            return {
                "status": "failed",
                "search_summary": None,
                "error": "搜索结果为空"
            }
        
        tasks = [self._simplify_single_task(task) for task in search_results]
        task_results = await asyncio.gather(*tasks)
        
        successful_tasks = [r for r in task_results if r.get("status") == "success"]
        failed_tasks = [r for r in task_results if r.get("status") == "failed"]
    
        
        if len(successful_tasks) == len(task_results):
            status = "success"
        elif len(successful_tasks) > 0:
            status = "partial_success"
        else:
            status = "failed"
        
        return {
            "status": status,
            "search_summary": task_results,
            "error": "; ".join([r.get("error", "") for r in failed_tasks]) if failed_tasks else None
        }
    
    def simplify_sync(self, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """同步版本的简化方法"""
        return asyncio.run(self.simplify(search_results))


async def search_simplify_node(state: MessageState) -> MessageState:
    """
    模块入口函数：对搜索结果进行简化处理
    
    Args:
        state: Graph状态字典，包含search_results和user_document
        
    Returns:
        更新后的状态字典，包含search_summary
    """
    agent = SearchSimplifyAgent()
    
    search_results = state.get("search_results", [])
    
    user_document = state.get("user_document")
    if user_document:
        user_task = agent._process_user_documents(user_document)
        if user_task:
            search_results = search_results + [user_task]
    
    result = await agent.simplify(search_results)
    
    state["search_summary"] = result.get("search_summary")
    state["status"] = result.get("status")
    
    if result.get("error"):
        state["error"] = result.get("error")
    
    return state


if __name__ == "__main__":
    test_search_results = [
        {
            "task_id": "task_1",
            "issue": "2026年AI行业市场环境与发展趋势调研",
            "description": "调研当前时间节点人工智能行业的宏观市场数据",
            "status": "success",
            "results": [
                {
                    "url": "https://www.news.cn/20260128/3b2f11906fd74ca397fef9996c805a60/c.html",
                    "title": "新华深读｜2026年中国AI发展趋势前瞻",
                    "content": "美国聚焦闭源，而中国主导开源市场。2026年人工智能行业将迎来新的发展机遇...",
                    "score": 0.8755427
                },
                {
                    "url": "https://www.ibm.com/cn-zh/think/news/ai-tech-trends-predictions-2026",
                    "title": "2026 年塑造AI 与技术的趋势 - IBM联盟开源人工智能总监",
                    "content": "人工智能技术将在2026年取得重大突破...",
                    "score": 0.87218446
                }
            ]
        },
        {
            "task_id": "task_2",
            "issue": "AI行业竞争格局与标杆企业商业模式分析",
            "description": "收集主要竞争对手信息",
            "status": "success",
            "results": [
                {
                    "url": "https://example.com/ai-competition",
                    "title": "AI行业竞争格局分析",
                    "content": "当前AI行业竞争激烈，主要玩家包括...",
                    "score": 0.85
                }
            ]
        }
    ]
    
    agent = SearchSimplifyAgent()
    result = asyncio.run(agent.simplify(test_search_results))
    
    print(f"状态: {result['status']}")
    if result.get("search_summary"):
        print("\n" + "=" * 60)
        print("各任务简化结果:")
        print("=" * 60)
        for task_summary in result["search_summary"]:
            print(f"\n【{task_summary['task_id']}】状态: {task_summary['status']}")
            if task_summary.get("summary"):
                print(f"关键内容总结:\n{task_summary['summary']}")
            if task_summary.get("structure"):
                print(f"逻辑结构梳理:\n{task_summary['structure']}")
            if task_summary.get("retrieval_keywords"):
                print(f"检索关键词: {task_summary['retrieval_keywords']}")
            if task_summary.get("error"):
                print(f"错误: {task_summary['error']}")
