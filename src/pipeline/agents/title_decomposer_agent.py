"""
标题分解智能体模块
将用户查询分解为多个具体的子查询任务

输出格式示例:
{
    "subtasks":[
        {
            "id": "task_1",
            "issue": "2026年AI行业市场环境与发展趋势调研",
            "description": "调研当前时间节点人工智能行业的宏观市场数据",
            "expected_results": 3
        },
        {
            "id": "task_2",
            "issue": "AI行业竞争格局与标杆企业商业模式分析",
            "description": "收集主要竞争对手信息",
            "expected_results": 5
        }
    ]
}
"""

import json
import re
from typing import List, Dict, Any, Optional

from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from src.pipeline.state_model import MessageState
from src.core.model_client import model_client
from src.prompts.load_prompt import load_prompt
import asyncio
load_dotenv()


class TitleDecomposerAgent:
    """
    标题分解智能体
    
    将用户查询分解为多个具体的子查询任务，
    每个子任务包含: id, issue, description, expected_results
    """
    
    PROMPT_NAME = "title_decomposer"
    
    def __init__(self):
        self._prompt_template: str = load_prompt(self.PROMPT_NAME)
    
    def _build_messages(self, user_query: str, depth_demand: str = None) -> List:
        """
        构建LLM消息列表
        
        SystemMessage: 角色定义 + 任务说明
        HumanMessage: 具体的用户查询和深度需求
        """
        system_prompt = self._prompt_template.replace("{{ user_query }}", user_query)
        system_prompt = system_prompt.replace("{{ depth_demand }}", depth_demand or "无特殊要求")
        
        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content="请根据上述要求分解任务。")
        ]
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        解析LLM响应，提取JSON格式的子任务列表
        
        Args:
            response_text: LLM返回的原始文本
            
        Returns:
            包含subtasks列表的字典
        """
        json_patterns = [
            r'\{[\s\S]*"subtasks"[\s\S]*\}',
            r'\[[\s\S]*\{[\s\S]*"id"[\s\S]*\][\s\S]*\}',
        ]
        
        for pattern in json_patterns:
            match = re.search(pattern, response_text)
            if match:
                try:
                    parsed = json.loads(match.group())
                    if "subtasks" in parsed:
                        return parsed
                    elif isinstance(parsed, list):
                        return {"subtasks": parsed}
                except json.JSONDecodeError:
                    continue
        
        return {"subtasks": [], "raw_response": response_text}
    
    def _normalize_subtasks(self, subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """标准化子任务格式"""
        normalized = []
        for i, task in enumerate(subtasks):
            normalized_task = {
                "id": task.get("id", f"task_{i + 1}"),
                "issue": task.get("issue", task.get("query", "")),
                "description": task.get("description", ""),
                "expected_results": task.get("expected_results", 5),
            }
            normalized.append(normalized_task)
        return normalized
    
    async def decompose(self, user_query: str, depth_demand: str = None) -> Dict[str, Any]:
        """
        分解用户查询为子任务
        
        Args:
            user_query: 用户输入的查询
            
        Returns:
            {
                "status": "success" | "failed",
                "subtasks": [...],
                "error": Optional[str]
            }
        """
        try:
            messages = self._build_messages(user_query, depth_demand)
            response = await model_client.llm.ainvoke(messages)
            response_text = '```json\n{\n  "subtasks": [\n    {\n      "id": "task_1",\n      "issue": "张雪峰去世新闻最新消息",\n      "description": "核实张雪峰去世消息的真实性，搜索相关官方通报或权威媒体报道，确认事件是否发生及具体时间。",\n      "expected_results": 5\n    },\n    {\n      "id": "task_2",\n      "issue": "张雪峰个人简介及生平经历",\n      "description": "收集张雪峰的基本个人信息、职业背景、主要成就及社会影响力，为报道提供人物背景素材。",\n      "expected_results": 3\n    },\n    {\n      "id": "task_3",\n      "issue": "张雪峰去世原因及细节",\n      "description": "如果消息属实，搜索关于去世的具体原因、地点及相关细节；若为谣言，搜索辟谣信息。",\n      "expected_results": 3\n    },\n    {\n      "id": "task_4",\n      "issue": "张雪峰相关公众评价及近期动态",\n      "description": "收集公众对张雪峰的评论、哀悼动态（若属实）或其近期的公开活动行程，丰富报道内容。",\n      "expected_results": 5\n    }\n  ]\n}\n```'
            response_text = response.content
            
            parsed = self._parse_response(response_text)
            subtasks = self._normalize_subtasks(parsed.get("subtasks", []))
            
            return {
                "status": "success",
                "subtasks": subtasks,
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "subtasks": [],
                "error": str(e)
            }
    
    def decompose_sync(self, user_query: str, depth_demand: str = None) -> Dict[str, Any]:
        """同步版本的分解方法"""
        import asyncio
        return asyncio.run(self.decompose(user_query, depth_demand))



async def title_decomposer_node(state: MessageState) -> Dict[str, Any]:
    """
    模块入口函数：分解用户查询
    
    Args:
        user_query: 用户查询字符串
        
    Returns:
        分解结果，包含子任务列表
    """
    agent = TitleDecomposerAgent()
    return await agent.decompose(state["user_query"], state["depth_demand"])


if __name__ == "__main__":
    pass
    # test_query = "帮我写一篇关于张雪峰去世的相关报道"
    # depth_demand = "写得详细些，深入些"
    # result = asyncio.run(title_decompose_node(MessageState({"user_query": test_query, "depth_demand": depth_demand})))
    
    # print(f"状态: {result['status']}")
    # print(f"子任务数量: {len(result.get('subtasks', []))}")
    
    # for task in result.get("subtasks", []):
    #     print(f"\n任务 {task['id']}:")
    #     print(f"  主题: {task['issue']}")
    #     print(f"  描述: {task['description']}")
    #     print(f"  期望结果数: {task['expected_results']}")
