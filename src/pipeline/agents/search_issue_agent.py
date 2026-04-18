"""
搜索任务智能体模块
根据任务列表异步执行多个搜索任务，并整合搜索结果，同时将搜索结果存入向量数据库
"""

import asyncio
from typing import List, Dict, Any
from dataclasses import dataclass
from src.pipeline.tools.search_tool import async_search_client
from src.pipeline.tools.rag_tool import RAGTool, create_rag_tool
from src.pipeline.state_model import MessageState
from dotenv import load_dotenv
from src.utils.logs import logger

load_dotenv()


@dataclass
class SearchTask:
    """搜索任务数据类"""
    id: str
    issue: str
    description: str
    expected_results: int


class SearchIssueAgent:
    """
    搜索任务智能体
    
    根据分解后的任务列表，异步执行多个主题搜索，
    并将搜索结果整合返回
    """
    
    def __init__(self):
        self.rag_tool = create_rag_tool()
        pass
    
    async def _execute_single_search(
        self, 
        task: SearchTask
    ) -> Dict[str, Any]:
        """
        执行单个搜索任务，并将搜索结果存入向量数据库
        
        Args:
            task: 搜索任务对象，包含查询主题和期望结果数量
            
        Returns:
            包含任务ID和搜索结果的字典
        """
        try:
            result = await async_search_client.search(
                query=task.issue,
                max_results=task.expected_results,
                search_depth="advanced"
            )
            """
            result 格式：
            {
                'query':'查询主题'，
                'result':[
                    {'title':'结果标题','url':'结果URL','content':'结果内容','score':0.9986},
                    ...
                ]
            }
            """
            
            search_results = result.get("results", [])
            
            # 将搜索结果存入向量数据库
            chroma_stats = None
            if search_results:
                try:
                    chroma_stats = self.rag_tool.add_search_results(
                        search_results=search_results,
                        task_id=task.id
                    )
                    logger.info(f"任务 {task.id} 搜索结果已存入向量数据库: {chroma_stats}")
                except Exception as e:
                    logger.warning(f"任务 {task.id} 存入向量数据库失败: {str(e)}", exc_info=True)
            
            return {
                "task_id": task.id,
                "issue": task.issue,
                "description": task.description,
                "status": "success",
                "results": search_results,
                "chroma_stats": chroma_stats
            }
        except Exception as e:
            return {
                "task_id": task.id,
                "issue": task.issue,
                "status": "failed",
                "error": str(e),
                "results": [],
                "chroma_stats": None
            }
    
    async def execute_search_tasks(
        self, 
        subtasks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        异步执行多个搜索任务
        
        Args:
            subtasks: 任务列表，每个任务包含:
                - id: 任务唯一标识
                - issue: 搜索主题
                - description: 任务描述
                - expected_results: 期望返回结果数量
                
        Returns:
            整合后的搜索结果，包含:
                - total_tasks: 总任务数
                - completed_tasks: 完成的任务数
                - failed_tasks: 失败的任务数
                - results: 各任务的搜索结果列表
        """
        search_tasks = [
            SearchTask(
                id=task.get("id", f"task_{i}"),
                issue=task.get("issue", ""),
                description=task.get("description", ""),
                expected_results=task.get("expected_results", 5)
            )
            for i, task in enumerate(subtasks)
        ]
        
        search_coroutines = [
            self._execute_single_search(task) 
            for task in search_tasks
        ]
        results = await asyncio.gather(*search_coroutines)
        
        completed_tasks = sum(1 for r in results if r.get("status") == "success")
        failed_tasks = sum(1 for r in results if r.get("status") == "failed")
        
        return {
            "total_tasks": len(search_tasks),
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "results": list(results)
        }


async def search_node(state: MessageState) -> Dict[str, Any]:
    """
    模块入口函数：执行搜索任务列表
    
    Args:
        subtasks: 任务列表
        
    Returns:
        整合后的搜索结果
    """
    agent = SearchIssueAgent()
    results = await agent.execute_search_tasks(state["subtasks"])
    state["search_results"]= results["results"]
    return state    



if __name__ == "__main__":
    test_subtasks = [
        {
            "id": "task_1",
            "issue": "2026年AI行业市场环境与发展趋势调研",
            "description": "调研当前时间节点人工智能行业的宏观市场数据",
            "expected_results": 2
        },
        {
            "id": "task_2",
            "issue": "AI行业竞争格局与标杆企业商业模式分析",
            "description": "收集主要竞争对手信息",
            "expected_results": 1
        }
    ]
    
    agent = SearchIssueAgent()
    result = asyncio.run(agent.execute_search_tasks(test_subtasks))
    
    # print(f"总任务数: {result['total_tasks']}")
    # print(f"完成任务数: {result['completed_tasks']}")
    # print(f"失败任务数: {result['failed_tasks']}")
    
    # for task_result in result["results"]:
    #     print(f"\n任务 {task_result['task_id']}: {task_result['issue']}")
    #     print(f"状态: {task_result['status']}")
    #     if task_result['status'] == 'success':
    #         print(f"结果数量: {len(task_result['results'])}")
