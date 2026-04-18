"""
LangGraph 编排器（Orchestrator）。

该模块负责：
1. 注册所有 pipeline 节点（路由、检索、大纲、写作、组装、后处理等）。
2. 定义节点之间的执行流向（包括条件分支与并行触发）。
3. 提供统一的异步/同步运行入口，供 `main.py` 或其他调用方直接使用。
"""

import asyncio
import uuid
from typing import Any, Dict, List, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.pipeline.agents import (
    assemble_article_node,
    deepagents_node,
    head_agent_node,
    memory_manage_agent_node,
    outline_generate_node,
    outline_review_node,
    search_node,
    search_simplify_node,
    title_decomposer_node,
    write_chapter_node,
)
from src.pipeline.state_model import MessageState

# Langfuse 在部分环境下可能未安装或未配置。
# 这里采用“可选依赖”的方式：导入失败不影响主流程运行。
try:
    from langfuse import get_client
    from langfuse.langchain import CallbackHandler
except Exception:  # pragma: no cover - optional dependency behavior
    get_client = None
    CallbackHandler = None


class WriterAgentOrchestrator:
    """
    Writer 工作流编排器。

    说明：
    - `self.graph` 是编译后的 LangGraph，可通过 `ainvoke` 执行。
    - `MemorySaver` 用于线程级 checkpoint，支持同一 `thread_id` 的状态延续。
    - 若启用 Langfuse，则自动挂载 callback 用于 tracing。
    """

    def __init__(self, enable_langfuse: bool = True):
        # 图状态 checkpoint（内存实现）
        self.memory = MemorySaver()

        # tracing 相关对象，默认置空，按条件启用
        self.langfuse = None
        self.langfuse_handler = None

        # 仅在“允许 + 依赖存在”时启用 Langfuse
        if enable_langfuse and get_client is not None and CallbackHandler is not None:
            try:
                self.langfuse = get_client()
                self.langfuse_handler = CallbackHandler()
            except Exception:
                # tracing 初始化失败时降级为无 tracing，不中断主流程
                self.langfuse = None
                self.langfuse_handler = None

        # 构建并编译 LangGraph
        self.graph = self._build_graph()

    def _route_by_head(self, state: MessageState) -> List[str]:
        """
        根据 head_agent 的路由结果决定下一跳节点。

        逻辑：
        1. 主分支二选一：
           - `title_decomposer` -> `title_decomposer_node`
           - `deepagents` -> `deepagents_node`
           - 兜底默认走 `title_decomposer_node`
        2. 若 `need_memory_manage=True`，额外并行触发 `memory_manage_agent_node`。

        返回值：
        - 返回 `List[str]`，允许 LangGraph 在同一步触发多个目标节点。
        """
        route_map = {
            "title_decomposer": "title_decomposer_node",
            "deepagents": "deepagents_node",
        }

        # head_agent_node 产出的 next_node；缺省时走新文章流程
        next_key = state.get("next_node", "title_decomposer")
        targets = [route_map.get(next_key, "title_decomposer_node")]

        # 当上下文占用超过阈值时，附加并行记忆压缩分支
        if state.get("need_memory_manage", False):
            targets.append("memory_manage_agent_node")

        return targets

    def _build_graph(self):
        """
        构建并编译 LangGraph。

        节点拓扑概览：
        START
          -> head_agent_node
             -> title_decomposer_node -> search_node -> search_simplify_node
                -> outline_generate_node -> outline_review_node
                -> write_chapter_node -> assemble_article_node -> END
             -> deepagents_node -> END
             -> (可并行) memory_manage_agent_node -> END
        """
        builder = StateGraph(MessageState)

        # 1) 注册全部节点
        builder.add_node("memory_manage_agent_node", memory_manage_agent_node)
        builder.add_node("head_agent_node", head_agent_node)
        builder.add_node("title_decomposer_node", title_decomposer_node)
        builder.add_node("search_node", search_node)
        builder.add_node("search_simplify_node", search_simplify_node)
        builder.add_node("outline_generate_node", outline_generate_node)
        builder.add_node("outline_review_node", outline_review_node)
        builder.add_node("write_chapter_node", write_chapter_node)
        builder.add_node("assemble_article_node", assemble_article_node)
        builder.add_node("deepagents_node", deepagents_node)

        # 2) 固定入口：START -> head_agent_node
        builder.add_edge(START, "head_agent_node")

        # 3) 条件分发：
        #    由 _route_by_head 返回目标节点列表，实现主流程路由 + 可选并行压缩
        builder.add_conditional_edges(
            "head_agent_node",
            self._route_by_head,
            {
                "title_decomposer_node": "title_decomposer_node",
                "deepagents_node": "deepagents_node",
                "memory_manage_agent_node": "memory_manage_agent_node",
            },
        )

        # 4) 新文章流程主链路
        builder.add_edge("title_decomposer_node", "search_node")
        builder.add_edge("search_node", "search_simplify_node")
        builder.add_edge("search_simplify_node", "outline_generate_node")
        builder.add_edge("outline_generate_node", "outline_review_node")
        builder.add_edge("outline_review_node", "write_chapter_node")
        builder.add_edge("write_chapter_node", "assemble_article_node")
        builder.add_edge("assemble_article_node", END)

        # 5) deepagents 分支与 memory 管理分支都直接结束
        builder.add_edge("deepagents_node", END)
        builder.add_edge("memory_manage_agent_node", END)

        # 6) 编译图并按需挂载 Langfuse 回调
        compiled = builder.compile(checkpointer=self.memory)
        if self.langfuse_handler is not None:
            compiled = compiled.with_config({"callbacks": [self.langfuse_handler]})
        return compiled

    def _build_initial_state(
        self,
        topic: str,
        demand: Optional[str],
        thread_id: str,
        is_lookup_outline: bool,
    ) -> Dict[str, Any]:
        """
        构造图的初始状态。

        关键字段说明：
        - user_id: 用于文件落盘目录、会话隔离等。
        - user_query: 业务层明确使用的原始主题。
        - depth_demand: 可选写作深度/风格要求。
        - is_lookup_outline: 是否在大纲阶段启用人工查看/反馈机制。
        - messages: LangGraph/LangChain 侧常用消息字段（会参与路由和上下文计算）。
        """
        return {
            "user_id": thread_id,
            "user_query": topic,
            "depth_demand": demand,
            "is_lookup_outline": is_lookup_outline,
            "messages": [{"role": "user", "content": topic}],
        }

    async def run(
        self,
        topic: str,
        demand: Optional[str] = None,
        thread_id: Optional[str] = None,
        is_lookup_outline: bool = False,
    ) -> Dict[str, Any]:
        """
        异步执行工作流。

        参数：
        - topic: 必填主题；空字符串会直接抛错。
        - demand: 可选需求描述。
        - thread_id: 可选会话 ID；未传时自动生成 UUID。
        - is_lookup_outline: 是否在大纲节点触发人工审阅中断。

        返回：
        - 最终图状态（字典），并附加 `thread_id` 方便调用方追踪。
        """
        # 规范化并校验主题
        normalized_topic = (topic or "").strip()
        if not normalized_topic:
            raise ValueError("topic 不能为空")

        # 线程 ID 为空时自动生成，确保每次执行可追踪
        run_thread_id = (thread_id or "").strip() or uuid.uuid4().hex

        # LangGraph 线程配置：用于 checkpoint 命名空间隔离
        config = {"configurable": {"thread_id": run_thread_id}}

        # 初始化状态并触发图执行
        initial_state = self._build_initial_state(
            topic=normalized_topic,
            demand=demand,
            thread_id=run_thread_id,
            is_lookup_outline=is_lookup_outline,
        )
        result = await self.graph.ainvoke(initial_state, config=config)

        # 兼容非 dict 返回（正常情况下应为 dict）
        if not isinstance(result, dict):
            return {"thread_id": run_thread_id, "result": result}

        # 在结果中补齐 thread_id，便于上层统一消费
        result["thread_id"] = run_thread_id
        return result

    def run_sync(
        self,
        topic: str,
        demand: Optional[str] = None,
        thread_id: Optional[str] = None,
        is_lookup_outline: bool = False,
    ) -> Dict[str, Any]:
        """
        同步包装器：用于 CLI 或脚本直接调用。

        说明：
        - 内部通过 `asyncio.run` 执行异步 `run`。
        - 若调用方已在事件循环中，应改用 `await run(...)` 避免嵌套循环报错。
        """
        return asyncio.run(
            self.run(
                topic=topic,
                demand=demand,
                thread_id=thread_id,
                is_lookup_outline=is_lookup_outline,
            )
        )
