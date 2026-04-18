
"""
文章处理主控制器智能体
使用 deepagents 框架协调文章审核和润色子智能体
"""
import os

import json
from typing import TypedDict, List, Optional, Dict, Any
from pathlib import Path

from deepagents import create_deep_agent, CompiledSubAgent
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from src.core.model_client import ModelClient
from langchain_openai import ChatOpenAI

from src.prompts.load_prompt import load_prompt_with_metadata
from src.pipeline.agents.deepagent.polish_article_agent import create_polish_article_graph
from src.pipeline.agents.deepagent.review_article_agent import create_review_article_graph
from src.pipeline.agents.deepagent.draw_images_agent import create_draw_images_graph
from deepagents.backends import FilesystemBackend
from langgraph.checkpoint.memory import MemorySaver
from src.utils.logs import logger
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")     
load_dotenv()


class ControllerState(TypedDict):
    """主控制器状态"""
    messages: List
    user_id: Optional[str]
    user_modifications: Optional[str]
    iterations: int
    review_history: List[Dict[str, Any]]
    final_article: Optional[str]
    final_score: Optional[int]
    
#获取模型
def get_llm(model_name: str):
    # model_client = ModelClient()
    # model_entry = next(m for m in model_client._llms if m["name"] == model_name)
    # llm = model_entry["llm"]
    # print(f"使用模型: {model_name}")
    # print(llm)
    return ChatOpenAI(
    model="ZhipuAI/GLM-5",
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_BASE_API_KEY"),
    temperature=float(os.getenv("DEFAULT_LLM_TEMPERATURE", 0.5)),
    max_tokens=int(os.getenv("DEFAULT_LLM_MAX_TOKENS", 1024)),
    )
from langchain.chat_models import init_chat_model
    
llm = init_chat_model(
    model_provider="openai",
       # model=os.getenv("DEFAULT_LLM_MODEL"),
    base_url=os.getenv("SILICONFLOW_BASE_URL"),
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    model="Pro/deepseek-ai/DeepSeek-V3.2"
)
def create_controller_agent(user_id: str):
    """
    创建主控制器智能体

    Returns:
        编译后的主控制器智能体
    """
    # 加载控制器提示词
    prompt_data = load_prompt_with_metadata("controller_prompt")
    system_prompt = prompt_data.get("content", "")

    # 定义文章审核子智能体
    review_agent = CompiledSubAgent(
    runnable=create_review_article_graph(), # 修正参数名
    name="review_article_agent",
    description="""文章审核助手，当需要审核文章时调用此工具，它可以自己去读取data目录下的文章。
    输入：请将任务封装为 JSON 字符串传入，包含 user_id（用户ID）、order（指令内容）、user_modifications（用户修改建议，可选） 三个字段。
    """
    )

    # 定义文章润色子智能体
    polish_agent = CompiledSubAgent(
        runnable=create_polish_article_graph(),  # 修改1: compiled_graph 改为 runnable
        name="polish_article_agent",
        description="""文章润色助手，当需要润色文章时调用此工具，它可以自己去读取data目录下的文章。
        输入要求：请将任务封装为 JSON 字符串传入，包含 user_id（用户ID）、order（指令内容）、user_modifications（用户修改建议，可选）、review_report（审核报告，可选）
       """
    )

    # 定义图片生成子智能体
    draw_images_agent = CompiledSubAgent(
        runnable=create_draw_images_graph(),
        name="draw_images_agent",
        description="""图片生成助手，当需要根据文章内容生成配图时调用此工具。
        输入要求：请将任务封装为 JSON 字符串传入，包含 user_id（用户ID）和图片描述字典（如 {"image_1": "图片描述1", "image_2": "图片描述2"}）。
        图片描述通常从润色结果的 add_image 字段中提取。
        """
    )

    user_id=f"D:/study_notebook/DeepStudy_jupyter/SGG/OmniWriter/data/{user_id}"
    backend=FilesystemBackend(root_dir=user_id,virtual_mode=True)
    skills=["/skills/"]
    
    # 创建主智能体 (假设你在外部已经定义了 llm 变量)
    controller_agent = create_deep_agent(
        model=llm,                            
        system_prompt=system_prompt,   
        subagents=[review_agent, polish_agent, draw_images_agent],
        # backend=backend,
        # skills=skills,
        checkpointer=MemorySaver()
    )

    return controller_agent

async def deepagents_node(state: MessageState) -> Dict[str, Any]:
    user_id = state["user_id"]
    agent = create_controller_agent(user_id)
    messages = [
        HumanMessage(content=state["article_post_process"])
    ]
    result = await agent.ainvoke(
        {"messages": messages},
        config={"configurable": {"thread_id": user_id}}
    )
    final_messages = result.get("messages", [])
    if final_messages:
        last_message = final_messages[-1]
        content = last_message.content if hasattr(last_message, 'content') else str(last_message)
        try:
            output = json.loads(content)
            logger.info(f"文章处理完成，结果: {output}")
            return output
        except json.JSONDecodeError:
            logger.warning(f"无法解析最终输出为JSON，返回原始内容")
            return {
                "status": "success",
                "messages": content
            }

    return {
        "status": "failed",
        "error": "未获取到有效输出"
    }
    

async def run_controller_agent(
    user_id: str,
    user_modifications: Optional[str] = None,
    max_iterations: int = 3
) -> Dict[str, Any]:
    """
    运行主控制器智能体

    Args:
        user_id: 用户ID
        user_modifications: 用户修改建议
        max_iterations: 最大迭代次数

    Returns:
        处理结果
    """
    logger.info(f"启动文章处理流程，user_id: {user_id}")

    # 创建主智能体
    agent = create_controller_agent(user_id)

    initial_input=f"用户的文章都存放在data目录下。请开始对用户ID为{user_id}的用户的文章进行润色和审核，用户的修改建议: {user_modifications}，最大迭代次数: {max_iterations}"
    # initial_input=f"帮我查一查马斯克最近有什么动作没"
    messages = [
        HumanMessage(content=initial_input)
    ]

    async for event in agent.astream(
    {"messages": messages},
    stream_mode=["updates", "messages", "custom"],  # 同时三种！
    config={"configurable": {"thread_id": user_id}},
    subgraphs=True,
    version="v2"
    ):
        if event["type"] == "updates":
            print(f"📊 状态更新 [{event['ns']}]: {event['data']}")
        elif event["type"] == "messages":
            print(f"💬 Token [{event['ns']}]: {event['data'].content if hasattr(event['data'], 'content') else ''}")
        elif event["type"] == "custom":
            print(f"🎯 自定义事件 [{event['ns']}]: {event['data']}")
        # # 运行主智能体
        # result = await agent.ainvoke({"messages": messages}, config={"configurable": {"thread_id": user_id}})

        # # 解析结果
        # final_messages = result.get("messages", [])
        # if final_messages:
        #     last_message = final_messages[-1]
        #     content = last_message.content if hasattr(last_message, 'content') else str(last_message)
        #     try:
        #         output = json.loads(content)
        #         logger.info(f"文章处理完成，结果: {output}")
        #         return output
        #     except json.JSONDecodeError:
        #         logger.warning(f"无法解析最终输出为JSON，返回原始内容")
        #         return {
        #             "status": "success",
        #             "raw_output": content
        #         }

        # return {
        #     "status": "failed",
        #     "error": "未获取到有效输出"
        # }

    # except Exception as e:
    #     logger.error(f"主控制器运行失败: {str(e)}", exc_info=True)
    #     return {
    #         "status": "failed",
    #         "error": str(e)
    #     }


if __name__ == "__main__":
    import asyncio
    # model=get_llm("GLM-5")
    # res = model.invoke("你好")
    # print(res)
    async def test():
        """测试主控制器"""
        await run_controller_agent(
            user_id="thread_id_123",
            user_modifications="语言要通俗易懂，风默有趣",
            max_iterations=3
        )
        # print(json.dumps(result, ensure_ascii=False, indent=2))

    asyncio.run(test())

