from deepagents import create_deep_agent, CompiledSubAgent
from langgraph.checkpoint.memory import MemorySaver
from deepagents.backends.filesystem import FilesystemBackend
import os
from langchain_openai import ChatOpenAI
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")           # 可选，不写就是 default
# Checkpointer is REQUIRED for human-in-the-loop

checkpointer = MemorySaver()
def get_llm():
    # model_client = ModelClient()
    # model_entry = next(m for m in model_client._llms if m["name"] == model_name)
    # llm = model_entry["llm"]
    # print(f"使用模型: {model_name}")
    # print(llm)
    return ChatOpenAI(
    base_url=os.getenv("SILICONFLOW_BASE_URL"),
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    model="Pro/deepseek-ai/DeepSeek-V3.2"
    )


# async_subagents = [
#     AsyncSubAgent(
#         name="search_user",
#         description="可以用来查询用户信息",
#         graph_id="search_user",
#         # 无 url → 同部署 ASGI 传输
#     ),
# ]
from search_user_agent import create_search_user_graph
from review_article_agent import create_review_article_graph
from polish_article_agent import create_polish_article_graph
from layout_agent import create_layout_graph
# 定义文章审核子智能体
review_agent = CompiledSubAgent(
runnable=create_review_article_graph(), # 修正参数名
name="review_article_agent",
description="""文章审核助手，当需要审核文章时调用此工具，它可以自己去读取data目录下的文章。
调用此智能体时,输入中必须包含一个JSON 字符串，其中包含 user_id（用户ID）、user_modifications（用户修改建议，可选） 二个字段。
如果调用此智能体时，传入的输入中没有user_id字段，或者user_id字段为空，那么此智能体会返回错误信息。
例如：传入的输入为"{"user_id": "tw946", "user_modifications": "文章要包含三个故事"}"，则此智能体会返回审核报告。
"""
)

# 定义文章润色子智能体
polish_agent = CompiledSubAgent(
    runnable=create_polish_article_graph(),  # 修改1: compiled_graph 改为 runnable
    name="polish_article_agent",
    description="""文章润色助手，当需要润色文章时调用此工具，它可以自己去读取data目录下的文章。
    调用此智能体时,输入中必须包含一个JSON 字符串，其中包含 user_id（用户ID）、user_modifications（用户修改建议，可选）、review_report（审核报告，可选） 三个字段。
    如果调用此智能体时，传入的输入中没有user_id字段，或者user_id字段为空，那么此智能体会返回错误信息。
    例如：传入的输入为"{"user_id": "tw946", "user_modifications": "文章要写的通俗易懂", "review_report": "审核报告中显示了文章只包含了两个故事，需要添加第三个故事。"}"，则此智能体会返回润色结果。
    """
)

# 定义排版子智能体
layout_agent = CompiledSubAgent(
    runnable=create_layout_graph(),
    name="layout_agent",
    description="""排版助手，当需要根据文章内容进行排版时调用此工具。
    调用此智能体时,输入中必须包含一个JSON 字符串，其中包含 user_id（用户ID）。
    如果调用此智能体时，传入的输入中没有user_id字段，或者user_id字段为空，那么此智能体会返回错误信息。
    例如：传入的输入为"{"user_id": "tw946"}"，则此智能体会返回排版结果。
    """
)
agent = create_deep_agent(
    model=get_llm(),
    subagents=[review_agent,polish_agent,layout_agent],
    # backend=FilesystemBackend(root_dir="D:/study_notebook/DeepStudy_jupyter/SGG/OmniWriter/test",virtual_mode=True),
    # skills=["/skills/"],
    checkpointer=checkpointer,  # Required!
    system_prompt="""你是一个专业的文章润色助手，你可以根据用户的需求，对文章进行润色，审核和排版。
    【注意事项】1、你不需要去找文章的路径，你只需要将用户的ID传递给子智能体，子智能体会根据用户ID调用读取对应的文章。禁止自己去读取文章进行润色。
    2、只有当审核智能体返回的审核报告显示通过，你才能去调用排版智能体进行排版，否则就再调用润色智能体进行润色，直到审核报告通过为止。  
    """
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "我的user_id为12345，请帮我把文章润色一下，我要求文章字数在1000字左右",
            }
        ]
    },
    config={"configurable": {"thread_id": "12345"}},
)
state = agent.get_state({"configurable": {"thread_id": "12345"}})
print(result)