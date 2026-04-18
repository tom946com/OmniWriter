"""
图片生成智能体
使用 LangGraph 框架，根据 controller_agent 传来的消息并行生成图片
"""

import json
import os
import asyncio
import re
from typing import TypedDict, Dict, Any, Optional, List
from pathlib import Path

import httpx
from PIL import Image
from io import BytesIO

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

from src.utils.logs import logger

load_dotenv()


async def generate_modelscope_image(prompt: str, file_name: str, size: str = "1024x1024", client: Optional[httpx.AsyncClient] = None) -> str:
    """
    核心画图函数（异步版本）：接收提示词，返回本地图片路径
    
    Args:
        prompt: 图片生成提示词
        file_name: 保存的文件路径
        size: 图片尺寸，默认 "1024x1024"
        client: httpx 异步客户端，用于复用 TCP 连接
        
    Returns:
        本地图片文件路径
    """
    base_url = os.getenv("IMAGE_BASE_URL")
    api_key = os.getenv("IMAGE_BASE_API_KEY")
    model = os.getenv("IMAGE_MODEL")
    
    common_headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # 1. 提交异步任务
    resp = await client.post(
        f"{base_url}v1/images/generations",
        headers={**common_headers, 
        "X-ModelScope-Async-Mode": "true"
        },
        json={
            "model": model,
            "prompt": prompt,
            "size": size
        }
    )
    resp.raise_for_status()
    task_id = resp.json()["task_id"]

    # 2. 轮询结果
    while True:
        result = await client.get(
            f"{base_url}v1/tasks/{task_id}",
            headers={**common_headers, "X-ModelScope-Task-Type": "image_generation"},
        )
        result.raise_for_status()
        data = result.json()

        if data["task_status"] == "SUCCEED":
            img_url = data["output_images"][0]
            img_resp = await client.get(img_url)
            image = Image.open(BytesIO(img_resp.content))
            image.save(file_name)
            return file_name
        elif data["task_status"] == "FAILED":
            raise Exception("ModelScope 图像生成失败")

        await asyncio.sleep(5)


class DrawImagesState(TypedDict):
    """图片生成状态"""
    messages: List
    user_id: Optional[str]
    image_dict: Dict[str, Dict[str, str]]


def _parse_image_dict(messages: List) -> Dict[str, Dict[str, str]]:
    """
    从 messages 中解析图片字典

    Args:
        messages: 消息列表

    Returns:
        图片字典，格式：{"image_1": {"description": "图片描述 1", "size": "1024x1024"}, ...}
        兼容旧格式：{"image_1": "图片描述 1"}
    """
    result = {}

    if not messages:
        return result

    for msg in reversed(messages):
        try:
            content = msg.content if hasattr(msg, 'content') else str(msg)
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    for key, value in data.items():
                        if key.startswith("image_"):
                            if isinstance(value, str):
                                result[key] = {"description": value, "size": "1024x1024"}
                            elif isinstance(value, dict):
                                desc = value.get("description", "")
                                size = value.get("size", "1024x1024")
                                result[key] = {"description": desc, "size": size}
            except json.JSONDecodeError:
                json_pattern = r'```json\s*([\s\S]*?)\s*```|```\s*([\s\S]*?)\s*```|(\{[\s\S]*\})'
                match = re.search(json_pattern, content)
                if match:
                    json_str = match.group(1) or match.group(2) or match.group(3)
                    data = json.loads(json_str)
                    if isinstance(data, dict):
                        for key, value in data.items():
                            if key.startswith("image_"):
                                if isinstance(value, str):
                                    result[key] = {"description": value, "size": "1024x1024"}
                                elif isinstance(value, dict):
                                    desc = value.get("description", "")
                                    size = value.get("size", "1024x1024")
                                    result[key] = {"description": desc, "size": size}
        except Exception:
            continue

    return result


def _get_user_dir(user_id: str) -> Path:
    """
    获取用户目录

    Args:
        user_id: 用户ID

    Returns:
        用户目录路径
    """
    script_dir = Path(r"D:\study_notebook\DeepStudy_jupyter\SGG\OmniWriter")
    data_dir = script_dir / "data"
    return data_dir / user_id


async def _generate_single_image(image_key: str, image_info: Dict[str, str], user_dir: Path, client: httpx.AsyncClient) -> str:
    """
    生成单张图片

    Args:
        image_key: 图片键（如 image_1）
        image_info: 图片信息字典，包含 description 和 size
        user_dir: 用户目录
        client: httpx 异步客户端

    Returns:
        生成结果消息
    """
    try:
        description = image_info.get("description", "")
        size = image_info.get("size", "1024x1024")
        
        images_dir = user_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        image_path = images_dir / f"{image_key}.png"

        await generate_modelscope_image(description, str(image_path), size, client)

        result = f"{image_key}已生成完成，并保存到了{user_dir}\\images\\{image_key}.png中"
        logger.info(f"图片生成成功: {image_key}")
        return result

    except Exception as e:
        error_msg = f"{image_key}生成失败"
        logger.error(f"图片 {image_key} 生成失败: {str(e)}", exc_info=True)
        return error_msg


async def draw_images_node(state: DrawImagesState) -> DrawImagesState:
    """
    图片生成节点

    Args:
        state: 状态

    Returns:
        更新后的状态
    """
    writer = get_stream_writer()
    writer({"message": "开始生成图片"})
    messages = state.get("messages", [])
    user_id = state.get("user_id")

    # 从 messages 中解析图片字典
    image_dict = _parse_image_dict(messages)

    if not image_dict:
        result = {
            "status": "no_images",
            "message": "未找到需要生成的图片"
        }
        state["messages"] = messages + [HumanMessage(content=json.dumps(result, ensure_ascii=False))]
        return state

    # 如果状态中没有 user_id，尝试从 messages 中解析
    if not user_id:
        for msg in reversed(messages):
            try:
                content = msg.content if hasattr(msg, 'content') else str(msg)
                data = json.loads(content)
                if isinstance(data, dict) and "user_id" in data:
                    user_id = data["user_id"]
                    break
            except Exception:
                continue

    if not user_id:
        error_result = {
            "status": "error",
            "message": "未找到 user_id"
        }
        state["messages"] = messages + [HumanMessage(content=json.dumps(error_result, ensure_ascii=False))]
        return state

    # 获取用户目录
    user_dir = _get_user_dir(user_id)

    # 按 key 排序，确保顺序一致
    sorted_items = sorted(image_dict.items(), key=lambda x: x[0])
    
    # 创建异步 HTTP 客户端，复用 TCP 连接提高效率
    async with httpx.AsyncClient(timeout=60.0) as client:
        tasks = []
        for image_key, image_info in sorted_items:
            task = _generate_single_image(image_key, image_info, user_dir, client)
            tasks.append(task)

        # 等待所有图片生成完成
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # 构建结果字典 - 严格按照排序后的顺序
    output_dict = {}
    success_count = 0
    fail_count = 0

    for i, (image_key, image_info) in enumerate(sorted_items):
        result_content = results[i]
        if isinstance(result_content, Exception):
            output_dict[image_key] = f"{image_key}生成失败"
            fail_count += 1
        else:
            output_dict[image_key] = result_content
            if "已生成完成" in result_content:
                success_count += 1
            else:
                fail_count += 1

    # 构建最终输出
    final_output = {
        "status": "success" if fail_count == 0 else "partial_success",
        "total": len(image_dict),
        "success": success_count,
        "failed": fail_count,
        "results": output_dict
    }

    state["messages"] = messages + [HumanMessage(content=json.dumps(final_output, ensure_ascii=False))]

    return state


def create_draw_images_graph():
    """
    创建图片生成工作流图

    Returns:
        编译后的工作流图
    """
    logger.info("注册图片生成智能体")

    workflow = StateGraph(DrawImagesState)

    workflow.add_node("draw_images", draw_images_node)

    workflow.set_entry_point("draw_images")

    workflow.add_edge("draw_images", END)

    return workflow.compile()


async def run_draw_images(messages: List, user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    运行图片生成智能体

    Args:
        messages: 消息列表
        user_id: 用户ID（可选）

    Returns:
        生成结果
    """
    state = {
        "messages": messages,
        "user_id": user_id
    }

    graph = create_draw_images_graph()
    result = await graph.ainvoke(state)

    final_messages = result.get("messages", [])
    if final_messages:
        last_message = final_messages[-1]
        content = last_message.content if hasattr(last_message, 'content') else str(last_message)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"raw_output": content}

    return {"status": "failed", "error": "未获取到有效输出"}


if __name__ == "__main__":
    import asyncio

    async def test():
        """测试图片生成智能体"""
        test_messages = [
            HumanMessage(content=json.dumps({
                "user_id": "thread_id_123",
                "image_1":{
                    "description":"水彩画风格的江南水乡",
                    "size":"1024x512"},
                "image_2":{
                    "description":"赛博朋克风格的东京街头",
                    "size":"920x1024"},
                "image_3":{
                    "description":"史诗感十足的冰雪城堡",
                    "size":"1920x768"}  
            }, ensure_ascii=False))
        ]

        result = await run_draw_images(test_messages)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    asyncio.run(test())
