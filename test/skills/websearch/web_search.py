# 导入必需的库：异步执行、命令行参数解析、Tavily搜索
import asyncio
import argparse
from tavily import AsyncTavilyClient

# ===================== 配置区域 =====================
# Tavily API密钥（建议后续替换为环境变量，不要硬编码）
import os
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
# 初始化异步搜索客户端
async_search_client = AsyncTavilyClient(api_key=TAVILY_API_KEY)
# ====================================================

async def web_search(keywords: str, num_results: int = 1):
    """
    异步网页搜索核心函数
    :param keywords: 搜索关键词
    :param num_results: 返回结果数量
    :return: 搜索结果字典 / 出错返回None
    """
    # 执行Tavily高级搜索
    result = await async_search_client.search(
        query=keywords,
        max_results=num_results,
        search_depth="advanced"
    )
    return result


async def main():
    """主函数：解析命令行参数 + 执行搜索 + 输出结果"""
    # 1. 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Tavily 异步网页搜索工具")
    # 必传参数：搜索关键词
    parser.add_argument(
        "--keywords",
        type=str,
        required=True,
        help="搜索关键词（包含空格请用双引号包裹）"
    )
    # 可选参数：返回结果数（默认1）
    parser.add_argument(
        "--num_results",
        type=int,
        default=1,
        help="搜索结果数量（默认值：1）"
    )

    # 2. 解析命令行传入的参数
    args = parser.parse_args()

    # 3. 执行搜索
    print(f"🔍 正在搜索：{args.keywords}，获取结果数：{args.num_results}")
    search_data = await web_search(args.keywords, args.num_results)
    return search_data
# 程序入口：运行异步主函数
if __name__ == "__main__":
    asyncio.run(main())