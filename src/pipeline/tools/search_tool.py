import os
from tavily import AsyncTavilyClient
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("TAVILY_API_KEY")
if not api_key:
    raise ValueError("TAVILY_API_KEY 环境变量未设置")

async_search_client = AsyncTavilyClient(api_key=api_key)
