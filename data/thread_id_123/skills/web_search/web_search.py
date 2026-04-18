from tavily import AsyncTavilyClient

async_search_client = AsyncTavilyClient(api_key="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")

async def web_search(keywords: str, num_results: int = 5):
    result = await async_search_client.search(
        query=keywords,
        max_results=num_results,
        search_depth="advanced"
    )
    return result