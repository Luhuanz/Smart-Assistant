import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from configs.settings import *

logger = logging.getLogger("WebSearcher")
logger.setLevel(logging.INFO)


class BaseWebSearcher(ABC):
    """
    基类，确保所有子类实现统一的搜索接口。
    """

    @abstractmethod
    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        执行网络搜索，返回结构化的搜索结果列表。
        """
        pass


class TavilyBasicSearcher(BaseWebSearcher):
    """
    使用 Tavily API进行搜索
    """

    def __init__(self, api_key: Optional[str] = None):
        from tavily import TavilyClient
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("Tavily API Key 未提供!!!")
        self.client = TavilyClient(self.api_key)

    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        logger.info(f"[TavilyBasicSearcher] Searching for: {query} with top_k={top_k}")
        if not query.strip():
            logger.warning("查询字符串为空，将返回空结果。")
            return []

        try:
            # search_depth指定为 "basic", "deep" 等
            raw_response = self.client.search(query=query, max_results=top_k, search_depth="basic")
        except Exception as e:
            logger.error(f"Tavily 搜索异常: {e}")
            return []

        if "results" not in raw_response:
            logger.warning("Tavily 响应中未找到 'results' 字段")
            return []

        # 将结果组装为统一结构
        # item 可能包含: title, content, url, score
        search_results = []
        for item in raw_response["results"]:
            result = {
                "title": item.get("title", ""),
                "content": item.get("content", ""),
                "url": item.get("url", ""),
                "score": item.get("score", 0),
            }
            search_results.append(result)

        # 只取 top_k 个
        return search_results[:top_k]


class LiteBaseSearcher(BaseWebSearcher):
    """
    使用已有的 search() 执行最简洁的网络搜索
    """

    def __init__(self, some_config: Optional[Any] = None):
        pass

    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        logger.info(f"[LiteWebSearcher] Searching for: {query} with top_k={top_k}")
        if not query.strip():
            logger.warning("查询字符串为空，将返回空结果。")
            return []

        from api.websearch.utils import search

        try:
            raw_results = await search(query, top_k)
        except Exception as e:
            logger.error(f"LiteWebSearcher 搜索异常: {e}")
            return []

        search_results = []
        for doc in raw_results:
            result = {
                "title": doc.get("title", ""),
                "content": doc.get("snippet", ""),
                "url": doc.get("link", ""),
            }
            search_results.append(result)

        return search_results[:top_k]


if __name__ == "__main__":
    import asyncio


    async def main():
        query_text = "皮卡丘进化是什么？"

        # Tavily搜索
        tavily_searcher = TavilyBasicSearcher(api_key=TAVILY_API_KEY)
        tavily_results = await tavily_searcher.search(query_text, top_k=3)
        print("=== Tavily 搜索结果 ===")
        for i, item in enumerate(tavily_results, 1):
            print(f"{i}. 标题: {item['title']}\n   摘要: {item['content']}\n   链接: {item['url']}\n")
        # LiteBaseSearcher
        lite_searcher = LiteBaseSearcher()
        lite_results = await lite_searcher.search(query_text, top_k=3)
        print("=== LiteWebSearcher 搜索结果 ===")
        for i, item in enumerate(lite_results, 1):
            print(f"{i}. 标题: {item['title']}\n   摘要: {item['content']}\n   链接: {item['url']}\n")


    asyncio.run(main())
