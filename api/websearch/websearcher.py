import os
import logging
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from configs.settings import *

logger = logging.getLogger("WebSearcher")
logger.setLevel(logging.INFO)


# ---------- 抽象基类 ----------
class BaseWebSearcher(ABC):
    """所有搜索器统一接口：同步 search(query) -> List[dict]"""

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """执行网络搜索，返回结构化结果列表"""
        raise NotImplementedError


# ---------- Tavily ----------
class TavilyBasicSearcher(BaseWebSearcher):
    """使用 Tavily API 进行搜索（同步实现）"""

    def __init__(self, api_key: Optional[str] = None):
        from tavily import TavilyClient
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("Tavily API Key 未提供!!!")
        self.client = TavilyClient(self.api_key)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        logger.info(f"[TavilyBasicSearcher] Searching for: {query} (top_k={top_k})")
        if not query.strip():
            return []

        try:
            # search_depth: "basic" / "deep"
            raw = self.client.search(query=query,
                                     max_results=top_k,
                                     search_depth="basic")
        except Exception as e:
            logger.error(f"Tavily 搜索异常: {e}")
            return []

        if "results" not in raw:
            logger.warning("Tavily 响应中未找到 results 字段")
            return []

        return [
            {
                "title":   item.get("title", ""),
                "content": item.get("content", ""),
                "url":     item.get("url", ""),
                "score":   item.get("score", 0),
            }
            for item in raw["results"][:top_k]
        ]


# ---------- Lite 基础搜索 ----------
class LiteBaseSearcher(BaseWebSearcher):
    """
    使用你已有的 async search() 工具，但对外暴露同步接口
    """

    def __init__(self, some_config: Optional[Any] = None):
        pass

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        logger.info(f"[LiteWebSearcher] Searching for: {query} (top_k={top_k})")
        if not query.strip():
            return []

        # 引入异步工具函数
        from api.websearch.utils import search  # <-- async def search(q, k) -> list

        try:
            # 直接在当前线程里跑完协程，拿到结果
            raw_results = asyncio.run(search(query, top_k))
        except Exception as e:
            logger.error(f"LiteWebSearcher 搜索异常: {e}")
            return []

        return [
            {
                "title":   doc.get("title", ""),
                "content": doc.get("snippet", ""),
                "url":     doc.get("link", ""),
            }
            for doc in raw_results[:top_k]
        ]


# ---------- 简单自测 ----------
if __name__ == "__main__":
    query_text = "皮卡丘进化是什么？"

    # Tavily
    tavily_searcher = TavilyBasicSearcher(api_key=TAVILY_API_KEY)
    tavily_results = tavily_searcher.search(query_text, top_k=3)
    print("=== Tavily 搜索结果 ===")
    for i, item in enumerate(tavily_results, 1):
        print(f"{i}. {item['title']}  |  {item['url']}")

    # Lite
    lite_searcher = LiteBaseSearcher()
    lite_results = lite_searcher.search(query_text, top_k=3)
    print("\n=== LiteWebSearcher 搜索结果 ===")
    for i, item in enumerate(lite_results, 1):
        print(f"{i}. {item['title']}  |  {item['url']}")
