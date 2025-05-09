import os
import logging
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

from configs.settings import *

logger = logging.getLogger("WebSearcher")
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------

class BaseWebSearcher(ABC):
    """所有搜索器统一接口：同步 search(query) -> List[dict]"""
    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Tavily
# ---------------------------------------------------------------------------

class TavilyBasicSearcher(BaseWebSearcher):
    """使用 Tavily API 进行搜索（完全同步实现）"""

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
            raw = self.client.search(
                query=query,
                max_results=top_k,
                search_depth="basic"
            )
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


# ---------------------------------------------------------------------------
# Lite 基础搜索  ——  用 async 搜索工具，但对外提供同步接口
# ---------------------------------------------------------------------------

class LiteBaseSearcher(BaseWebSearcher):
    def __init__(self, some_config: Optional[Any] = None):
        pass

    def _run_sync(self, coro):
        """在任意环境下安全执行协程，返回结果"""
        try:
            loop = asyncio.get_running_loop()
            # FastAPI / Uvicorn 等环境已经在事件循环中
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            return future.result()
        except RuntimeError:
            # 当前线程没有事件循环（脚本、子线程等）
            return asyncio.run(coro)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        logger.info(f"[LiteWebSearcher] Searching for: {query} (top_k={top_k})")
        if not query.strip():
            return []

        # 异步搜索工具
        from api.websearch.utils import search      # async def search(q, k) -> list

        try:
            raw_results = self._run_sync(search(query, top_k))
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


# ---------------------------------------------------------------------------
# 简单自测
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    query_text = "皮卡丘进化是什么？"

    tavily_searcher = TavilyBasicSearcher(api_key=TAVILY_API_KEY)
    print("[Tavily] ->", tavily_searcher.search(query_text, top_k=3))

    lite_searcher = LiteBaseSearcher()
    print("[Lite]   ->", lite_searcher.search(query_text, top_k=3))
