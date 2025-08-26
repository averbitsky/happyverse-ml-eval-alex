"""
Web search tool with caching support
Location: src/intervieweval/tools/search.py
"""

import logging
import time
from typing import Optional
from ddgs import DDGS

from intervieweval.cache.manager import PersistentCache
from intervieweval.utils.logging import ColoredLogger
from intervieweval.utils.metrics import web_searches, web_search_latency, cache_hits, cache_misses

logger = logging.getLogger(__name__)


class WebSearchTool:
    """
    Web search tool with persistent caching and metrics.
    """

    def __init__(
        self,
        cache: Optional[PersistentCache] = None,
        max_results: int = 5,
        cache_ttl: int = 3600,
        cache_namespace_suffix: str = "",
    ):
        """
        Initialize web search tool.

        Args:
            cache: Optional persistent cache
            max_results: Maximum number of search results
            cache_ttl: Cache TTL in seconds
            cache_namespace_suffix: Suffix for cache namespace to prevent cross-contamination
        """
        self.cache = cache
        self.max_results = max_results
        self.cache_ttl = cache_ttl
        self.cache_namespace_suffix = cache_namespace_suffix
        self.ddgs = DDGS()

        # Build namespace with suffix if provided
        base_namespace = "search"
        if cache_namespace_suffix:
            self.namespace = f"{base_namespace}_{cache_namespace_suffix}"
        else:
            self.namespace = base_namespace

        logger.info(f"Initialized web search tool (max_results={max_results}, namespace={self.namespace})")

    def search(self, query: str) -> str:
        """
        Perform a web search with caching.

        Args:
            query: Search query

        Returns:
            Search results as formatted string
        """
        # Check cache first
        if self.cache:
            cached_result = self.cache.get(self.namespace, query)
            if cached_result is not None:
                ColoredLogger.log_cache_hit("search", query)
                cache_hits.labels(cache_type="search").inc()
                return cached_result
            cache_misses.labels(cache_type="search").inc()

        # Perform search
        ColoredLogger.log_agent_action("Web Search", query)
        web_searches.inc()

        start_time = time.time()

        try:
            # Use DuckDuckGo search
            results = list(self.ddgs.text(query, max_results=self.max_results))

            if results:
                # Format results
                formatted = []
                for r in results:
                    title = r.get("title", "")
                    body = r.get("body", "")
                    formatted.append(f"{title}: {body}")
                result = " | ".join(formatted)
            else:
                result = "No results found"

            # Cache the result
            if self.cache:
                self.cache.set(namespace=self.namespace, key=query, value=result, ttl=self.cache_ttl)

            ColoredLogger.log_agent_observation(result)

        except Exception as e:
            logger.error(f"Search failed for query '{query}': {str(e)}")
            result = f"Search failed: {str(e)}"

            # Cache error for shorter time
            if self.cache:
                self.cache.set(namespace=self.namespace, key=query, value=result, ttl=300)  # 5 minutes

        finally:
            # Record latency
            latency = time.time() - start_time
            web_search_latency.observe(latency)

        return result

    async def search_async(self, query: str) -> str:
        """
        Async version of search (runs in executor).

        Args:
            query: Search query

        Returns:
            Search results as formatted string
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.search, query)

    def clear_cache(self) -> int:
        """
        Clear all cached search results.

        Returns:
            Number of items cleared
        """
        if not self.cache:
            return 0

        cleared = self.cache.clear_namespace(self.namespace)
        logger.info(f"Cleared {cleared} cached search results")
        return cleared

    def get_cache_stats(self) -> dict:
        """
        Get cache statistics for search namespace.

        Returns:
            Dictionary of cache stats
        """
        if not self.cache:
            return {"enabled": False}

        stats = self.cache.get_stats()
        namespace_stats = stats.get("namespaces", {}).get(self.namespace, 0)

        return {
            "enabled": True,
            "cached_searches": namespace_stats,
            "hit_rate": stats.get("hit_rate", 0),
            "total_hits": stats.get("total_hits", 0),
            "total_misses": stats.get("total_misses", 0),
        }
