"""
Persistent cache manager using SQLite for persistent storage with TTL (Time-to-Live) and LRU (Least Recently Used)
eviction. Includes a web search tool that uses this cache to store and retrieve search results.
"""

import sqlite3
import json
import time
import hashlib
from typing import Any, Optional, Dict
from pathlib import Path
import logging
from threading import Lock
from collections import OrderedDict
import pickle

logger = logging.getLogger(__name__)


class PersistentCache:
    """
    Persistent cache implementation using SQLite with TTL and LRU eviction.
    Thread-safe and supports different cache namespaces.
    """

    def __init__(
        self,
        cache_dir: str = "cache",
        cache_name: str = "evaluator_cache.db",
        max_memory_items: int = 1000,
        default_ttl: int = 3600,
        max_db_size_mb: int = 100
    ):
        """
        Initialize persistent cache.

        Args:
            cache_dir: Directory to store cache database
            cache_name: Name of the cache database file
            max_memory_items: Maximum items to keep in memory cache
            default_ttl: Default TTL in seconds
            max_db_size_mb: Maximum database size in MB
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.db_path = self.cache_dir / cache_name
        self.default_ttl = default_ttl
        self.max_db_size_mb = max_db_size_mb
        self.max_memory_items = max_memory_items

        # Thread lock for database operations
        self.lock = Lock()

        # In-memory LRU cache for faster access
        self.memory_cache = OrderedDict()

        # Initialize database
        self._init_db()

        # Clean expired entries on startup
        self.clean_expired()

        logger.info(f"Initialized persistent cache at {self.db_path}")

    def _init_db(self):
        """Initialize SQLite database with cache tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value BLOB NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    last_accessed REAL NOT NULL,
                    PRIMARY KEY (namespace, key)
                )
            """)

            # Create indices for better performance
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires_at 
                ON cache(expires_at)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_last_accessed 
                ON cache(last_accessed)
            """)

            # Metadata table for statistics
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_stats (
                    stat_name TEXT PRIMARY KEY,
                    stat_value INTEGER
                )
            """)

            # Initialize stats
            conn.execute("""
                INSERT OR IGNORE INTO cache_stats (stat_name, stat_value)
                VALUES 
                    ('total_hits', 0),
                    ('total_misses', 0),
                    ('total_evictions', 0)
            """)

            conn.commit()

    def _make_key(self, key: str) -> str:
        """Create a hashed key for storage"""
        return hashlib.sha256(key.encode()).hexdigest()

    def get(
        self,
        namespace: str,
        key: str,
        default: Any = None
    ) -> Any:
        """
        Get value from cache.

        Args:
            namespace: Cache namespace (e.g., 'search', 'prompt', 'llm')
            key: Cache key
            default: Default value if not found

        Returns:
            Cached value or default
        """
        with self.lock:
            # Check memory cache first
            memory_key = f"{namespace}:{key}"
            if memory_key in self.memory_cache:
                # Move to end (LRU)
                self.memory_cache.move_to_end(memory_key)
                self._update_stats('hits')
                return self.memory_cache[memory_key]

            # Check persistent cache
            hashed_key = self._make_key(key)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT value, expires_at FROM cache
                    WHERE namespace = ? AND key = ?
                """, (namespace, hashed_key))

                row = cursor.fetchone()

                if row:
                    value_blob, expires_at = row

                    # Check if expired
                    if time.time() > expires_at:
                        # Delete expired entry
                        conn.execute("""
                            DELETE FROM cache
                            WHERE namespace = ? AND key = ?
                        """, (namespace, hashed_key))
                        conn.commit()
                        self._update_stats('misses')
                        return default

                    # Update access stats
                    conn.execute("""
                        UPDATE cache
                        SET access_count = access_count + 1,
                            last_accessed = ?
                        WHERE namespace = ? AND key = ?
                    """, (time.time(), namespace, hashed_key))
                    conn.commit()

                    # Deserialize value
                    try:
                        value = pickle.loads(value_blob)

                        # Add to memory cache
                        self._add_to_memory_cache(memory_key, value)

                        self._update_stats('hits')
                        return value
                    except Exception as e:
                        logger.warning(f"Failed to deserialize cache value: {e}")
                        self._update_stats('misses')
                        return default

                self._update_stats('misses')
                return default

    def set(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in cache.

        Args:
            namespace: Cache namespace
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)

        Returns:
            True if successful
        """
        with self.lock:
            try:
                hashed_key = self._make_key(key)
                ttl = ttl or self.default_ttl
                current_time = time.time()
                expires_at = current_time + ttl

                # Serialize value
                value_blob = pickle.dumps(value)

                # Check database size
                if self._get_db_size_mb() > self.max_db_size_mb:
                    self._evict_lru(10)  # Evict 10 least recently used items

                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO cache 
                        (namespace, key, value, created_at, expires_at, last_accessed)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (namespace, hashed_key, value_blob, current_time, expires_at, current_time))
                    conn.commit()

                # Add to memory cache
                memory_key = f"{namespace}:{key}"
                self._add_to_memory_cache(memory_key, value)

                return True

            except Exception as e:
                logger.error(f"Failed to set cache value: {e}")
                return False

    def _add_to_memory_cache(self, key: str, value: Any):
        """Add item to memory cache with LRU eviction"""
        # Remove oldest if at capacity
        if len(self.memory_cache) >= self.max_memory_items:
            self.memory_cache.popitem(last=False)

        self.memory_cache[key] = value

    def delete(self, namespace: str, key: str) -> bool:
        """Delete item from cache"""
        with self.lock:
            hashed_key = self._make_key(key)
            memory_key = f"{namespace}:{key}"

            # Remove from memory cache
            if memory_key in self.memory_cache:
                del self.memory_cache[memory_key]

            # Remove from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    DELETE FROM cache
                    WHERE namespace = ? AND key = ?
                """, (namespace, hashed_key))
                conn.commit()

                return cursor.rowcount > 0

    def clear_namespace(self, namespace: str) -> int:
        """Clear all items in a namespace"""
        with self.lock:
            # Clear from memory cache
            keys_to_remove = [k for k in self.memory_cache if k.startswith(f"{namespace}:")]
            for key in keys_to_remove:
                del self.memory_cache[key]

            # Clear from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    DELETE FROM cache WHERE namespace = ?
                """, (namespace,))
                conn.commit()

                return cursor.rowcount

    def clean_expired(self) -> int:
        """Clean all expired entries from cache"""
        with self.lock:
            current_time = time.time()

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    DELETE FROM cache WHERE expires_at < ?
                """, (current_time,))
                conn.commit()

                deleted = cursor.rowcount
                if deleted > 0:
                    logger.info(f"Cleaned {deleted} expired cache entries")

                return deleted

    def _evict_lru(self, count: int) -> int:
        """Evict least recently used items"""
        with sqlite3.connect(self.db_path) as conn:
            # Find LRU items
            cursor = conn.execute("""
                SELECT namespace, key FROM cache
                ORDER BY last_accessed ASC
                LIMIT ?
            """, (count,))

            items = cursor.fetchall()

            # Delete them
            for namespace, key in items:
                conn.execute("""
                    DELETE FROM cache WHERE namespace = ? AND key = ?
                """, (namespace, key))

            conn.commit()

            self._update_stats('evictions', len(items))

            if items:
                logger.info(f"Evicted {len(items)} LRU cache entries")

            return len(items)

    def _get_db_size_mb(self) -> float:
        """Get database size in MB"""
        return self.db_path.stat().st_size / (1024 * 1024)

    def _update_stats(self, stat_type: str, count: int = 1):
        """Update cache statistics"""
        stat_name = f"total_{stat_type}"

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE cache_stats
                SET stat_value = stat_value + ?
                WHERE stat_name = ?
            """, (count, stat_name))
            conn.commit()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                # Get basic stats
                cursor = conn.execute("""
                    SELECT stat_name, stat_value FROM cache_stats
                """)
                stats = dict(cursor.fetchall())

                # Get cache size
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_items,
                        SUM(LENGTH(value)) as total_bytes,
                        AVG(access_count) as avg_access_count
                    FROM cache
                """)

                row = cursor.fetchone()
                stats.update({
                    'total_items': row[0] or 0,
                    'total_bytes': row[1] or 0,
                    'avg_access_count': row[2] or 0,
                    'memory_cache_items': len(self.memory_cache),
                    'db_size_mb': self._get_db_size_mb()
                })

                # Get namespace breakdown
                cursor = conn.execute("""
                    SELECT namespace, COUNT(*) as count
                    FROM cache
                    GROUP BY namespace
                """)

                stats['namespaces'] = dict(cursor.fetchall())

                # Calculate hit rate
                total_requests = stats.get('total_hits', 0) + stats.get('total_misses', 0)
                stats['hit_rate'] = (
                    stats.get('total_hits', 0) / total_requests
                    if total_requests > 0 else 0
                )

                return stats

    def export_cache(self, export_path: str) -> bool:
        """Export cache to a file for backup"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as source:
                    with sqlite3.connect(export_path) as dest:
                        source.backup(dest)
                logger.info(f"Cache exported to {export_path}")
                return True
        except Exception as e:
            logger.error(f"Failed to export cache: {e}")
            return False

    def import_cache(self, import_path: str) -> bool:
        """Import cache from a backup file"""
        try:
            with self.lock:
                # Clear memory cache
                self.memory_cache.clear()

                # Import database
                with sqlite3.connect(import_path) as source:
                    with sqlite3.connect(self.db_path) as dest:
                        source.backup(dest)

                logger.info(f"Cache imported from {import_path}")
                return True
        except Exception as e:
            logger.error(f"Failed to import cache: {e}")
            return False


class CachedWebSearchTool:
    """Web search tool with persistent caching"""

    def __init__(self, cache: PersistentCache, max_results: int = 5):
        self.cache = cache
        self.max_results = max_results
        self.ddgs = DDGS()
        self.namespace = "search"
        logger.info("Initialized cached web search tool")

    def run(self, query: str) -> str:
        """Run a web search query with persistent caching"""
        from colorama import Fore, Style

        # Check cache first
        cached_result = self.cache.get(self.namespace, query)
        if cached_result is not None:
            print(f"{Fore.YELLOW}[AGENT THOUGHT]: Found cached result for: {query}{Style.RESET_ALL}")
            return cached_result

        print(f"{Fore.CYAN}[AGENT ACTION]: Web Search{Style.RESET_ALL}")
        print(f"   Input: {query}")

        try:
            # Perform search
            results = list(self.ddgs.text(query, max_results=self.max_results))
            if results:
                # Format results
                formatted = []
                for r in results:
                    formatted.append(f"{r.get('title', '')}: {r.get('body', '')}")
                result = " ".join(formatted)
            else:
                result = "No results found"

            # Cache the result with 1 hour TTL
            self.cache.set(self.namespace, query, result, ttl=3600)

            print(f"{Fore.MAGENTA}[OBSERVATION]: {result[:200]}...{Style.RESET_ALL}")

            return result

        except Exception as e:
            error_msg = f"Search failed: {str(e)}"
            print(f"{Fore.MAGENTA}[OBSERVATION]: {error_msg}{Style.RESET_ALL}")
            # Cache error for shorter time (5 minutes)
            self.cache.set(self.namespace, query, error_msg, ttl=300)
            return error_msg