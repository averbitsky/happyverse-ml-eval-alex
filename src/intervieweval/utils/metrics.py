"""
Prometheus metrics for monitoring the evaluation system
Location: src/intervieweval/utils/metrics.py
"""

from prometheus_client import Counter, Histogram, Gauge, Summary

# Evaluation metrics
evaluation_counter = Counter(
    'evaluations_total',
    'Total number of evaluations performed'
)

evaluation_duration = Histogram(
    'evaluation_duration_seconds',
    'Time taken to complete an evaluation',
    buckets=(1, 5, 10, 30, 60, 120, 300, 600)
)

# Cache metrics
cache_hits = Counter(
    'cache_hits_total',
    'Total cache hits',
    ['cache_type']  # search, prompt, llm
)

cache_misses = Counter(
    'cache_misses_total',
    'Total cache misses',
    ['cache_type']
)

cache_size = Gauge(
    'cache_size_bytes',
    'Current cache size in bytes',
    ['cache_namespace']
)

# LLM metrics
llm_calls = Counter(
    'llm_calls_total',
    'Total LLM API calls',
    ['chain_type']  # PLAUSIBILITY, TECHNICAL, COMMUNICATION, SYNTHESIS
)

llm_tokens = Counter(
    'llm_tokens_total',
    'Total tokens used',
    ['type']  # input, output
)

llm_latency = Histogram(
    'llm_latency_seconds',
    'LLM API call latency',
    ['chain_type'],
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30)
)

# Rate limiting metrics
rate_limit_errors = Counter(
    'rate_limit_errors_total',
    'Total rate limit errors encountered'
)

rate_limit_retries = Counter(
    'rate_limit_retries_total',
    'Total rate limit retries',
    ['chain_type']
)

# Parallel processing metrics
parallel_tasks = Gauge(
    'parallel_tasks_running',
    'Number of tasks currently running in parallel'
)

batch_size = Histogram(
    'batch_size',
    'Size of evaluation batches',
    buckets=(1, 2, 5, 10, 20, 50)
)

# Search metrics
web_searches = Counter(
    'web_searches_total',
    'Total web searches performed'
)

web_search_latency = Histogram(
    'web_search_latency_seconds',
    'Web search latency',
    buckets=(0.1, 0.5, 1, 2, 5, 10)
)

# Verification metrics
entities_extracted = Counter(
    'entities_extracted_total',
    'Total entities extracted for verification',
    ['entity_type']  # companies, technologies, claims
)

verifications_performed = Counter(
    'verifications_performed_total',
    'Total verifications performed',
    ['verification_type']
)

# Error metrics
evaluation_errors = Counter(
    'evaluation_errors_total',
    'Total evaluation errors',
    ['error_type']
)

validation_failures = Counter(
    'validation_failures_total',
    'Total validation failures',
    ['evaluator_type']
)

# Performance metrics
memory_usage = Gauge(
    'memory_usage_bytes',
    'Current memory usage in bytes'
)

cpu_usage = Gauge(
    'cpu_usage_percent',
    'Current CPU usage percentage'
)


def setup_metrics(port: int = None):
    """
    Setup Prometheus metrics server.

    Args:
        port: Port to expose metrics on (if provided, starts server)
    """
    import logging

    logger = logging.getLogger(__name__)

    if port:
        try:
            from prometheus_client import start_http_server
            start_http_server(port)
            logger.info(f"Metrics server started on port {port}")
            logger.info(f"Metrics available at http://localhost:{port}/metrics")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")