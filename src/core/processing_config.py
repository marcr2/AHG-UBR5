"""
Configuration file for optimizing embedding processing performance.
Adjust these settings based on your system capabilities and API limits.
"""

# --- PARALLEL PROCESSING CONFIGURATION ---
# Optimized parallel processing configuration
MAX_WORKERS = 8  # Increased from 6 to 8 for faster processing
BATCH_SIZE = 500  # Increased from 200 to 500 for fewer citation processing cycles
RATE_LIMIT_DELAY = 0.04  # 25 req/s to stay within Google's 1500 req/min limit
REQUEST_TIMEOUT = 30  # Reduced from 60 to 30 for faster failure detection
MIN_CHUNK_LENGTH = 50  # Minimum chunk length for text splitting
MAX_CHUNK_LENGTH = 8000  # Maximum chunk length for text splitting
SAVE_INTERVAL = 1000  # Save progress every N papers

# --- CITATION PROCESSING OPTIMIZATION ---
CITATION_MAX_WORKERS = 1  # Single worker for immediate processing (no parallel needed)
CITATION_TIMEOUT = 5  # Timeout per citation request (seconds) - balanced for reliability
CITATION_BATCH_SIZE = 1  # Process citations immediately (no batching needed)
CITATION_RATE_LIMIT = 0.5  # 500ms between citation requests to avoid overwhelming APIs

# --- VECTOR DATABASE CONFIG ---
DB_BATCH_SIZE = 5000  # Number of embeddings to add to ChromaDB in one operation (max allowed by ChromaDB is 5461)
# Larger values = faster loading but more memory usage
# Recommended: 5000 (ChromaDB limit is 5461)

# --- TEXT PROCESSING CONFIG ---
MIN_CHUNK_LENGTH = 50  # Minimum character length for a chunk
MAX_CHUNK_LENGTH = 8000  # Maximum character length for a chunk
# Google's text-embedding-004 has a limit of ~8000 tokens

# --- RATE LIMITING ---
# Different limits for different API endpoints
EMBEDDING_MAX_REQUESTS_PER_MINUTE = 1500  # Embedding API: 1500 requests per minute (25 req/s)
GEMINI_MAX_REQUESTS_PER_MINUTE = 1000     # Gemini API: 1000 requests per minute
MAX_BATCH_ENQUEUED_TOKENS = 3_000_000     # Max batch enqueued tokens
RATE_LIMIT_WINDOW = 60  # Rate limiting window in seconds

# Use embedding limit for processing (since that's what we use most)
MAX_REQUESTS_PER_MINUTE = EMBEDDING_MAX_REQUESTS_PER_MINUTE

# IMPORTANT: RATE_LIMIT_DELAY must be >= 0.04s to stay within Google's 25 req/s limit
# Google allows 1500 requests/minute = 25 requests/second
# Therefore: 1 second ÷ 25 requests = 0.04 seconds between requests

# --- MEMORY OPTIMIZATION ---
SAVE_INTERVAL = 1000  # Save metadata every N papers
# More frequent saves = better recovery from crashes
# Less frequent saves = better performance

# --- SOURCE CONFIG ---
DUMPS = ["biorxiv", "medrxiv"]  # Only process these sources

# --- PERFORMANCE PROFILES ---
PERFORMANCE_PROFILES = {
    "parallel_fixed": {
        "max_workers": 1,
        "request_timeout": 15,  # Reduced timeout for faster failure detection
        "rate_limit_delay": 0.04,  # 25 requests/sec (within Google's 1500 req/min limit)
        "batch_size": 100,
        "db_batch_size": 5000
    }
}

def get_config(profile="parallel_fixed"):
    """Get configuration for the fixed parallel processing profile."""
    return PERFORMANCE_PROFILES["parallel_fixed"]

def print_config_info():
    """Print current configuration information."""
    print("🔧 Current Processing Configuration:")
    print(f"   Parallel workers: {MAX_WORKERS}")
    print(f"   Request timeout: {REQUEST_TIMEOUT}s")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Rate limit delay: {RATE_LIMIT_DELAY}s")
    print(f"   Sources: {', '.join(DUMPS)}")
    print()
    print("📊 Citation Processing:")
    print(f"   Citation workers: {CITATION_MAX_WORKERS} (immediate processing)")
    print(f"   Citation timeout: {CITATION_TIMEOUT}s")
    print(f"   Citation rate limit: {CITATION_RATE_LIMIT}s between requests")
    print()
    print("💡 Performance Tips:")
    print("   - Citations are processed immediately for each paper")
    print("   - Use 'rate_limit_safe' if hitting 429 errors")
    print("   - Increase MAX_WORKERS for faster processing (if API allows)")
    print("   - Decrease REQUEST_TIMEOUT if getting timeouts")
    print("   - Increase RATE_LIMIT_DELAY if hitting rate limits")
    print("   - Use 'aggressive' profile for fastest processing")
    print("   - Use 'conservative' profile for most reliable processing")

if __name__ == "__main__":
    print_config_info() 