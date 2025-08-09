"""
Configuration file for optimizing embedding processing performance.
Adjust these settings based on your system capabilities and API limits.
"""

# --- PARALLEL PROCESSING CONFIGURATION ---
# Optimized parallel processing configuration
MAX_WORKERS = 6  # Number of parallel workers for processing
BATCH_SIZE = 200  # Number of papers to process in each batch
RATE_LIMIT_DELAY = 0.04  # Delay between requests (seconds) - optimized for 1500 req/min
REQUEST_TIMEOUT = 60  # Timeout for API requests (seconds)
MIN_CHUNK_LENGTH = 50  # Minimum chunk length for text splitting
MAX_CHUNK_LENGTH = 8000  # Maximum chunk length for text splitting
SAVE_INTERVAL = 1000  # Save progress every N papers

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
EMBEDDING_MAX_REQUESTS_PER_MINUTE = 1500  # Embedding API: 1500 requests per minute
GEMINI_MAX_REQUESTS_PER_MINUTE = 1000     # Gemini API: 1000 requests per minute
MAX_BATCH_ENQUEUED_TOKENS = 3_000_000     # Max batch enqueued tokens
RATE_LIMIT_WINDOW = 60  # Rate limiting window in seconds

# Use embedding limit for processing (since that's what we use most)
MAX_REQUESTS_PER_MINUTE = EMBEDDING_MAX_REQUESTS_PER_MINUTE

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
        "request_timeout": 60,
        "rate_limit_delay": 0.04,  # 25 requests/sec
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
    print("💡 Performance Tips:")
    print("   - Use 'rate_limit_safe' if hitting 429 errors")
    print("   - Increase MAX_WORKERS for faster processing (if API allows)")
    print("   - Decrease REQUEST_TIMEOUT if getting timeouts")
    print("   - Increase RATE_LIMIT_DELAY if hitting rate limits")
    print("   - Use 'aggressive' profile for fastest processing")
    print("   - Use 'conservative' profile for most reliable processing")

if __name__ == "__main__":
    print_config_info() 