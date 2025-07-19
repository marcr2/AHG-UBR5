"""
Configuration file for optimizing embedding processing performance.
Adjust these settings based on your system capabilities and API limits.
"""

# --- PARALLEL PROCESSING CONFIG ---
MAX_WORKERS = 10  # Increased workers to utilize 1500 req/min limit
# With 1500 requests per minute limit, parallel processing is now much more feasible
# Adjust MAX_WORKERS based on your system and API stability

# --- API CONFIG ---
REQUEST_TIMEOUT = 60  # Reduced timeout for higher throughput
# Increase if you're getting timeout errors
# Decrease if you want to fail fast on slow requests

# --- BATCHING CONFIG ---
BATCH_SIZE = 200  # Number of embeddings per file
# Larger batches = fewer files but more memory usage
# Smaller batches = more files but less memory usage

# --- VECTOR DATABASE CONFIG ---
DB_BATCH_SIZE = 5000  # Number of embeddings to add to ChromaDB in one operation (max allowed by ChromaDB is 5461)
# Larger values = faster loading but more memory usage
# Recommended: 5000 (ChromaDB limit is 5461)

# --- TEXT PROCESSING CONFIG ---
MIN_CHUNK_LENGTH = 50  # Minimum character length for a chunk
MAX_CHUNK_LENGTH = 8000  # Maximum character length for a chunk
# Google's text-embedding-004 has a limit of ~8000 tokens

# --- RATE LIMITING ---
RATE_LIMIT_DELAY = 0.04  # 0.04 second delay between requests (1500 req/min = 1 req/0.04s)
# With 1500 requests per minute limit, we need at least 0.04 seconds between requests
# This ensures we stay well under the limit

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