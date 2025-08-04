import os
import pkg_resources
import pandas as pd
from paperscraper.xrxiv.xrxiv_query import XRXivQuery
import requests
import json
import re
from tqdm.auto import tqdm
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random
from processing_config import get_config, print_config_info

import matplotlib.pyplot as plt

def extract_publication_date(paper_data):
    """Extract publication date from paper data."""
    date_fields = [
        'date', 'publication_date', 'published_date', 'date_published',
        'pub_date', 'date_created', 'created_date', 'submitted_date',
        'date_submitted', 'posted_date', 'date_posted'
    ]
    
    def is_valid_year(year_str):
        """Check if a year string represents a valid scientific publication year."""
        try:
            year = int(year_str)
            # Valid years should be between 1900 and current year + 1
            current_year = datetime.now().year
            return 1900 <= year <= current_year + 1
        except (ValueError, TypeError):
            return False
    
    for field in date_fields:
        if field in paper_data and paper_data[field]:
            date_str = str(paper_data[field])
            try:
                # Try DD-MM-YYYY format first (as specified by user)
                if re.match(r'\d{2}-\d{2}-\d{4}', date_str):
                    year = date_str[6:10]
                    if is_valid_year(year):
                        # Convert DD-MM-YYYY to YYYY-MM-DD
                        return f"{year}-{date_str[3:5]}-{date_str[0:2]}"
                # Try YYYY-MM-DD format
                elif re.match(r'\d{4}-\d{2}-\d{2}', date_str):
                    year = date_str[:4]
                    if is_valid_year(year):
                        return date_str
                # Try YYYY format
                elif re.match(r'\d{4}', date_str):
                    if is_valid_year(date_str):
                        return f"{date_str}-01-01"
                else:
                    # Fallback to finding any 4-digit year
                    year_match = re.search(r'(\d{4})', date_str)
                    if year_match and is_valid_year(year_match.group(1)):
                        return f"{year_match.group(1)}-01-01"
            except:
                continue
    
    doi = paper_data.get('doi', '')
    if doi:
        year_match = re.search(r'(\d{4})', doi)
        if year_match and is_valid_year(year_match.group(1)):
            return f"{year_match.group(1)}-01-01"
    
    return f"{datetime.now().year}-01-01"

def extract_citation_count(paper_data):
    """Extract citation count from paper data."""
    citation_fields = [
        'citations', 'citation_count', 'cited_by_count', 'times_cited',
        'citation_count', 'reference_count', 'cited_count'
    ]
    
    for field in citation_fields:
        if field in paper_data and paper_data[field] is not None:
            try:
                count = int(paper_data[field])
                return str(max(0, count))
            except (ValueError, TypeError):
                continue
    
    return "not found"

def extract_journal_info(paper_data):
    """Extract journal information from paper data."""
    journal_fields = [
        'journal', 'journal_name', 'publication', 'source', 'venue',
        'journal_title', 'publication_venue'
    ]
    
    for field in journal_fields:
        if field in paper_data and paper_data[field]:
            return str(paper_data[field])
    
    source = paper_data.get('source', '')
    if source in ['biorxiv', 'medrxiv']:
        return f"{source.upper()}"
    
    return "Unknown journal"
import matplotlib.dates as mdates
from collections import deque
import threading

# --- ADAPTIVE WORKER MANAGER ---
try:
    from adaptive_worker_manager import get_adaptive_manager, record_request, get_worker_count, get_status, stop_adaptive_manager
    ADAPTIVE_SCALING_ENABLED = True
    print("✅ Adaptive worker scaling enabled")
except ImportError:
    ADAPTIVE_SCALING_ENABLED = False
    print("⚠️  Adaptive worker scaling not available, using static configuration")

# --- REQUEST RATE TRACKING ---
request_count_lock = threading.Lock()
request_count = 0
request_start_time = None

def reset_request_counter():
    global request_count, request_start_time
    with request_count_lock:
        request_count = 0
        request_start_time = time.time()

def increment_request_counter():
    global request_count
    with request_count_lock:
        request_count += 1

def get_requests_per_sec():
    global request_count, request_start_time
    with request_count_lock:
        elapsed = time.time() - request_start_time if request_start_time else 1
        return request_count / elapsed if elapsed > 0 else 0

class LiveRatePlot:
    def __init__(self, window_seconds=300, title="Requests/sec (rolling, last 5 min)"):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 4))
        self.ax.set_title(title)
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Requests/sec")
        self.rate_times = deque()
        self.rate_values = deque()
        self.error_times = deque()
        self.error_values = deque()
        self.lock = threading.Lock()
        self.window_seconds = window_seconds
        self.last_plot_update = 0
    def update(self, rate, error=False):
        now = datetime.now()
        with self.lock:
            self.rate_times.append(now)
            self.rate_values.append(rate)
            if error:
                self.error_times.append(now)
                self.error_values.append(float(rate))
    def refresh(self):
        with self.lock:
            now = datetime.now()
            # Remove old points
            while self.rate_times and (now - self.rate_times[0]).total_seconds() > self.window_seconds:
                self.rate_times.popleft()
                self.rate_values.popleft()
            while self.error_times and (now - self.error_times[0]).total_seconds() > self.window_seconds:
                self.error_times.popleft()
                self.error_values.popleft()
            self.ax.clear()
            self.ax.set_title("Requests/sec (rolling, last 5 min)")
            self.ax.set_xlabel("Time")
            self.ax.set_ylabel("Requests/sec")
            if self.rate_times:
                self.ax.plot(self.rate_times, self.rate_values, label="Requests/sec", color="blue")
            if self.error_times:
                self.ax.scatter(self.error_times, self.error_values, color="red", label="429 error", marker="x", s=60)
            self.ax.legend()
            self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            self.fig.autofmt_xdate()
            plt.tight_layout()
            plt.pause(0.01)
    def close(self):
        plt.ioff()
        plt.show()

# --- CONFIG ---
# Import optimized configuration
try:
    from processing_config import MAX_WORKERS, BATCH_SIZE, RATE_LIMIT_DELAY, REQUEST_TIMEOUT, MIN_CHUNK_LENGTH, MAX_CHUNK_LENGTH, SAVE_INTERVAL, DUMPS
    print("✅ Using optimized configuration from processing_config.py")
except ImportError:
    print("⚠️  processing_config.py not found, using fallback configuration")
    # Fallback configuration
    MAX_WORKERS = 3
    BATCH_SIZE = 200
    RATE_LIMIT_DELAY = 0.02
    REQUEST_TIMEOUT = 60
    MIN_CHUNK_LENGTH = 50
    MAX_CHUNK_LENGTH = 8000
    SAVE_INTERVAL = 1000
    DUMPS = ["biorxiv", "medrxiv"]

DUMP_ROOT = pkg_resources.resource_filename("paperscraper", "server_dumps")
EMBEDDINGS_DIR = "xrvix_embeddings"

# Rate limiting settings
# Different limits for different API endpoints
EMBEDDING_MAX_REQUESTS_PER_MINUTE = 1500  # Embedding API: 1500 requests per minute
GEMINI_MAX_REQUESTS_PER_MINUTE = 1000     # Gemini API: 1000 requests per minute
MAX_REQUESTS_PER_MINUTE = EMBEDDING_MAX_REQUESTS_PER_MINUTE  # Use embedding limit for processing
MAX_BATCH_ENQUEUED_TOKENS = 3_000_000     # Max batch enqueued tokens
RATE_LIMIT_WINDOW = 60  # seconds
MAX_RETRIES = 3
BASE_BACKOFF_DELAY = 1  # seconds

# Global rate limiting
request_times = []

def is_rate_limited():
    """Check if we're currently rate limited"""
    global request_times
    current_time = time.time()
    
    # Remove old requests outside the window
    request_times = [t for t in request_times if current_time - t < RATE_LIMIT_WINDOW]
    
    # Check if we've made too many requests recently
    return len(request_times) >= MAX_REQUESTS_PER_MINUTE

def wait_for_rate_limit():
    """Wait until we're no longer rate limited"""
    from tqdm.auto import tqdm
    while is_rate_limited():
        # With 1500 req/min limit, we need to wait at least 1 minute after hitting the limit
        sleep_time = 60 + random.uniform(0, 10)  # 60-70 seconds to be safe
        # tqdm progress bar for rate limit wait
        with tqdm(total=sleep_time, desc="Rate limited: waiting to resume", bar_format="{l_bar}{bar}| {remaining} left", colour="#FFA500", leave=False) as pbar:
            waited = 0.0
            while waited < sleep_time:
                time.sleep(0.1)
                waited += 0.1
                pbar.update(0.1)
        # Clear old request times after waiting
        global request_times
        current_time = time.time()
        request_times = [t for t in request_times if current_time - t < RATE_LIMIT_WINDOW]

def record_request():
    """Record that we made a request"""
    global request_times
    request_times.append(time.time())

def get_google_embedding(text, api_key, retry_count=0, skip_base_delay=False):
    """Get embedding with global rate limiting and exponential backoff"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {"model": "models/text-embedding-004", "content": {"parts": [{"text": text}]}}
    try:
        # Strict global paper/sec rate limit before every API call
        # wait_for_paper_rate_limit() # Removed global rate limiting
        # Add base rate limiting delay (ensures we stay under 25 req/sec)
        if RATE_LIMIT_DELAY > 0 and not skip_base_delay:
            time.sleep(RATE_LIMIT_DELAY)
        response = requests.post(url, headers=headers, json=data, timeout=REQUEST_TIMEOUT)
        increment_request_counter()
        
        # Record request for adaptive worker manager
        if ADAPTIVE_SCALING_ENABLED:
            record_request()
        
        if response.status_code == 429:
            # Record rate limit error for immediate scale down
            if ADAPTIVE_SCALING_ENABLED:
                from adaptive_worker_manager import get_adaptive_manager
                get_adaptive_manager().record_rate_limit_error()
            
            if retry_count < MAX_RETRIES:
                print(f"⚠️  Rate limited (attempt {retry_count + 1}/{MAX_RETRIES}). Waiting 60s...")
                time.sleep(60)
                return get_google_embedding(text, api_key, retry_count + 1)
            else:
                print(f"❌ Max retries reached for rate limit")
                return None
        response.raise_for_status()
        return response.json()["embedding"]["values"]
    except requests.exceptions.Timeout:
        print(f"⚠️  Timeout for text: {text[:50]}...")
        return None
    except requests.exceptions.RequestException as e:
        if "429" in str(e):
            # Record rate limit error for immediate scale down
            if ADAPTIVE_SCALING_ENABLED:
                from adaptive_worker_manager import get_adaptive_manager
                get_adaptive_manager().record_rate_limit_error()
            
            if retry_count < MAX_RETRIES:
                print(f"⚠️  Rate limited (attempt {retry_count + 1}/{MAX_RETRIES}). Waiting 60s...")
                time.sleep(60)
                return get_google_embedding(text, api_key, retry_count + 1)
            else:
                print(f"❌ Max retries reached for rate limit")
                return None
        else:
            print(f"⚠️  API error: {e}")
            return None
    except Exception as e:
        print(f"⚠️  Unexpected error: {e}")
        return None

def chunk_paragraphs(text):
    if not isinstance(text, str):
        return []
    
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    
    # Filter by length
    filtered_paragraphs = []
    for para in paragraphs:
        if MIN_CHUNK_LENGTH <= len(para) <= MAX_CHUNK_LENGTH:
            filtered_paragraphs.append(para)
    
    return filtered_paragraphs

def process_paper_embeddings(paper_data, api_key):
    """
    Process a single paper and generate embeddings for its paragraphs.
    
    Args:
        paper_data: Tuple of (idx, row, source)
        api_key: Google API key
        
    Returns:
        List of embedding results or None if failed
    """
    idx, row, source = paper_data
    title = row.get("title", "")
    doi = row.get("doi", "")
    abstract = row.get("abstract", "")
    authors = row.get("authors", "")  # Add authors field
    
    # Extract additional metadata fields
    publication_date = extract_publication_date(row)
    citation_count = extract_citation_count(row)
    journal = extract_journal_info(row)
    year = publication_date[:4] if publication_date else str(datetime.now().year)
    
    paragraphs = chunk_paragraphs(abstract)
    
    # Skip papers with no meaningful content
    if not paragraphs:
        return []
    
    results = []
    
    for i, para in enumerate(paragraphs):
        if not para.strip():
            continue
            
        embedding = get_google_embedding(para, api_key)
        if embedding is None:
            continue
            
        results.append({
            "embedding": embedding,
            "chunk": para,
            "metadata": {
                "title": title,
                "doi": doi,
                "authors": authors,  # Add authors to metadata
                "publication_date": publication_date,
                "citation_count": citation_count,
                "journal": journal,
                "source": source,
                "paper_index": idx,
                "para_idx": i,
                "chunk_length": len(para)
            }
        })
    
    return results

def ensure_directory(path):
    """Ensure directory exists"""
    os.makedirs(path, exist_ok=True)

def save_batch(batch_data, source, batch_num, embeddings_dir):
    """Save a batch of embeddings to a file"""
    batch_file = os.path.join(embeddings_dir, source, f"batch_{batch_num:04d}.json")
    
    batch_content = {
        "source": source,
        "batch_num": batch_num,
        "timestamp": datetime.now().isoformat(),
        "embeddings": batch_data["embeddings"],
        "chunks": batch_data["chunks"],
        "metadata": batch_data["metadata"]
    }
    
    with open(batch_file, 'w', encoding='utf-8') as f:
        json.dump(batch_content, f, indent=2, ensure_ascii=False)
    
    return batch_file

def load_metadata(embeddings_dir):
    """Load or create metadata index"""
    metadata_file = os.path.join(embeddings_dir, "metadata.json")
    
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Error loading metadata: {e}")
    
    # Return default structure if no existing file
    return {
        "created": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "total_embeddings": 0,
        "total_chunks": 0,
        "total_papers": 0,
        "sources": {},
        "batches": {},
        "processed_papers": {}  # Track processed papers by source
    }

def save_metadata(metadata, embeddings_dir):
    """Save metadata index"""
    metadata["last_updated"] = datetime.now().isoformat()
    # Convert sets to lists for JSON serialization
    if "processed_papers" in metadata:
        for source, papers in metadata["processed_papers"].items():
            if isinstance(papers, set):
                metadata["processed_papers"][source] = list(papers)
    metadata_file = os.path.join(embeddings_dir, "metadata.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

def get_next_batch_num(source, embeddings_dir):
    """Get the next batch number for a source"""
    source_dir = os.path.join(embeddings_dir, source)
    if not os.path.exists(source_dir):
        return 0
    
    existing_batches = [f for f in os.listdir(source_dir) if f.startswith("batch_") and f.endswith(".json")]
    if not existing_batches:
        return 0
    
    batch_nums = [int(f.split("_")[1].split(".")[0]) for f in existing_batches]
    return max(batch_nums) + 1

def get_processed_papers(metadata, source):
    """Get set of already processed paper indices for a source"""
    processed = metadata.get("processed_papers", {}).get(source, [])
    # Convert to set (JSON saves sets as lists)
    if isinstance(processed, list):
        processed = set(processed)
    elif not isinstance(processed, set):
        processed = set()
    return processed

def mark_paper_processed(metadata, source, paper_index):
    """Mark a paper as processed"""
    if "processed_papers" not in metadata:
        metadata["processed_papers"] = {}
    if source not in metadata["processed_papers"]:
        metadata["processed_papers"][source] = set()
    
    # Ensure it's a set (JSON might have saved it as a list)
    if isinstance(metadata["processed_papers"][source], list):
        metadata["processed_papers"][source] = set(metadata["processed_papers"][source])
    
    metadata["processed_papers"][source].add(paper_index)

# Remove wait_for_paper_rate_limit and all references to paper_rate_lock, paper_timestamps, paper_rate_event, and related logic

def sequential_process_papers(unprocessed_papers, api_key, current_batch, metadata, db, batch_num, embeddings_dir, start_time):
    """Process papers sequentially to avoid rate limiting"""
    db_embeddings = 0
    db_chunks = 0
    reset_request_counter()
    with tqdm(total=len(unprocessed_papers), desc=f"Processing {db} papers (sequential)", unit="paper") as pbar:
        for paper_data in unprocessed_papers:
            idx, row, source = paper_data
            
            try:
                results = process_paper_embeddings(paper_data, api_key)
                if results:
                    # Add results to current batch
                    for result in results:
                        current_batch["embeddings"].append(result["embedding"])
                        current_batch["chunks"].append(result["chunk"])
                        current_batch["metadata"].append(result["metadata"])
                        
                        db_embeddings += 1
                        metadata["total_embeddings"] += 1
                    
                    # Save batch if it reaches the size limit
                    if len(current_batch["embeddings"]) >= BATCH_SIZE:
                        batch_file = save_batch(current_batch, db, batch_num, embeddings_dir)
                        metadata["batches"][f"{db}_batch_{batch_num:04d}"] = {
                            "file": batch_file,
                            "embeddings": len(current_batch["embeddings"]),
                            "created": datetime.now().isoformat()
                        }
                        
                        # Reset batch
                        current_batch = {"embeddings": [], "chunks": [], "metadata": []}
                        batch_num += 1
                
                # Mark paper as processed
                mark_paper_processed(metadata, db, idx)
                db_chunks += len(chunk_paragraphs(row.get("abstract", "")))
                metadata["total_chunks"] += len(chunk_paragraphs(row.get("abstract", "")))
                
                # Save metadata periodically
                if pbar.n % SAVE_INTERVAL == 0 and pbar.n > 0:
                    save_metadata(metadata, embeddings_dir)
                
            except Exception as e:
                print(f"\n⚠️  Error processing paper {idx}: {e}")
            
            pbar.update(1)
            pbar.set_postfix({
                "chunks": db_chunks, 
                "embeddings": db_embeddings,
                "batches": batch_num,
                "req_rate": f"{get_requests_per_sec():.1f} reqs/s"
            })
    
    return db_embeddings, db_chunks, batch_num

def adaptive_parallel_process_papers(unprocessed_papers, api_key, current_batch, metadata, db, batch_num, embeddings_dir, start_time):
    """Process papers with adaptive worker scaling based on request rates"""
    db_embeddings = 0
    db_chunks = 0
    
    print(f"🚀 Adaptive parallel processing:")
    print(f"   Starting workers: {get_worker_count()}")
    print(f"   Target RPS: 24")
    print(f"   Will scale automatically every 10 seconds (after 30s warmup)")
    
    plot = LiveRatePlot()
    last_plot_update = time.time()
    reset_request_counter()
    
    # Use a single ThreadPoolExecutor with maximum workers to avoid freezing
    max_possible_workers = 20  # Maximum workers the adaptive manager can use
    print(f"   Using single executor with max {max_possible_workers} workers")
    
    with ThreadPoolExecutor(max_workers=max_possible_workers) as executor:
        # Submit all papers at once, but control concurrency through adaptive delays
        future_to_paper = {}
        
        print(f"📤 Submitting {len(unprocessed_papers)} papers to processing queue...")
        for paper_data in unprocessed_papers:
            # Use current worker count for rate limiting calculations
            current_workers = get_worker_count()
            # Very conservative delay calculation to prevent rate limiting
            worker_delay = max(RATE_LIMIT_DELAY / max(current_workers, 1), 0.2) if current_workers > 1 else RATE_LIMIT_DELAY
            future = executor.submit(process_paper_embeddings_optimized, paper_data, api_key, current_workers, worker_delay)
            future_to_paper[future] = paper_data
        
        print(f"✅ All papers submitted. Starting processing...")
        
        with tqdm(total=len(unprocessed_papers), desc=f"Processing {db} papers (adaptive)", unit="paper") as pbar:
            completed_count = 0
            last_update_time = time.time()
            
            for future in as_completed(future_to_paper, timeout=300):  # 5-minute timeout per future
                paper_data = future_to_paper[future]
                idx, row, source = paper_data
                error_429 = False
                
                try:
                    results = future.result(timeout=60)  # 1-minute timeout for result
                    if results:
                        for result in results:
                            current_batch["embeddings"].append(result["embedding"])
                            current_batch["chunks"].append(result["chunk"])
                            current_batch["metadata"].append(result["metadata"])
                            db_embeddings += 1
                            metadata["total_embeddings"] += 1
                        if len(current_batch["embeddings"]) >= BATCH_SIZE:
                            batch_file = save_batch(current_batch, db, batch_num, embeddings_dir)
                            metadata["batches"][f"{db}_batch_{batch_num:04d}"] = {
                                "file": batch_file,
                                "embeddings": len(current_batch["embeddings"]),
                                "created": datetime.now().isoformat()
                            }
                            current_batch = {"embeddings": [], "chunks": [], "metadata": []}
                            batch_num += 1
                except Exception as e:
                    if '429' in str(e):
                        error_429 = True
                    print(f"\n⚠️  Error processing paper {idx}: {e}")
                
                mark_paper_processed(metadata, db, idx)
                db_chunks += len(chunk_paragraphs(row.get("abstract", "")))
                metadata["total_chunks"] += len(chunk_paragraphs(row.get("abstract", "")))
                
                completed_count += 1
                
                if pbar.n % SAVE_INTERVAL == 0 and pbar.n > 0:
                    save_metadata(metadata, embeddings_dir)
                
                # Update progress bar with adaptive scaling info
                status = get_status()
                plot.update(get_requests_per_sec(), error=error_429)
                pbar.set_postfix({
                    "req_rate": f"{get_requests_per_sec():.2f} reqs/s",
                    "workers": f"{status['current_workers']}",
                    "rps_ratio": f"{status['performance_ratio']:.2f}",
                    "completed": f"{completed_count}"
                })
                pbar.update(1)
                
                # Force update every 10 seconds even if no completions
                current_time = time.time()
                if current_time - last_update_time > 10:
                    pbar.set_postfix({
                        "req_rate": f"{get_requests_per_sec():.2f} reqs/s",
                        "workers": f"{status['current_workers']}",
                        "rps_ratio": f"{status['performance_ratio']:.2f}",
                        "completed": f"{completed_count}",
                        "waiting": "⏳"
                    })
                    last_update_time = current_time
                
                if time.time() - last_plot_update > 2:
                    plot.refresh()
                    last_plot_update = time.time()
                
                # Show scaling status periodically
                if pbar.n % 50 == 0 and pbar.n > 0:
                    status = get_status()
                    print(f"\n📊 Progress update - Workers: {status['current_workers']}, RPS: {status['current_rps']:.1f}, Ratio: {status['performance_ratio']:.2f}")
    
    plot.refresh()
    plot.close()
    return db_embeddings, db_chunks, batch_num

def optimized_parallel_process_papers(unprocessed_papers, api_key, current_batch, metadata, db, batch_num, embeddings_dir, effective_workers, start_time):
    """Optimized parallel processing that can achieve 5 papers/second while respecting rate limits"""
    db_embeddings = 0
    db_chunks = 0
    worker_delay = RATE_LIMIT_DELAY # RATE_LIMIT_DELAY / effective_workers if effective_workers > 1 else RATE_LIMIT_DELAY
    print(f"🚀 Optimized parallel processing:")
    print(f"   Workers: {effective_workers}")
    print(f"   Base delay: {RATE_LIMIT_DELAY}s")
    print(f"   Worker delay: {worker_delay:.2f}s")
    print(f"   Target: 25 papers/second")
    plot = LiveRatePlot()
    last_plot_update = time.time()
    reset_request_counter()
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        future_to_paper = {
            executor.submit(process_paper_embeddings_optimized, paper_data, api_key, effective_workers, worker_delay): paper_data 
            for paper_data in unprocessed_papers
        }
        from tqdm.auto import tqdm
        with tqdm(total=len(unprocessed_papers), desc=f"Processing {db} papers (optimized)", unit="paper") as pbar:
            for future in as_completed(future_to_paper):
                # Global paper/sec rate limit (enforced before processing each paper)
                # wait_for_paper_rate_limit() # Removed global rate limiting
                paper_data = future_to_paper[future]
                idx, row, source = paper_data
                error_429 = False
                try:
                    results = future.result()
                    if results:
                        for result in results:
                            current_batch["embeddings"].append(result["embedding"])
                            current_batch["chunks"].append(result["chunk"])
                            current_batch["metadata"].append(result["metadata"])
                            db_embeddings += 1
                            metadata["total_embeddings"] += 1
                        if len(current_batch["embeddings"]) >= BATCH_SIZE:
                            batch_file = save_batch(current_batch, db, batch_num, embeddings_dir)
                            metadata["batches"][f"{db}_batch_{batch_num:04d}"] = {
                                "file": batch_file,
                                "embeddings": len(current_batch["embeddings"]),
                                "created": datetime.now().isoformat()
                            }
                            current_batch = {"embeddings": [], "chunks": [], "metadata": []}
                            batch_num += 1
                except Exception as e:
                    if '429' in str(e):
                        error_429 = True
                    print(f"\n⚠️  Error processing paper {idx}: {e}")
                mark_paper_processed(metadata, db, idx)
                db_chunks += len(chunk_paragraphs(row.get("abstract", "")))
                metadata["total_chunks"] += len(chunk_paragraphs(row.get("abstract", "")))
                if pbar.n % SAVE_INTERVAL == 0 and pbar.n > 0:
                    save_metadata(metadata, embeddings_dir)
                elapsed = time.time() - start_time
                plot.update(get_requests_per_sec(), error=error_429)
                pbar.set_postfix({"req_rate": f"{get_requests_per_sec():.2f} reqs/s", "delay": f"{worker_delay:.2f}s"})
                pbar.update(1)
                if time.time() - last_plot_update > 2:
                    plot.refresh()
                    last_plot_update = time.time()
    plot.refresh()
    plot.close()
    return db_embeddings, db_chunks, batch_num

def process_paper_embeddings_optimized(paper_data, api_key, num_workers, worker_delay):
    """
    Optimized version that processes paragraphs in parallel within a paper.
    
    Args:
        paper_data: Tuple of (idx, row, source)
        api_key: Google API key
        num_workers: Number of workers for parallel processing
        worker_delay: Delay for this specific worker
        
    Returns:
        List of embedding results or None if failed
    """
    idx, row, source = paper_data
    title = row.get("title", "")
    doi = row.get("doi", "")
    abstract = row.get("abstract", "")
    authors = row.get("authors", "")  # Add authors field
    
    # Extract additional metadata fields
    publication_date = extract_publication_date(row)
    citation_count = extract_citation_count(row)
    journal = extract_journal_info(row)
    
    paragraphs = chunk_paragraphs(abstract)
    
    # Skip papers with no meaningful content
    if not paragraphs:
        return []
    
    # For papers with few paragraphs, process sequentially with proper delays
    if len(paragraphs) <= 2:
        results = []
        for i, para in enumerate(paragraphs):
            if para.strip():
                # Add worker-specific delay to stagger requests
                if i > 0:
                    time.sleep(worker_delay)
                # Don't skip base delay - use proper rate limiting
                embedding = get_google_embedding(para, api_key, skip_base_delay=False)
                if embedding is not None:
                    results.append({
                        "embedding": embedding,
                        "chunk": para,
                        "metadata": {
                            "title": title,
                            "doi": doi,
                            "authors": authors,  # Add authors to metadata
                            "publication_date": publication_date,
                            "citation_count": citation_count,
                            "journal": journal,
                            "source": source,
                            "paper_index": idx,
                            "para_idx": i,
                            "chunk_length": len(para)
                        }
                    })
        return results
    
    # For papers with many paragraphs, process in parallel
    results = []
    
    # Calculate optimal delay between requests for this worker
    worker_id = hash(f"{idx}_{source}") % num_workers
    para_delay = worker_delay / len(paragraphs) if len(paragraphs) > 1 else worker_delay
    
    with ThreadPoolExecutor(max_workers=min(5, len(paragraphs))) as executor:
        # Submit paragraph processing tasks
        future_to_para = {}
        for i, para in enumerate(paragraphs):
            if para.strip():
                future = executor.submit(
                    get_google_embedding_with_delay, 
                    para, 
                    api_key, 
                    para_delay * i  # Stagger paragraph requests
                )
                future_to_para[future] = (i, para)
        
        # Collect results
        for future in as_completed(future_to_para):
            i, para = future_to_para[future]
            try:
                embedding = future.result()
                if embedding is not None:
                    results.append({
                        "embedding": embedding,
                        "chunk": para,
                        "metadata": {
                            "title": title,
                            "doi": doi,
                            "authors": authors,  # Add authors to metadata
                            "publication_date": publication_date,
                            "citation_count": citation_count,
                            "journal": journal,
                            "source": source,
                            "paper_index": idx,
                            "para_idx": i,
                            "chunk_length": len(para)
                        }
                    })
            except Exception as e:
                print(f"⚠️  Error processing paragraph {i} in paper {idx}: {e}")
    
    return results

def get_google_embedding_with_delay(text, api_key, delay_offset=0):
    """Get embedding with a specific delay offset for parallel processing"""
    # Add worker-specific delay to spread requests
    if delay_offset > 0:
        time.sleep(delay_offset)
    
    # Don't skip base delay - use proper rate limiting
    return get_google_embedding(text, api_key, skip_base_delay=False)

def parallel_process_papers(unprocessed_papers, api_key, current_batch, metadata, db, batch_num, embeddings_dir, effective_workers, start_time):
    """Process papers in parallel with the specified number of workers"""
    db_embeddings = 0
    db_chunks = 0
    worker_delay = RATE_LIMIT_DELAY / effective_workers if effective_workers > 1 else RATE_LIMIT_DELAY
    print(f"🚀 Parallel processing with {effective_workers} workers:")
    print(f"   Workers: {effective_workers}")
    print(f"   Base delay: {RATE_LIMIT_DELAY}s")
    print(f"   Worker delay: {worker_delay:.2f}s")
    print(f"   Target: 3-5 papers/second")
    plot = LiveRatePlot()
    last_plot_update = time.time()
    reset_request_counter()
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        future_to_paper = {
            executor.submit(process_paper_embeddings, paper_data, api_key): paper_data 
            for paper_data in unprocessed_papers
        }
        from tqdm.auto import tqdm
        with tqdm(total=len(unprocessed_papers), desc=f"Processing {db} papers (parallel)", unit="paper") as pbar:
            for future in as_completed(future_to_paper):
                paper_data = future_to_paper[future]
                idx, row, source = paper_data
                error_429 = False
                try:
                    results = future.result()
                    if results:
                        for result in results:
                            current_batch["embeddings"].append(result["embedding"])
                            current_batch["chunks"].append(result["chunk"])
                            current_batch["metadata"].append(result["metadata"])
                            db_embeddings += 1
                            metadata["total_embeddings"] += 1
                        if len(current_batch["embeddings"]) >= BATCH_SIZE:
                            batch_file = save_batch(current_batch, db, batch_num, embeddings_dir)
                            metadata["batches"][f"{db}_batch_{batch_num:04d}"] = {
                                "file": batch_file,
                                "embeddings": len(current_batch["embeddings"]),
                                "created": datetime.now().isoformat()
                            }
                            current_batch = {"embeddings": [], "chunks": [], "metadata": []}
                            batch_num += 1
                except Exception as e:
                    if '429' in str(e):
                        error_429 = True
                    print(f"\n⚠️  Error processing paper {idx}: {e}")
                mark_paper_processed(metadata, db, idx)
                db_chunks += len(chunk_paragraphs(row.get("abstract", "")))
                metadata["total_chunks"] += len(chunk_paragraphs(row.get("abstract", "")))
                if pbar.n % SAVE_INTERVAL == 0 and pbar.n > 0:
                    save_metadata(metadata, embeddings_dir)
                elapsed = time.time() - start_time
                rate = pbar.n / elapsed if elapsed > 0 else 0
                plot.update(get_requests_per_sec(), error=error_429)
                pbar.set_postfix({"req_rate": f"{get_requests_per_sec():.2f} reqs/s", "delay": f"{worker_delay:.2f}s"})
                pbar.update(1)
                if time.time() - last_plot_update > 2:
                    plot.refresh()
                    last_plot_update = time.time()
    plot.refresh()
    plot.close()
    return db_embeddings, db_chunks, batch_num

def legacy_sequential_process_xrvix():
    """
    Legacy sequential xrvix processing: processes all papers and paragraphs strictly sequentially,
    with optional rate limit safe mode (enforces delay between requests).
    Shows a live matplotlib line graph of papers/sec over time (rolling 5 minutes), with red markers for 429 errors.
    """
    print("\n=== Legacy Sequential xrvix Processing (No Parallel, No Rate Limit) ===")
    print("⚠️  WARNING: This mode ignores all rate limits and processes as fast as possible by default!")
    print("   You may hit 429 errors if you exceed the API quota.")

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from collections import deque
    import threading
    import time
    from processing_config import PERFORMANCE_PROFILES

    # Ask user if they want rate limit safe mode
    rate_limit_safe = False
    delay = 0.0
    print("\n💡 Option: Run in rate limit safe mode?")
    print("   This will add a delay after each API request to avoid 429 errors.")
    while True:
        ans = input("Run in rate limit safe mode? (y/n): ").strip().lower()
        if ans in ["y", "yes"]:
            rate_limit_safe = True
            delay = PERFORMANCE_PROFILES["rate_limit_safe"]["rate_limit_delay"]
            print(f"✅ Rate limit safe mode enabled. Delay: {delay} seconds between requests.")
            break
        elif ans in ["n", "no"]:
            print("⚠️  Running in burst mode (no delay, may hit rate limits)")
            break
        else:
            print("Please enter 'y' or 'n'.")

    # Live plot setup
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title("Requests/sec (rolling, last 5 min)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Requests/sec")
    rate_times = deque()
    rate_values = deque()
    error_times = deque()
    error_values = deque()
    lock = threading.Lock()
    window_seconds = 300  # 5 minutes
    last_plot_update = [0]

    def update_plot():
        with lock:
            now = datetime.now()
            # Remove old points
            while rate_times and (now - rate_times[0]).total_seconds() > window_seconds:
                rate_times.popleft()
                rate_values.popleft()
            while error_times and (now - error_times[0]).total_seconds() > window_seconds:
                error_times.popleft()
                error_values.popleft()
            ax.clear()
            ax.set_title("Requests/sec (rolling, last 5 min)")
            ax.set_xlabel("Time")
            ax.set_ylabel("Requests/sec")
            if rate_times:
                ax.plot(rate_times, rate_values, label="Requests/sec", color="blue")
            if error_times:
                ax.scatter(error_times, error_values, color="red", label="429 error", marker="x", s=60)
            ax.legend()
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            fig.autofmt_xdate()
            plt.tight_layout()
            plt.pause(0.01)

    # Load API key
    with open("keys.json") as f:
        api_key = json.load(f)["GOOGLE_API_KEY"]

    # Setup directory structure
    ensure_directory(EMBEDDINGS_DIR)
    
    # Load or create metadata
    metadata = load_metadata(EMBEDDINGS_DIR)
    print(f"📊 Loaded metadata: {metadata['total_embeddings']} total embeddings")

    for db in DUMPS:
        print(f"\n🔄 Processing {db} (legacy sequential)...")
        source_dir = os.path.join(EMBEDDINGS_DIR, db)
        ensure_directory(source_dir)
        dump_paths = [os.path.join(DUMP_ROOT, f) for f in os.listdir(DUMP_ROOT) if f.startswith(db)]
        if not dump_paths:
            print(f"❌ No dump found for {db}, skipping.")
            continue
        path = sorted(dump_paths, reverse=True)[0]
        print(f"📁 Using dump file: {os.path.basename(path)}")
        querier = XRXivQuery(path)
        if querier.errored:
            print(f"❌ Error loading {db} dump, skipping.")
            continue
        print(f"📊 Found {len(querier.df)} papers in {db}")
        processed_papers = get_processed_papers(metadata, db)
        print(f"📊 Already processed: {len(processed_papers)} papers")
        unprocessed_papers = []
        for idx, row in querier.df.iterrows():
            if idx not in processed_papers:
                unprocessed_papers.append((idx, row, db))
        print(f"📊 Remaining to process: {len(unprocessed_papers)} papers")
        if not unprocessed_papers:
            print(f"✅ All papers for {db} already processed!")
            continue
        current_batch = {"embeddings": [], "chunks": [], "metadata": []}
        batch_num = get_next_batch_num(db, EMBEDDINGS_DIR)
        db_embeddings = 0
        db_chunks = 0
        start_time = time.time()
        from tqdm.auto import tqdm
        last_rate_update = time.time()
        papers_processed = 0
        error_flag = [False]
        def get_embedding_with_429_flag(*args, **kwargs):
            try:
                return get_google_embedding(*args, **kwargs)
            except Exception as e:
                if '429' in str(e):
                    error_flag[0] = True
                raise
        reset_request_counter()
        with tqdm(total=len(unprocessed_papers), desc=f"Processing {db} (legacy)", unit="paper") as pbar:
            for n, (idx, row, source) in enumerate(unprocessed_papers, 1):
                title = row.get("title", "")
                doi = row.get("doi", "")
                abstract = row.get("abstract", "")
                paragraphs = chunk_paragraphs(abstract)
                paper_had_429 = False
                for i, para in enumerate(paragraphs):
                    if not para.strip():
                        continue
                    # No explicit delay or rate limit unless rate_limit_safe is enabled!
                    try:
                        embedding = get_google_embedding(para, api_key, skip_base_delay=True)
                        if rate_limit_safe and delay > 0:
                            time.sleep(delay)
                    except Exception as e:
                        if '429' in str(e):
                            paper_had_429 = True
                            # Mark error for plotting
                            with lock:
                                error_times.append(datetime.now())
                                error_values.append(float(0.0)) # Always append float for consistency
                        continue
                    if embedding is None:
                        continue
                    current_batch["embeddings"].append(embedding)
                    current_batch["chunks"].append(para)
                    current_batch["metadata"].append({
                        "title": title,
                        "doi": doi,
                        "source": source,
                        "paper_index": idx,
                        "para_idx": i,
                        "chunk_length": len(para)
                    })
                    db_embeddings += 1
                    db_chunks += 1
                    metadata["total_embeddings"] += 1
                    metadata["total_chunks"] += 1
                    # Save batch if it reaches the size limit
                    if len(current_batch["embeddings"]) >= BATCH_SIZE:
                        batch_file = save_batch(current_batch, db, batch_num, EMBEDDINGS_DIR)
                        metadata["batches"][f"{db}_batch_{batch_num:04d}"] = {
                            "file": batch_file,
                            "embeddings": len(current_batch["embeddings"]),
                            "created": datetime.now().isoformat()
                        }
                        current_batch = {"embeddings": [], "chunks": [], "metadata": []}
                        batch_num += 1
                # Mark paper as processed
                mark_paper_processed(metadata, db, idx)
                papers_processed += 1
                # Save metadata periodically
                if n % SAVE_INTERVAL == 0:
                    save_metadata(metadata, EMBEDDINGS_DIR)
                # Progress bar update and live plot update
                elapsed = time.time() - start_time
                rate = papers_processed / elapsed if elapsed > 0 else 0
                now = datetime.now()
                with lock:
                    rate_times.append(now)
                    rate_values.append(get_requests_per_sec())
                if paper_had_429:
                    with lock:
                        error_times.append(now)
                        error_values.append(float(rate))  # Always append float for consistency
                pbar.set_postfix({"req_rate": f"{get_requests_per_sec():.1f} reqs/s", "delay": f"{delay:.2f}s" if rate_limit_safe else "burst"})
                pbar.update(1)
                # Update plot every 2 seconds
                if time.time() - last_plot_update[0] > 2:
                    # Use requests/sec for plotting
                    now = datetime.now()
                    with lock:
                        rate_times.append(now)
                        rate_values.append(get_requests_per_sec())
                    update_plot()
                    last_plot_update[0] = time.time()
        # Final plot update
        update_plot()
        # Save any remaining embeddings in the current batch
        if current_batch["embeddings"]:
            batch_file = save_batch(current_batch, db, batch_num, EMBEDDINGS_DIR)
            metadata["batches"][f"{db}_batch_{batch_num:04d}"] = {
                "file": batch_file,
                "embeddings": len(current_batch["embeddings"]),
                "created": datetime.now().isoformat()
            }
        total_processed = len(get_processed_papers(metadata, db))
        metadata["sources"][db] = {
            "papers": len(querier.df),
            "processed_papers": total_processed,
            "chunks": db_chunks,
            "embeddings": db_embeddings,
            "batches": batch_num + 1
        }
        metadata["total_papers"] += len(querier.df)
        save_metadata(metadata, EMBEDDINGS_DIR)
        processing_time = time.time() - start_time
        papers_per_second = len(unprocessed_papers) / processing_time if processing_time > 0 else 0
        print(f"✅ Finished {db}: {db_chunks} chunks processed, {db_embeddings} embeddings created in {batch_num + 1} batches")
        print(f"⏱️  Processing time: {processing_time/3600:.1f} hours")
        print(f"🚀 Processing rate: {papers_per_second:.1f} papers/second")
    print("\n🎉 All dumps processed (legacy sequential mode)!")
    print("\nClose the plot window to finish.")
    plt.ioff()
    plt.show()

def main():
    print("=== Starting xrvix Dumps Processing (Multi-File Storage with Adaptive Parallel Processing) ===")
    
    # Initialize adaptive worker manager if enabled
    if ADAPTIVE_SCALING_ENABLED:
        from adaptive_worker_manager import reset_adaptive_manager
        adaptive_manager = reset_adaptive_manager()  # Reset for clean start
        print(f"🔧 Adaptive Worker Configuration:")
        print(f"   Target RPS: {adaptive_manager.target_rps}")
        print(f"   Scaling interval: {adaptive_manager.scaling_interval}s")
        print(f"   Max workers: {adaptive_manager.max_workers}")
        print(f"   Min workers: {adaptive_manager.min_workers}")
        print(f"   Scale up threshold: {adaptive_manager.scale_up_threshold} (immediate when RPS < 19.2)")
        print(f"   Scale down threshold: {adaptive_manager.scale_down_threshold} (immediate when RPS > 28.8)")
        print(f"   Scaling behavior: Immediate up/down based on RPS thresholds")
        print(f"   Warmup period: {adaptive_manager.warmup_period}s before scaling starts")
    else:
        print(f"🔧 Static Processing Configuration:")
        print(f"   Parallel workers: {MAX_WORKERS}")
        print(f"   Batch size: {BATCH_SIZE}")
        print(f"   Rate limit delay: {RATE_LIMIT_DELAY}s")
        print(f"   Request timeout: {REQUEST_TIMEOUT}s")
    
    print(f"   Sources: {', '.join(DUMPS)}")
    print(f"🚦 Rate limiting: {MAX_REQUESTS_PER_MINUTE} requests per {RATE_LIMIT_WINDOW}s")
    print(f"   - Embedding API: {EMBEDDING_MAX_REQUESTS_PER_MINUTE} requests/minute")
    print(f"   - Gemini API: {GEMINI_MAX_REQUESTS_PER_MINUTE} requests/minute")
    print()
    
    # Load API key
    with open("keys.json") as f:
        api_key = json.load(f)["GOOGLE_API_KEY"]

    # Setup directory structure
    ensure_directory(EMBEDDINGS_DIR)
    
    # Load or create metadata
    metadata = load_metadata(EMBEDDINGS_DIR)
    print(f"📊 Loaded metadata: {metadata['total_embeddings']} total embeddings")

    try:
        for db in DUMPS:
            print(f"\n🔄 Processing {db}...")
            
            # Create source directory
            source_dir = os.path.join(EMBEDDINGS_DIR, db)
            ensure_directory(source_dir)
            
            # Find dump file
            dump_paths = [os.path.join(DUMP_ROOT, f) for f in os.listdir(DUMP_ROOT) if f.startswith(db)]
            if not dump_paths:
                print(f"❌ No dump found for {db}, skipping.")
                continue
                
            path = sorted(dump_paths, reverse=True)[0]
            print(f"📁 Using dump file: {os.path.basename(path)}")
            
            # Load dump
            querier = XRXivQuery(path)
            if querier.errored:
                print(f"❌ Error loading {db} dump, skipping.")
                continue
                
            print(f"📊 Found {len(querier.df)} papers in {db}")
            
            # Get already processed papers
            processed_papers = get_processed_papers(metadata, db)
            print(f"📊 Already processed: {len(processed_papers)} papers")
            
            # Filter out already processed papers
            unprocessed_papers = []
            for idx, row in querier.df.iterrows():
                if idx not in processed_papers:
                    unprocessed_papers.append((idx, row, db))
            
            print(f"📊 Remaining to process: {len(unprocessed_papers)} papers")
            
            if not unprocessed_papers:
                print(f"✅ All papers for {db} already processed!")
                continue
            
            # Initialize batch tracking
            current_batch = {
                "embeddings": [],
                "chunks": [],
                "metadata": []
            }
            batch_num = get_next_batch_num(db, EMBEDDINGS_DIR)
            db_embeddings = 0
            db_chunks = 0
            
            start_time = time.time()
            
            # Use adaptive worker scaling if enabled
            if ADAPTIVE_SCALING_ENABLED:
                print(f"🚀 Starting adaptive parallel processing...")
                print(f"   Starting with {get_worker_count()} workers")
                print(f"   Will automatically scale based on request rate")
                
                # Process papers with adaptive worker scaling
                db_embeddings, db_chunks, batch_num = adaptive_parallel_process_papers(
                    unprocessed_papers, api_key, current_batch, metadata, db, 
                    batch_num, EMBEDDINGS_DIR, start_time
                )
            else:
                # Use static configuration
                effective_workers = MAX_WORKERS
                print(f"🚀 Starting parallel processing with {effective_workers} workers...")
                
                # Check if user wants to force parallel processing (bypass automatic fallback)
                force_parallel = os.environ.get("FORCE_PARALLEL", "false").lower() == "true"
                
                # Only apply automatic fallback if not forcing parallel
                if not force_parallel:
                    # Conservative fallback for very high rate limiting
                    if RATE_LIMIT_DELAY > 2.0:  # Only for extremely conservative profiles
                        print("🐌 Using sequential processing due to very high rate limiting (delay > 2s)")
                        db_embeddings, db_chunks, batch_num = sequential_process_papers(unprocessed_papers, api_key, current_batch, metadata, db, batch_num, EMBEDDINGS_DIR, start_time)
                    else:
                        print(f"🚀 Using parallel processing with {effective_workers} workers (delay: {RATE_LIMIT_DELAY}s)")
                        db_embeddings, db_chunks, batch_num = optimized_parallel_process_papers(unprocessed_papers, api_key, current_batch, metadata, db, batch_num, EMBEDDINGS_DIR, effective_workers, start_time)
                else:
                    print(f"🚀 FORCING parallel processing with {effective_workers} workers (bypassing automatic fallback)")
                    db_embeddings, db_chunks, batch_num = optimized_parallel_process_papers(unprocessed_papers, api_key, current_batch, metadata, db, batch_num, EMBEDDINGS_DIR, effective_workers, start_time)
            
            # Save any remaining embeddings in the current batch
            if current_batch["embeddings"]:
                batch_file = save_batch(current_batch, db, batch_num, EMBEDDINGS_DIR)
                metadata["batches"][f"{db}_batch_{batch_num:04d}"] = {
                    "file": batch_file,
                    "embeddings": len(current_batch["embeddings"]),
                    "created": datetime.now().isoformat()
                }
            
            # Update source statistics
            total_processed = len(get_processed_papers(metadata, db))
            metadata["sources"][db] = {
                "papers": len(querier.df),
                "processed_papers": total_processed,
                "chunks": db_chunks,
                "embeddings": db_embeddings,
                "batches": batch_num + 1
            }
            metadata["total_papers"] += len(querier.df)
            
            # Save final metadata for this source
            save_metadata(metadata, EMBEDDINGS_DIR)
            
            processing_time = time.time() - start_time
            papers_per_second = len(unprocessed_papers) / processing_time if processing_time > 0 else 0
            
            print(f"✅ Finished {db}: {db_chunks} chunks processed, {db_embeddings} embeddings created in {batch_num + 1} batches")
            print(f"⏱️  Processing time: {processing_time/3600:.1f} hours")
            print(f"🚀 Processing rate: {papers_per_second:.1f} papers/second")
            
            # Show adaptive scaling status if enabled
            if ADAPTIVE_SCALING_ENABLED:
                status = get_status()
                print(f"🔧 Final adaptive scaling status:")
                print(f"   Workers: {status['current_workers']}")
                print(f"   Average RPS: {status['current_rps']:.1f}")
                print(f"   Performance ratio: {status['performance_ratio']:.2f}")
        
        # Print final statistics
        print(f"\n🎉 All dumps processed!")
        print(f"📊 Total embeddings: {metadata['total_embeddings']}")
        print(f"📊 Total chunks: {metadata['total_chunks']}")
        print(f"📊 Total papers: {metadata['total_papers']}")
        print(f"📁 Embeddings directory: {os.path.abspath(EMBEDDINGS_DIR)}")
        
        # Calculate total file size
        total_size = 0
        for root, dirs, files in os.walk(EMBEDDINGS_DIR):
            for file in files:
                total_size += os.path.getsize(os.path.join(root, file))
        print(f"📊 Total size: {total_size / 1024 / 1024:.1f} MB")
        
        # Print source breakdown
        print(f"\n📊 Source Breakdown:")
        for source, stats in metadata["sources"].items():
            print(f"   {source}: {stats['papers']} papers, {stats.get('processed_papers', 0)} processed, {stats['embeddings']} embeddings, {stats['batches']} batches")
    
    finally:
        # Clean up adaptive worker manager
        if ADAPTIVE_SCALING_ENABLED:
            stop_adaptive_manager()
            print("🔧 Adaptive worker manager stopped")

if __name__ == "__main__":
    main() 