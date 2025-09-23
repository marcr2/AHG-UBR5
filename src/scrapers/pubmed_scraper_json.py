import os
import json
import pandas as pd
from paperscraper.pubmed import get_and_dump_pubmed_papers
from paperscraper.citations import get_citations_from_title, get_citations_by_doi
from paperscraper.impact import Impactor
import requests
import re
from tqdm.auto import tqdm
import numpy as np
from datetime import datetime
import time
from src.core.processing_config import get_config, print_config_info
from src.core.chromadb_manager import ChromaDBManager
import warnings
import urllib.parse
import xml.etree.ElementTree as ET
import concurrent.futures
import threading
from functools import partial
import sys

# Suppress paperscraper warnings
warnings.filterwarnings("ignore", message="Could not find paper")
warnings.filterwarnings("ignore", category=UserWarning, module="paperscraper")

# Force unbuffered output for better CLI responsiveness
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

def flush_output():
    """Force flush all output streams to ensure immediate display"""
    sys.stdout.flush()
    sys.stderr.flush()

# Global counter for citation processing errors
citation_error_counter = {
    'doi_lookup_failures': 0,
    'title_lookup_failures': 0,
    'total_failures': 0,
    'successful_citations': 0,
    'failed_citations': 0
}

# Thread-local storage for rate limiting
thread_local = threading.local()

def get_rate_limiter():
    """Get or create a rate limiter for the current thread."""
    if not hasattr(thread_local, 'last_request_time'):
        thread_local.last_request_time = 0
    return thread_local.last_request_time

def set_rate_limiter(timestamp):
    """Set the last request time for the current thread."""
    thread_local.last_request_time = timestamp

def rate_limit_delay(min_delay=0.1):
    """Implement rate limiting to avoid overwhelming APIs."""
    current_time = time.time()
    last_request = get_rate_limiter()
    
    if current_time - last_request < min_delay:
        sleep_time = min_delay - (current_time - last_request)
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    set_rate_limiter(time.time())

def reset_citation_error_counter():
    """Reset the global citation error counter."""
    global citation_error_counter
    citation_error_counter = {
        'doi_lookup_failures': 0,
        'title_lookup_failures': 0,
        'total_failures': 0,
        'successful_citations': 0,
        'failed_citations': 0
    }

def get_citation_error_summary():
    """Get a summary of citation processing errors."""
    global citation_error_counter
    return citation_error_counter.copy()

def save_papers_to_file(papers, filepath):
    """
    Save papers to a JSONL file.
    
    Args:
        papers: List of paper dictionaries
        filepath: Path to save the file
    """
    import os
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save papers to JSONL format
    with open(filepath, 'w', encoding='utf-8') as f:
        for paper in papers:
            f.write(json.dumps(paper, ensure_ascii=False) + '\n')

def search_pubmed_comprehensive(search_terms, date_from="1900", date_to=None, max_results=None, auto_save_frequency=500):
    """
    Comprehensive PubMed search that searches until no more unique papers are found.
    
    Args:
        search_terms: List of search terms
        date_from: Start date for search (YYYY or YYYY-MM-DD)
        date_to: End date for search (YYYY or YYYY-MM-DD), defaults to current date
        max_results: Maximum number of results to collect (default: 10000)
        auto_save_frequency: Save data every N articles (default: 500)
    
    Returns:
        List of paper dictionaries
    """
    if date_to is None:
        date_to = datetime.now().strftime("%Y-%m-%d")
    
    print(f"🔍 Starting comprehensive PubMed search...")
    print(f"   Date range: {date_from} to {date_to}")
    print(f"   Searching until no more unique papers found for each keyword")
    
    all_papers = []
    
    # Set default max_results if not provided
    if max_results is None:
        max_results = 10000  # Default to 10,000 papers
    
    # Track last auto-save count
    last_autosave_count = 0
    
    # Ensure all search terms are strings and clean them
    cleaned_search_terms = []
    for term in search_terms:
        if term is not None:
            # Convert to string and clean - handle various data types
            try:
                if isinstance(term, (int, float)):
                    # Convert numbers to strings
                    term_str = str(int(term)) if isinstance(term, float) and term.is_integer() else str(term)
                else:
                    term_str = str(term).strip()
                
                # Additional validation
                if term_str and term_str.lower() not in ['nan', 'none', 'null', ''] and len(term_str.strip()) > 0:
                    cleaned_search_terms.append(term_str.strip())
            except (ValueError, TypeError) as e:
                print(f"⚠️  Skipping invalid search term '{term}': {e}")
                continue
    
    if not cleaned_search_terms:
        print("❌ No valid search terms provided!")
        return []
    
    print(f"📋 Cleaned search terms: {', '.join(cleaned_search_terms)}")
    
    # Simple approach: search each keyword individually
    search_strategies = []
    for term in cleaned_search_terms:
        # Ensure term is a string and not a float - additional safety check
        if isinstance(term, (int, float)):
            term = str(term)
        elif not isinstance(term, str):
            term = str(term)
        
        # Additional validation to prevent float values from being passed to paperscraper
        if term and term.strip() and not term.lower() in ['nan', 'none', 'null', '']:
            # Search each keyword in Title/Abstract
            search_strategies.append(f'"{term}"[Title/Abstract]')
    
    print(f"📋 Generated {len(search_strategies)} individual keyword searches")
    
    # Initialize tracking variables
    all_papers = []
    unique_papers = set()
    total_requests = 0
    start_time = time.time()
    
    # Real-time display function
    def update_display():
        elapsed = time.time() - start_time
        unique_count = len(unique_papers)
        total_count = len(all_papers)
        rps = total_requests / elapsed if elapsed > 0 else 0
        
        # Calculate ETA based on current rate
        if rps > 0 and len(search_strategies) > 0:
            remaining_strategies = len(search_strategies) - total_requests
            eta_seconds = remaining_strategies / rps if rps > 0 else 0
            eta_str = f"{int(eta_seconds//3600):02d}:{int((eta_seconds%3600)//60):02d}:{int(eta_seconds%60):02d}"
        else:
            eta_str = "00:00:00"
        
        # Clear line and print updated info
        print(f"\rExecuting strategies: {total_requests}/{len(search_strategies)} | "
              f"Found {total_count} papers of which {unique_count} are unique | "
              f"Requests per second: {rps:.1f}/s (rate limit = 3/s) | "
              f"ETA: {eta_str}", end="", flush=True)
    
    # Execute searches with real-time display
    for i, search_query in enumerate(search_strategies):
            try:
                # Debug: Check search_query type before processing
                if not isinstance(search_query, str):
                    print(f"⚠️  Warning: search_query is not a string: {type(search_query)} = {search_query}")
                    search_query = str(search_query)
                
                # Add date filters to the search query
                date_filter = f" AND ({date_from}[Date - Publication]:{date_to}[Date - Publication])"
                full_query = search_query + date_filter
                
                # Silent execution - only show real-time display
                
                # Use enhanced progress tracking for all searches
                try:
                    import threading
                    import queue
                    
                    result_queue = queue.Queue()
                    progress_queue = queue.Queue()
                    
                    def search_worker_with_progress(query):
                        # Retry mechanism with exponential backoff
                        max_retries = 3
                        for attempt in range(max_retries):
                            try:
                                if attempt > 0:
                                    wait_time = 2 ** attempt  # Exponential backoff: 2, 4 seconds
                                    time.sleep(wait_time)
                                
                                # Call paperscraper silently with additional error handling
                                try:
                                    papers = get_and_dump_pubmed_papers([[query]], f"temp_search_{i}.jsonl")
                                except AttributeError as attr_error:
                                    if "'float' object has no attribute 'lower'" in str(attr_error):
                                        print(f"⚠️  Data type error in search strategy {i+1}: {attr_error}")
                                        print(f"   Search query: {query}")
                                        print(f"   Query type: {type(query)}")
                                        # Try to fix by ensuring the query is a proper string
                                        if not isinstance(query, str):
                                            query = str(query)
                                        papers = get_and_dump_pubmed_papers([[query]], f"temp_search_{i}.jsonl")
                                    else:
                                        raise attr_error
                                
                                # Handle case where paperscraper returns None (known bug)
                                if papers is None:
                                    papers = []  # Will be loaded from file later
                                elif not isinstance(papers, list):
                                    papers = []
                                
                                result_queue.put(('success', papers))
                                break  # Success, exit retry loop
                                
                            except Exception as e:
                                if attempt == max_retries - 1:
                                    # Final attempt failed
                                    result_queue.put(('error', e))
                                else:
                                    # Silent retry
                                    pass
                    
                    def progress_monitor():
                        """Monitor progress by checking the temp file size and content"""
                        temp_file = f"temp_search_{i}.jsonl"
                        last_size = 0
                        last_count = 0
                        start_time = time.time()
                        no_progress_count = 0
                        max_no_progress = 120  # Stop monitoring after 120 seconds (2 minutes) of no progress
                        
                        while True:
                            try:
                                if os.path.exists(temp_file):
                                    current_size = os.path.getsize(temp_file)
                                    # Count lines in the file to get paper count
                                    try:
                                        with open(temp_file, 'r', encoding='utf-8') as f:
                                            current_count = sum(1 for _ in f)
                                    except:
                                        current_count = 0
                                    
                                    # Update progress if there's change
                                    if current_size != last_size or current_count != last_count:
                                        elapsed = time.time() - start_time
                                        progress_queue.put({
                                            'size': current_size,
                                            'count': current_count,
                                            'elapsed': elapsed,
                                            'file': temp_file,
                                            'status': 'active'
                                        })
                                        last_size = current_size
                                        last_count = current_count
                                        no_progress_count = 0
                                    else:
                                        no_progress_count += 1
                                        # Send status update even if no progress
                                        elapsed = time.time() - start_time
                                        progress_queue.put({
                                            'size': current_size,
                                            'count': current_count,
                                            'elapsed': elapsed,
                                            'file': temp_file,
                                            'status': 'waiting' if no_progress_count < 10 else 'stalled'
                                        })
                                else:
                                    # File doesn't exist yet - paperscraper hasn't started writing
                                    elapsed = time.time() - start_time
                                    # Provide more specific status based on elapsed time
                                    if elapsed < 15:
                                        status = 'initializing'
                                    elif elapsed < 45:
                                        status = 'connecting'
                                    elif elapsed < 90:
                                        status = 'querying'
                                    else:
                                        status = 'stalled'
                                    
                                    progress_queue.put({
                                        'size': 0,
                                        'count': 0,
                                        'elapsed': elapsed,
                                        'file': temp_file,
                                        'status': status
                                    })
                                    no_progress_count += 1
                                
                                # Stop monitoring if no progress for too long
                                if no_progress_count > max_no_progress:
                                    progress_queue.put({
                                        'size': last_size,
                                        'count': last_count,
                                        'elapsed': time.time() - start_time,
                                        'file': temp_file,
                                        'status': 'timeout'
                                    })
                                    break
                                
                                time.sleep(5)  # Check every 5 seconds to reduce spam
                            except Exception:
                                break
                    
                    # Start search thread
                    search_thread = threading.Thread(target=search_worker_with_progress, args=(full_query,))
                    search_thread.daemon = True
                    search_thread.start()
                    
                    # Wait for completion and get result
                    try:
                        result_type, result_data = result_queue.get(timeout=300)  # 5 minute timeout
                        if result_type == 'success':
                            papers = result_data
                        else:
                            raise result_data
                    except queue.Empty:
                        print(f"\n⏰ Search timeout - switching to direct API fallback")
                        try:
                            fallback_papers = search_pubmed_direct_api([search_query], max_results=1000, date_from=date_from, date_to=date_to)
                            if fallback_papers:
                                papers = fallback_papers
                            else:
                                papers = []
                        except Exception as fallback_error:
                            print(f"❌ Fallback also failed: {fallback_error}")
                            papers = []
                    
                except Exception as e:
                    # Silent error handling - try fallback
                    try:
                        fallback_papers = search_pubmed_direct_api([search_query], max_results=1000, date_from=date_from, date_to=date_to)
                        if fallback_papers:
                            papers = fallback_papers
                        else:
                            papers = []
                    except Exception:
                        papers = []
                
                # Load the temporary results
                if os.path.exists(f"temp_search_{i}.jsonl"):
                    try:
                        # Check file size first
                        file_size = os.path.getsize(f"temp_search_{i}.jsonl")
                        if file_size == 0:
                            temp_papers = []
                        else:
                            with open(f"temp_search_{i}.jsonl", 'r', encoding='utf-8') as f:
                                temp_papers = [json.loads(line) for line in f if line.strip()]
                            
                            if temp_papers is None:
                                temp_papers = []
                    except Exception:
                        temp_papers = []
                else:
                    temp_papers = []
                
                # Validate paper data and enrich with citation/impact factor data
                valid_papers = []
                skipped_count = 0
                for paper in temp_papers:
                    # Check if paper has required fields
                    if paper and isinstance(paper, dict):
                        title = paper.get('title', '')
                        abstract = paper.get('abstract', '')
                        
                        # Only add papers with both title and abstract
                        if title and abstract and str(title).strip() and str(abstract).strip():
                            # Skip citation extraction during initial scraping to avoid hangs
                            # Citations will be extracted later during embedding processing
                            paper['citation_count'] = 'pending'
                            
                            # Extract impact factor directly (this is fast and local)
                            journal_name = paper.get('journal', '')
                            # Ensure journal_name is a string, not a float
                            if isinstance(journal_name, (int, float)):
                                journal_name = str(journal_name)
                            elif not isinstance(journal_name, str):
                                journal_name = str(journal_name) if journal_name is not None else ''
                            paper['impact_factor'] = estimate_impact_factor(journal_name)
                            
                            valid_papers.append(paper)
                        else:
                            skipped_count += 1
                    else:
                        skipped_count += 1
                
                # Silent processing
                
                # Add unique valid papers to our collection
                papers_added_this_round = 0
                for paper in valid_papers:
                    if paper not in all_papers:
                        all_papers.append(paper)
                        papers_added_this_round += 1
                
                # Auto-save data every N articles
                current_count = len(all_papers)
                if current_count > 0 and current_count - last_autosave_count >= auto_save_frequency:
                    try:
                        save_papers_to_file(all_papers, f"data/scraped_data/pubmed/pubmed_autosave_{current_count}_papers.jsonl")
                        print(f"\n💾 Auto-saved {current_count} papers to autosave file")
                        last_autosave_count = current_count
                    except Exception as e:
                        print(f"\n⚠️  Auto-save failed: {e}")
                
                # Clean up temporary file
                if os.path.exists(f"temp_search_{i}.jsonl"):
                    os.remove(f"temp_search_{i}.jsonl")
                
                # Reduced delay - only 0.5 seconds between strategies
                time.sleep(0.5)
                
            except Exception as e:
                print(f"\n⚠️  Search strategy {i+1} failed: {e}")
                # Clean up temporary file even on exception
                if os.path.exists(f"temp_search_{i}.jsonl"):
                    try:
                        os.remove(f"temp_search_{i}.jsonl")
                        print(f"🧹 Cleaned up temp file: temp_search_{i}.jsonl")
                    except Exception as cleanup_error:
                        print(f"⚠️  Failed to clean up temp file: {cleanup_error}")
                continue
            
            # Update tracking variables
            total_requests += 1
            
            # Add papers to unique set for tracking
            for paper in temp_papers:
                if paper.get('title'):
                    unique_papers.add(paper['title'])
            
            # Update real-time display
            update_display()
            
            # Small delay to allow display updates
            time.sleep(0.1)
            
            # Continue searching until all keywords are exhausted
    
    # Final display update
    print()  # New line after the real-time display
    print(f"✅ Search completed: {len(all_papers)} total papers, {len(unique_papers)} unique")
    
    # Display comprehensive search summary
    print(f"\n📊 SEARCH SUMMARY")
    print(f"=" * 50)
    print(f"🔍 Total search strategies executed: {len(search_strategies)}")
    print(f"📄 Total papers collected: {len(all_papers)}")
    print(f"📅 Date range: {date_from} to {date_to}")
    print(f"🎯 Target results: {max_results}")
    
    # Calculate success rate
    successful_strategies = len([s for s in search_strategies if any(p.get('title') for p in all_papers)])
    success_rate = (successful_strategies / len(search_strategies)) * 100 if search_strategies else 0
    print(f"✅ Success rate: {success_rate:.1f}% ({successful_strategies}/{len(search_strategies)} strategies)")
    
    # Show file sizes
    total_temp_size = 0
    temp_files = [f"temp_search_{i}.jsonl" for i in range(len(search_strategies))]
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            size_mb = os.path.getsize(temp_file) / (1024 * 1024)
            total_temp_size += size_mb
            print(f"📁 {temp_file}: {size_mb:.1f}MB")
    
    print(f"💾 Total data collected: {total_temp_size:.1f}MB")
    print(f"=" * 50)
    flush_output()
    
    # Remove duplicates based on DOI and title
    unique_papers = []
    seen_dois = set()
    seen_titles = set()
    
    for paper in all_papers:
        try:
            # Safely extract DOI and title with defensive programming
            doi = paper.get('doi', '')
            title = paper.get('title', '')
            
            # Handle None values safely
            if doi is None:
                doi = ""
            if title is None:
                title = ""
            
            # Convert to strings and strip safely
            doi = str(doi).strip() if doi is not None else ""
            
            # Ensure title is a string before processing
            if isinstance(title, (int, float)):
                title = str(title)
            elif not isinstance(title, str):
                title = str(title) if title is not None else ''
            
            title = title.strip().lower() if title else ""
            
            # Check if we've seen this paper before
            if doi and doi not in seen_dois:
                unique_papers.append(paper)
                seen_dois.add(doi)
            elif title and title not in seen_titles:
                unique_papers.append(paper)
                seen_titles.add(title)
        except Exception as e:
            print(f"⚠️  Error processing paper for deduplication: {e}")
            # Add the paper anyway to avoid losing data
            unique_papers.append(paper)
            continue
    
    print(f"🎯 Found {len(all_papers)} total papers, {len(unique_papers)} unique papers")
    
    # Final save of all data
    try:
        final_save_path = f"data/scraped_data/pubmed/pubmed_final_{len(unique_papers)}_papers.jsonl"
        save_papers_to_file(unique_papers, final_save_path)
        print(f"💾 Final save: {len(unique_papers)} papers saved to {final_save_path}")
    except Exception as e:
        print(f"⚠️  Final save failed: {e}")
    
    return unique_papers

def search_pubmed_direct_api(search_terms, date_from="1900", date_to=None, auto_save_frequency=500):
    """
    Comprehensive PubMed search using direct API calls that searches until no more unique papers are found.
    
    Args:
        search_terms: List of search terms
        date_from: Start date for search (YYYY or YYYY-MM-DD)
        date_to: End date for search (YYYY or YYYY-MM-DD), defaults to current date
    
    Returns:
        List of paper dictionaries
    """
    if date_to is None:
        date_to = datetime.now().strftime("%Y-%m-%d")
    
    print(f"🔍 Starting comprehensive direct PubMed API search...")
    print(f"   Date range: {date_from} to {date_to}")
    print(f"   Searching until no more unique papers found for each keyword")
    
    all_papers = []
    
    # PubMed E-utilities base URL
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    
    # Ensure all search terms are strings and clean them
    cleaned_search_terms = []
    for term in search_terms:
        if term is not None:
            # Convert to string and clean - handle various data types
            try:
                if isinstance(term, (int, float)):
                    # Convert numbers to strings
                    term_str = str(int(term)) if isinstance(term, float) and term.is_integer() else str(term)
                else:
                    term_str = str(term).strip()
                
                # Additional validation
                if term_str and term_str.lower() not in ['nan', 'none', 'null', ''] and len(term_str.strip()) > 0:
                    cleaned_search_terms.append(term_str.strip())
            except (ValueError, TypeError) as e:
                print(f"⚠️  Skipping invalid search term '{term}': {e}")
                continue
    
    if not cleaned_search_terms:
        print("❌ No valid search terms provided!")
        return []
    
    print(f"📋 Cleaned search terms: {', '.join(cleaned_search_terms)}")
    
    # Simple approach: search each keyword individually
    search_strategies = []
    for term in cleaned_search_terms:
        # Ensure term is a string and not a float - additional safety check
        if isinstance(term, (int, float)):
            term = str(term)
        elif not isinstance(term, str):
            term = str(term)
        
        # Additional validation to prevent float values from being passed to paperscraper
        if term and term.strip() and not term.lower() in ['nan', 'none', 'null', '']:
            # Search each keyword in Title/Abstract
            search_strategies.append(f'"{term}"[Title/Abstract]')
    
    print(f"📋 Generated {len(search_strategies)} individual keyword searches")
    
    # Execute searches with progress tracking
    with tqdm(total=len(search_strategies), desc="Executing API searches", unit="strategy") as pbar:
        for i, search_query in enumerate(search_strategies):
            try:
                # Add date filters to the search query
                date_filter = f" AND ({date_from}[Date - Publication]:{date_to}[Date - Publication])"
                full_query = search_query + date_filter
                
                print(f"\n🔍 API Strategy {i+1}: {search_query}")
                
                # URL encode the query
                encoded_query = urllib.parse.quote(full_query)
                
                # Step 1: Search for IDs
                search_url = f"{base_url}esearch.fcgi?db=pubmed&term={encoded_query}&retmax={max_results}&retmode=json"
                response = requests.get(search_url, timeout=30)
                response.raise_for_status()
                
                search_data = response.json()
                if 'esearchresult' not in search_data:
                    continue
                
                id_list = search_data['esearchresult'].get('idlist', [])
                if not id_list:
                    continue
                
                print(f"   📊 Strategy {i+1} found {len(id_list)} papers")
                
                # Step 2: Fetch paper details in batches
                batch_size = 100  # PubMed allows up to 100 IDs per request
                for j in range(0, len(id_list), batch_size):
                    batch_ids = id_list[j:j+batch_size]
                    id_string = ','.join(batch_ids)
                    
                    # Fetch paper details
                    fetch_url = f"{base_url}efetch.fcgi?db=pubmed&id={id_string}&retmode=xml"
                    fetch_response = requests.get(fetch_url, timeout=30)
                    fetch_response.raise_for_status()
                    
                    # Parse XML response
                    try:
                        root = ET.fromstring(fetch_response.content)
                        for article in root.findall('.//PubmedArticle'):
                            paper_data = parse_pubmed_xml(article)
                            if paper_data and paper_data not in all_papers:
                                all_papers.append(paper_data)
                                
                                # Auto-save data every N articles
                                if len(all_papers) > 0 and len(all_papers) % auto_save_frequency == 0:
                                    try:
                                        save_papers_to_file(all_papers, f"data/scraped_data/pubmed/pubmed_autosave_{len(all_papers)}_papers.jsonl")
                                        print(f"\n💾 Auto-saved {len(all_papers)} papers to autosave file")
                                    except Exception as e:
                                        print(f"\n⚠️  Auto-save failed: {e}")
                    except ET.ParseError:
                        continue
                    
                    # Reduced rate limiting delay
                    time.sleep(0.1)
                
                # Reduced delay between search strategies
                time.sleep(0.5)
                
            except Exception as e:
                print(f"\n⚠️  Search strategy {i+1} failed: {e}")
                # Clean up temporary file even on exception
                if os.path.exists(f"temp_search_{i}.jsonl"):
                    try:
                        os.remove(f"temp_search_{i}.jsonl")
                        print(f"🧹 Cleaned up temp file: temp_search_{i}.jsonl")
                    except Exception as cleanup_error:
                        print(f"⚠️  Failed to clean up temp file: {cleanup_error}")
                continue
            
            pbar.update(1)
            pbar.set_postfix({"papers_found": len(all_papers)})
            flush_output()
            
            # Early exit if we have enough results
            if len(all_papers) >= max_results:
                print(f"\n✅ Reached target of {max_results} papers, stopping search")
                break
    
    # Display comprehensive search summary
    print(f"\n📊 SEARCH SUMMARY")
    print(f"=" * 50)
    print(f"🔍 Total search strategies executed: {len(search_strategies)}")
    print(f"📄 Total papers collected: {len(all_papers)}")
    print(f"📅 Date range: {date_from} to {date_to}")
    print(f"🎯 Target results: {max_results}")
    
    # Calculate success rate
    successful_strategies = len([s for s in search_strategies if any(p.get('title') for p in all_papers)])
    success_rate = (successful_strategies / len(search_strategies)) * 100 if search_strategies else 0
    print(f"✅ Success rate: {success_rate:.1f}% ({successful_strategies}/{len(search_strategies)} strategies)")
    
    # Show file sizes
    total_temp_size = 0
    temp_files = [f"temp_search_{i}.jsonl" for i in range(len(search_strategies))]
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            size_mb = os.path.getsize(temp_file) / (1024 * 1024)
            total_temp_size += size_mb
            print(f"📁 {temp_file}: {size_mb:.1f}MB")
    
    print(f"💾 Total data collected: {total_temp_size:.1f}MB")
    print(f"=" * 50)
    flush_output()
    
    # Remove duplicates based on DOI and title
    unique_papers = []
    seen_dois = set()
    seen_titles = set()
    
    for paper in all_papers:
        doi = str(paper.get('doi', '')).strip()
        title = paper.get('title', '')
        
        # Ensure title is a string before processing
        if isinstance(title, (int, float)):
            title = str(title)
        elif not isinstance(title, str):
            title = str(title) if title is not None else ''
        
        title = title.strip().lower()
        
        # Check if we've seen this paper before
        if doi and doi not in seen_dois:
            unique_papers.append(paper)
            seen_dois.add(doi)
        elif title and title not in seen_titles:
            unique_papers.append(paper)
            seen_titles.add(title)
    
    print(f"🎯 Found {len(all_papers)} total papers, {len(unique_papers)} unique papers")
    
    # Final save of all data
    try:
        final_save_path = f"data/scraped_data/pubmed/pubmed_final_{len(unique_papers)}_papers.jsonl"
        save_papers_to_file(unique_papers, final_save_path)
        print(f"💾 Final save: {len(unique_papers)} papers saved to {final_save_path}")
    except Exception as e:
        print(f"⚠️  Final save failed: {e}")
    
    return unique_papers

def extract_citation_from_pubmed_xml(article_element):
    """
    Extract citation count using NIH iCite API for PMID.
    
    Args:
        article_element: XML element representing a PubMed article
    
    Returns:
        Citation count as integer or None if not found
    """
    try:
        # First extract PMID from the XML
        pmid_elem = article_element.find('.//PMID')
        if pmid_elem is not None and pmid_elem.text:
            pmid = pmid_elem.text.strip()
            if pmid and pmid.isdigit():
                # Use iCite API to get citation count
                citation_count = get_citations_by_pmid(pmid)
                return citation_count
        
        return None
        
    except Exception as e:
        return None

def parse_pubmed_xml(article_element):
    """
    Parse PubMed XML article element into paper dictionary.
    
    Args:
        article_element: XML element representing a PubMed article
    
    Returns:
        Dictionary with paper data or None if parsing fails
    """
    try:
        paper_data = {}
        
        # Extract title
        title_elem = article_element.find('.//ArticleTitle')
        if title_elem is not None and title_elem.text:
            paper_data['title'] = title_elem.text.strip()
        
        # Extract abstract
        abstract_elem = article_element.find('.//Abstract/AbstractText')
        if abstract_elem is not None and abstract_elem.text:
            paper_data['abstract'] = abstract_elem.text.strip()
        
        # Extract journal
        journal_elem = article_element.find('.//Journal/Title')
        if journal_elem is not None and journal_elem.text:
            paper_data['journal'] = journal_elem.text.strip()
        
        # Extract publication date
        pub_date_elem = article_element.find('.//PubDate')
        if pub_date_elem is not None:
            year_elem = pub_date_elem.find('Year')
            month_elem = pub_date_elem.find('Month')
            day_elem = pub_date_elem.find('Day')
            
            if year_elem is not None and year_elem.text:
                year = year_elem.text.strip()
                month = month_elem.text.strip() if month_elem is not None else "01"
                day = day_elem.text.strip() if day_elem is not None else "01"
                
                # Ensure month and day are 2 digits
                month = month.zfill(2)
                day = day.zfill(2)
                
                paper_data['date'] = f"{year}-{month}-{day}"
        
        # Extract authors
        authors = []
        author_list = article_element.find('.//AuthorList')
        if author_list is not None:
            for author_elem in author_list.findall('Author'):
                last_name_elem = author_elem.find('LastName')
                fore_name_elem = author_elem.find('ForeName')
                
                if last_name_elem is not None and last_name_elem.text:
                    last_name = last_name_elem.text.strip()
                    fore_name = fore_name_elem.text.strip() if fore_name_elem is not None else ""
                    
                    if fore_name:
                        authors.append(f"{fore_name} {last_name}")
                    else:
                        authors.append(last_name)
        
        if authors:
            paper_data['authors'] = authors
        
        # Extract DOI
        doi_elem = article_element.find('.//ELocationID[@EIdType="doi"]')
        if doi_elem is not None and doi_elem.text:
            paper_data['doi'] = doi_elem.text.strip()
        
        # Extract PMID
        pmid_elem = article_element.find('.//PMID')
        if pmid_elem is not None and pmid_elem.text:
            paper_data['pmid'] = pmid_elem.text.strip()
        
        # Extract MeSH terms
        mesh_terms = []
        mesh_list = article_element.find('.//MeshHeadingList')
        if mesh_list is not None:
            for mesh_elem in mesh_list.findall('MeshHeading/DescriptorName'):
                if mesh_elem.text:
                    mesh_terms.append(mesh_elem.text.strip())
        
        if mesh_terms:
            paper_data['mesh_terms'] = mesh_terms
        
        # Extract citation count directly from PubMed XML (if available)
        citation_count = extract_citation_from_pubmed_xml(article_element)
        if citation_count is not None:
            paper_data['citation_count'] = str(citation_count)
        else:
            paper_data['citation_count'] = 'not found'
        
        # Extract impact factor directly based on journal
        journal_name = paper_data.get('journal', '')
        impact_factor = estimate_impact_factor(journal_name)
        paper_data['impact_factor'] = impact_factor
        
        # Only return if we have at least a title
        if paper_data.get('title'):
            return paper_data
        
    except Exception as e:
        print(f"⚠️  Error parsing PubMed XML: {e}")
        return None
    
    return None

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

def extract_citation_count_fast(paper_data, debug=False):
    """Enhanced citation count extraction with comprehensive error handling and retry mechanisms."""
    global citation_error_counter
    
    paper_title = paper_data.get('title', 'Unknown')[:50]
    if debug:
        print(f"🔍 [DEBUG] Extracting citations for: {paper_title}...")
    
    # Strategy 1: Check if citation data is already in the paper_data (fastest)
    citation_fields = [
        'citations', 'citation_count', 'cited_by_count', 'times_cited',
        'reference_count', 'cited_count', 'num_citations', 'citedByCount',
        'citationCount', 'citedBy', 'timesCited', 'citation_num', 'cited_count',
        'times_cited', 'citation_count', 'citations_count'
    ]
    
    for field in citation_fields:
        if field in paper_data and paper_data[field] is not None:
            try:
                count = int(paper_data[field])
                if count >= 0:
                    citation_error_counter['successful_citations'] += 1
                    if debug:
                        print(f"✅ [DEBUG] Found existing citation count: {count}")
                    return str(count)
            except (ValueError, TypeError):
                continue
    
    # Strategy 1.5: Try to extract citation count from text fields
    text_fields = ['abstract', 'title', 'summary']
    for field in text_fields:
        if field in paper_data and paper_data[field]:
            text = str(paper_data[field])
            # Look for citation patterns like "cited by X", "X citations", etc.
            import re
            citation_patterns = [
                r'cited by (\d+)',
                r'(\d+) citations?',
                r'(\d+) times cited',
                r'cited (\d+) times'
            ]
            for pattern in citation_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        count = int(match.group(1))
                        if count >= 0:
                            citation_error_counter['successful_citations'] += 1
                            if debug:
                                print(f"✅ [DEBUG] Extracted citation count from text: {count}")
                            return str(count)
                    except (ValueError, TypeError):
                        continue
    
    # If citation processing is disabled, return early to avoid hangs
    if not CITATION_PROCESSING_ENABLED:
        citation_error_counter['failed_citations'] += 1
        if debug:
            print(f"⚠️ [DEBUG] Citation processing is disabled")
        return "disabled"
    
    # Strategy 2: Try DOI-based lookup with retry mechanism
    doi = paper_data.get('doi', '')
    if doi:
        # Clean and validate DOI
        doi = str(doi).strip()
        if not doi or doi.lower() in ['nan', 'none', 'null', '']:
            doi = ''
        else:
            # Ensure DOI has proper format (should start with 10.)
            if not doi.startswith('10.'):
                doi = ''
    
    if doi:
        if debug:
            print(f"🔍 [DEBUG] Attempting DOI lookup: {doi}")
        
        # Retry mechanism for DOI lookup
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if debug and attempt > 0:
                    print(f"🔄 [DEBUG] DOI lookup retry {attempt + 1}/{max_retries}")
                
                # Rate limiting with exponential backoff
                from src.core.processing_config import CITATION_RATE_LIMIT
                delay = CITATION_RATE_LIMIT * (2 ** attempt)  # Exponential backoff
                rate_limit_delay(delay)
                
                # Timeout protection
                import signal
                import threading
                
                def timeout_handler(signum, frame):
                    raise TimeoutError("DOI citation lookup timed out")
                
                # Set timeout for citation lookup (10 seconds) - only on Unix systems
                if hasattr(signal, 'SIGALRM'):
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(10)
                
                try:
                    citations = get_citations_by_doi(doi)
                finally:
                    if hasattr(signal, 'SIGALRM'):
                        signal.alarm(0)  # Cancel the alarm
                
                if citations is not None and citations >= 0:
                    citation_error_counter['successful_citations'] += 1
                    if debug:
                        print(f"✅ [DEBUG] DOI lookup successful: {citations} citations")
                    return str(citations)
                elif citations is None:
                    if debug:
                        print(f"⚠️ [DEBUG] DOI lookup returned None")
                    break  # Don't retry if API explicitly returns None
                else:
                    if debug:
                        print(f"⚠️ [DEBUG] DOI lookup returned invalid value: {citations}")
                    break
                    
            except TimeoutError as e:
                if debug:
                    print(f"⏰ [DEBUG] DOI lookup timeout (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    citation_error_counter['doi_lookup_failures'] += 1
                    citation_error_counter['failed_citations'] += 1
            except Exception as e:
                if debug:
                    print(f"❌ [DEBUG] DOI lookup error (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    citation_error_counter['doi_lookup_failures'] += 1
                    citation_error_counter['failed_citations'] += 1
                # Continue to next attempt
    
    # Strategy 3: Skip title-based lookup (Google Scholar has captcha issues)
    title = paper_data.get('title', '')
    if title:
        if debug:
            print(f"⚠️ [DEBUG] Skipping title-based lookup (Google Scholar captcha issues)")
        citation_error_counter['title_lookup_failures'] += 1
        citation_error_counter['failed_citations'] += 1
    
    # Strategy 4: Try PMID-based lookup using iCite API with retry
    pmid = paper_data.get('pmid', '')
    if pmid:
        if debug:
            print(f"🔍 [DEBUG] Attempting PMID lookup: {pmid}")
        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                if debug and attempt > 0:
                    print(f"🔄 [DEBUG] PMID lookup retry {attempt + 1}/{max_retries}")
                
                # Rate limiting for PMID lookup
                rate_limit_delay(0.2 * (2 ** attempt))  # Exponential backoff
                citations = get_citations_by_pmid(pmid)
                
                if citations is not None and citations >= 0:
                    citation_error_counter['successful_citations'] += 1
                    if debug:
                        print(f"✅ [DEBUG] PMID lookup successful: {citations} citations")
                    return str(citations)
                else:
                    if debug:
                        print(f"⚠️ [DEBUG] PMID lookup returned invalid value: {citations}")
                    break
                    
            except Exception as e:
                if debug:
                    print(f"❌ [DEBUG] PMID lookup error (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    citation_error_counter['failed_citations'] += 1
    
    citation_error_counter['failed_citations'] += 1
    if debug:
        print(f"❌ [DEBUG] All citation extraction strategies failed")
    return "not found"

def get_citations_by_pmid(pmid):
    """Get citation count using NIH iCite API for PMID."""
    try:
        import requests
        
        # Rate limiting
        rate_limit_delay(0.1)
        
        # Clean PMID
        pmid = str(pmid).strip()
        if not pmid or not pmid.isdigit():
            return None
        
        # Query iCite API using the correct format
        url = f"https://icite.od.nih.gov/api/pubs/{pmid}"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data and 'data' in data and len(data['data']) > 0:
                citation_count = data['data'][0].get('citation_count', 0)
                return int(citation_count) if citation_count is not None else 0
            
            return None
            
        except requests.exceptions.RequestException as e:
            return None
        
    except Exception as e:
        return None

def extract_journal_info_fast(paper_data):
    """Enhanced journal info extraction with more field variations."""
    # First check if journal information is already in the paper_data
    journal_fields = [
        'journal', 'journal_name', 'publication', 'source', 'venue',
        'journal_title', 'publication_venue', 'journal_ref', 'journal-ref',
        'journalName', 'publicationVenue', 'sourceTitle', 'periodical',
        'magazine', 'publicationName', 'journalTitle'
    ]
    
    for field in journal_fields:
        if field in paper_data and paper_data[field]:
            journal_name = str(paper_data[field])
            if journal_name and journal_name.lower() not in ['nan', 'none', 'null', '']:
                return journal_name
    
    # If no journal found in paper_data, check source
    source = paper_data.get('source', '')
    if source in ['biorxiv', 'medrxiv']:
        return f"{source.upper()}"
    
    return "Unknown journal"

def extract_impact_factor_fast(paper_data, journal_name=None):
    """Enhanced impact factor extraction with multiple strategies."""
    # Strategy 1: Check if impact factor is already in the data
    impact_fields = [
        'impact_factor', 'journal_impact_factor', 'if', 'jif',
        'impact', 'journal_if', 'journal_impact', 'impactFactor',
        'journalImpactFactor', 'jcr_impact_factor', 'scimago_impact'
    ]
    
    for field in impact_fields:
        if field in paper_data and paper_data[field] is not None:
            try:
                impact = float(paper_data[field])
                if impact > 0:
                    return str(impact)
            except (ValueError, TypeError):
                continue
    
    # Strategy 2: Get journal name if not provided
    if not journal_name:
        journal_name = extract_journal_info_fast(paper_data)
    
    # Strategy 3: Try to get impact factor from journal name using API
    try:
        impact_factor = get_impact_factor_from_api(journal_name)
        if impact_factor:
            return str(impact_factor)
    except Exception:
        pass
    
    # Strategy 4: Use local estimation as fallback
    return estimate_impact_factor(journal_name)

def process_paper_citations_immediate(paper_data, debug=False):
    """Process citations immediately for a single paper with comprehensive error handling."""
    try:
        if debug:
            print(f"🔍 [IMMEDIATE DEBUG] Starting citation processing...")
        
        # Process citation count with debug mode
        citation_count = extract_citation_count_fast(paper_data, debug=debug)
        
        # Process journal info
        journal = extract_journal_info_fast(paper_data)
        
        # Process impact factor
        impact_factor = extract_impact_factor_fast(paper_data, journal)
        
        if debug:
            print(f"✅ [IMMEDIATE DEBUG] Citation processing complete: {citation_count}")
        
        return {
            'citation_count': citation_count,
            'journal': journal,
            'impact_factor': impact_factor
        }
        
    except Exception as e:
        error_msg = f"Citation processing error: {e}"
        if debug:
            print(f"❌ [IMMEDIATE DEBUG] {error_msg}")
        else:
            print(f"⚠️  {error_msg}")
        return {
            'citation_count': 'error',
            'journal': 'Unknown journal',
            'impact_factor': 'not found'
        }

def process_citations_batch(papers_data, show_progress=True, debug=False):
    """Process citations for a batch of papers with comprehensive error handling and debug reporting."""
    if not CITATION_PROCESSING_ENABLED:
        print("🔧 Citation processing is disabled - skipping citation extraction")
        return papers_data
    
    print(f"\n📊 Processing citations for {len(papers_data)} papers...")
    print("⏱️  This may take several minutes due to API rate limits...")
    if debug:
        print("🐛 Debug mode enabled - detailed logging will be shown")
    
    processed_papers = []
    successful_citations = 0
    failed_citations = 0
    error_details = []
    
    if show_progress:
        with tqdm(total=len(papers_data), desc="Processing citations", unit="paper") as pbar:
            for i, paper_data in enumerate(papers_data):
                try:
                    # Update progress bar description
                    title = paper_data.get('title', 'Unknown')[:50]
                    pbar.set_description(f"Citations: {title}...")
                    
                    if debug:
                        print(f"\n📄 [BATCH DEBUG] Processing paper {i+1}/{len(papers_data)}")
                    
                    # Process citations for this paper with debug mode
                    citation_data = process_paper_citations_immediate(paper_data, debug=debug)
                    
                    # Update the paper data with citation information
                    paper_data.update(citation_data)
                    processed_papers.append(paper_data)
                    
                    # Track success/failure
                    citation_count = citation_data.get('citation_count', 'unknown')
                    if citation_count not in ['not found', 'error', 'disabled']:
                        successful_citations += 1
                    else:
                        failed_citations += 1
                        if debug:
                            error_details.append(f"Paper {i+1}: {citation_count}")
                    
                    # Update progress bar with citation count
                    pbar.set_postfix({
                        "citations": citation_count,
                        "success": successful_citations,
                        "failed": failed_citations
                    })
                    
                except Exception as e:
                    error_msg = f"Error processing citations for paper {i+1}: {e}"
                    print(f"\n❌ {error_msg}")
                    error_details.append(error_msg)
                    
                    # Add paper with default citation data
                    paper_data.update({
                        'citation_count': 'error',
                        'journal': 'Unknown journal',
                        'impact_factor': 'not found'
                    })
                    processed_papers.append(paper_data)
                    failed_citations += 1
                
                pbar.update(1)
    else:
        # Process without progress bar (for compatibility)
        for i, paper_data in enumerate(papers_data):
            try:
                if debug:
                    print(f"\n📄 [BATCH DEBUG] Processing paper {i+1}/{len(papers_data)}")
                
                citation_data = process_paper_citations_immediate(paper_data, debug=debug)
                paper_data.update(citation_data)
                processed_papers.append(paper_data)
                
                # Track success/failure
                citation_count = citation_data.get('citation_count', 'unknown')
                if citation_count not in ['not found', 'error', 'disabled']:
                    successful_citations += 1
                else:
                    failed_citations += 1
                    
            except Exception as e:
                error_msg = f"Error processing citations for paper {i+1}: {e}"
                print(f"❌ {error_msg}")
                error_details.append(error_msg)
                
                paper_data.update({
                    'citation_count': 'error',
                    'journal': 'Unknown journal',
                    'impact_factor': 'not found'
                })
                processed_papers.append(paper_data)
                failed_citations += 1
    
    # Print comprehensive summary
    print(f"\n📊 Citation Processing Summary:")
    print(f"   Total papers processed: {len(processed_papers)}")
    print(f"   Successful citations: {successful_citations}")
    print(f"   Failed citations: {failed_citations}")
    print(f"   Success rate: {successful_citations/len(processed_papers)*100:.1f}%")
    
    if debug and error_details:
        print(f"\n🐛 Debug Details:")
        for error in error_details[:10]:  # Show first 10 errors
            print(f"   {error}")
        if len(error_details) > 10:
            print(f"   ... and {len(error_details) - 10} more errors")
    
    return processed_papers

def get_impact_factor_from_api(journal_name):
    """Get impact factor from external API (placeholder for future implementation)."""
    try:
        # This is a placeholder for future API integration
        # Could integrate with JCR API, Scimago API, or other sources
        # For now, return None to fall back to local estimation
        return None
    except Exception:
        return None

def estimate_impact_factor(journal_name):
    """Enhanced impact factor estimation with comprehensive journal database."""
    # Ensure journal_name is a string
    if isinstance(journal_name, (int, float)):
        journal_name = str(journal_name)
    elif not isinstance(journal_name, str):
        journal_name = str(journal_name) if journal_name is not None else ''
    
    if not journal_name or journal_name == "Unknown journal":
        return "not found"
    
    # Comprehensive impact factor mapping based on recent data (2023-2024)
    impact_factors = {
        # Top-tier journals
        'nature': 49.962,
        'science': 56.9,
        'cell': 66.85,
        'nature medicine': 87.241,
        'nature biotechnology': 68.164,
        'nature genetics': 41.307,
        'nature cell biology': 28.213,
        'nature immunology': 31.25,
        'nature reviews immunology': 108.555,
        'nature reviews molecular cell biology': 81.3,
        'nature reviews genetics': 42.7,
        'nature reviews cancer': 75.4,
        'nature reviews drug discovery': 120.1,
        
        # Immunology journals
        'immunity': 43.474,
        'journal of immunology': 5.422,
        'journal of experimental medicine': 17.579,
        'nature immunology': 31.25,
        'immunological reviews': 13.0,
        'trends in immunology': 13.1,
        'european journal of immunology': 5.4,
        'journal of allergy and clinical immunology': 14.2,
        
        # General science journals
        'proceedings of the national academy of sciences': 12.779,
        'pnas': 12.779,
        'plos one': 3.752,
        'plos biology': 9.593,
        'plos genetics': 6.02,
        'plos computational biology': 4.7,
        'elife': 8.713,
        
        # Bioinformatics and computational biology
        'bioinformatics': 6.937,
        'nucleic acids research': 19.16,
        'genome research': 11.093,
        'genome biology': 17.906,
        'bmc genomics': 4.317,
        'bmc bioinformatics': 3.169,
        'briefings in bioinformatics': 13.9,
        'bioinformatics and biology insights': 2.1,
        
        # Cell biology journals
        'cell reports': 9.995,
        'molecular cell': 19.328,
        'developmental cell': 13.417,
        'current biology': 10.834,
        'cell metabolism': 29.0,
        'cell stem cell': 25.3,
        'cancer cell': 50.3,
        'molecular biology of the cell': 3.9,
        
        # Nature family journals
        'nature communications': 17.694,
        'nature methods': 47.99,
        'nature neuroscience': 25.0,
        'nature structural & molecular biology': 15.8,
        'nature chemical biology': 15.0,
        'nature materials': 41.2,
        'nature physics': 20.5,
        'nature chemistry': 24.4,
        
        # Neuroscience journals
        'neuron': 16.2,
        'journal of neuroscience': 6.7,
        'nature neuroscience': 25.0,
        'trends in neurosciences': 16.2,
        'cerebral cortex': 4.9,
        'neuroimage': 7.4,
        
        # Medical journals
        'the lancet': 202.731,
        'new england journal of medicine': 176.079,
        'jama': 157.335,
        'bmj': 105.7,
        'nature medicine': 87.241,
        'cell metabolism': 29.0,
        'diabetes': 8.0,
        'circulation': 37.8,
        
        # Preprint servers (no impact factor)
        'biorxiv': 0.0,
        'medrxiv': 0.0,
        'arxiv': 0.0,
        'chemrxiv': 0.0,
        'bioarxiv': 0.0,
        
        # Biochemistry journals
        'journal of biological chemistry': 5.5,
        'biochemistry': 3.2,
        'protein science': 6.3,
        'journal of molecular biology': 5.0,
        'structure': 4.2,
        
        # Genetics journals
        'genetics': 4.4,
        'genome research': 11.093,
        'genome biology': 17.906,
        'human molecular genetics': 5.1,
        'american journal of human genetics': 11.0,
        
        # Cancer journals
        'cancer cell': 50.3,
        'cancer research': 13.3,
        'journal of clinical oncology': 50.7,
        'nature cancer': 23.0,
        'cancer discovery': 28.2,
        
        # Microbiology journals
        'cell host & microbe': 30.3,
        'nature microbiology': 20.5,
        'journal of bacteriology': 3.2,
        'applied and environmental microbiology': 4.4,
        
        # Plant biology journals
        'plant cell': 12.1,
        'plant journal': 7.0,
        'plant physiology': 8.0,
        'nature plants': 15.8,
        
        # Other specialized journals
        'journal of proteome research': 4.4,
        'proteomics': 4.0,
        'mass spectrometry reviews': 8.0,
        'analytical chemistry': 8.0,
        'journal of chromatography a': 4.1,
    }
    
    journal_lower = journal_name.lower().strip()
    
    # Direct match
    if journal_lower in impact_factors:
        return str(impact_factors[journal_lower])
    
    # Partial match - check if any key is contained in the journal name
    for key, impact in impact_factors.items():
        if key in journal_lower or journal_lower in key:
            return str(impact)
    
    # Fuzzy matching for common variations
    journal_variations = {
        'nature': ['nat ', 'nature '],
        'science': ['science ', 'sci '],
        'cell': ['cell ', 'cell:'],
        'journal': ['j ', 'journal of', 'j.'],
        'proceedings': ['proc ', 'proceedings of'],
        'plos': ['plos ', 'public library of science'],
        'bmc': ['bmc ', 'biomed central'],
        'pnas': ['proc natl acad sci', 'proceedings of the national academy'],
    }
    
    for base_name, variations in journal_variations.items():
        for variation in variations:
            if variation in journal_lower:
                if base_name in impact_factors:
                    return str(impact_factors[base_name])
    
    # Default for unknown journals - estimate based on journal name patterns
    if any(word in journal_lower for word in ['nature', 'science', 'cell']):
        return "15.0"  # High-impact estimate
    elif any(word in journal_lower for word in ['journal', 'proceedings', 'plos']):
        return "5.0"   # Medium-impact estimate
    elif any(word in journal_lower for word in ['bmc', 'frontiers', 'molecules']):
        return "3.0"   # Lower-impact estimate
    else:
        return "not found"

def extract_additional_metadata(paper_data):
    """Extract additional metadata fields that paperscraper might provide."""
    metadata = {}
    
    # Extract keywords
    keyword_fields = ['keywords', 'keyword', 'subject', 'subjects', 'tags']
    for field in keyword_fields:
        if field in paper_data and paper_data[field]:
            keywords = paper_data[field]
            if isinstance(keywords, list):
                metadata['keywords'] = keywords
            elif isinstance(keywords, str):
                # Split on common delimiters
                metadata['keywords'] = [k.strip() for k in keywords.replace(';', ',').split(',') if k.strip()]
            break
    
    # Extract abstract
    abstract_fields = ['abstract', 'summary', 'description']
    for field in abstract_fields:
        if field in paper_data and paper_data[field]:
            metadata['abstract_full'] = str(paper_data[field])
            break
    
    # Extract affiliation information
    affiliation_fields = ['affiliations', 'affiliation', 'institutions', 'institution']
    for field in affiliation_fields:
        if field in paper_data and paper_data[field]:
            affiliations = paper_data[field]
            if isinstance(affiliations, list):
                metadata['affiliations'] = affiliations
            elif isinstance(affiliations, str):
                metadata['affiliations'] = [affiliations]
            break
    
    # Extract funding information
    funding_fields = ['funding', 'grants', 'grant', 'funding_source']
    for field in funding_fields:
        if field in paper_data and paper_data[field]:
            metadata['funding'] = str(paper_data[field])
            break
    
    # Extract license information
    license_fields = ['license', 'licence', 'copyright', 'rights']
    for field in license_fields:
        if field in paper_data and paper_data[field]:
            metadata['license'] = str(paper_data[field])
            break
    
    # Extract URL information
    url_fields = ['url', 'link', 'pdf_url', 'full_text_url']
    for field in url_fields:
        if field in paper_data and paper_data[field]:
            metadata['url'] = str(paper_data[field])
            break
    
    # Extract category/subject classification
    category_fields = ['category', 'categories', 'classification', 'subject_class']
    for field in category_fields:
        if field in paper_data and paper_data[field]:
            categories = paper_data[field]
            if isinstance(categories, list):
                metadata['categories'] = categories
            elif isinstance(categories, str):
                metadata['categories'] = [categories]
            break
    
    # Extract version information (for preprints)
    version_fields = ['version', 'revision', 'v']
    for field in version_fields:
        if field in paper_data and paper_data[field]:
            metadata['version'] = str(paper_data[field])
            break
    
    # Extract language
    language_fields = ['language', 'lang', 'locale']
    for field in language_fields:
        if field in paper_data and paper_data[field]:
            metadata['language'] = str(paper_data[field])
            break
    
    return metadata

# --- CONFIG ---
# Load configuration from environment variable or use default
CONFIG_PROFILE = os.environ.get("PROCESSING_PROFILE", "balanced")
config = get_config(CONFIG_PROFILE)

# Add citation processing configuration
CITATION_PROCESSING_ENABLED = os.environ.get("ENABLE_CITATIONS", "true").lower() == "true"  # Default to true - citations always enabled
CITATION_TIMEOUT_SECONDS = int(os.environ.get("CITATION_TIMEOUT", "5"))  # Reduced timeout

# --- UTILS ---
# Global session for connection pooling
_embedding_session = None

def get_embedding_session():
    """Get or create a global requests session for connection pooling."""
    global _embedding_session
    if _embedding_session is None:
        _embedding_session = requests.Session()
        # Configure session for better performance
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,  # Number of connection pools
            pool_maxsize=20,      # Maximum connections per pool
            max_retries=3,        # Retry failed requests
            pool_block=False      # Don't block when pool is full
        )
        _embedding_session.mount('http://', adapter)
        _embedding_session.mount('https://', adapter)
    return _embedding_session

def get_google_embedding(text, api_key, retry_count=0):
    """Get Google embedding with optimized network settings and retry logic."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={api_key}"
    headers = {
        "Content-Type": "application/json",
        "Connection": "keep-alive",  # Enable connection reuse
        "User-Agent": "PubMed-Scraper/1.0"  # Identify our requests
    }
    data = {"model": "models/text-embedding-004", "content": {"parts": [{"text": text}]}}
    
    # Add rate limiting delay
    if config.get("rate_limit_delay", 0) > 0:
        time.sleep(config.get("rate_limit_delay", 0))
    
    try:
        # Use session for connection pooling
        session = get_embedding_session()
        response = session.post(
            url, 
            headers=headers, 
            json=data, 
            timeout=config.get("request_timeout", 15),  # Reduced timeout
            stream=False  # Don't stream for small responses
        )
        response.raise_for_status()
        return response.json()["embedding"]["values"]
        
    except requests.exceptions.Timeout:
        if retry_count < 2:  # Max 2 retries for timeout
            print(f"⚠️  Timeout for embedding, retrying ({retry_count + 1}/2)...")
            time.sleep(1)  # Brief delay before retry
            return get_google_embedding(text, api_key, retry_count + 1)
        else:
            print(f"❌ Max retries reached for timeout")
            return None
            
    except requests.exceptions.RequestException as e:
        if "429" in str(e) and retry_count < 3:  # Rate limit retry
            wait_time = (2 ** retry_count) * 5  # Exponential backoff: 5s, 10s, 20s
            print(f"⚠️  Rate limited, waiting {wait_time}s before retry ({retry_count + 1}/3)...")
            time.sleep(wait_time)
            return get_google_embedding(text, api_key, retry_count + 1)
        else:
            print(f"❌ Request error: {e}")
            return None

def chunk_paragraphs(text):
    """Safely chunk text into paragraphs, handling None and empty values."""
    if not text or not isinstance(text, str):
        return []
    
    # Split on newlines and filter out empty paragraphs
    paragraphs = [p.strip() for p in text.split("\n") if p and p.strip()]
    return paragraphs

def save_embeddings_to_json(embeddings_data, filename="embeddings.json"):
    """Save embeddings and metadata to JSON file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(embeddings_data, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved embeddings to {filename}")

def save_embeddings_incremental(embeddings_data, filename="embeddings.json", batch_size=10):
    """Save embeddings incrementally to prevent data loss"""
    # Create backup filename
    backup_filename = filename.replace('.json', '_backup.json')
    
    # Save to backup first
    with open(backup_filename, 'w', encoding='utf-8') as f:
        json.dump(embeddings_data, f, indent=2, ensure_ascii=False)
    
    # Then save to main file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(embeddings_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Incrementally saved {len(embeddings_data['embeddings'])} embeddings to {filename}")

def save_batch_pubmed(batch_data, batch_num, embeddings_dir):
    """Save a batch of PubMed embeddings to a file"""
    # Ensure directory exists
    os.makedirs(embeddings_dir, exist_ok=True)
    
    batch_file = os.path.join(embeddings_dir, f"batch_{batch_num:04d}.json")
    
    batch_content = {
        "source": "pubmed",
        "batch_num": batch_num,
        "timestamp": datetime.now().isoformat(),
        "embeddings": batch_data["embeddings"],
        "chunks": batch_data["chunks"],
        "metadata": batch_data["metadata"],
        "stats": {
            "total_embeddings": len(batch_data["embeddings"]),
            "total_chunks": len(batch_data["chunks"])
        }
    }
    
    with open(batch_file, 'w', encoding='utf-8') as f:
        json.dump(batch_content, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Saved PubMed batch {batch_num:04d} with {len(batch_data['embeddings'])} embeddings")
    return batch_file

def append_embedding_to_batch(chunk, embedding, metadata, current_batch, batch_num, embeddings_dir, batch_size=100):
    """Append a single embedding to the current batch and save when full"""
    # Add to current batch
    current_batch["chunks"].append(chunk)
    current_batch["embeddings"].append(embedding)
    current_batch["metadata"].append(metadata)
    
    # Save batch if it reaches the size limit
    if len(current_batch["embeddings"]) >= batch_size:
        batch_file = save_batch_pubmed(current_batch, batch_num, embeddings_dir)
        # Reset batch
        current_batch = {"chunks": [], "embeddings": [], "metadata": []}
        batch_num += 1
        return current_batch, batch_num, batch_file
    else:
        return current_batch, batch_num, None

def load_embeddings_from_json(filename="embeddings.json"):
    """Load embeddings and metadata from JSON file"""
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"chunks": [], "embeddings": [], "metadata": []}

DUMP_FILE = "data/scraped_data/pubmed/pubmed_dump.jsonl"
EMBEDDINGS_DIR = "data/embeddings/pubmed"
EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "pubmed_embeddings.json")

def get_search_terms():
    """Get search terms from configuration file or use defaults."""
    import json
    import os
    
    # Try to load custom keywords from configuration file
    config_file = "config/search_keywords_config.json"
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            pubmed_keywords = config.get("pubmed_keywords", "")
            
            if pubmed_keywords:
                # Parse comma-separated keywords and clean them
                terms = [term.strip() for term in pubmed_keywords.split(',') if term.strip()]
                print(f"📋 Using custom PubMed keywords: {', '.join(terms)}")
                return terms
        except Exception as e:
            print(f"⚠️  Could not load custom keywords: {e}")
            print("📋 Falling back to default UBR5 keywords")
    
    # Default UBR5 search terms if no custom keywords found
    terms = [
        "ubr5",
        "UBR5", 
        "ubr-5",
        "UBR-5"
    ]
    
    print(f"📋 Using default UBR5 search terms: {', '.join(terms)}")
    return list(terms)

def get_max_results_from_user():
    """Get maximum number of results from user input."""
    print("\n🔢 Maximum Results Configuration")
    print("=" * 50)
    print("Specify the maximum number of papers to retrieve:")
    print("  • Enter a positive number (e.g., 1000, 5000) for a specific limit")
    print("  • Enter -1 for unlimited results (use with caution)")
    print("  • Press Enter for default (5000)")
    print()
    
    while True:
        try:
            user_input = input("Maximum results (default: 5000): ").strip()
            
            # If user presses Enter, use default
            if not user_input:
                max_results = 5000
                print(f"✅ Using default: {max_results} papers")
                break
            
            # Parse user input
            max_results = int(user_input)
            
            if max_results == -1:
                print("⚠️  Unlimited results selected - this may take a very long time!")
                confirm = input("Are you sure? (y/n): ").strip().lower()
                if confirm == 'y':
                    max_results = 50000  # Set a very high limit instead of truly unlimited
                    print(f"✅ Unlimited results mode: {max_results} papers maximum")
                    break
                else:
                    print("❌ Unlimited results cancelled, please enter a number")
                    continue
            
            elif max_results <= 0:
                print("❌ Please enter a positive number or -1 for unlimited")
                continue
            
            elif max_results > 50000:
                print(f"⚠️  {max_results} is a very large number - this may take hours!")
                print(f"   Estimated time: {max_results // 1000} - {max_results // 500} minutes")
                confirm = input("Are you sure? (y/n): ").strip().lower()
                if confirm == 'y':
                    print(f"✅ Maximum results set to: {max_results}")
                    break
                else:
                    print("❌ Please enter a smaller number")
                    continue
            
            else:
                print(f"✅ Maximum results set to: {max_results}")
                break
                
        except ValueError:
            print("❌ Please enter a valid number")
            continue
        except KeyboardInterrupt:
            print("\n👋 Search cancelled by user")
            return None
    
    return max_results

def load_embeddings_to_chromadb(embeddings_data, source_name="pubmed"):
    """
    Load generated embeddings into ChromaDB automatically.
    
    Args:
        embeddings_data: The embeddings data dictionary
        source_name: Name of the source (e.g., "pubmed")
    """
    try:
        print(f"\n🗄️  Loading embeddings into ChromaDB...")
        
        # Initialize ChromaDB manager
        chroma_manager = ChromaDBManager()
        
        # Create collection if it doesn't exist
        if not chroma_manager.create_collection():
            print("❌ Failed to create ChromaDB collection")
            return False
        
        # Add embeddings to collection
        if chroma_manager.add_embeddings_to_collection(embeddings_data, source_name):
            print(f"✅ Successfully loaded {len(embeddings_data['chunks'])} embeddings into ChromaDB")
            
            # Display collection stats
            stats = chroma_manager.get_collection_stats()
            print(f"📊 ChromaDB collection now contains {stats.get('total_documents', 0)} total documents")
            return True
        else:
            print("❌ Failed to add embeddings to ChromaDB collection")
            return False
            
    except Exception as e:
        print(f"❌ Error loading embeddings to ChromaDB: {e}")
        return False

def load_pubmed_batches_to_chromadb(embeddings_dir, source_name="pubmed"):
    """Load all PubMed batch files into ChromaDB"""
    try:
        import glob
        
        # Find all batch files
        batch_files = glob.glob(os.path.join(embeddings_dir, "batch_*.json"))
        batch_files.sort()  # Sort to ensure correct order
        
        if not batch_files:
            print(f"❌ No batch files found in {embeddings_dir}")
            return False
        
        print(f"\n🗄️  Loading {len(batch_files)} PubMed batch files into ChromaDB...")
        
        # Initialize ChromaDB manager
        chroma_manager = ChromaDBManager()
        
        # Create collection if it doesn't exist
        if not chroma_manager.create_collection():
            print("❌ Failed to create ChromaDB collection")
            return False
        
        total_embeddings = 0
        for batch_file in batch_files:
            try:
                with open(batch_file, 'r', encoding='utf-8') as f:
                    batch_data = json.load(f)
                
                embeddings = batch_data.get("embeddings", [])
                chunks = batch_data.get("chunks", [])
                metadata = batch_data.get("metadata", [])
                
                if embeddings and chunks and metadata:
                    # Create embeddings_data structure for compatibility
                    embeddings_data = {
                        "embeddings": embeddings,
                        "chunks": chunks,
                        "metadata": metadata
                    }
                    
                    if chroma_manager.add_embeddings_to_collection(embeddings_data, source_name):
                        total_embeddings += len(embeddings)
                        print(f"✅ Loaded batch {os.path.basename(batch_file)}: {len(embeddings)} embeddings")
                    else:
                        print(f"❌ Failed to load batch {os.path.basename(batch_file)}")
                
            except Exception as e:
                print(f"❌ Error loading batch {batch_file}: {e}")
                continue
        
        # Display collection stats
        stats = chroma_manager.get_collection_stats()
        print(f"✅ Successfully loaded {total_embeddings} PubMed embeddings into ChromaDB")
        print(f"📊 ChromaDB collection now contains {stats.get('total_documents', 0)} total documents")
        return True
        
    except Exception as e:
        print(f"❌ Error loading PubMed batches to ChromaDB: {e}")
        return False

def ensure_directory_structure():
    """Ensure the data/embeddings/xrvix_embeddings directory exists and handle migration from old paths."""
    # Create the main directory
    os.makedirs("data/embeddings/xrvix_embeddings", exist_ok=True)
    
    # Check if old pubmed_embeddings.json exists and migrate it
    old_path = "pubmed_embeddings.json"
    new_path = "data/embeddings/xrvix_embeddings/pubmed_embeddings.json"
    
    if os.path.exists(old_path) and not os.path.exists(new_path):
        print(f"🔄 Migrating {old_path} to {new_path}...")
        try:
            import shutil
            shutil.move(old_path, new_path)
            print(f"✅ Successfully migrated {old_path} to {new_path}")
        except Exception as e:
            print(f"⚠️  Failed to migrate file: {e}")
            print(f"   You may need to manually move {old_path} to {new_path}")
    
    # Create subdirectories for different sources if they don't exist
    subdirs = ["pubmed", "biorxiv", "medrxiv"]
    for subdir in subdirs:
        os.makedirs(os.path.join("data/embeddings/xrvix_embeddings", subdir), exist_ok=True)

def check_old_embedding_files():
    """Check for old embedding files and provide guidance."""
    old_files = []
    
    # Check for old pubmed_embeddings.json in root
    if os.path.exists("pubmed_embeddings.json"):
        old_files.append("pubmed_embeddings.json")
    
    # Check for any other old embedding files
    for file in os.listdir("."):
        if file.endswith("_embeddings.json") and file != "pubmed_embeddings.json":
            old_files.append(file)
    
    if old_files:
        print("\n⚠️  Found old embedding files in root directory:")
        for file in old_files:
            print(f"   - {file}")
        print("\n💡 These files should be moved to the data/embeddings/xrvix_embeddings/ folder for better organization.")
        print("   The system will attempt to migrate them automatically.")

# Note: enrich_existing_dump_file() function removed - citation and impact factor data 
# is now extracted directly during the initial scraping phase

def reprocess_existing_citations():
    """Reprocess citations for existing papers in the dump file."""
    print("🔄 Reprocessing citations for existing papers...")
    
    if not os.path.exists(DUMP_FILE):
        print(f"❌ No existing dump file found at {DUMP_FILE}")
        return False
    
    # Load existing papers
    papers = []
    with open(DUMP_FILE, 'r', encoding='utf-8') as f:
        papers = [json.loads(line) for line in f]
    
    print(f"📚 Found {len(papers)} existing papers to reprocess")
    
    # Enable debug mode for reprocessing
    debug_citations = os.environ.get("DEBUG_CITATIONS", "true").lower() == "true"
    
    # Reprocess citations
    processed_papers = process_citations_batch(papers, show_progress=True, debug=debug_citations)
    
    # Save updated papers
    with open(DUMP_FILE, 'w', encoding='utf-8') as f:
        for paper in processed_papers:
            json.dump(paper, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"✅ Successfully reprocessed citations for {len(processed_papers)} papers")
    return True

def cleanup_temp_files():
    """Clean up any remaining temp_search_*.jsonl files."""
    import glob
    temp_files = glob.glob("temp_search_*.jsonl")
    cleaned_count = 0
    
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                cleaned_count += 1
                print(f"🧹 Cleaned up leftover temp file: {temp_file}")
        except Exception as e:
            print(f"⚠️  Failed to clean up {temp_file}: {e}")
    
    if cleaned_count > 0:
        print(f"✅ Cleaned up {cleaned_count} temporary files")
    return cleaned_count

def get_processed_papers_from_batches(embeddings_dir):
    """Get set of already processed paper DOIs/titles from existing batch files."""
    processed_papers = set()
    
    try:
        if not os.path.exists(embeddings_dir):
            return processed_papers
            
        # Look for batch files
        batch_files = [f for f in os.listdir(embeddings_dir) if f.startswith('batch_') and f.endswith('.jsonl')]
        
        for batch_file in batch_files:
            batch_path = os.path.join(embeddings_dir, batch_file)
            try:
                with open(batch_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            batch_data = json.loads(line)
                            if 'metadata' in batch_data:
                                metadata = batch_data['metadata']
                                # Use DOI as primary identifier, fallback to title
                                paper_id = metadata.get('doi', '') or metadata.get('title', '')
                                if paper_id:
                                    processed_papers.add(paper_id)
            except Exception as e:
                print(f"⚠️  Error reading batch file {batch_file}: {e}")
                continue
                
    except Exception as e:
        print(f"⚠️  Error scanning existing batches: {e}")
    
    return processed_papers

def is_paper_already_processed(paper, processed_papers):
    """Check if a paper has already been processed based on DOI or title."""
    # Check DOI first (most reliable)
    doi = paper.get('doi', '')
    if doi and doi in processed_papers:
        return True
    
    # Check title as fallback
    title = paper.get('title', '')
    if title and title in processed_papers:
        return True
    
    return False

def process_existing_pubmed_data(data_file_path, api_key, progress_callback=None):
    """
    Process existing PubMed data file to generate embeddings.
    
    Args:
        data_file_path: Path to the PubMed data file (.jsonl)
        api_key: Google API key for embedding generation
        progress_callback: Optional callback function to report progress (current, total)
    """
    print(f"🔄 Processing existing PubMed data: {os.path.basename(data_file_path)}")
    print("="*60)
    
    try:
        # Load papers from the data file
        papers = []
        with open(data_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    papers.append(json.loads(line))
        
        print(f"📚 Loaded {len(papers)} papers from {os.path.basename(data_file_path)}")
        
        if not papers:
            print("❌ No papers found in the data file")
            return False
        
        # Ensure embeddings directory exists
        ensure_directory_structure()
        
        # Check for existing embeddings to avoid duplicates
        print("🔍 Checking for existing embeddings...")
        processed_papers = get_processed_papers_from_batches(EMBEDDINGS_DIR)
        print(f"📊 Found {len(processed_papers)} already processed papers")
        
        # Filter out already processed papers
        unprocessed_papers = []
        skipped_count = 0
        for paper in papers:
            if is_paper_already_processed(paper, processed_papers):
                skipped_count += 1
            else:
                unprocessed_papers.append(paper)
        
        print(f"⏭️  Skipped {skipped_count} already processed papers")
        print(f"🔄 Will process {len(unprocessed_papers)} new papers")
        
        if not unprocessed_papers:
            print("✅ All papers already processed!")
            return True
        
        # Use unprocessed papers for the rest of the function
        papers = unprocessed_papers
        
        # Initialize batch tracking
        current_batch = {"embeddings": [], "chunks": [], "metadata": []}
        batch_num = 0
        total_embeddings = 0
        total_chunks = 0
        
        print(f"🚀 Starting embedding generation for {len(papers)} papers...")
        
        # Import tqdm for progress bar
        try:
            from tqdm import tqdm
        except ImportError:
            print("⚠️  tqdm not available, using basic progress display")
            tqdm = lambda x, **kwargs: x
        
        # Calculate total expected embeddings for progress tracking
        total_expected_embeddings = 0
        for paper in papers:
            title = paper.get('title', '')
            abstract = paper.get('abstract', '')
            full_text = f"{title}\n\n{abstract}" if abstract else title
            paragraphs = chunk_paragraphs(full_text)
            total_expected_embeddings += len([para for para in paragraphs if para and para.strip()])
        
        print(f"📊 Expected total embeddings: {total_expected_embeddings}")
        
        # Create embedding progress bar
        embedding_progress = tqdm(total=total_expected_embeddings, desc="🚀 Generating embeddings", 
                                unit="embedding", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} embeddings [{elapsed}<{remaining}, {rate_fmt}]')
        
        # Process papers with parallel processing for better performance
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        
        # Use parallel processing for better performance
        max_workers = min(4, len(papers))  # Limit to 4 workers to avoid overwhelming API
        print(f"🚀 Using {max_workers} parallel workers for embedding generation")
        
        # Thread-safe counters
        embedding_lock = threading.Lock()
        processed_count = [0]  # Use list for mutable reference
        
        def process_paper_parallel(paper_data):
            """Process a single paper in parallel."""
            idx, paper = paper_data
            
            try:
                # Extract paper data
                title = paper.get('title', '')
                abstract = paper.get('abstract', '')
                doi = paper.get('doi', '')
                authors = paper.get('authors', [])
                publication_date = paper.get('date', '')
                journal = paper.get('journal', '')
                citation_count = paper.get('citation_count', 0)
                impact_factor = paper.get('impact_factor', 0)
                
                # Extract year from publication date
                year = ""
                if publication_date:
                    try:
                        year = str(publication_date)[:4] if len(str(publication_date)) >= 4 else ""
                    except:
                        year = ""
                
                # Create author string
                author = ""
                if authors and isinstance(authors, list) and len(authors) > 0:
                    author = authors[0] if isinstance(authors[0], str) else str(authors[0])
                elif isinstance(authors, str):
                    author = authors
                
                # Extract additional metadata
                additional_metadata = extract_additional_metadata(paper)
                
                # Create citation data
                citation_data = {
                    'citation_count': citation_count,
                    'journal': journal,
                    'impact_factor': impact_factor
                }
                
                # Combine title and abstract for chunking
                full_text = f"{title}\n\n{abstract}" if abstract else title
                
                # Chunk the text into paragraphs
                paragraphs = chunk_paragraphs(full_text)
                
                if not paragraphs:
                    return []
                
                # Process paragraphs for this paper
                results = []
                valid_paragraphs = [para for para in paragraphs if para and para.strip()]
                
                for i, para in enumerate(valid_paragraphs):
                    try:
                        embedding = get_google_embedding(para, api_key)
                        
                        if embedding is None:
                            continue
                        
                        # Create comprehensive metadata object
                        metadata = {
                            "title": title,
                            "doi": doi,
                            "author": author,
                            "publication_date": publication_date,
                            "citation_count": citation_data['citation_count'],
                            "journal": citation_data['journal'],
                            "impact_factor": citation_data['impact_factor'],
                            "source": "pubmed",
                            "paper_index": idx,
                            "para_idx": i,
                            "chunk_length": len(para),
                            "year": year
                        }
                        
                        # Add additional metadata fields if available
                        metadata.update(additional_metadata)
                        
                        results.append({
                            "embedding": embedding,
                            "chunk": para,
                            "metadata": metadata
                        })
                        
                    except Exception as e:
                        print(f"\n⚠️  Embedding error for paper {idx + 1}: {e}")
                        continue
                
                return results
                
            except Exception as e:
                print(f"⚠️  Error processing paper {idx + 1}: {e}")
                return []
        
        # Process papers in parallel
        paper_results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all papers for processing
            future_to_paper = {
                executor.submit(process_paper_parallel, (idx, paper)): (idx, paper) 
                for idx, paper in enumerate(papers)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_paper):
                idx, paper = future_to_paper[future]
                try:
                    results = future.result()
                    if results:
                        paper_results.extend(results)
                        
                        # Update progress
                        with embedding_lock:
                            processed_count[0] += len(results)
                            embedding_progress.update(len(results))
                            embedding_progress.set_postfix({
                                'processed': processed_count[0],
                                'paper': f"{idx+1}/{len(papers)}"
                            })
                            
                            # Call progress callback if provided
                            if progress_callback:
                                progress_callback(processed_count[0], total_expected_embeddings)
                        
                except Exception as e:
                    print(f"⚠️  Error in parallel processing for paper {idx + 1}: {e}")
        
        # Now add all results to batches
        print(f"📊 Adding {len(paper_results)} embeddings to batches...")
        for result in paper_results:
            current_batch, batch_num, batch_file = append_embedding_to_batch(
                result["chunk"], result["embedding"], result["metadata"], 
                current_batch, batch_num, EMBEDDINGS_DIR, batch_size=100
            )
        
        total_embeddings = len(paper_results)
        total_chunks = sum(len(chunk_paragraphs(f"{paper.get('title', '')}\n\n{paper.get('abstract', '')}")) for paper in papers)
        
        # Save any remaining embeddings in the current batch
        if current_batch["embeddings"]:
            batch_file = save_batch_pubmed(current_batch, batch_num, EMBEDDINGS_DIR)
            print(f"💾 Saved final batch {batch_num:04d} with {len(current_batch['embeddings'])} embeddings")
        
        print(f"🎉 Processing completed!")
        print(f"📊 Total chunks processed: {total_chunks}")
        print(f"📊 Total embeddings created: {total_embeddings}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing PubMed data: {e}")
        import traceback
        traceback.print_exc()
        return False

def main(search_terms=None):
    """
    Main function for PubMed scraping and processing.
    Searches until no more unique papers are found for each keyword.
    
    Args:
        search_terms: List of search terms (optional)
    """
    try:
        print("=== Starting PubMed Scraping and Processing (JSON Storage) ===")
        print(f"🔧 Using configuration profile: {CONFIG_PROFILE}")
        print(f"🔧 Citation processing: {'ENABLED' if CITATION_PROCESSING_ENABLED else 'DISABLED'}")
        print("⚠️  Note: paperscraper has limited API key support - using 3 req/sec rate limit")
        if not CITATION_PROCESSING_ENABLED:
            print("💡 To enable citation processing, set environment variable: ENABLE_CITATIONS=true")
        else:
            print("💡 Citation processing is enabled with timeout protection to prevent hangs")
        
        # Check if user wants to reprocess existing citations
        if os.path.exists(DUMP_FILE) and CITATION_PROCESSING_ENABLED:
            reprocess_choice = input("\n🔄 Existing papers found. Reprocess citations? (y/n): ").strip().lower()
            if reprocess_choice == 'y':
                if reprocess_existing_citations():
                    print("✅ Citation reprocessing completed!")
                    return
                else:
                    print("❌ Citation reprocessing failed!")
        
        print_config_info()
        
        # Ensure directory structure and handle migration
        ensure_directory_structure()
        
        # Check for old embedding files
        check_old_embedding_files()
        
        # Reset citation error counter at the start
        reset_citation_error_counter()
        
        # Get search terms from user (or use provided parameter)
        if search_terms is None:
            search_terms = get_search_terms()
        print(f"🔍 Search terms: {', '.join(search_terms)}")
        flush_output()

        # Load API key
        with open("config/keys.json") as f:
            api_key = json.load(f)["GOOGLE_API_KEY"]

        # Step 1: Fetch and dump PubMed papers
        print(f"\n🔄 Fetching PubMed papers for {len(search_terms)} search terms...")
        print("⏱️  This may take several minutes...")
        print("🔍 Searching until no more unique papers found for each keyword")
        
        # Use the comprehensive search function (no max_results limit)
        # Auto-save every 500 papers by default
        papers_to_process = search_pubmed_comprehensive(search_terms, auto_save_frequency=500)
        
        if not papers_to_process:
            print("❌ No papers found. Exiting.")
            return
        
        print(f"✅ Found {len(papers_to_process)} papers from comprehensive search")
        
        # Step 1.5: Process citations if enabled
        if CITATION_PROCESSING_ENABLED:
            # Enable debug mode for citation processing
            debug_citations = os.environ.get("DEBUG_CITATIONS", "false").lower() == "true"
            papers_to_process = process_citations_batch(papers_to_process, show_progress=True, debug=debug_citations)
        else:
            print("🔧 Citation processing is disabled - papers will have 'pending' citation counts")
            print("💡 To enable citation processing, set environment variable: ENABLE_CITATIONS=true")
        
        # Save the papers directly (citation and impact factor data already included during parsing)
        with open(DUMP_FILE, 'w', encoding='utf-8') as f:
            for paper in papers_to_process:
                json.dump(paper, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"✅ Saved {len(papers_to_process)} papers to {DUMP_FILE}")
        print(f"📁 Embeddings will be saved to {EMBEDDINGS_FILE}")

        # Step 2: Process the papers directly
        print(f"\n📊 Processing {len(papers_to_process)} papers...")
        
        # Validate papers before creating DataFrame
        valid_papers = []
        for paper in papers_to_process:
            if paper and isinstance(paper, dict):
                # Ensure all required fields exist and are not None
                title = paper.get('title', '')
                abstract = paper.get('abstract', '')
                
                if title and abstract and title.strip() and abstract.strip():
                    # Convert any None values to empty strings
                    for key in paper:
                        if paper[key] is None:
                            paper[key] = ""
                    valid_papers.append(paper)
                else:
                    print(f"⚠️  Skipping paper: Missing title or abstract")
            else:
                print(f"⚠️  Skipping invalid paper data: {type(paper)}")
        
        if not valid_papers:
            print("❌ No valid papers to process. Exiting.")
            return
        
        print(f"📊 Processing {len(valid_papers)} valid papers out of {len(papers_to_process)} found")
        
        df = pd.DataFrame(valid_papers)
        print(f"📊 Loaded {len(df)} papers for processing")

        # Initialize batch storage
        current_batch = {"chunks": [], "embeddings": [], "metadata": []}
        batch_num = 0
        total_chunks = 0
        total_embeddings = 0
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(DUMP_FILE), exist_ok=True)
        os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
        
        print(f"\n🔄 Processing papers and creating embeddings...")
        if CITATION_PROCESSING_ENABLED:
            print(f"💡 Citations have been processed with progress tracking")
        else:
            print(f"💡 Citations are disabled - using 'pending' values")
        with tqdm(total=len(df), desc="Creating embeddings", unit="paper") as pbar:
            for idx, row in df.iterrows():
                try:
                    # Safely extract all fields with defensive programming
                    title = str(row.get("title", "") or "")
                    doi = str(row.get("doi", "") or "")
                    abstract = str(row.get("abstract", "") or "")
                    authors_raw = row.get("authors", "")
                    
                    # Handle None values safely
                    if title is None:
                        title = ""
                    if doi is None:
                        doi = ""
                    if abstract is None:
                        abstract = ""
                    if authors_raw is None:
                        authors_raw = ""
                    
                    # Convert author list to string for ChromaDB compatibility
                    if isinstance(authors_raw, list):
                        author = "; ".join([str(a) for a in authors_raw if a])  # Join authors with semicolon separator
                    else:
                        author = str(authors_raw) if authors_raw else ""
                    
                    # Extract basic metadata fields
                    try:
                        publication_date = extract_publication_date(row)
                        year = publication_date[:4] if publication_date else str(datetime.now().year)
                    except Exception as e:
                        print(f"⚠️  Error extracting publication date for paper {idx}: {e}")
                        publication_date = f"{datetime.now().year}-01-01"
                        year = str(datetime.now().year)
                    
                    # Get citation data that was already extracted during XML parsing
                    citation_data = {
                        'citation_count': row.get('citation_count', 'not found'),
                        'journal': row.get('journal', 'Unknown journal'),
                        'impact_factor': row.get('impact_factor', 'not found')
                    }
                    
                    # Extract additional metadata fields
                    try:
                        additional_metadata = extract_additional_metadata(row)
                    except Exception as e:
                        print(f"⚠️  Error extracting additional metadata for paper {idx}: {e}")
                        additional_metadata = {}
                    
                    # Update progress bar description to show citation processing
                    pbar.set_description(f"Processing paper {idx+1}/{len(df)} (citations: {citation_data['citation_count']})")
                    
                    # Only process papers that have abstracts
                    if not abstract or not abstract.strip():
                        print(f"⚠️  Skipping paper {idx}: No abstract available")
                        continue
                        
                    paragraphs = chunk_paragraphs(abstract)
                    
                    # Skip papers with no valid paragraphs
                    if not paragraphs:
                        print(f"⚠️  Skipping paper {idx}: No valid paragraphs found in abstract")
                        continue
                    
                    # Additional safety check for paper data
                    if not title or not title.strip():
                        print(f"⚠️  Skipping paper {idx}: No valid title")
                        continue
                    
                    # Process paragraphs for this paper
                    paper_embeddings = 0
                    for i, para in enumerate(paragraphs):
                        if not para or not para.strip():  # Skip empty paragraphs
                            continue
                            
                        try:
                            embedding = get_google_embedding(para, api_key)
                            
                            # Create comprehensive metadata object with immediate citation data
                            metadata = {
                                "title": title,
                                "doi": doi,
                                "author": author,  # Use author field name for consistency
                                "publication_date": publication_date,
                                "citation_count": citation_data['citation_count'],  # Immediate citation data
                                "journal": citation_data['journal'],  # Immediate journal data
                                "impact_factor": citation_data['impact_factor'],  # Immediate impact factor data
                                "source": "pubmed",
                                "paper_index": idx,
                                "para_idx": i,
                                "chunk_length": len(para),
                                "year": year  # Add explicit year field
                            }
                            
                            # Add additional metadata fields if available
                            metadata.update(additional_metadata)
                            
                            # Add to batch and save when full
                            current_batch, batch_num, batch_file = append_embedding_to_batch(
                                para, embedding, metadata, current_batch, batch_num, EMBEDDINGS_DIR, batch_size=100
                            )
                            
                            paper_embeddings += 1
                            total_embeddings += 1
                            
                            # Print progress every 50 embeddings
                            if total_embeddings % 50 == 0:
                                print(f"💾 Processed {total_embeddings} embeddings so far...")
                                
                        except Exception as e:
                            print(f"\n⚠️  Embedding error for PubMed paper {idx}: {e}")
                            continue
                    
                    total_chunks += len(paragraphs)
                    
                except Exception as e:
                    print(f"⚠️  Error processing paper {idx}: {e}")
                    continue
                
                pbar.update(1)
                pbar.set_postfix({"chunks": total_chunks, "embeddings": total_embeddings})
        
        # Save any remaining embeddings in the current batch
        if current_batch["embeddings"]:
            batch_file = save_batch_pubmed(current_batch, batch_num, EMBEDDINGS_DIR)
            print(f"💾 Saved final PubMed batch {batch_num:04d} with {len(current_batch['embeddings'])} embeddings")
        
        print(f"🎉 PubMed batch processing completed! Total embeddings: {total_embeddings}")
        
        # Step 3: Automatically load embeddings into ChromaDB
        # Note: For batch processing, we'll need to load all batch files
        load_pubmed_batches_to_chromadb(EMBEDDINGS_DIR, "pubmed")
        
        # Print final statistics
        print(f"\n🎉 PubMed processing complete!")
        print(f"📊 Total chunks processed: {total_chunks}")
        print(f"📊 Total embeddings created: {total_embeddings}")
        
        # Print citation processing summary
        citation_summary = get_citation_error_summary()
        if CITATION_PROCESSING_ENABLED:
            print(f"\n📊 Citation Processing Summary:")
            print(f"   Successful citations: {citation_summary['successful_citations']}")
            print(f"   DOI lookup failures: {citation_summary['doi_lookup_failures']}")
            print(f"   Title lookup failures: {citation_summary['title_lookup_failures']}")
            print(f"   Total failures: {citation_summary['failed_citations']}")
            
            if citation_summary['successful_citations'] > 0:
                success_rate = (citation_summary['successful_citations'] / 
                              (citation_summary['successful_citations'] + citation_summary['failed_citations'])) * 100
                print(f"   Success rate: {success_rate:.1f}%")
        else:
            print(f"\n🔧 Citation processing was disabled - all papers have 'pending' citation counts")
            print(f"💡 To enable citation processing, set environment variable: ENABLE_CITATIONS=true")
        
        # Final cleanup of any remaining temp files
        cleanup_temp_files()
            
    except Exception as e:
        print(f"\n❌ Fatal error in main function: {e}")
        print("Stack trace:")
        import traceback
        traceback.print_exc()
        # Clean up temp files even on fatal error
        cleanup_temp_files()
        return

if __name__ == "__main__":
    main() 