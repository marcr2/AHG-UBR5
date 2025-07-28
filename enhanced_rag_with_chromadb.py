import json
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
import os
from tqdm.auto import tqdm
from chromadb_manager import ChromaDBManager
import logging
from hypothesis_tools import HypothesisGenerator, HypothesisCritic, MetaHypothesisGenerator, LAB_GOALS
import time
import random
import threading
from collections import deque
import pandas as pd  # For Excel export
from openpyxl import load_workbook
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HypothesisTimer:
    """Timer for hypothesis critique process with 5-minute limit."""
    
    def __init__(self, timeout_minutes=5):
        self.timeout_seconds = timeout_minutes * 60
        self.start_time = None
        self.is_expired = False
        self.lock = threading.Lock()
    
    def start(self):
        """Start the timer."""
        with self.lock:
            self.start_time = time.time()
            self.is_expired = False
    
    def check_expired(self):
        """Check if the timer has expired."""
        if self.start_time is None:
            return False
        with self.lock:
            if not self.is_expired:
                elapsed = time.time() - self.start_time
                if elapsed >= self.timeout_seconds:
                    self.is_expired = True
            return self.is_expired
    
    def get_remaining_time(self):
        """Get remaining time in seconds."""
        if self.start_time is None:
            return self.timeout_seconds
        elapsed = time.time() - self.start_time
        return max(0, self.timeout_seconds - elapsed)
    
    def reset(self):
        """Reset the timer."""
        with self.lock:
            self.start_time = None
            self.is_expired = False

class GeminiRateLimiter:
    """Rate limiter for Gemini API calls (1000 requests per minute)."""
    
    def __init__(self, max_requests_per_minute=1000):
        self.max_requests = max_requests_per_minute
        self.request_times = deque()
        self.lock = threading.Lock() if 'threading' in globals() else None
    
    def wait_if_needed(self):
        """Wait if we're at the rate limit."""
        current_time = time.time()
        
        # Remove old requests (older than 1 minute)
        while self.request_times and current_time - self.request_times[0] > 60:
            self.request_times.popleft()
        
        # If we're at the limit, wait
        if len(self.request_times) >= self.max_requests:
            wait_time = 60 - (current_time - self.request_times[0]) + 1
            if wait_time > 0:
                print(f"⏳ Rate limit reached. Waiting {wait_time:.1f}s for Gemini API...")
                time.sleep(wait_time)
                current_time = time.time()
        
        # Add current request
        self.request_times.append(current_time)
    
    def get_remaining_requests(self):
        """Get number of remaining requests in current window."""
        current_time = time.time()
        
        # Remove old requests
        while self.request_times and current_time - self.request_times[0] > 60:
            self.request_times.popleft()
        
        return max(0, self.max_requests - len(self.request_times))

class EnhancedRAGQuery:
    """
    Enhanced RAG Query System with ChromaDB integration.
    
    Features:
    - Traditional similarity search (original functionality)
    - ChromaDB vector database for persistent storage
    - Metadata filtering
    - Hybrid search capabilities
    - Performance comparison between methods
    """
    
    def __init__(self, use_chromadb: bool = True, load_data_at_startup: bool = True):
        """Initialize the enhanced RAG query system."""
        self.use_chromadb = use_chromadb
        self.load_data_at_startup = load_data_at_startup
        self.embeddings_data = None
        self.chroma_manager = None
        self.hypothesis_generator = None
        self.hypothesis_critic = None
        self.meta_hypothesis_generator = None
        self.last_context_chunks = None
        self.hypothesis_records = []  # Track all hypotheses and results for export
        
        # Track the initial "add" quantity for dynamic chunk selection
        self.initial_add_quantity = None
        self.hypothesis_timer = HypothesisTimer(timeout_minutes=5)  # 5-minute timer
        
        # Package system for collecting search results
        self.current_package = {
            "chunks": [],
            "metadata": [],
            "sources": set(),
            "total_chars": 0
        }
        
        # Initialize Gemini rate limiter (1000 requests per minute)
        self.gemini_rate_limiter = GeminiRateLimiter(max_requests_per_minute=1000)
        
        print("🚀 Initializing Enhanced RAG System...")
        
        # Load API keys first
        try:
            with open("keys.json") as f:
                keys = json.load(f)
                self.api_key = keys["GOOGLE_API_KEY"]  # For embeddings
                self.gemini_api_key = keys["GEMINI_API_KEY"]  # For text generation
            print("✅ API keys loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load API keys: {e}")
            self.api_key = None
            self.gemini_api_key = None

        # Initialize Gemini client
        try:
            from google import genai
            self.gemini_client = genai.Client(api_key=self.gemini_api_key)
            print("✅ Gemini client initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini client: {e}")
            self.gemini_client = None

        # Initialize hypothesis tools after Gemini client is ready
        self._initialize_hypothesis_tools()

        # Initialize ChromaDB if requested
        if self.use_chromadb:
            self._initialize_chromadb()

        # Load embeddings data (optional)
        if self.load_data_at_startup:
            print("📚 Loading embeddings data (this may take a while for large datasets)...")
            self._load_embeddings_data()
        else:
            print("⏭️  Skipping data loading at startup (will load on-demand)")
            self.embeddings_data = None

        print("✅ Enhanced RAG System initialized!")
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB manager."""
        try:
            self.chroma_manager = ChromaDBManager()
            if not self.chroma_manager.create_collection():
                logger.error("❌ Failed to initialize ChromaDB")
                self.use_chromadb = False
            else:
                logger.info("✅ ChromaDB initialized successfully")
        except Exception as e:
            logger.error(f"❌ ChromaDB initialization failed: {e}")
            self.use_chromadb = False
    
    def _load_embeddings_data(self):
        """Load all available embedding files."""
        # Check if ChromaDB already has data
        if self.use_chromadb and self.chroma_manager:
            stats = self.chroma_manager.get_collection_stats()
            if stats.get('total_documents', 0) > 0:
                logger.info(f"📚 ChromaDB already has {stats.get('total_documents', 0)} documents - skipping data loading")
                self.embeddings_data = None  # We'll use ChromaDB directly
                return
        
        # Only load embeddings if ChromaDB is empty
        self.embeddings_data = self.load_all_embeddings()
        
        # If ChromaDB is available, populate it
        if self.use_chromadb and self.embeddings_data:
            self._populate_chromadb()
    
    def _populate_chromadb(self):
        """Populate ChromaDB with embeddings data."""
        if not self.chroma_manager or not self.embeddings_data:
            return
        
        try:
            # Check if collection is empty
            stats = self.chroma_manager.get_collection_stats()
            if stats.get('total_documents', 0) > 0:
                logger.info(f"📚 ChromaDB collection already populated with {stats.get('total_documents', 0)} documents")
                return
            
            logger.info("🔄 Populating ChromaDB with embeddings data...")
            
            # Add embeddings from each source
            for source_name, source_stats in self.embeddings_data["sources"].items():
                # Filter data for this source
                source_data = {
                    "chunks": [],
                    "embeddings": [],
                    "metadata": []
                }
                
                for i, meta in enumerate(self.embeddings_data["metadata"]):
                    if meta.get("source_file") == source_name:
                        source_data["chunks"].append(self.embeddings_data["chunks"][i])
                        source_data["embeddings"].append(self.embeddings_data["embeddings"][i])
                        source_data["metadata"].append(meta)
                
                if source_data["chunks"]:
                    self.chroma_manager.add_embeddings_to_collection(source_data, source_name)
                    logger.info(f"✅ Added {len(source_data['chunks'])} embeddings from {source_name}")
            
            # Show final stats
            final_stats = self.chroma_manager.get_collection_stats()
            logger.info(f"🎉 ChromaDB populated with {final_stats.get('total_documents', 0)} total documents")
        
        except Exception as e:
            logger.error(f"❌ Failed to populate ChromaDB: {e}")
    
    def get_google_embedding(self, text):
        """Get embedding for a query text using Google's API."""
        if not self.api_key:
            logger.error("❌ API key not available")
            return None
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        data = {"model": "models/text-embedding-004", "content": {"parts": [{"text": text}]}}
        
        try:
            # Add timeout to prevent hanging
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            embedding = response.json()["embedding"]["values"]
            return embedding
        except requests.exceptions.Timeout:
            logger.error("❌ Query embedding request timed out (30s)")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Network error getting query embedding: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Error getting query embedding: {e}")
            return None
    
    def load_embeddings_from_json(self, filename):
        """Load embeddings from JSON file."""
        if not os.path.exists(filename):
            logger.warning(f"⚠️ Embeddings file '{filename}' not found!")
            return None
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"📊 Loaded {len(data['chunks'])} chunks from {filename}")
            return data
        except Exception as e:
            logger.error(f"❌ Failed to load {filename}: {e}")
            return None
    
    def load_all_embeddings(self):
        """Load all available embedding files (both single files and batch-based)."""
        all_data = {
            "chunks": [],
            "embeddings": [],
            "metadata": [],
            "sources": {}
        }
        
        total_chunks = 0
        
        # Load PubMed embeddings (single file)
        if os.path.exists("pubmed_embeddings.json"):
            logger.info(f"🔄 Loading pubmed_embeddings.json...")
            data = self.load_embeddings_from_json("pubmed_embeddings.json")
            if data:
                # Add source prefix to metadata
                for meta in data["metadata"]:
                    meta["source_file"] = "pubmed"
                
                # Append to combined data
                all_data["chunks"].extend(data["chunks"])
                all_data["embeddings"].extend(data["embeddings"])
                all_data["metadata"].extend(data["metadata"])
                
                # Track source statistics
                all_data["sources"]["pubmed"] = {
                    "chunks": len(data["chunks"]),
                    "embeddings": len(data["embeddings"]),
                    "stats": data.get("stats", {})
                }
                
                total_chunks += len(data["chunks"])
        
        # Load xrvix embeddings (batch-based system)
        xrvix_dir = "xrvix_embeddings"
        if os.path.exists(xrvix_dir):
            logger.info(f"🔄 Loading xrvix embeddings from {xrvix_dir}...")
            
            # Load metadata first
            metadata_file = os.path.join(xrvix_dir, "metadata.json")
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    logger.info(f"📊 Found metadata: {metadata.get('total_embeddings', 0)} total embeddings")
                except Exception as e:
                    logger.error(f"❌ Failed to load metadata: {e}")
                    metadata = {}
            else:
                metadata = {}
            
            # Process each source directory (biorxiv, medrxiv, etc.)
            for source_dir in os.listdir(xrvix_dir):
                source_path = os.path.join(xrvix_dir, source_dir)
                if os.path.isdir(source_path) and source_dir in ["biorxiv", "medrxiv"]:
                    logger.info(f"🔄 Processing {source_dir}...")
                    
                    # Find all batch files
                    batch_files = [f for f in os.listdir(source_path) if f.startswith("batch_") and f.endswith(".json")]
                    batch_files.sort()  # Sort to process in order
                    
                    source_chunks = 0
                    source_embeddings = 0
                    
                    # Add progress bar for batch loading
                    for batch_file in batch_files:
                        batch_path = os.path.join(source_path, batch_file)
                        try:
                            with open(batch_path, 'r', encoding='utf-8') as f:
                                batch_data = json.load(f)
                            
                            # Add source information to metadata
                            for meta in batch_data.get("metadata", []):
                                meta["source_file"] = source_dir
                            
                            # Append to combined data
                            batch_chunks = len(batch_data.get("chunks", []))
                            batch_embeddings = len(batch_data.get("embeddings", []))
                            
                            all_data["chunks"].extend(batch_data.get("chunks", []))
                            all_data["embeddings"].extend(batch_data.get("embeddings", []))
                            all_data["metadata"].extend(batch_data.get("metadata", []))
                            
                            source_chunks += batch_chunks
                            source_embeddings += batch_embeddings
                            
                        except Exception as e:
                            logger.error(f"❌ Failed to load batch {batch_file}: {e}")
                    
                    # Track source statistics
                    if source_chunks > 0:
                        all_data["sources"][source_dir] = {
                            "chunks": source_chunks,
                            "embeddings": source_embeddings,
                            "batches": len(batch_files),
                            "stats": metadata.get("sources", {}).get(source_dir, {})
                        }
                        total_chunks += source_chunks
                        logger.info(f"✅ Loaded {source_chunks} chunks from {source_dir} ({len(batch_files)} batches)")
        
        if total_chunks == 0:
            logger.warning("⚠️ No embedding files found!")
            return None
        
        logger.info(f"🎉 Combined {total_chunks} chunks from all sources")
        return all_data
    
    def search_chromadb(self, query, top_k=5, filter_dict=None, show_progress=True):
        """Search using ChromaDB."""
        if not self.use_chromadb or not self.chroma_manager:
            print("❌ ChromaDB not available. Please initialize with ChromaDB enabled.")
            return []
        
        # Check if ChromaDB has data
        if not self.is_chromadb_ready():
            print("❌ ChromaDB is empty. Please load data first using the master processor (option 4).")
            return []
        
        # Get query embedding
        query_embedding = self.get_google_embedding(query)
        if not query_embedding:
            return []
        
        # Search in ChromaDB
        results = self.chroma_manager.search_similar(
            query_embedding=query_embedding,
            n_results=top_k,
            where_filter=filter_dict
        )
        
        return results
    
    def search_hybrid(self, query, top_k=20, filter_dict=None):
        """Search using ChromaDB only (simplified from hybrid)."""
        if not self.use_chromadb or not self.chroma_manager:
            print("❌ ChromaDB not available. Please initialize with ChromaDB enabled.")
            return []
        
        try:
            # Use only ChromaDB search
            results = self.search_chromadb(query, top_k=top_k, filter_dict=filter_dict)
            
            # Convert ChromaDB results to the expected format
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "chunk": result.get("document", ""),
                    "metadata": result.get("metadata", {}),
                    "similarity": 1.0 - result.get("distance", 0) if result.get("distance") is not None else 0,
                    "method": "chromadb"
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ ChromaDB search failed: {e}")
            return []
    
    def display_results(self, results, show_method=True):
        """Display search results in a formatted way."""
        if not results:
            print("[display_results] No results found.")
            return
        print(f"[display_results] Found {len(results)} relevant chunks.")
        total_chars = sum(len(result.get("chunk", "")) for result in results)
        sources = set()
        methods = set()
        for result in results:
            sources.add(result.get("metadata", {}).get("source", "Unknown"))
            methods.add(result.get("method", "Unknown"))
        print(f"[display_results] Summary:")
        print(f"   📚 Total chunks: {len(results)}")
        print(f"   📝 Total characters: {total_chars:,}")
        if show_method:
            print(f"   🔍 Search methods: {', '.join(methods)}")
        print(f"   📖 Sources: {', '.join(sources)}")
        print(f"[display_results] Top 3 results (showing examples):")
        for i, result in enumerate(results[:3], 1):
            chunk = result.get("chunk", "")
            metadata = result.get("metadata", {})
            similarity = result.get("similarity", 0)
            title = metadata.get("title", chunk[:60])
            if len(title) > 60:
                title = title[:57] + "..."
            print(f"   {i}. {title} (similarity: {similarity:.3f})")
        if len(results) > 3:
            print(f"   ... and {len(results) - 3} more chunks")
    
    def display_statistics(self):
        """Display comprehensive statistics about the knowledge base."""
        print(f"[display_statistics] Knowledge Base Statistics:")
        print("=" * 50)
        
        # Show ChromaDB stats if available (primary data source)
        if self.use_chromadb and self.chroma_manager:
            print(f"[display_statistics] ChromaDB Statistics (Primary Data Source):")
            chroma_stats = self.chroma_manager.get_collection_stats()
            print(f"   📊 Total documents: {chroma_stats.get('total_documents', 0)}")
            print(f"   📊 Collection name: {chroma_stats.get('collection_name', 'N/A')}")
            print(f"   📊 Available collections: {self.chroma_manager.list_collections()}")
            
            # Show source breakdown if available
            batch_stats = self.chroma_manager.get_batch_statistics()
            if batch_stats and batch_stats.get('sources'):
                print(f"[display_statistics] Source Breakdown:")
                for source, stats in batch_stats['sources'].items():
                    print(f"   {source}: {stats['total_documents']} documents, {len(stats['batches'])} batches")
        
        # Show in-memory data stats if available (secondary/fallback)
        if self.embeddings_data:
            print(f"[display_statistics] In-Memory Data Statistics (Secondary):")
            total_chunks = len(self.embeddings_data["chunks"])
            total_embeddings = len(self.embeddings_data["embeddings"])
            print(f"   📈 Total chunks: {total_chunks}")
            print(f"   📈 Total embeddings: {total_embeddings}")
            if total_embeddings > 0:
                print(f"   📈 Embedding dimensions: {len(self.embeddings_data['embeddings'][0])}")
            print(f"[display_statistics] Source Breakdown:")
            for source, stats in self.embeddings_data["sources"].items():
                print(f"   {source}: {stats['chunks']} chunks")
                if "stats" in stats and stats["stats"]:
                    source_stats = stats["stats"]
                    print(f"     - Papers: {source_stats.get('total_papers', 'N/A')}")
                    print(f"     - Embeddings: {source_stats.get('total_embeddings', 'N/A')}")
        else:
            print(f"[display_statistics] In-Memory Data: Not loaded (using ChromaDB directly)")
    
    def extract_citations_from_chunks(self, context_chunks):
        """Extract citation information from context chunks used for hypothesis critique."""
        citations = []
        unique_sources = set()
        
        for chunk in context_chunks:
            if isinstance(chunk, dict):
                metadata = chunk.get('metadata', {})
                # Extract source information
                source_name = metadata.get('source_name', 'Unknown')
                title = metadata.get('title', 'No title')
                doi = metadata.get('doi', 'No DOI')
                authors = metadata.get('authors', 'Unknown authors')
                journal = metadata.get('journal', 'Unknown journal')
                year = metadata.get('year', 'Unknown year')
                
                # Create citation entry
                citation = {
                    'source_name': source_name,
                    'title': title,
                    'doi': doi,
                    'authors': authors,
                    'journal': journal,
                    'year': year
                }
                
                # Add to citations if not already present (based on DOI or title)
                citation_key = doi if doi != 'No DOI' else title
                if citation_key not in unique_sources:
                    citations.append(citation)
                    unique_sources.add(citation_key)
            else:
                # Fallback for chunks without metadata
                citations.append({
                    'source_name': 'Unknown',
                    'title': 'Unknown source',
                    'doi': 'No DOI',
                    'authors': 'Unknown authors',
                    'journal': 'Unknown journal',
                    'year': 'Unknown year'
                })
        
        return citations
    
    def format_citations_for_export(self, citations):
        """Format citations for Excel export."""
        if not citations:
            return "No citations available"
        
        formatted_citations = []
        for citation in citations:
            # Format as academic citation
            if citation['doi'] != 'No DOI':
                formatted = f"{citation['authors']} ({citation['year']}). {citation['title']}. {citation['journal']}. DOI: {citation['doi']}"
            else:
                formatted = f"{citation['authors']} ({citation['year']}). {citation['title']}. {citation['journal']}."
            formatted_citations.append(formatted)
        
        return "; ".join(formatted_citations)

    def retrieve_relevant_chunks(self, query, top_k=1500):
        """Retrieve the most relevant chunks from all loaded batches using ChromaDB."""
        if not self.use_chromadb or not self.chroma_manager:
            print("❌ ChromaDB not available.")
            return []
        
        # Check if ChromaDB has data
        if not self.is_chromadb_ready():
            print("❌ ChromaDB is empty. Please load data first using the master processor (option 4).")
            return []
        
        # Get query embedding
        query_embedding = self.get_google_embedding(query)
        if not query_embedding:
            print("❌ Failed to get query embedding.")
            return []
        
        # Use ChromaDB search directly - much faster than loading all documents
        print(f"🔍 Searching ChromaDB for '{query}' (requesting {top_k} results)...")
        results = self.chroma_manager.search_similar(
            query_embedding=query_embedding,
            n_results=top_k
        )
        
        if not results:
            print("❌ No results found in ChromaDB.")
            return []
        
        print(f"✅ Found {len(results)} relevant chunks from ChromaDB")
        
        # Return results in the expected format
        return results

    def select_dynamic_chunks_for_generation(self, query, num_chunks=None):
        """Select new chunks from ChromaDB for hypothesis generation, using the initial add quantity."""
        if not self.use_chromadb or not self.chroma_manager:
            print("❌ ChromaDB not available for dynamic chunk selection.")
            return None
        
        # Use initial add quantity if not specified
        if num_chunks is None:
            if self.initial_add_quantity is None:
                print("⚠️  No initial add quantity stored, using default of 500 chunks")
                num_chunks = 500
            else:
                num_chunks = self.initial_add_quantity
                print(f"🔄 Selecting {num_chunks} new chunks from ChromaDB (based on initial add quantity)")
        
        try:
            # Get new chunks from ChromaDB using the original query
            if "prompt" in self.current_package and self.current_package["prompt"]:
                search_query = self.current_package["prompt"]
            else:
                search_query = query
            
            print(f"🔍 Searching ChromaDB for '{search_query}' to select {num_chunks} new chunks...")
            
            # Use hybrid search to get diverse results
            results = self.search_hybrid(search_query, top_k=num_chunks, filter_dict=None)
            
            if not results:
                print("❌ No new chunks found in ChromaDB.")
                return None
            
            # Extract just the chunk text for generation
            new_chunks = [result['chunk'] for result in results]
            print(f"✅ Selected {len(new_chunks)} new chunks from ChromaDB for this hypothesis generation")
            
            return new_chunks
            
        except Exception as e:
            print(f"❌ Error selecting dynamic chunks: {e}")
            return None

    @staticmethod
    def automated_verdict(accuracy, novelty, relevancy, accuracy_threshold=80, novelty_threshold=85, relevancy_threshold=50):
        if accuracy is not None and novelty is not None and relevancy is not None:
            return "ACCEPTED" if (accuracy >= accuracy_threshold and novelty >= novelty_threshold and relevancy >= relevancy_threshold) else "REJECTED"
        return "REJECTED"

    def iterative_hypothesis_generation(self, user_prompt, max_rounds=5, n=3):
        """Run generator-critic feedback loop for each hypothesis until accepted (>=90% accuracy/novelty and verdict ACCEPT)."""
        novelty_threshold = 85
        accuracy_threshold = 80
        relevancy_threshold = 50
        if not self.hypothesis_generator or not self.hypothesis_critic:
            print("[iterative_hypothesis_generation] ERROR: Hypothesis tools not initialized. Check if Gemini API key is available.")
            return
        # Use stored context chunks if available, otherwise search for new ones
        if hasattr(self, 'last_context_chunks') and self.last_context_chunks:
            context_chunks = self.last_context_chunks
            print(f"[iterative_hypothesis_generation] Using {len(context_chunks)} previously found context chunks for hypothesis generation.")
        else:
            print(f"[iterative_hypothesis_generation] Searching all loaded batches for relevant context...")
            context_chunks = self.retrieve_relevant_chunks(user_prompt, top_k=1500)
            if not context_chunks:
                print("[iterative_hypothesis_generation] ERROR: No relevant context found.")
                return
            self.last_context_chunks = context_chunks
        print(f"[iterative_hypothesis_generation] Generating initial hypotheses...")
        try:
            context_texts = [chunk.get("document", "") if isinstance(chunk, dict) else chunk for chunk in context_chunks]
            hypotheses = self.hypothesis_generator.generate(context_texts, n=n)
        except Exception as e:
            print(f"[iterative_hypothesis_generation] ERROR: Failed to generate initial hypotheses: {e}")
            return
        accepted = [False] * n
        rounds = [0] * n
        critiques = [{} for _ in range(n)]
        from tqdm.auto import tqdm
        total_iterations = max_rounds * n
        current_iteration = 0
        print(f"[iterative_hypothesis_generation] Starting iterative refinement (max {max_rounds} rounds per hypothesis, 5-minute time limit per critique, acceptance: Novelty >= {novelty_threshold}, Accuracy >= {accuracy_threshold}, Relevancy >= {relevancy_threshold})...")
        with tqdm(total=total_iterations, desc="Hypothesis refinement", unit="iteration") as pbar:
            while not all(accepted) and max(rounds) < max_rounds:
                for i in range(n):
                    if accepted[i]:
                        continue
                    rounds[i] += 1
                    current_iteration += 1
                    print(f"[iterative_hypothesis_generation] Critique Round {rounds[i]} for Hypothesis {i+1} (5-minute timer started)...")
                    self.hypothesis_timer.start()
                    try:
                        context_texts = [chunk.get("document", "") if isinstance(chunk, dict) else chunk for chunk in context_chunks]
                        if self.hypothesis_timer.check_expired():
                            print(f"[iterative_hypothesis_generation] TIMEOUT: Hypothesis {i+1} critique timed out.")
                            citations = self.extract_citations_from_chunks(context_chunks)
                            formatted_citations = self.format_citations_for_export(citations)
                            self.hypothesis_records.append({
                                'Hypothesis': hypotheses[i],
                                'Accuracy': None,
                                'Novelty': None,
                                'Verdict': 'TIMEOUT',
                                'Critique': 'Critique process timed out after 5 minutes',
                                'Citations': formatted_citations
                            })
                            pbar.update(1)
                            continue
                        critique_result = self.hypothesis_critic.critique(hypotheses[i], context_texts, prompt=user_prompt, lab_goals=LAB_GOALS)
                        if self.hypothesis_timer.check_expired():
                            print(f"[iterative_hypothesis_generation] TIMEOUT: Hypothesis {i+1} critique timed out after critique.")
                            citations = self.extract_citations_from_chunks(context_chunks)
                            formatted_citations = self.format_citations_for_export(citations)
                            self.hypothesis_records.append({
                                'Hypothesis': hypotheses[i],
                                'Accuracy': critique_result.get('accuracy', None),
                                'Novelty': critique_result.get('novelty', None),
                                'Verdict': 'TIMEOUT',
                                'Critique': critique_result.get('critique', '') + ' [Process timed out]',
                                'Citations': formatted_citations
                            })
                            pbar.update(1)
                            continue
                        critiques[i] = critique_result
                        novelty = critique_result.get('novelty', 0)
                        accuracy = critique_result.get('accuracy', 0)
                        relevancy = critique_result.get('relevancy', 0)
                        verdict = self.automated_verdict(accuracy, novelty, relevancy, accuracy_threshold, novelty_threshold, relevancy_threshold)
                        remaining_time = self.hypothesis_timer.get_remaining_time()
                        print(f"\033[1mHypothesis {i+1}:\033[0m \033[96m{hypotheses[i]}\033[0m")
                        print(f"[iterative_hypothesis_generation] Critique: {critique_result['critique']}")
                        print(f"[iterative_hypothesis_generation] Scores: Novelty={novelty}/{novelty_threshold}, Accuracy={accuracy}/{accuracy_threshold}, Relevancy={relevancy}/{relevancy_threshold}, Automated Verdict={verdict}, Time left={remaining_time:.1f}s")
                        citations = self.extract_citations_from_chunks(context_chunks)
                        formatted_citations = self.format_citations_for_export(citations)
                        self.hypothesis_records.append({
                            'Hypothesis': hypotheses[i],
                            'Accuracy': accuracy,
                            'Novelty': novelty,
                            'Relevancy': relevancy,
                            'Verdict': verdict,
                            'Critique': critique_result.get('critique', ''),
                            'Citations': formatted_citations
                        })
                        if verdict == 'ACCEPTED':
                            print(f"\033[92mACCEPTED\033[0m: Hypothesis {i+1} (Novelty: {novelty}/{novelty_threshold}, Accuracy: {accuracy}/{accuracy_threshold}, Relevancy: {relevancy}/{relevancy_threshold})")
                            accepted[i] = True
                        else:
                            print(f"\033[91mREJECTED\033[0m: Hypothesis {i+1} (Novelty: {novelty}/{novelty_threshold}, Accuracy: {accuracy}/{accuracy_threshold}, Relevancy: {relevancy}/{relevancy_threshold}) - Regenerating...")
                            if self.hypothesis_timer.check_expired():
                                print(f"[iterative_hypothesis_generation] TIMEOUT: Hypothesis {i+1} regeneration timed out.")
                                pbar.update(1)
                                continue
                            try:
                                context_texts = [chunk.get("document", "") if isinstance(chunk, dict) else chunk for chunk in context_chunks]
                                new_hypothesis = self.hypothesis_generator.generate(context_texts, n=1)
                                if new_hypothesis:
                                    hypotheses[i] = new_hypothesis[0]
                            except Exception as e:
                                print(f"[iterative_hypothesis_generation] ERROR: Failed to regenerate hypothesis {i+1}: {e}")
                                continue
                        avg_rating = (novelty + accuracy + relevancy) / 3 if novelty is not None and accuracy is not None and relevancy is not None else 0
                        pbar.set_postfix({
                            "accepted": sum(accepted),
                            "round": max(rounds),
                            "current": f"H{i+1}",
                            "avg_rating": f"{avg_rating:.1f}",
                            "time_left": f"{remaining_time:.0f}s"
                        })
                        pbar.update(1)
                    except Exception as e:
                        print(f"[iterative_hypothesis_generation] ERROR: Failed to critique hypothesis {i+1}: {e}")
                        citations = self.extract_citations_from_chunks(context_chunks)
                        formatted_citations = self.format_citations_for_export(citations)
                        self.hypothesis_records.append({
                            'Hypothesis': hypotheses[i],
                            'Accuracy': None,
                            'Novelty': None,
                            'Relevancy': None,
                            'Verdict': 'ERROR',
                            'Critique': f'Error during critique: {str(e)}',
                            'Citations': formatted_citations
                        })
                        pbar.update(1)
                        continue
                    if all(accepted):
                        break
        print("[iterative_hypothesis_generation] Final Hypotheses:")
        for i, hyp in enumerate(hypotheses, 1):
            status = "ACCEPTED" if accepted[i-1] else f"FAILED (max {max_rounds} rounds)"
            print(f"   {i}. {hyp} [{status}]")
            if isinstance(critiques[i-1], dict) and critiques[i-1]:
                print(f"      Critique: {critiques[i-1].get('critique','')}")
                print(f"      Novelty: {critiques[i-1].get('novelty','')}/{novelty_threshold}, Accuracy: {critiques[i-1].get('accuracy','')}/{accuracy_threshold}, Relevancy: {critiques[i-1].get('relevancy','')}/{relevancy_threshold}, Automated Verdict: {self.automated_verdict(critiques[i-1].get('accuracy',0), critiques[i-1].get('novelty',0), critiques[i-1].get('relevancy',0), accuracy_threshold, novelty_threshold, relevancy_threshold)}")
        return hypotheses

    def add_to_package(self, search_results):
        """Add search results to the current package."""
        current_size = self.current_package["total_chars"]
        new_size = current_size + sum(len(result.get("chunk", "")) for result in search_results)
        if new_size > 2000000:
            print(f"[add_to_package] WARNING: Adding {len(search_results)} chunks would make package very large ({new_size:,} characters)")
            print("[add_to_package] This may cause API errors. Consider using 'clear' first.")
            response = input("Continue anyway? (y/n): ").lower()
            if response != 'y':
                return
        
        # Store the initial add quantity for dynamic chunk selection
        if self.initial_add_quantity is None:
            self.initial_add_quantity = len(search_results)
            print(f"[add_to_package] Stored initial add quantity: {self.initial_add_quantity} chunks for dynamic selection")
        
        for result in search_results:
            chunk = result.get("chunk", "")
            metadata = result.get("metadata", {})
            self.current_package["chunks"].append(chunk)
            self.current_package["metadata"].append(metadata)
            self.current_package["sources"].add(metadata.get("source", "Unknown"))
            self.current_package["total_chars"] += len(chunk)
        print(f"[add_to_package] Added {len(search_results)} chunks to package.")
        print(f"[add_to_package] Package now contains {len(self.current_package['chunks'])} chunks from {len(self.current_package['sources'])} sources.")
        if self.current_package["total_chars"] > 1000000:
            print(f"[add_to_package] WARNING: Package size: {self.current_package['total_chars']:,} characters (consider using 'clear' if you get API errors)")

    def clear_package(self):
        """Clear the current package."""
        self.current_package = {
            "chunks": [],
            "metadata": [],
            "sources": set(),
            "total_chars": 0
        }
        # Reset the initial add quantity when clearing package
        self.initial_add_quantity = None
        print("[clear_package] Package cleared.")
        print("[clear_package] Initial add quantity reset.")

    def show_package(self):
        """Display information about the current package."""
        if not self.current_package["chunks"]:
            print("[show_package] Package is empty.")
            return
        print(f"[show_package] Current Package:")
        print(f"   📚 Total chunks: {len(self.current_package['chunks'])}")
        print(f"   📝 Total characters: {self.current_package['total_chars']:,}")
        print(f"   📖 Sources: {', '.join(self.current_package['sources'])}")
        print(f"\n[show_package] Sample chunks:")
        for i, chunk in enumerate(self.current_package["chunks"][:3], 1):
            metadata = self.current_package["metadata"][i-1]
            title = metadata.get("title", "No title")[:60]
            print(f"   {i}. {title}...")
        if len(self.current_package["chunks"]) > 3:
            print(f"   ... and {len(self.current_package['chunks']) - 3} more chunks")

    def generate_hypotheses_from_package(self, n=5):
        novelty_threshold = 85
        accuracy_threshold = 80
        relevancy_threshold = 50
        if not self.current_package["chunks"]:
            print("❌ Package is empty. Add some chunks first.")
            return
        if not self.hypothesis_generator or not self.hypothesis_critic:
            print("❌ Hypothesis tools not initialized. Check if Gemini API key is available.")
            return
        # Prompt for the query if not already stored
        if "prompt" not in self.current_package or not self.current_package["prompt"]:
            self.current_package["prompt"] = input("Enter the original query for this package: ").strip()
        package_prompt = self.current_package["prompt"]
        print(f"\n🧠 Generating {n} hypotheses...")
        print(f"📦 Using {len(self.current_package['chunks'])} package chunks for critique")
        print(f"[INFO] Acceptance criteria: Novelty >= {novelty_threshold}, Accuracy >= {accuracy_threshold}, Relevancy >= {relevancy_threshold}")
        
        # Check package size and warn if too large
        package_size = sum(len(chunk) for chunk in self.current_package["chunks"])
        if package_size > 1000000:  # 1MB limit for package
            print(f"⚠️  Package is very large ({package_size:,} characters). This may cause API errors.")
            print("💡 Consider using 'clear' and adding fewer chunks.")
            response = input("Continue anyway? (y/n): ").lower()
            if response != 'y':
                return
        
        # For generation, use a sample of the database instead of the entire thing
        # This prevents the 300MB payload limit error
        if not self.embeddings_data:
            # Check if ChromaDB has data first
            if self.use_chromadb and self.chroma_manager and self.is_chromadb_ready():
                print("📚 ChromaDB has data - using package chunks for generation (no need to load embeddings)")
                # Use only the package chunks for generation when ChromaDB is available
                all_chunks = self.current_package["chunks"]
            else:
                print("📚 Loading embeddings data...")
                self.embeddings_data = self.load_all_embeddings()
                if not self.embeddings_data:
                    print("❌ No embeddings data available")
                    return
                all_chunks = self.embeddings_data["chunks"]
        else:
            all_chunks = self.embeddings_data["chunks"]
        package_chunks = self.current_package["chunks"]
        package_metadata = self.current_package["metadata"]
        
        # Calculate safe sample size (aim for ~50MB payload)
        # Each chunk is roughly 1000 characters, so 50MB = ~50,000 chunks
        max_chunks_for_generation = 50000
        
        # Check if we're using package chunks (from ChromaDB) or all chunks (from embeddings)
        if all_chunks is self.current_package["chunks"]:
            # Using package chunks from ChromaDB
            generation_chunks = all_chunks
            print(f"📚 Using {len(generation_chunks)} package chunks from ChromaDB for generation")
        else:
            # Using all chunks from embeddings data
            if len(all_chunks) > max_chunks_for_generation:
                print(f"📚 Database contains {len(all_chunks)} total chunks")
                print(f"📚 Using sample of {max_chunks_for_generation} chunks for generation (to avoid API limits)")
                # Take a representative sample from different parts of the database
                step = len(all_chunks) // max_chunks_for_generation
                generation_chunks = all_chunks[::step][:max_chunks_for_generation]
            else:
                generation_chunks = all_chunks
                print(f"📚 Database contains {len(all_chunks)} total chunks")
        
        print(f"📦 Package contains {len(package_chunks)} chunks")
        
        # Sequential, real-time generator-critic loop for 5 accepted hypotheses
        print(f"\n🧠 Generating hypotheses one at a time until 5 are accepted (novelty >= {novelty_threshold}, accuracy >= {accuracy_threshold}, 5-minute time limit per critique)...")
        
        # Initialize Excel file for incremental saving
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Ensure hypothesis_export directory exists
        export_dir = "hypothesis_export"
        os.makedirs(export_dir, exist_ok=True)
        excel_filename = os.path.join(export_dir, f"hypothesis_export_{timestamp}.xlsx")
        print(f"💾 Will save hypotheses incrementally to: {excel_filename}")
        
        accepted_hypotheses = []
        attempts = 0
        max_attempts = 100  # Prevent infinite loops
        while len(accepted_hypotheses) < n and attempts < max_attempts:
            attempts += 1
            print(f"\n🧠 Generating hypothesis attempt {attempts}...")
            
            # Select new chunks from ChromaDB for this hypothesis generation
            if self.use_chromadb and self.chroma_manager and self.initial_add_quantity:
                print(f"🔄 Selecting new chunks for hypothesis attempt {attempts}...")
                dynamic_chunks = self.select_dynamic_chunks_for_generation(package_prompt)
                if dynamic_chunks:
                    generation_chunks = dynamic_chunks
                    print(f"📚 Using {len(generation_chunks)} dynamically selected chunks for generation")
                else:
                    print(f"⚠️  Failed to select dynamic chunks, using original chunks")
            else:
                print(f"📚 Using original chunks for generation (no dynamic selection available)")
            
            # Start timer for this hypothesis
            self.hypothesis_timer.start()
            print(f"⏱️  Starting 5-minute timer for hypothesis attempt {attempts}...")
            
            hypothesis_list = self._generate_hypotheses_with_retry(generation_chunks, n=1)
            if not hypothesis_list:
                print("❌ Failed to generate hypothesis. Skipping...")
                # Record failed generation
                failed_record = {
                    'Hypothesis': f'Failed generation attempt {attempts}',
                    'Accuracy': None,
                    'Novelty': None,
                    'Verdict': 'GENERATION_FAILED',
                    'Critique': 'Failed to generate hypothesis',
                    'Citations': 'No citations available'
                }
                self.hypothesis_records.append(failed_record)
                # Save incrementally
                self.save_hypothesis_record_incrementally(failed_record, excel_filename)
                continue
            
            hypothesis = hypothesis_list[0]
            print(f"Generated Hypothesis: {hypothesis}")
            print(f"\n🔄 Critiquing Hypothesis: {hypothesis}")
            
            # Check timer before critique
            if self.hypothesis_timer.check_expired():
                print(f"⏰ Time limit reached for hypothesis attempt {attempts}. Moving to next attempt.")
                # Record timeout result
                timeout_record = {
                    'Hypothesis': hypothesis,
                    'Accuracy': None,
                    'Novelty': None,
                    'Verdict': 'TIMEOUT',
                    'Critique': 'Critique process timed out after 5 minutes',
                    'Citations': 'No citations available'
                }
                self.hypothesis_records.append(timeout_record)
                # Save incrementally
                self.save_hypothesis_record_incrementally(timeout_record, excel_filename)
                continue
            
            critique_result = self._critique_hypothesis_with_retry(hypothesis, package_chunks, package_prompt)
            if not critique_result:
                print(f"❌ Failed to critique hypothesis after retries. Skipping...")
                # Record failed critique
                critique_failed_record = {
                    'Hypothesis': hypothesis,
                    'Accuracy': None,
                    'Novelty': None,
                    'Verdict': 'CRITIQUE_FAILED',
                    'Critique': 'Failed to critique hypothesis after retries',
                    'Citations': 'No citations available'
                }
                self.hypothesis_records.append(critique_failed_record)
                # Save incrementally
                self.save_hypothesis_record_incrementally(critique_failed_record, excel_filename)
                continue
            
            # Check timer after critique
            if self.hypothesis_timer.check_expired():
                print(f"⏰ Time limit reached for hypothesis attempt {attempts} after critique. Moving to next attempt.")
                # Record timeout result with partial critique
                partial_timeout_record = {
                    'Hypothesis': hypothesis,
                    'Accuracy': critique_result.get('accuracy', None),
                    'Novelty': critique_result.get('novelty', None),
                    'Verdict': 'TIMEOUT',
                    'Critique': critique_result.get('critique', '') + ' [Process timed out]',
                    'Citations': 'No citations available'
                }
                self.hypothesis_records.append(partial_timeout_record)
                # Save incrementally
                self.save_hypothesis_record_incrementally(partial_timeout_record, excel_filename)
                continue
            
            novelty = critique_result.get('novelty', 0)
            accuracy = critique_result.get('accuracy', 0)
            relevancy = critique_result.get('relevancy', 0)
            # Use the same thresholds as the acceptance logic
            verdict = self.automated_verdict(accuracy, novelty, relevancy, accuracy_threshold, novelty_threshold, relevancy_threshold)
            remaining_time = self.hypothesis_timer.get_remaining_time()
            print(f"   Novelty: {novelty}/{novelty_threshold}, Accuracy: {accuracy}/{accuracy_threshold}, Relevancy: {relevancy}/{relevancy_threshold}, Automated Verdict: {verdict}")
            print(f"   ⏱️  Time remaining: {remaining_time:.1f}s")
            
            # Extract citations from package metadata
            citations = []
            unique_sources = set()
            for i, metadata in enumerate(package_metadata):
                if metadata:
                    source_name = metadata.get('source_name', 'Unknown')
                    title = metadata.get('title', 'No title')
                    doi = metadata.get('doi', 'No DOI')
                    authors = metadata.get('authors', 'Unknown authors')
                    journal = metadata.get('journal', 'Unknown journal')
                    year = metadata.get('year', 'Unknown year')
                    
                    citation = {
                        'source_name': source_name,
                        'title': title,
                        'doi': doi,
                        'authors': authors,
                        'journal': journal,
                        'year': year
                    }
                    
                    # Add to citations if not already present
                    citation_key = doi if doi != 'No DOI' else title
                    if citation_key not in unique_sources:
                        citations.append(citation)
                        unique_sources.add(citation_key)
            
            formatted_citations = self.format_citations_for_export(citations)
            
            # Track record for export
            hypothesis_record = {
                'Hypothesis': hypothesis,
                'Accuracy': accuracy,
                'Novelty': novelty,
                'Relevancy': relevancy,
                'Verdict': verdict,
                'Critique': critique_result.get('critique', ''),
                'Citations': formatted_citations
            }
            self.hypothesis_records.append(hypothesis_record)
            
            # Save incrementally to Excel
            self.save_hypothesis_record_incrementally(hypothesis_record, excel_filename)
            
            if verdict == 'ACCEPTED':
                accepted_hypotheses.append({
                    "hypothesis": hypothesis,
                    "critique": critique_result,
                    "score": (novelty + accuracy + relevancy) / 3
                })
                print(f"✅ Hypothesis accepted! ({len(accepted_hypotheses)}/{n})-----------------------------------\n")
            else:
                print(f"❌ Hypothesis rejected (verdict: {verdict})-----------------------------------\n")
        
        if not accepted_hypotheses:
            print("❌ No hypotheses were successfully critiqued.")
            return
        
        print(f"\n🏆 Top {len(accepted_hypotheses)} Hypotheses:")
        print("=" * 80)
        for i, result in enumerate(accepted_hypotheses, 1):
            hypothesis = result["hypothesis"]
            score = result["score"]
            critique = result["critique"]
            print(f"\n{i}. {hypothesis}")
            print(f"   Score: {score:.1f}")
            print(f"   Novelty: {critique.get('novelty', 'N/A')}")
            print(f"   Accuracy: {critique.get('accuracy', 'N/A')}")
            print(f"   Relevancy: {critique.get('relevancy', 'N/A')}")
            print(f"   Automated Verdict: {self.automated_verdict(critique.get('accuracy', 0), critique.get('novelty', 0), critique.get('relevancy', 0), accuracy_threshold, novelty_threshold, relevancy_threshold)}")
            print(f"   Critique: {critique.get('critique', 'N/A')}")
            print("-" * 80)
        return accepted_hypotheses[:n]

    def _generate_hypotheses_with_retry(self, generation_chunks, n, max_retries=3):
        """Generate hypotheses with retry logic for rate limiting."""
        if not self.hypothesis_generator:
            print("❌ Hypothesis generator not available")
            return None
            
        for attempt in range(max_retries):
            try:
                # Check rate limit before making API call
                remaining_requests = self.gemini_rate_limiter.get_remaining_requests()
                if remaining_requests <= 0:
                    print(f"⏳ Gemini API rate limit reached. Waiting for reset...")
                    self.gemini_rate_limiter.wait_if_needed()
                
                # Extract chunk text from the sample
                generation_chunk_texts = [chunk for chunk in generation_chunks]
                hypotheses = self.hypothesis_generator.generate(generation_chunk_texts, n=n)
                return hypotheses
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                    wait_time = (2 ** attempt) * 30 + random.randint(0, 10)  # Exponential backoff
                    print(f"⚠️  API rate limited (attempt {attempt + 1}/{max_retries}). Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    if attempt == max_retries - 1:
                        print("❌ Max retries reached. API quota exceeded.")
                        return None
                elif "payload size exceeds the limit" in error_str:
                    print("💡 Payload too large. Try reducing package size with 'clear' and adding fewer chunks")
                    return None
                else:
                    print(f"❌ Unexpected error: {e}")
                    return None
        return None

    def _critique_hypothesis_with_retry(self, hypothesis, package_chunks, prompt, max_retries=3):
        """Critique hypothesis with retry logic for rate limiting."""
        if not self.hypothesis_critic:
            print("❌ Hypothesis critic not available")
            return None
            
        for attempt in range(max_retries):
            try:
                # Check rate limit before making API call
                remaining_requests = self.gemini_rate_limiter.get_remaining_requests()
                if remaining_requests <= 0:
                    print(f"⏳ Gemini API rate limit reached. Waiting for reset...")
                    self.gemini_rate_limiter.wait_if_needed()
                
                # Extract chunk text from package chunks
                package_chunk_texts = [chunk for chunk in package_chunks]
                critique_result = self.hypothesis_critic.critique(hypothesis, package_chunk_texts, prompt=prompt, lab_goals=LAB_GOALS)
                return critique_result
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                    wait_time = (2 ** attempt) * 30 + random.randint(0, 10)  # Exponential backoff
                    print(f"⚠️  API rate limited (attempt {attempt + 1}/{max_retries}). Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    if attempt == max_retries - 1:
                        print("❌ Max retries reached. API quota exceeded.")
                        return None
                elif "payload size exceeds the limit" in error_str:
                    print("💡 Package is too large for critique. Try 'clear' and add fewer chunks.")
                    return None
                else:
                    print(f"❌ Unexpected error: {e}")
                    return None
        return None

    def generate_hypotheses_offline(self, n=5):
        """Generate hypotheses without using the API (fallback mode)."""
        if not self.current_package["chunks"]:
            print("❌ Package is empty. Add some chunks first.")
            return
        
        print(f"\n🧠 Generating {n} hypotheses (offline mode)...")
        print(f"📦 Using {len(self.current_package['chunks'])} package chunks")
        
        # Extract key terms and patterns from the package
        package_text = " ".join(self.current_package["chunks"])
        
        # Simple keyword extraction
        import re
        from collections import Counter
        
        # Extract UBR-5 related terms
        ubr5_terms = re.findall(r'\b(?:UBR5?|ubiquitin|ligase|protein|E3|ubiquitination|degradation|regulation|pathway|signaling|cancer|immunology|immune|response|activation|inhibition|expression|function|mechanism|target|therapeutic|drug|treatment|therapy)\b', package_text, re.IGNORECASE)
        term_counts = Counter(ubr5_terms)
        
        # Generate simple hypotheses based on common patterns
        hypotheses = []
        
        # Hypothesis 1: UBR5 regulation
        if any(term in term_counts for term in ['regulation', 'expression', 'function']):
            hypotheses.append("UBR5 expression and function are regulated by specific signaling pathways in immune cells, affecting immune response and cancer progression.")
        
        # Hypothesis 2: Therapeutic targeting
        if any(term in term_counts for term in ['therapeutic', 'drug', 'treatment', 'target']):
            hypotheses.append("UBR5 represents a novel therapeutic target for cancer immunotherapy, with potential for drug development and clinical applications.")
        
        # Hypothesis 3: Immune response
        if any(term in term_counts for term in ['immune', 'response', 'activation', 'immunology']):
            hypotheses.append("UBR5 plays a critical role in regulating immune cell activation and response, influencing cancer immunosurveillance and immunotherapy efficacy.")
        
        # Hypothesis 4: Ubiquitination pathway
        if any(term in term_counts for term in ['ubiquitin', 'ligase', 'ubiquitination', 'degradation']):
            hypotheses.append("UBR5-mediated ubiquitination regulates key proteins in immune signaling pathways, affecting cell fate decisions and immune function.")
        
        # Hypothesis 5: Cancer mechanism
        if any(term in term_counts for term in ['cancer', 'mechanism', 'pathway', 'signaling']):
            hypotheses.append("UBR5 functions as a key regulator in cancer cell signaling pathways, with implications for tumor progression and therapeutic resistance.")
        
        # Fill remaining slots with generic hypotheses if needed
        while len(hypotheses) < n:
            hypotheses.append(f"UBR5 may have additional roles in cellular processes related to {list(term_counts.keys())[:3] if term_counts else 'protein regulation'}.")
        
        print(f"\n🏆 Generated {len(hypotheses)} Hypotheses (Offline Mode):")
        print("=" * 80)
        for i, hypothesis in enumerate(hypotheses[:n], 1):
            print(f"\n{i}. {hypothesis}")
            print(f"   Score: N/A (offline mode)")
            print(f"   Novelty: N/A (offline mode)")
            print(f"   Accuracy: N/A (offline mode)")
            print(f"   Verdict: N/A (offline mode)")
            print(f"   Note: This is a fallback hypothesis based on keyword analysis")
            print("-" * 80)
        
        print(f"\n💡 Note: These are simplified hypotheses generated without API access.")
        print(f"💡 For more sophisticated analysis, try 'generate' when API is available.")
        
        return hypotheses[:n]

    def generate_hypotheses_with_meta_generator(self, user_prompt: str, n_per_meta: int = 3, chunks_per_meta: int = 1500):
        """
        Generate hypotheses using the meta-hypothesis generator approach.
        
        This method:
        1. Takes the user's prompt and generates 5 meta-hypotheses
        2. For each meta-hypothesis, generates n_per_meta hypotheses using the existing system
        3. Returns all generated hypotheses with their critiques
        
        Args:
            user_prompt: The original user query
            n_per_meta: Number of hypotheses to generate per meta-hypothesis (default: 3)
            chunks_per_meta: Number of context chunks to use per meta-hypothesis (default: 1500)
        """
        if not self.meta_hypothesis_generator or not self.hypothesis_generator or not self.hypothesis_critic:
            print("❌ Meta-hypothesis tools not initialized. Check if Gemini API key is available.")
            return None
        
        print(f"\n🧠 META-HYPOTHESIS GENERATION SESSION")
        print("=" * 80)
        print(f"Original Query: {user_prompt}")
        print(f"Generating {n_per_meta} hypotheses per meta-hypothesis")
        print("=" * 80)
        
        # Step 1: Generate meta-hypotheses
        print(f"\n🔍 Step 1: Generating 5 meta-hypotheses from user query...")
        try:
            meta_hypotheses = self.meta_hypothesis_generator.generate_meta_hypotheses(user_prompt)
            if not meta_hypotheses or len(meta_hypotheses) < 5:
                print("❌ Failed to generate sufficient meta-hypotheses")
                return None
            
            print(f"✅ Generated {len(meta_hypotheses)} meta-hypotheses:")
            for i, meta_hyp in enumerate(meta_hypotheses, 1):
                print(f"  {i}. {meta_hyp}")
        except Exception as e:
            print(f"❌ Error generating meta-hypotheses: {e}")
            return None
        
        # Step 2: For each meta-hypothesis, generate hypotheses
        all_hypotheses = []
        total_meta_hypotheses = len(meta_hypotheses)
        
        # Initialize Excel file for incremental saving
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Ensure hypothesis_export directory exists
        export_dir = "hypothesis_export"
        os.makedirs(export_dir, exist_ok=True)
        excel_filename = os.path.join(export_dir, f"meta_hypothesis_export_{timestamp}.xlsx")
        print(f"💾 Will save meta-hypotheses incrementally to: {excel_filename}")
        
        # Initialize hypothesis records for this session
        self.hypothesis_records = []
        
        for meta_idx, meta_hypothesis in enumerate(meta_hypotheses, 1):
            print(f"\n🔍 Step 2.{meta_idx}: Generating hypotheses for meta-hypothesis {meta_idx}/{total_meta_hypotheses}")
            print(f"Meta-hypothesis: {meta_hypothesis}")
            print("-" * 60)
            
            # Get relevant context for this meta-hypothesis
            context_chunks = self.retrieve_relevant_chunks(meta_hypothesis, top_k=chunks_per_meta)
            if not context_chunks:
                print(f"⚠️  No relevant context found for meta-hypothesis {meta_idx}. Skipping...")
                continue
            
            print(f"📚 Found {len(context_chunks)} relevant context chunks (using {chunks_per_meta} chunks per meta-hypothesis)")
            
            # Generate hypotheses for this meta-hypothesis
            meta_hypotheses_generated = []
            attempts = 0
            max_attempts = n_per_meta * 3  # Allow more attempts to get enough hypotheses
            
            while len(meta_hypotheses_generated) < n_per_meta and attempts < max_attempts:
                attempts += 1
                print(f"\n🧠 Generating hypothesis attempt {attempts} for meta-hypothesis {meta_idx}...")
                
                # Select new chunks from ChromaDB for this hypothesis generation
                if self.use_chromadb and self.chroma_manager and self.initial_add_quantity:
                    print(f"🔄 Selecting new chunks for hypothesis attempt {attempts}...")
                    dynamic_chunks = self.select_dynamic_chunks_for_generation(meta_hypothesis)
                    if dynamic_chunks:
                        context_chunks = dynamic_chunks
                        print(f"📚 Using {len(context_chunks)} dynamically selected chunks for generation")
                    else:
                        print(f"⚠️  Failed to select dynamic chunks, using original chunks")
                else:
                    print(f"📚 Using original chunks for generation (no dynamic selection available)")
                
                # Start timer for this hypothesis
                self.hypothesis_timer.start()
                print(f"⏱️  Starting 5-minute timer for hypothesis attempt {attempts}...")
                
                # Generate hypothesis
                try:
                    context_texts = [chunk.get("document", "") if isinstance(chunk, dict) else chunk for chunk in context_chunks]
                    hypothesis_list = self.hypothesis_generator.generate(context_texts, n=1)
                    
                    if not hypothesis_list:
                        print("❌ Failed to generate hypothesis. Skipping...")
                        continue
                    
                    hypothesis = hypothesis_list[0]
                    print(f"📝 Generated: {hypothesis}")
                    
                    # Check timer before critique
                    if self.hypothesis_timer.check_expired():
                        print(f"⏰ Time limit reached before critique")
                        continue
                    
                    # Critique hypothesis
                    print(f"🔍 Critiquing hypothesis...")
                    critique_result = self.hypothesis_critic.critique(hypothesis, context_texts, prompt=meta_hypothesis, lab_goals=LAB_GOALS)
                    
                    if critique_result:
                        novelty = critique_result.get('novelty')
                        accuracy = critique_result.get('accuracy')
                        relevancy = critique_result.get('relevancy')
                        critique = critique_result.get('critique', 'No critique available')
                        
                        # Determine verdict
                        verdict = self.automated_verdict(accuracy, novelty, relevancy)
                        
                        # Extract citations
                        citations = self.extract_citations_from_chunks(context_chunks)
                        formatted_citations = self.format_citations_for_export(citations)
                        
                        # Create record
                        record = {
                            'Meta_Hypothesis': meta_hypothesis,
                            'Meta_Hypothesis_Index': meta_idx,
                            'Hypothesis': hypothesis,
                            'Accuracy': accuracy,
                            'Novelty': novelty,
                            'Relevancy': relevancy,
                            'Verdict': verdict,
                            'Critique': critique,
                            'Citations': formatted_citations,
                            'Original_Query': user_prompt
                        }
                        
                        # Add to records
                        self.hypothesis_records.append(record)
                        
                        # Save incrementally to Excel
                        self.save_hypothesis_record_incrementally(record, excel_filename)
                        
                        # Display result
                        print(f"\n📊 Hypothesis Result:")
                        print(f"   Novelty: {novelty}/85")
                        print(f"   Accuracy: {accuracy}/80")
                        print(f"   Relevancy: {relevancy}/50")
                        print(f"   Verdict: {verdict}")
                        
                        # Accept hypothesis if it meets criteria
                        if verdict == "ACCEPTED":
                            meta_hypotheses_generated.append(record)
                            print(f"✅ Hypothesis accepted for meta-hypothesis {meta_idx}!")
                        else:
                            print(f"❌ Hypothesis rejected for meta-hypothesis {meta_idx}")
                        
                        # Check timer
                        if self.hypothesis_timer.check_expired():
                            print(f"⏰ Time limit reached for meta-hypothesis {meta_idx}. Moving to next meta-hypothesis.")
                            break
                    
                except Exception as e:
                    print(f"❌ Error generating/critiquing hypothesis: {e}")
                    continue
            
            print(f"✅ Generated {len(meta_hypotheses_generated)} hypotheses for meta-hypothesis {meta_idx}")
            all_hypotheses.extend(meta_hypotheses_generated)
        
        # Summary
        print(f"\n🎉 META-HYPOTHESIS GENERATION COMPLETE")
        print("=" * 80)
        print(f"Total meta-hypotheses processed: {total_meta_hypotheses}")
        print(f"Total hypotheses generated: {len(all_hypotheses)}")
        print(f"Average hypotheses per meta-hypothesis: {len(all_hypotheses)/total_meta_hypotheses:.1f}")
        
        # Final export summary
        if all_hypotheses:
            print(f"\n💾 Final export summary:")
            print(f"📊 All results have been incrementally saved to: {excel_filename}")
            print(f"📁 File location: {os.path.dirname(os.path.abspath(excel_filename))}")
            
            # Also create a final comprehensive export
            final_export_filename = self.export_hypotheses_to_excel()
            if final_export_filename:
                print(f"📋 Comprehensive results also exported to: {final_export_filename}")
            else:
                print(f"⚠️  Comprehensive export failed, but incremental saves are complete")
        else:
            print(f"❌ No hypotheses were successfully generated")
        
        return all_hypotheses

    def check_api_status(self):
        """Check Gemini API status and provide guidance."""
        print("\n🔍 Checking Gemini API Status...")
        
        if not self.gemini_client:
            print("❌ Gemini client not initialized")
            print("💡 Check your API key in keys.json")
            return
        
        # Show current rate limit status
        remaining_requests = self.gemini_rate_limiter.get_remaining_requests()
        print(f"📊 Gemini API Rate Limit Status:")
        print(f"   - Limit: 1000 requests per minute")
        print(f"   - Remaining requests: {remaining_requests}")
        print(f"   - Used requests: {1000 - remaining_requests}")
        
        try:
            # Try a simple test request
            test_response = self.gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents="Hello"
            )
            print("✅ Gemini API is working")
            print("💡 You can use 'generate' command for hypothesis generation")
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                print("❌ API quota exceeded (rate limited)")
                print("💡 Solutions:")
                print("   - Wait 30-60 minutes for quota reset")
                print("   - Use 'generate_offline' for fallback hypotheses")
                print("   - Check your billing/plan at https://ai.google.dev/")
                print("   - Consider upgrading your plan for higher limits")
            elif "401" in error_str or "403" in error_str:
                print("❌ API authentication failed")
                print("💡 Check your API key in keys.json")
            else:
                print(f"❌ API error: {e}")
                print("💡 Try again later or use 'generate_offline'")

    def is_chromadb_ready(self):
        """Check if ChromaDB is ready and has data for searching."""
        if not self.use_chromadb or not self.chroma_manager:
            return False
        
        try:
            stats = self.chroma_manager.get_collection_stats()
            return stats.get('total_documents', 0) > 0
        except Exception as e:
            logger.error(f"Error checking ChromaDB readiness: {e}")
            return False

    def _initialize_hypothesis_tools(self):
        """Initialize the HypothesisGenerator, HypothesisCritic, and MetaHypothesisGenerator."""
        if self.gemini_client:
            self.hypothesis_generator = HypothesisGenerator(model=self.gemini_client)
            def embedding_fn(text):
                return np.array(self.get_google_embedding(text))
            self.hypothesis_critic = HypothesisCritic(model=self.gemini_client, embedding_fn=embedding_fn)
            self.meta_hypothesis_generator = MetaHypothesisGenerator(model=self.gemini_client)
            print("✅ Hypothesis tools initialized successfully.")
        else:
            print("⚠️ Gemini client not initialized, skipping hypothesis tools.")

    def export_hypotheses_to_excel(self, filename=None):
        """Export hypothesis records to Excel file with grouping by meta-hypothesis."""
        if not self.hypothesis_records:
            print("❌ No hypothesis records to export.")
            return None
        
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if filename is None:
                # Ensure hypothesis_export directory exists
                export_dir = "hypothesis_export"
                os.makedirs(export_dir, exist_ok=True)
                filename = os.path.join(export_dir, f"hypothesis_export_{timestamp}.xlsx")
            
            import pandas as pd
            from openpyxl import Workbook
            from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
            
            # Check if we have meta-hypothesis data
            has_meta_data = any('Meta_Hypothesis' in record for record in self.hypothesis_records)
            
            if has_meta_data:
                # Group by meta-hypothesis
                meta_groups = {}
                for record in self.hypothesis_records:
                    meta_hypothesis = record.get('Meta_Hypothesis', 'Unknown')
                    if meta_hypothesis not in meta_groups:
                        meta_groups[meta_hypothesis] = []
                    meta_groups[meta_hypothesis].append(record)
                
                # Create workbook with multiple sheets
                workbook = Workbook()
                workbook.remove(workbook.active)  # Remove default sheet
                
                # Define styles
                header_font = Font(bold=True, color="FFFFFF")
                header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
                meta_header_font = Font(bold=True, color="FFFFFF", size=12)
                meta_header_fill = PatternFill(start_color="C5504B", end_color="C5504B", fill_type="solid")
                border = Border(
                    left=Side(style='thin'),
                    right=Side(style='thin'),
                    top=Side(style='thin'),
                    bottom=Side(style='thin')
                )
                
                # Create summary sheet
                summary_sheet = workbook.create_sheet("Summary")
                summary_sheet['A1'] = "Meta-Hypothesis Summary"
                summary_sheet['A1'].font = meta_header_font
                summary_sheet['A1'].fill = meta_header_fill
                
                summary_sheet['A3'] = "Meta-Hypothesis"
                summary_sheet['B3'] = "Hypotheses Generated"
                summary_sheet['C3'] = "Accepted"
                summary_sheet['D3'] = "Rejected"
                summary_sheet['E3'] = "Average Novelty"
                summary_sheet['F3'] = "Average Accuracy"
                
                for col in ['A3', 'B3', 'C3', 'D3', 'E3', 'F3']:
                    summary_sheet[col].font = header_font
                    summary_sheet[col].fill = header_fill
                    summary_sheet[col].border = border
                
                row = 4
                for meta_hypothesis, records in meta_groups.items():
                    summary_sheet[f'A{row}'] = meta_hypothesis[:100] + "..." if len(meta_hypothesis) > 100 else meta_hypothesis
                    summary_sheet[f'B{row}'] = len(records)
                    
                    accepted_count = sum(1 for r in records if r.get('Verdict') == 'ACCEPTED')
                    rejected_count = len(records) - accepted_count
                    summary_sheet[f'C{row}'] = accepted_count
                    summary_sheet[f'D{row}'] = rejected_count
                    
                    avg_novelty = sum(r.get('Novelty', 0) for r in records if r.get('Novelty') is not None) / len(records)
                    avg_accuracy = sum(r.get('Accuracy', 0) for r in records if r.get('Accuracy') is not None) / len(records)
                    summary_sheet[f'E{row}'] = round(avg_novelty, 1)
                    summary_sheet[f'F{row}'] = round(avg_accuracy, 1)
                    
                    for col in ['A', 'B', 'C', 'D', 'E', 'F']:
                        summary_sheet[f'{col}{row}'].border = border
                    
                    row += 1
                
                # Auto-adjust column widths for summary
                for column in summary_sheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    summary_sheet.column_dimensions[column_letter].width = adjusted_width
                
                # Create detailed sheets for each meta-hypothesis
                for i, (meta_hypothesis, records) in enumerate(meta_groups.items(), 1):
                    # Create sheet name (Excel has 31 character limit)
                    sheet_name = f"Meta_{i}"
                    if len(meta_hypothesis) > 20:
                        sheet_name += f"_{meta_hypothesis[:20].replace(' ', '_')}"
                    else:
                        sheet_name += f"_{meta_hypothesis.replace(' ', '_')}"
                    
                    # Clean sheet name for Excel compatibility
                    sheet_name = "".join(c for c in sheet_name if c.isalnum() or c in ('_', '-'))[:31]
                    
                    sheet = workbook.create_sheet(sheet_name)
                    
                    # Add meta-hypothesis header
                    sheet['A1'] = f"Meta-Hypothesis {i}: {meta_hypothesis}"
                    sheet['A1'].font = meta_header_font
                    sheet['A1'].fill = meta_header_fill
                    sheet.merge_cells('A1:H1')
                    
                    # Add hypothesis data
                    df_group = pd.DataFrame(records)
                    if not df_group.empty:
                        # Write headers
                        headers = list(df_group.columns)
                        for col, header in enumerate(headers, 1):
                            cell = sheet.cell(row=3, column=col, value=header)
                            cell.font = header_font
                            cell.fill = header_fill
                            cell.border = border
                        
                        # Write data
                        for row_idx, record in enumerate(records, 4):
                            for col_idx, (key, value) in enumerate(record.items(), 1):
                                cell = sheet.cell(row=row_idx, column=col_idx, value=value)
                                cell.border = border
                        
                        # Auto-adjust column widths
                        for column in sheet.columns:
                            max_length = 0
                            column_letter = column[0].column_letter
                            for cell in column:
                                try:
                                    if len(str(cell.value)) > max_length:
                                        max_length = len(str(cell.value))
                                except:
                                    pass
                            adjusted_width = min(max_length + 2, 50)
                            sheet.column_dimensions[column_letter].width = adjusted_width
                
                # Save workbook
                workbook.save(filename)
                
            else:
                # Fallback to simple export for non-meta-hypothesis data
                df = pd.DataFrame(self.hypothesis_records)
                with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Hypotheses', index=False)
                    worksheet = writer.sheets['Hypotheses']
                    for column in worksheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass
                        adjusted_width = min(max_length + 2, 50)
                        worksheet.column_dimensions[column_letter].width = adjusted_width
            
            print(f"✅ Successfully exported {len(self.hypothesis_records)} hypothesis records to: {filename}")
            print(f"📁 File saved in: {os.path.dirname(os.path.abspath(filename))}")
            if has_meta_data:
                print(f"📊 Organized into {len(meta_groups)} meta-hypothesis groups with summary sheet")
            return filename
        except Exception as e:
            print(f"❌ Failed to export hypotheses to Excel: {e}")
            return None
    
    def save_hypothesis_record_incrementally(self, record, filename=None):
        """Save a single hypothesis record to Excel file incrementally."""
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Ensure hypothesis_export directory exists
            export_dir = "hypothesis_export"
            os.makedirs(export_dir, exist_ok=True)
            filename = os.path.join(export_dir, f"hypothesis_export_{timestamp}.xlsx")
        
        try:
            import pandas as pd
            from openpyxl import load_workbook
            
            # Check if file exists
            if os.path.exists(filename):
                # Load existing workbook
                workbook = load_workbook(filename)
                
                # Get or create the Hypotheses sheet
                if 'Hypotheses' in workbook.sheetnames:
                    worksheet = workbook['Hypotheses']
                    # Find the next empty row
                    next_row = worksheet.max_row + 1
                else:
                    worksheet = workbook.create_sheet('Hypotheses')
                    next_row = 1
                    # Add headers for new sheet
                    headers = list(record.keys())
                    for col, header in enumerate(headers, 1):
                        worksheet.cell(row=1, column=col, value=header)
                    next_row = 2
                
                # Add the record data
                for col, value in enumerate(record.values(), 1):
                    worksheet.cell(row=next_row, column=col, value=value)
                
                # Save the workbook
                workbook.save(filename)
                
            else:
                # Create new file
                df_row = pd.DataFrame([record])
                with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                    df_row.to_excel(writer, sheet_name='Hypotheses', index=False)
                    worksheet = writer.sheets['Hypotheses']
                    # Auto-adjust column widths
                    for column in worksheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass
                        adjusted_width = min(max_length + 2, 50)
                        worksheet.column_dimensions[column_letter].width = adjusted_width
            
            print(f"💾 Saved hypothesis record to: {filename}")
            print(f"📁 File saved in: {os.path.dirname(os.path.abspath(filename))}")
            return filename
            
        except Exception as e:
            print(f"❌ Failed to save hypothesis record incrementally: {e}")
            return None

    def clear_hypothesis_records(self):
        """Clear all stored hypothesis records."""
        self.hypothesis_records = []
        print("🗑️ Hypothesis records cleared.")
    
    def show_hypothesis_records(self):
        """Display current hypothesis records."""
        if not self.hypothesis_records:
            print("📝 No hypothesis records available.")
            return
        
        print(f"\n📝 Current Hypothesis Records ({len(self.hypothesis_records)} total):")
        print("=" * 80)
        
        for i, record in enumerate(self.hypothesis_records, 1):
            print(f"\n{i}. Hypothesis: {record['Hypothesis'][:100]}{'...' if len(record['Hypothesis']) > 100 else ''}")
            print(f"   Accuracy: {record['Accuracy']}, Novelty: {record['Novelty']}, Relevancy: {record['Relevancy']}, Verdict: {record['Verdict']}")
            print(f"   Citations: {len(record['Citations'].split(';')) if record['Citations'] != 'No citations available' else 0} sources")
            print("-" * 80)

    def _display_commands(self):
        """Display available commands."""
        print(f"\n💡 Available commands:")
        print(f"   - 'add <query>': Search, batch, and generate hypotheses from all relevant results (e.g., 'add cancer')")
        print(f"   - 'add <N> <query>': Search, batch, and generate hypotheses from up to N results (e.g., 'add 2000 cancer')")
        print(f"   - 'add all <query>': Search, batch, and generate hypotheses from ALL found results (e.g., 'add all cancer')")
        print(f"   - 'meta <query>': Use meta-hypothesis generator to create 5 diverse research directions, then generate hypotheses for each (e.g., 'meta UBR-5 in cancer')")
        print(f"   - 'meta <chunks> <query>': Advanced usage - specify number of chunks per meta-hypothesis (e.g., 'meta 2000 UBR-5 in cancer')")
        print(f"   - 'clear': Clear current package")
        print(f"   - 'export': Export hypothesis records to Excel")
        print(f"   - 'records': Show current hypothesis records")
        print(f"   - 'clear_records': Clear hypothesis records")
        print(f"   💡 Note: Large queries may take longer and use more API calls. Use 'clear' if you get errors.")
        print(f"   🆕 Dynamic chunk selection: Each hypothesis will use new chunks from the database!")
        print(f"   🆕 Meta-hypothesis generator: Creates diverse research directions for comprehensive exploration!")
        print(f"   📚 Default: 1500 chunks per hypothesis generation for richer context!")
        print()

    def interactive_search(self):
        novelty_threshold = 85
        accuracy_threshold = 80
        relevancy_threshold = 50
        print("=== Enhanced RAG Query System ===")
        print("🔍 Search across all your knowledge bases!")
        self._display_commands()
        while True:
            query = input("❓ Your question (or command): ").strip()
            if query.lower() == 'clear':
                self.clear_package()
            elif query.lower() == 'export':
                self.export_hypotheses_to_excel()
            elif query.lower() == 'records':
                self.show_hypothesis_records()
            elif query.lower() == 'clear_records':
                self.clear_hypothesis_records()
            elif query.lower().startswith('meta '):
                try:
                    parts = query.split(' ', 1)
                    if len(parts) < 2:
                        print("❌ Usage: meta <query> (e.g., 'meta UBR-5 in cancer')")
                        print("   Advanced: meta <chunks> <query> (e.g., 'meta 2000 UBR-5 in cancer' for 2000 chunks per meta-hypothesis)")
                        continue
                    
                    # Check if first part is a number (chunks specification)
                    query_parts = parts[1].split(' ', 1)
                    if len(query_parts) > 1 and query_parts[0].isdigit():
                        chunks_per_meta = int(query_parts[0])
                        # Validate chunks number
                        if chunks_per_meta < 5:
                            print(f"❌ Too few chunks ({chunks_per_meta}). Minimum is 5 chunks per meta-hypothesis.")
                            continue
                        elif chunks_per_meta > 5000:
                            print(f"⚠️  Very large number of chunks ({chunks_per_meta}). This may cause API errors or timeouts.")
                            response = input("Continue anyway? (y/n): ").lower()
                            if response != 'y':
                                continue
                        meta_query = query_parts[1]
                        print(f"\n🧠 Meta-hypothesis generation for: '{meta_query}'")
                        print(f"📚 Using {chunks_per_meta} chunks per meta-hypothesis")
                    else:
                        chunks_per_meta = 1500  # default
                        meta_query = parts[1]
                        print(f"\n🧠 Meta-hypothesis generation for: '{meta_query}'")
                        print(f"📚 Using default {chunks_per_meta} chunks per meta-hypothesis")
                    
                    print("This will generate 5 diverse research directions, then create hypotheses for each.")
                    print("This process may take several minutes and use significant API calls.")
                    response = input("Continue? (y/n): ").lower()
                    if response == 'y':
                        try:
                            results = self.generate_hypotheses_with_meta_generator(meta_query, n_per_meta=3, chunks_per_meta=chunks_per_meta)
                            if results:
                                print(f"\n✅ Meta-hypothesis generation complete! Generated {len(results)} hypotheses across 5 research directions.")
                            else:
                                print("❌ Meta-hypothesis generation failed.")
                        except Exception as e:
                            print(f"❌ Error during meta-hypothesis generation: {e}")
                    else:
                        print("Meta-hypothesis generation cancelled.")
                except Exception as e:
                    print(f"❌ Error processing meta command: {e}")
            elif query.lower().startswith('add '):
                try:
                    parts = query.split(' ', 2)
                    if len(parts) < 2:
                        print("❌ Usage: add <query> (e.g., 'add cancer')")
                        continue
                    # Check for 'all' keyword
                    if len(parts) > 2 and parts[1].lower() == 'all':
                        num_results = None  # None means all
                        search_query = parts[2]
                    elif len(parts) > 2:
                        try:
                            num_results = int(parts[1])
                            if num_results <= 0:
                                print("❌ Number of results must be positive")
                                continue
                            if num_results > 100000:
                                print("⚠️  Number of results is very large (>100,000). This may use a lot of memory and cause API errors.")
                            search_query = parts[2]
                        except ValueError:
                            num_results = 100000
                            search_query = parts[1] + " " + parts[2]
                    else:
                        num_results = 100000
                        search_query = parts[1]
                    print(f"\n🔍 Searching for: '{search_query}'" + (f" (requesting ALL results)" if num_results is None else f" (requesting up to {num_results} results)"))
                    try:
                        # Decide which method to use
                        if num_results is None or num_results > 5000:
                            # Use get_all_documents and compute similarity in Python
                            print("⚠️  Using full-document scan and in-memory similarity search. This may use a lot of RAM and be slow for very large collections.")
                            # Get all documents (limit to 100,000 for safety)
                            max_docs = num_results if (num_results is not None and num_results < 100000) else 100000
                            all_docs = self.chroma_manager.get_all_documents(limit=max_docs)
                            if not all_docs:
                                print("❌ No documents found in ChromaDB.")
                                continue
                            print(f"📦 Retrieved {len(all_docs)} documents from ChromaDB. Computing similarities...")
                            # Get query embedding
                            query_embedding = self.get_google_embedding(search_query)
                            if not query_embedding:
                                print("❌ Failed to get query embedding.")
                                continue
                            import numpy as np
                            from tqdm.auto import tqdm
                            doc_embeddings = []
                            for doc in all_docs:
                                emb = doc['metadata'].get('embedding', None)
                                if emb is not None:
                                    doc_embeddings.append(emb)
                                else:
                                    doc_embeddings.append([0.0]*len(query_embedding))  # fallback if missing
                            doc_embeddings = np.array(doc_embeddings)
                            from sklearn.metrics.pairwise import cosine_similarity
                            similarities = np.zeros(len(doc_embeddings))
                            print(f"🔍 Computing similarities across {len(doc_embeddings):,} documents...")
                            with tqdm(total=len(doc_embeddings), desc="Similarity", unit="doc") as pbar:
                                batch_size = 1000
                                for start in range(0, len(doc_embeddings), batch_size):
                                    end = min(start + batch_size, len(doc_embeddings))
                                    similarities[start:end] = cosine_similarity([query_embedding], doc_embeddings[start:end])[0]
                                    pbar.update(end - start)
                            top_indices = np.argsort(similarities)[::-1][:num_results if num_results is not None else len(all_docs)]
                            results = []
                            for idx in top_indices:
                                results.append({
                                    'chunk': all_docs[idx]['document'],
                                    'metadata': all_docs[idx]['metadata'],
                                    'similarity': similarities[idx],
                                    'method': 'manual_cosine'
                                })
                        else:
                            # Use fast ChromaDB search
                            results = self.search_hybrid(search_query, top_k=num_results, filter_dict=None)
                        if not results:
                            print("❌ No results found. Try a different search term or check if data is loaded.")
                            continue
                        print(f"📦 Found {len(results)} relevant chunks. Beginning batch processing...")
                        # Batch process for hypothesis generation
                        batch_size = 500
                        all_chunks = [r['chunk'] for r in results]
                        # Sequential, real-time generator-critic loop for 5 accepted hypotheses
                        print(f"\n🧠 Generating hypotheses one at a time until 5 are accepted (novelty >= {novelty_threshold}, accuracy >= {accuracy_threshold}, 5-minute time limit per critique)...")
                        accepted_hypotheses = []
                        attempts = 0
                        max_attempts = 100  # Prevent infinite loops
                        while len(accepted_hypotheses) < 5 and attempts < max_attempts:
                            attempts += 1
                            print(f"\n🧠 Generating hypothesis attempt {attempts}...")
                            
                            # Select new chunks from ChromaDB for this hypothesis generation
                            if self.use_chromadb and self.chroma_manager and self.initial_add_quantity:
                                print(f"🔄 Selecting new chunks for hypothesis attempt {attempts}...")
                                dynamic_chunks = self.select_dynamic_chunks_for_generation(search_query)
                                if dynamic_chunks:
                                    all_chunks = dynamic_chunks
                                    print(f"📚 Using {len(all_chunks)} dynamically selected chunks for generation")
                                else:
                                    print(f"⚠️  Failed to select dynamic chunks, using original chunks")
                            else:
                                print(f"📚 Using original chunks for generation (no dynamic selection available)")
                            
                            # Start timer for this hypothesis
                            self.hypothesis_timer.start()
                            print(f"⏱️  Starting 5-minute timer for hypothesis attempt {attempts}...")
                            
                            hypothesis_list = self._generate_hypotheses_with_retry(all_chunks, n=1)
                            if not hypothesis_list:
                                print("❌ Failed to generate hypothesis. Skipping...")
                                # Record failed generation
                                self.hypothesis_records.append({
                                    'Hypothesis': f'Failed generation attempt {attempts}',
                                    'Accuracy': None,
                                    'Novelty': None,
                                    'Verdict': 'GENERATION_FAILED',
                                    'Critique': 'Failed to generate hypothesis',
                                    'Citations': 'No citations available'
                                })
                                continue
                            
                            hypothesis = hypothesis_list[0]
                            print(f"Generated Hypothesis: {hypothesis}")
                            print(f"\n🔄 Critiquing Hypothesis: {hypothesis}")
                            
                            # Check timer before critique
                            if self.hypothesis_timer.check_expired():
                                print(f"⏰ Time limit reached for hypothesis attempt {attempts}. Moving to next attempt.")
                                # Record timeout result
                                self.hypothesis_records.append({
                                    'Hypothesis': hypothesis,
                                    'Accuracy': None,
                                    'Novelty': None,
                                    'Verdict': 'TIMEOUT',
                                    'Critique': 'Critique process timed out after 5 minutes',
                                    'Citations': 'No citations available'
                                })
                                continue
                            
                            critique_result = self._critique_hypothesis_with_retry(hypothesis, all_chunks, search_query)
                            if not critique_result:
                                print(f"❌ Failed to critique hypothesis after retries. Skipping...")
                                # Record failed critique
                                self.hypothesis_records.append({
                                    'Hypothesis': hypothesis,
                                    'Accuracy': None,
                                    'Novelty': None,
                                    'Verdict': 'CRITIQUE_FAILED',
                                    'Critique': 'Failed to critique hypothesis after retries',
                                    'Citations': 'No citations available'
                                })
                                continue
                            
                            # Check timer after critique
                            if self.hypothesis_timer.check_expired():
                                print(f"⏰ Time limit reached for hypothesis attempt {attempts} after critique. Moving to next attempt.")
                                # Record timeout result with partial critique
                                self.hypothesis_records.append({
                                    'Hypothesis': hypothesis,
                                    'Accuracy': critique_result.get('accuracy', None),
                                    'Novelty': critique_result.get('novelty', None),
                                    'Verdict': 'TIMEOUT',
                                    'Critique': critique_result.get('critique', '') + ' [Process timed out]',
                                    'Citations': 'No citations available'
                                })
                                continue
                            
                            novelty = critique_result.get('novelty', 0)
                            accuracy = critique_result.get('accuracy', 0)
                            relevancy = critique_result.get('relevancy', 0)
                            verdict = self.automated_verdict(accuracy, novelty, relevancy, accuracy_threshold, novelty_threshold, relevancy_threshold)
                            remaining_time = self.hypothesis_timer.get_remaining_time()
                            print(f"   Novelty: {novelty}/{novelty_threshold}, Accuracy: {accuracy}/{accuracy_threshold}, Relevancy: {relevancy}/{relevancy_threshold}, Automated Verdict: {verdict}")
                            print(f"   ⏱️  Time remaining: {remaining_time:.1f}s")
                            
                            # Extract citations from results metadata
                            citations = []
                            unique_sources = set()
                            for result in results:
                                metadata = result.get('metadata', {})
                                if metadata:
                                    source_name = metadata.get('source_name', 'Unknown')
                                    title = metadata.get('title', 'No title')
                                    doi = metadata.get('doi', 'No DOI')
                                    authors = metadata.get('authors', 'Unknown authors')
                                    journal = metadata.get('journal', 'Unknown journal')
                                    year = metadata.get('year', 'Unknown year')
                                    
                                    citation = {
                                        'source_name': source_name,
                                        'title': title,
                                        'doi': doi,
                                        'authors': authors,
                                        'journal': journal,
                                        'year': year
                                    }
                                    
                                    # Add to citations if not already present
                                    citation_key = doi if doi != 'No DOI' else title
                                    if citation_key not in unique_sources:
                                        citations.append(citation)
                                        unique_sources.add(citation_key)
                            
                            formatted_citations = self.format_citations_for_export(citations)
                            
                            # Track record for export
                            self.hypothesis_records.append({
                                'Hypothesis': hypothesis,
                                'Accuracy': accuracy,
                                'Novelty': novelty,
                                'Relevancy': relevancy,
                                'Verdict': verdict,
                                'Critique': critique_result.get('critique', ''),
                                'Citations': formatted_citations
                            })
                            
                            if verdict == 'ACCEPTED':
                                accepted_hypotheses.append({
                                    "hypothesis": hypothesis,
                                    "critique": critique_result,
                                    "score": (novelty + accuracy + relevancy) / 3
                                })
                                print(f"✅ Hypothesis accepted! ({len(accepted_hypotheses)}/5)")
                            else:
                                print(f"❌ Hypothesis rejected (verdict: {verdict})")
                        
                        if not accepted_hypotheses:
                            print("❌ No hypotheses were successfully critiqued.")
                            continue
                        
                        print(f"\n🏆 Top {len(accepted_hypotheses)} Hypotheses:")
                        print("=" * 80)
                        for i, result in enumerate(accepted_hypotheses, 1):
                            hypothesis = result["hypothesis"]
                            score = result["score"]
                            critique = result["critique"]
                            print(f"\n{i}. {hypothesis}")
                            print(f"   Score: {score:.1f}")
                            print(f"   Novelty: {critique.get('novelty', 'N/A')}")
                            print(f"   Accuracy: {critique.get('accuracy', 'N/A')}")
                            print(f"   Relevancy: {critique.get('relevancy', 'N/A')}")
                            print(f"   Automated Verdict: {self.automated_verdict(critique.get('accuracy', 0), critique.get('novelty', 0), critique.get('relevancy', 0), accuracy_threshold, novelty_threshold, relevancy_threshold)}")
                            print(f"   Critique: {critique.get('critique', 'N/A')}")
                            print("-" * 80)
                        
                        # Auto-export results
                        print(f"\n💾 Auto-exporting hypothesis records...")
                        export_filename = self.export_hypotheses_to_excel()
                        if export_filename:
                            print(f"📊 Results exported to: {export_filename}")
                            print(f"📁 File saved in: {os.path.dirname(os.path.abspath(export_filename))}")
                        
                        # Enter discussion loop
                        print("\n💬 You can now discuss these hypotheses with the AI. Type the number of a hypothesis (1-5) to select it, or 'exit' to leave discussion mode.")
                        selected_idx = 0
                        while True:
                            user_input = input("[Discussion] > ").strip()
                            if user_input.lower() in ['exit', 'quit']:
                                print("Exiting discussion mode.")
                                break
                            if user_input.isdigit() and 1 <= int(user_input) <= len(accepted_hypotheses):
                                selected_idx = int(user_input) - 1
                                print(f"Selected Hypothesis {user_input}:\n{accepted_hypotheses[selected_idx]['hypothesis']}")
                                print("You can now ask questions about this hypothesis. Type 'back' to select another, or 'exit' to leave.")
                                while True:
                                    q = input(f"[H{selected_idx+1} Q&A] > ").strip()
                                    if q.lower() in ['exit', 'quit']:
                                        print("Exiting discussion mode.")
                                        return
                                    if q.lower() == 'back':
                                        print("Returning to hypothesis selection.")
                                        break
                                    # Use the critic to answer the question about the selected hypothesis
                                    hypothesis = accepted_hypotheses[selected_idx]['hypothesis']
                                    context_chunks = all_chunks if 'all_chunks' in locals() else []
                                    if self.hypothesis_critic and self.hypothesis_critic.model:
                                        # Build a prompt for Q&A
                                        prompt = f"You are an expert scientific reviewer. The user has a question about the following hypothesis.\n\nHypothesis:\n{hypothesis}\n\nUser Question:\n{q}\n\nPlease answer in detail, using the literature context if relevant."
                                        response = self.hypothesis_critic.model.models.generate_content(
                                            model="gemini-2.5-flash",
                                            contents=prompt
                                        )
                                        print(f"AI: {response.text.strip()}")
                                    else:
                                        print(f"AI: (No LLM available) This is a placeholder answer about '{hypothesis}'. User asked: {q}")
                            else:
                                print(f"Please enter a number between 1 and {len(accepted_hypotheses)}, or 'exit'.")
                    except Exception as e:
                        print(f"❌ Search or generation failed: {e}")
                        print("💡 Try again with a different query.")
                    self._display_commands()
                except Exception as e:
                    print(f"❌ Command parsing error: {e}")
                    print("❌ Usage: add <query> (e.g., 'add cancer')")
            elif query == '':
                continue
            else:
                print("❌ Unknown command. Available commands: 'add', 'clear', 'export', 'records', 'clear_records'")
                self._display_commands()

    def run_comprehensive_hypothesis_session(self, query, max_hypotheses=5):
        """
        Run a comprehensive hypothesis generation session with all features:
        - Timer-controlled critique process
        - Automated verdict determination
        - Citation tracking
        - Excel export
        """
        print("=" * 80)
        print("🧠 COMPREHENSIVE HYPOTHESIS GENERATION SESSION")
        print("=" * 80)
        print(f"Query: {query}")
        print(f"Features: 5-minute timer per hypothesis, automated verdicts, citation tracking, Excel export")
        print("=" * 80)
        
        # Clear previous records
        self.clear_hypothesis_records()
        
        # Run hypothesis generation
        print(f"\n🔍 Searching for relevant context...")
        context_chunks = self.retrieve_relevant_chunks(query, top_k=1500)
        if not context_chunks:
            print("❌ No relevant context found.")
            return None
        
        print(f"📚 Found {len(context_chunks)} relevant context chunks")
        
        # Generate hypotheses with all features
        print(f"\n🧠 Generating {max_hypotheses} hypotheses with comprehensive evaluation...")
        
        accepted_hypotheses = []
        attempts = 0
        max_attempts = 50  # Prevent infinite loops
        
        while len(accepted_hypotheses) < max_hypotheses and attempts < max_attempts:
            attempts += 1
            print(f"\n🔄 Hypothesis Attempt {attempts}/{max_attempts}")
            print("-" * 60)
            
            # Select new chunks from ChromaDB for this hypothesis generation
            if self.use_chromadb and self.chroma_manager and self.initial_add_quantity:
                print(f"🔄 Selecting new chunks for hypothesis attempt {attempts}...")
                dynamic_chunks = self.select_dynamic_chunks_for_generation(query)
                if dynamic_chunks:
                    context_chunks = dynamic_chunks
                    print(f"📚 Using {len(context_chunks)} dynamically selected chunks for generation")
                else:
                    print(f"⚠️  Failed to select dynamic chunks, using original chunks")
            else:
                print(f"📚 Using original chunks for generation (no dynamic selection available)")
            
            # Start timer
            self.hypothesis_timer.start()
            print(f"⏱️  Timer started (5-minute limit)")
            
            # Generate hypothesis
            try:
                context_texts = [chunk.get("document", "") if isinstance(chunk, dict) else chunk for chunk in context_chunks]
                hypothesis_list = self.hypothesis_generator.generate(context_texts, n=1)
                
                if not hypothesis_list:
                    print("❌ Failed to generate hypothesis")
                    self.hypothesis_records.append({
                        'Hypothesis': f'Failed generation attempt {attempts}',
                        'Accuracy': None,
                        'Novelty': None,
                        'Verdict': 'GENERATION_FAILED',
                        'Critique': 'Failed to generate hypothesis',
                        'Citations': 'No citations available'
                    })
                    continue
                
                hypothesis = hypothesis_list[0]
                print(f"📝 Generated: {hypothesis}")
                
                # Check timer before critique
                if self.hypothesis_timer.check_expired():
                    print(f"⏰ Time limit reached before critique")
                    self.hypothesis_records.append({
                        'Hypothesis': hypothesis,
                        'Accuracy': None,
                        'Novelty': None,
                        'Verdict': 'TIMEOUT',
                        'Critique': 'Process timed out before critique',
                        'Citations': 'No citations available'
                    })
                    continue
                
                # Critique hypothesis
                print(f"🔍 Critiquing hypothesis...")
                critique_result = self.hypothesis_critic.critique(hypothesis, context_texts, prompt=query, lab_goals=LAB_GOALS)
                
                # Check timer after critique
                if self.hypothesis_timer.check_expired():
                    print(f"⏰ Time limit reached after critique")
                    self.hypothesis_records.append({
                        'Hypothesis': hypothesis,
                        'Accuracy': critique_result.get('accuracy', None),
                        'Novelty': critique_result.get('novelty', None),
                        'Verdict': 'TIMEOUT',
                        'Critique': critique_result.get('critique', '') + ' [Process timed out]',
                        'Citations': 'No citations available'
                    })
                    continue
                
                # Extract scores and determine verdict
                novelty = critique_result.get('novelty', 0)
                accuracy = critique_result.get('accuracy', 0)
                relevancy = critique_result.get('relevancy', 0)
                verdict = self.automated_verdict(accuracy, novelty, relevancy, accuracy_threshold=80, novelty_threshold=85, relevancy_threshold=50)
                remaining_time = self.hypothesis_timer.get_remaining_time()
                
                print(f"📊 Scores: Accuracy={accuracy}, Novelty={novelty}, Relevancy={relevancy}")
                print(f"⚖️  Automated Verdict: {verdict}")
                print(f"⏱️  Time remaining: {remaining_time:.1f}s")
                
                # Extract citations
                citations = self.extract_citations_from_chunks(context_chunks)
                formatted_citations = self.format_citations_for_export(citations)
                citation_count = len(citations)
                
                print(f"📚 Citations: {citation_count} unique sources")
                
                # Record hypothesis
                self.hypothesis_records.append({
                    'Hypothesis': hypothesis,
                    'Accuracy': accuracy,
                    'Novelty': novelty,
                    'Relevancy': relevancy,
                    'Verdict': verdict,
                    'Critique': critique_result.get('critique', ''),
                    'Citations': formatted_citations
                })
                
                # Check if accepted
                if verdict == 'ACCEPTED':
                    accepted_hypotheses.append({
                        "hypothesis": hypothesis,
                        "critique": critique_result,
                        "score": (novelty + accuracy + relevancy) / 3
                    })
                    print(f"✅ ACCEPTED! ({len(accepted_hypotheses)}/{max_hypotheses})")
                else:
                    print(f"❌ REJECTED (below thresholds)")
                
            except Exception as e:
                print(f"❌ Error during hypothesis processing: {e}")
                self.hypothesis_records.append({
                    'Hypothesis': f'Error in attempt {attempts}',
                    'Accuracy': None,
                    'Novelty': None,
                    'Verdict': 'ERROR',
                    'Critique': f'Error: {str(e)}',
                    'Citations': 'No citations available'
                })
                continue
        
        # Final summary
        print(f"\n" + "=" * 80)
        print(f"🏁 SESSION COMPLETE")
        print(f"=" * 80)
        print(f"Total attempts: {attempts}")
        print(f"Accepted hypotheses: {len(accepted_hypotheses)}")
        print(f"Total records: {len(self.hypothesis_records)}")
        
        # Export results
        print(f"\n💾 Exporting results to Excel...")
        export_filename = self.export_hypotheses_to_excel()
        
        if export_filename:
            print(f"✅ Results exported to: {export_filename}")
            print(f"📁 File saved in: {os.path.dirname(os.path.abspath(export_filename))}")
        else:
            print(f"❌ Export failed")
        
        return accepted_hypotheses

    def generate_hypotheses_with_per_hypothesis_timer(self, n=5, max_rounds=10, filename=None):
        """
        Generate n hypotheses, using a 5-minute timer for the entire process of generating and refining each hypothesis.
        After each iteration, print the result, time left, scores, and hypothesis text.
        Save every iteration in self.hypothesis_records and append to Excel after each iteration.
        Adds empty verifier columns separated by an empty column.
        """
        import pandas as pd
        from openpyxl import load_workbook
        from datetime import datetime
        if not self.current_package["chunks"]:
            print("❌ Package is empty. Add some chunks first.")
            return
        if not self.hypothesis_generator or not self.hypothesis_critic:
            print("❌ Hypothesis tools not initialized. Check if Gemini API key is available.")
            return
        novelty_threshold = 85
        accuracy_threshold = 80
        relevancy_threshold = 50
        print(f"\n🧠 Generating {n} hypotheses (5-minute timer per hypothesis)...")
        print(f"📦 Using {len(self.current_package['chunks'])} package chunks for critique")
        print(f"[INFO] Acceptance criteria: Novelty >= {novelty_threshold}, Accuracy >= {accuracy_threshold}, Relevancy >= {relevancy_threshold}")
        package_chunks = self.current_package["chunks"]
        self.hypothesis_records = []  # Clear previous records
        # Prepare Excel file
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Ensure hypothesis_export directory exists
            export_dir = "hypothesis_export"
            os.makedirs(export_dir, exist_ok=True)
            filename = os.path.join(export_dir, f"hypothesis_export_{timestamp}.xlsx")
        # Write header row at the start, with empty separator and verifier columns
        columns = [
            'HypothesisNumber', 'Iteration', 'Status', 'Hypothesis', 'Novelty', 'Accuracy', 'Critique', 'TimeLeft',
            '',  # Empty separator column
            'Verifier Novelty Score (0-100)', 'Verifier Accuracy Score (0-100)', 'Verifier Verdict (accept, refuse)'
        ]
        pd.DataFrame(columns=columns).to_excel(filename, index=False)
        for hyp_idx in range(n):
            print(f"\nGenerating hypothesis {hyp_idx+1}/{n}\n")
            
            # Select new chunks from ChromaDB for this hypothesis generation
            if self.use_chromadb and self.chroma_manager and self.initial_add_quantity:
                print(f"🔄 Selecting new chunks for hypothesis {hyp_idx+1}...")
                dynamic_chunks = self.select_dynamic_chunks_for_generation("package query")
                if dynamic_chunks:
                    context_texts = dynamic_chunks
                    print(f"📚 Using {len(context_texts)} dynamically selected chunks for generation")
                else:
                    print(f"⚠️  Failed to select dynamic chunks, using original chunks")
                    context_texts = [chunk for chunk in package_chunks]
            else:
                print(f"📚 Using original chunks for generation (no dynamic selection available)")
                context_texts = [chunk for chunk in package_chunks]
            
            self.hypothesis_timer.start()
            time_left = self.hypothesis_timer.get_remaining_time()
            hypothesis = self.hypothesis_generator.generate(context_texts, n=1)[0]
            accepted = False
            iteration = 0
            while not accepted and not self.hypothesis_timer.check_expired() and iteration < max_rounds:
                iteration += 1
                critique_result = self.hypothesis_critic.critique(hypothesis, context_texts, prompt=query, lab_goals=LAB_GOALS)
                novelty = critique_result.get('novelty', 0)
                accuracy = critique_result.get('accuracy', 0)
                relevancy = critique_result.get('relevancy', 0)
                time_left = self.hypothesis_timer.get_remaining_time()
                verdict = self.automated_verdict(accuracy, novelty, relevancy, accuracy_threshold=80, novelty_threshold=85, relevancy_threshold=50)
                print(f"Iteration {iteration}: {verdict} | Novelty: {novelty}/85 | Accuracy: {accuracy}/80 | Relevancy: {relevancy}/50 | Time left: {int(time_left)}s")
                print(f"Hypothesis: {hypothesis}")
                # Save this iteration to records
                record = {
                    'HypothesisNumber': hyp_idx+1,
                    'Iteration': iteration,
                    'Status': verdict,
                    'Hypothesis': hypothesis,
                    'Novelty': novelty,
                    'Accuracy': accuracy,
                    'Critique': critique_result.get('critique', ''),
                    'TimeLeft': int(time_left),
                    '': '',  # Empty separator column
                    'Verifier Novelty Score (0-100)': '',
                    'Verifier Accuracy Score (0-100)': '',
                    'Verifier Verdict (accept, refuse)': ''
                }
                self.hypothesis_records.append(record)
                # Append to Excel after each iteration
                df_row = pd.DataFrame([record])
                with pd.ExcelWriter(filename, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                    writer.book = load_workbook(filename)
                    writer.sheets = {ws.title: ws for ws in writer.book.worksheets}
                    startrow = writer.book['Hypotheses'].max_row
                    df_row.to_excel(writer, sheet_name='Hypotheses', index=False, header=False, startrow=startrow)
                if verdict == "ACCEPTED":
                    print(f"✅ Hypothesis accepted! (verdict: {verdict})")
                    accepted = True
                    break
                else:
                    print(f"❌ Hypothesis rejected (verdict: {verdict})")
                if self.hypothesis_timer.check_expired():
                    print(f"⏰ Time limit reached for hypothesis {hyp_idx+1}. Moving to next hypothesis.")
                    break
                # Regenerate hypothesis for next iteration
                new_hypothesis = self.hypothesis_generator.generate(context_texts, n=1)
                if new_hypothesis:
                    hypothesis = new_hypothesis[0]
        print(f"\n✅ All iterations and results have been saved to: {filename}")
        print(f"📁 File saved in: {os.path.dirname(os.path.abspath(filename))}")
        return self.hypothesis_records

def main():
    """Main function to run the enhanced RAG query system."""
    print("=== Enhanced RAG System Startup ===")
    print("🚀 Starting Enhanced RAG System with ChromaDB...")
    
    # Initialize with ChromaDB enabled and fast startup
    rag_system = EnhancedRAGQuery(use_chromadb=True, load_data_at_startup=False)
    
    # Show new features
    print("\n" + "=" * 80)
    print("🎉 NEW FEATURES AVAILABLE:")
    print("=" * 80)
    print("⏱️  5-minute timer per hypothesis critique")
    print("🤖 Automated verdict determination (no manual input)")
    print("📚 Citation tracking and academic formatting")
    print("📊 Excel export with comprehensive data")
    print("📈 Real-time progress tracking")
    print("=" * 80)
    
    # Run interactive search
    rag_system.interactive_search()

if __name__ == "__main__":
    main() 