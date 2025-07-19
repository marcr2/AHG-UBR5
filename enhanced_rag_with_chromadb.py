import json
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
import os
from tqdm.auto import tqdm
from chromadb_manager import ChromaDBManager
import logging
from hypothesis_tools import HypothesisGenerator, HypothesisCritic
import time
import random
import threading
from collections import deque

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        self.last_context_chunks = None
        
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
                logger.info("📚 ChromaDB collection already populated")
                return
            
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
            from tqdm.auto import tqdm
            with tqdm(total=1, desc="Generating query embedding", unit="request") as pbar:
                # Add timeout to prevent hanging
                response = requests.post(url, headers=headers, json=data, timeout=30)
                response.raise_for_status()
                embedding = response.json()["embedding"]["values"]
                pbar.update(1)
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
                    from tqdm.auto import tqdm
                    with tqdm(total=len(batch_files), desc=f"Loading {source_dir} batches", unit="batch") as pbar:
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
                                
                                # Update progress bar with batch info
                                pbar.set_postfix({
                                    "chunks": batch_chunks,
                                    "embeddings": batch_embeddings,
                                    "total": source_chunks
                                })
                                
                            except Exception as e:
                                logger.error(f"❌ Failed to load batch {batch_file}: {e}")
                            
                            pbar.update(1)
                    
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
            return []
        
        # Get query embedding
        query_embedding = self.get_google_embedding(query)
        if not query_embedding:
            return []
        
        # Search in ChromaDB with progress bar
        from tqdm.auto import tqdm
        if show_progress:
            with tqdm(total=1, desc="ChromaDB search", unit="operation") as pbar:
                results = self.chroma_manager.search_similar(
                    query_embedding=query_embedding,
                    n_results=top_k,
                    where_filter=filter_dict
                )
                pbar.update(1)
        else:
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
            print("❌ No results found.")
            return
        
        print(f"🔍 Found {len(results)} relevant chunks:")
        
        # Calculate summary statistics
        total_chars = sum(len(result.get("chunk", "")) for result in results)
        sources = set()
        methods = set()
        
        for result in results:
            sources.add(result.get("metadata", {}).get("source", "Unknown"))
            methods.add(result.get("method", "Unknown"))
        
        print(f"📊 Summary:")
        print(f"   📚 Total chunks: {len(results)}")
        print(f"   📝 Total characters: {total_chars:,}")
        if show_method:
            print(f"   🔍 Search methods: {', '.join(methods)}")
        print(f"   📖 Sources: {', '.join(sources)}")
        
        # Show top 3 results as examples
        print(f"📄 Top 3 results (showing examples):")
        for i, result in enumerate(results[:3], 1):
            chunk = result.get("chunk", "")
            metadata = result.get("metadata", {})
            similarity = result.get("similarity", 0)
            
            # Get title or first part of chunk
            title = metadata.get("title", chunk[:60])
            if len(title) > 60:
                title = title[:57] + "..."
            
            print(f"   {i}. {title} (similarity: {similarity:.3f})")
        
        if len(results) > 3:
            print(f"   ... and {len(results) - 3} more chunks")
    
    def display_statistics(self):
        """Display comprehensive statistics about the knowledge base."""
        print(f"\n📊 Knowledge Base Statistics:")
        print("=" * 50)
        
        if self.embeddings_data:
            total_chunks = len(self.embeddings_data["chunks"])
            total_embeddings = len(self.embeddings_data["embeddings"])
            
            print(f"📈 Total chunks: {total_chunks}")
            print(f"📈 Total embeddings: {total_embeddings}")
            if total_embeddings > 0:
                print(f"📈 Embedding dimensions: {len(self.embeddings_data['embeddings'][0])}")
            
            print(f"\n📊 Source Breakdown:")
            for source, stats in self.embeddings_data["sources"].items():
                print(f"   {source}: {stats['chunks']} chunks")
                if "stats" in stats and stats["stats"]:
                    source_stats = stats["stats"]
                    print(f"     - Papers: {source_stats.get('total_papers', 'N/A')}")
                    print(f"     - Embeddings: {source_stats.get('total_embeddings', 'N/A')}")
        
        if self.use_chromadb and self.chroma_manager:
            print(f"\n🗄️ ChromaDB Statistics:")
            chroma_stats = self.chroma_manager.get_collection_stats()
            print(f"   Total documents: {chroma_stats.get('total_documents', 0)}")
            print(f"   Collection name: {chroma_stats.get('collection_name', 'N/A')}")
            print(f"   Available collections: {self.chroma_manager.list_collections()}")
    
    def retrieve_relevant_chunks(self, query, top_k=20):
        """Retrieve the most relevant chunks from all loaded batches using ChromaDB."""
        if not self.use_chromadb or not self.chroma_manager:
            print("❌ ChromaDB not available.")
            return []
        
        # Load data on-demand if not loaded at startup
        if not self.embeddings_data:
            print("📚 Loading embeddings data on-demand...")
            self.embeddings_data = self.load_all_embeddings()
            if not self.embeddings_data:
                return []
        # Get all documents (chunks) from the collection
        all_docs = self.chroma_manager.get_all_documents()
        if not all_docs:
            print("❌ No documents found in ChromaDB.")
            return []
        # Get query embedding
        query_embedding = self.get_google_embedding(query)
        if not query_embedding:
            print("❌ Failed to get query embedding.")
            return []
        # Compute cosine similarity for all chunks
        doc_embeddings = np.array([doc['metadata'].get('embedding', None) for doc in all_docs])
        # If embeddings are not stored in metadata, fallback to ChromaDB search
        if doc_embeddings[0] is None:
            # Use ChromaDB search_similar as fallback
            return self.search_chromadb(query, top_k=top_k)
        
        # Add progress bar for similarity computation
        from tqdm.auto import tqdm
        total_docs = len(all_docs)
        print(f"🔍 Computing similarities across {total_docs:,} ChromaDB documents...")
        
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [all_docs[i]['document'] for i in top_indices]

    def iterative_hypothesis_generation(self, user_prompt, max_rounds=5, n=3):
        """Run generator-critic feedback loop for each hypothesis until accepted (>=90% accuracy/novelty and verdict ACCEPT)."""
        
        if not self.hypothesis_generator or not self.hypothesis_critic:
            print("❌ Hypothesis tools not initialized. Check if Gemini API key is available.")
            return
        
        # Use stored context chunks if available, otherwise search for new ones
        if hasattr(self, 'last_context_chunks') and self.last_context_chunks:
            context_chunks = self.last_context_chunks
            print(f"\n🧠 Using {len(context_chunks)} previously found context chunks for hypothesis generation...")
        else:
            print(f"\n🔍 Searching all loaded batches for relevant context...")
            context_chunks = self.retrieve_relevant_chunks(user_prompt, top_k=20)
            if not context_chunks:
                print("❌ No relevant context found.")
                return
            self.last_context_chunks = context_chunks
        
        print(f"\n🧠 Generating initial hypotheses...")
        try:
            # Extract chunk text from context_chunks
            context_texts = [chunk.get("chunk", "") if isinstance(chunk, dict) else chunk for chunk in context_chunks]
            hypotheses = self.hypothesis_generator.generate(context_texts, n=n)
        except Exception as e:
            print(f"❌ Failed to generate initial hypotheses: {e}")
            return
        
        accepted = [False] * n
        rounds = [0] * n
        critiques = [{} for _ in range(n)]
        
        # Add progress tracking for the iterative process
        from tqdm.auto import tqdm
        total_iterations = max_rounds * n
        current_iteration = 0
        
        print(f"\n🔄 Starting iterative refinement (max {max_rounds} rounds per hypothesis)...")
        with tqdm(total=total_iterations, desc="Hypothesis refinement", unit="iteration") as pbar:
            while not all(accepted) and max(rounds) < max_rounds:
                for i in range(n):
                    if accepted[i]:
                        continue
                    rounds[i] += 1
                    current_iteration += 1
                    
                    print(f"\n🔄 Critique Round {rounds[i]} for Hypothesis {i+1}:")
                    try:
                        # Extract chunk text from context_chunks
                        context_texts = [chunk.get("chunk", "") if isinstance(chunk, dict) else chunk for chunk in context_chunks]
                        critique_result = self.hypothesis_critic.critique(hypotheses[i], context_texts)
                        critiques[i] = critique_result
                        print(f"Hypothesis {i+1}: {hypotheses[i]}")
                        print(f"Critique: {critique_result['critique']}")
                        novelty = critique_result.get('novelty', 0)
                        accuracy = critique_result.get('accuracy', 0)
                        verdict = critique_result.get('verdict', 'REJECT')
                        print(f"Novelty Score: {novelty}, Accuracy Score: {accuracy}, Verdict: {verdict}")
                        
                        if accuracy is not None and novelty is not None and accuracy >= 90 and novelty >= 90 and verdict == 'ACCEPT':
                            print(f"✅ Hypothesis {i+1} accepted (accuracy: {accuracy}, novelty: {novelty}, verdict: {verdict})")
                            accepted[i] = True
                        else:
                            print(f"❌ Hypothesis {i+1} rejected (accuracy: {accuracy}, novelty: {novelty}, verdict: {verdict}) - Regenerating...")
                            try:
                                # Extract chunk text from context_chunks
                                context_texts = [chunk.get("chunk", "") if isinstance(chunk, dict) else chunk for chunk in context_chunks]
                                new_hypothesis = self.hypothesis_generator.generate(context_texts, n=1)
                                if new_hypothesis:
                                    hypotheses[i] = new_hypothesis[0]
                            except Exception as e:
                                print(f"❌ Failed to regenerate hypothesis {i+1}: {e}")
                                continue
                        
                        # Calculate average rating for progress tracking
                        novelty = critique_result.get('novelty', 0)
                        accuracy = critique_result.get('accuracy', 0)
                        avg_rating = (novelty + accuracy) / 2 if novelty is not None and accuracy is not None else 0
                        
                        # Update progress bar with average rating
                        pbar.set_postfix({
                            "accepted": sum(accepted),
                            "round": max(rounds),
                            "current": f"H{i+1}",
                            "avg_rating": f"{avg_rating:.1f}"
                        })
                        pbar.update(1)
                        
                    except Exception as e:
                        print(f"❌ Failed to critique hypothesis {i+1}: {e}")
                        pbar.update(1)
                        continue
                    
                    # Break if all hypotheses are accepted
                    if all(accepted):
                        break
        
        print("\nFinal Hypotheses:")
        for i, hyp in enumerate(hypotheses, 1):
            status = "ACCEPTED" if accepted[i-1] else "FAILED (max rounds)"
            print(f"{i}. {hyp} [{status}]")
            if isinstance(critiques[i-1], dict) and critiques[i-1]:
                print(f"   Critique: {critiques[i-1].get('critique','')}")
                print(f"   Novelty: {critiques[i-1].get('novelty','')}, Accuracy: {critiques[i-1].get('accuracy','')}, Verdict: {critiques[i-1].get('verdict','')}")
        return hypotheses

    def add_to_package(self, search_results):
        """Add search results to the current package."""
        # Check if adding these results would make the package too large
        current_size = self.current_package["total_chars"]
        new_size = current_size + sum(len(result.get("chunk", "")) for result in search_results)
        
        # Warn if package would become very large (>2MB)
        if new_size > 2000000:
            print(f"⚠️  Adding {len(search_results)} chunks would make package very large ({new_size:,} characters)")
            print("💡 This may cause API errors. Consider using 'clear' first.")
            response = input("Continue anyway? (y/n): ").lower()
            if response != 'y':
                return
        
        for result in search_results:
            chunk = result.get("chunk", "")
            metadata = result.get("metadata", {})
            
            # Add to package
            self.current_package["chunks"].append(chunk)
            self.current_package["metadata"].append(metadata)
            self.current_package["sources"].add(metadata.get("source", "Unknown"))
            self.current_package["total_chars"] += len(chunk)
        
        print(f"📦 Added {len(search_results)} chunks to package")
        print(f"📦 Package now contains {len(self.current_package['chunks'])} chunks from {len(self.current_package['sources'])} sources")
        
        # Show size warning if package is getting large
        if self.current_package["total_chars"] > 1000000:
            print(f"⚠️  Package size: {self.current_package['total_chars']:,} characters (consider using 'clear' if you get API errors)")

    def clear_package(self):
        """Clear the current package."""
        self.current_package = {
            "chunks": [],
            "metadata": [],
            "sources": set(),
            "total_chars": 0
        }
        print("🗑️ Package cleared")

    def show_package(self):
        """Display information about the current package."""
        if not self.current_package["chunks"]:
            print("📦 Package is empty")
            return
        
        print(f"📦 Current Package:")
        print(f"   📚 Total chunks: {len(self.current_package['chunks'])}")
        print(f"   📝 Total characters: {self.current_package['total_chars']:,}")
        print(f"   📖 Sources: {', '.join(self.current_package['sources'])}")
        
        # Show top 3 chunks as examples
        print(f"\n📄 Sample chunks:")
        for i, chunk in enumerate(self.current_package["chunks"][:3], 1):
            metadata = self.current_package["metadata"][i-1]
            title = metadata.get("title", "No title")[:60]
            print(f"   {i}. {title}...")
        if len(self.current_package["chunks"]) > 3:
            print(f"   ... and {len(self.current_package['chunks']) - 3} more chunks")

    def generate_hypotheses_from_package(self, n=5):
        """Generate hypotheses using the package for critique and entire database for generation."""
        if not self.current_package["chunks"]:
            print("❌ Package is empty. Add some chunks first.")
            return
        
        if not self.hypothesis_generator or not self.hypothesis_critic:
            print("❌ Hypothesis tools not initialized. Check if Gemini API key is available.")
            return
        
        print(f"\n🧠 Generating {n} hypotheses...")
        print(f"📦 Using {len(self.current_package['chunks'])} package chunks for critique")
        
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
            print("📚 Loading embeddings data...")
            self.embeddings_data = self.load_all_embeddings()
            if not self.embeddings_data:
                print("❌ No embeddings data available")
                return
        
        all_chunks = self.embeddings_data["chunks"]
        package_chunks = self.current_package["chunks"]
        
        # Calculate safe sample size (aim for ~50MB payload)
        # Each chunk is roughly 1000 characters, so 50MB = ~50,000 chunks
        max_chunks_for_generation = 50000
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
        
        # Generate initial hypotheses using database sample with retry logic
        print(f"\n🧠 Generating initial hypotheses using database sample...")
        hypotheses = self._generate_hypotheses_with_retry(generation_chunks, n)
        if not hypotheses:
            print("❌ Failed to generate hypotheses after retries. API may be rate limited.")
            print("💡 Try again later or use a smaller package.")
            return
        
        # Critique each hypothesis using package chunks with retry logic
        print(f"\n🧑‍🔬 Critiquing hypotheses using package chunks...")
        from tqdm.auto import tqdm
        
        final_hypotheses = []
        for i, hypothesis in enumerate(hypotheses, 1):
            print(f"\n🔄 Critiquing Hypothesis {i}: {hypothesis}")
            critique_result = self._critique_hypothesis_with_retry(hypothesis, package_chunks)
            if not critique_result:
                print(f"❌ Failed to critique hypothesis {i} after retries. Skipping...")
                continue
            
            novelty = critique_result.get('novelty', 0)
            accuracy = critique_result.get('accuracy', 0)
            verdict = critique_result.get('verdict', 'REJECT')
            
            print(f"   Novelty: {novelty}, Accuracy: {accuracy}, Verdict: {verdict}")
            
            final_hypotheses.append({
                "hypothesis": hypothesis,
                "critique": critique_result,
                "score": (novelty + accuracy) / 2 if novelty is not None and accuracy is not None else 0
            })
        
        if not final_hypotheses:
            print("❌ No hypotheses were successfully critiqued.")
            return
        
        # Sort by score and return top 5
        final_hypotheses.sort(key=lambda x: x["score"], reverse=True)
        
        print(f"\n🏆 Top {n} Hypotheses:")
        print("=" * 80)
        for i, result in enumerate(final_hypotheses[:n], 1):
            hypothesis = result["hypothesis"]
            score = result["score"]
            critique = result["critique"]
            
            print(f"\n{i}. {hypothesis}")
            print(f"   Score: {score:.1f}")
            print(f"   Novelty: {critique.get('novelty', 'N/A')}")
            print(f"   Accuracy: {critique.get('accuracy', 'N/A')}")
            print(f"   Verdict: {critique.get('verdict', 'N/A')}")
            print(f"   Critique: {critique.get('critique', 'N/A')}")
            print("-" * 80)
        
        return final_hypotheses[:n]

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

    def _critique_hypothesis_with_retry(self, hypothesis, package_chunks, max_retries=3):
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
                critique_result = self.hypothesis_critic.critique(hypothesis, package_chunk_texts)
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

    def _initialize_hypothesis_tools(self):
        """Initialize the HypothesisGenerator and HypothesisCritic."""
        if self.gemini_client:
            self.hypothesis_generator = HypothesisGenerator(model=self.gemini_client)
            self.hypothesis_critic = HypothesisCritic(model=self.gemini_client)
            print("✅ Hypothesis tools initialized successfully.")
        else:
            print("⚠️ Gemini client not initialized, skipping hypothesis tools.")

    def _display_commands(self):
        """Display available commands."""
        print(f"\n💡 Search method:")
        print(f"   - 'chromadb': Fast vector database search (default)")
        
        print(f"\n💡 Package commands:")
        print(f"   - 'search <query>': Search and add results to package (e.g., 'search cancer')")
        print(f"   - 'add <query>': Search and add results to package (same as search)")
        print(f"   - 'package': Show current package contents")
        print(f"   - 'clear': Clear current package")
        print(f"   - 'generate': Generate top 5 hypotheses using package for critique and database for generation")
        print(f"   - 'generate_offline': Generate hypotheses without API (fallback mode)")
        print(f"   💡 Note: Automatically adds up to 500 results per search")
        
        print(f"\n💡 Other commands:")
        print(f"   - 'quit': Exit the system")
        print(f"   - 'stats': Show database statistics")
        print(f"   - 'filter <source>': Filter by source (e.g., 'filter pubmed')")
        print(f"   - 'clear filter': Clear current filter")
        print(f"   - 'api_status': Check Gemini API status and quota")
        print()

    def interactive_search(self):
        """Interactive search interface with hypothesis and critique commands."""
        print("=== Enhanced RAG Query System ===")
        print("🔍 Search across all your knowledge bases!")
        
        self._display_commands()
        
        current_filter = None
        
        while True:
            query = input("❓ Your question (or command): ").strip()
            
            if query.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif query.lower() == 'stats':
                self.display_statistics()
            elif query.lower().startswith('filter '):
                source = query[7:].strip()
                # Load data if needed for filtering
                if not self.embeddings_data:
                    print("📚 Loading embeddings data for filtering...")
                    self.embeddings_data = self.load_all_embeddings()
                    if not self.embeddings_data:
                        print("❌ No embeddings data available!")
                        continue
                
                if source in self.embeddings_data["sources"]:
                    current_filter = {"source_file": source}
                    print(f"🔍 Filter set to: {source}")
                else:
                    print(f"❌ Unknown source: {source}")
                    current_filter = None
            elif query.lower() == 'clear filter':
                current_filter = None
                print("🔍 Filter cleared")
            elif query.lower() == 'package':
                self.show_package()
            elif query.lower() == 'clear':
                self.clear_package()
            elif query.lower() == 'generate':
                self.generate_hypotheses_from_package(n=5)
            elif query.lower() == 'generate_offline':
                self.generate_hypotheses_offline(n=5)
            elif query.lower() == 'api_status':
                self.check_api_status()
            elif query.lower().startswith('search ') or query.lower().startswith('add '):
                # Search and add to package
                try:
                    parts = query.split(' ', 2)  # Split into ['search', 'N', 'query'] or ['add', 'N', 'query']
                    if len(parts) < 2:
                        print("❌ Usage: search <query> or add <query> (e.g., 'search cancer')")
                        continue
                    
                    # Check if second part is a number
                    if len(parts) == 2:
                        # No number specified, use maximum
                        num_results = 500
                        search_query = parts[1]
                    else:
                        # Check if second part is a number
                        try:
                            num_results = int(parts[1])
                            search_query = parts[2]
                        except ValueError:
                            # Second part is not a number, treat as query with max results
                            num_results = 500
                            search_query = parts[1] + " " + parts[2]
                    
                    if num_results <= 0 or num_results > 500:
                        print("❌ Number of results must be between 1 and 500")
                        continue
                    
                    print(f"\n🔍 Searching for: '{search_query}' (requesting {num_results} results)")
                    try:
                        results = self.search_hybrid(search_query, top_k=num_results, filter_dict=current_filter)
                        if results:
                            self.display_results(results)
                            self.add_to_package(results)
                        else:
                            print("❌ No results found. Try a different search term or check if data is loaded.")
                    except Exception as e:
                        print(f"❌ Search failed: {e}")
                        print("💡 Try running 'stats' to check data availability")
                    
                    # Display commands after search
                    self._display_commands()
                except Exception as e:
                    print(f"❌ Command parsing error: {e}")
                    print("❌ Usage: search <query> or add <query> (e.g., 'search cancer')")
            elif query:
                # Regular search (legacy support)
                print(f"\n🔍 Searching for: '{query}'")
                try:
                    results = self.search_hybrid(query, top_k=20, filter_dict=current_filter)
                    if results:
                        self.display_results(results)
                        print(f"\n�� Tip: Use 'search <query>' to add results to package for hypothesis generation")
                    else:
                        print("❌ No results found. Try a different search term or check if data is loaded.")
                except Exception as e:
                    print(f"❌ Search failed: {e}")
                    print("💡 Try running 'stats' to check data availability")
                
                # Display commands after search
                self._display_commands()
            else:
                print("❌ Please enter a question.")

def main():
    """Main function to run the enhanced RAG query system."""
    print("=== Enhanced RAG System Startup ===")
    print("Choose startup mode:")
    print("1. Fast startup (skip data loading, load on-demand)")
    print("2. Full startup (load all data at startup - may be slow)")
    
    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice == "1":
            load_at_startup = False
            print("🚀 Starting in fast mode...")
            break
        elif choice == "2":
            load_at_startup = True
            print("📚 Starting in full mode...")
            break
        else:
            print("❌ Please enter 1 or 2")
    
    # Initialize with ChromaDB enabled
    rag_system = EnhancedRAGQuery(use_chromadb=True, load_data_at_startup=load_at_startup)
    
    # Run interactive search
    rag_system.interactive_search()

if __name__ == "__main__":
    main() 