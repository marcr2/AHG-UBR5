import json
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
import os
from tqdm.auto import tqdm
from chromadb_manager import ChromaDBManager
import logging
from hypothesis_tools import HypothesisGenerator, HypothesisCritic

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        """
        Initialize the enhanced RAG query system.
        
        Args:
            use_chromadb: Whether to use ChromaDB for vector storage
            load_data_at_startup: Whether to load all embeddings data at startup (can be slow for large datasets)
        """
        self.use_chromadb = use_chromadb
        self.chroma_manager = None
        self.embeddings_data = None
        self.last_context_chunks = None  # Store last context for critique
        self.load_data_at_startup = load_data_at_startup

        print("🚀 Initializing Enhanced RAG System...")
        
        # Load API keys
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

        # Wire up LLM to generator/critic
        self.hypothesis_generator = HypothesisGenerator(model=self.gemini_client)
        self.hypothesis_critic = HypothesisCritic(model=self.gemini_client)
        
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
    
    def search_traditional(self, query, top_k=5, show_progress=True):
        """Traditional similarity search using cosine similarity."""
        # Load data on-demand if not loaded at startup
        if not self.embeddings_data:
            print("📚 Loading embeddings data on-demand...")
            self.embeddings_data = self.load_all_embeddings()
            if not self.embeddings_data:
                return []
        
        # Get query embedding
        query_embedding = self.get_google_embedding(query)
        if not query_embedding:
            print("⚠️  Embedding generation failed, falling back to text-based search...")
            return self._fallback_text_search(query, top_k)
        
        # Convert embeddings to numpy arrays
        stored_embeddings = np.array(self.embeddings_data["embeddings"])
        query_embedding = np.array(query_embedding).reshape(1, -1)
        
        # Calculate cosine similarities with progress bar
        from tqdm.auto import tqdm
        total_embeddings = len(stored_embeddings)
        
        if show_progress:
            print(f"🔍 Computing similarities across {total_embeddings:,} embeddings...")
            with tqdm(total=1, desc="Traditional search", unit="operation") as pbar:
                similarities = cosine_similarity(query_embedding, stored_embeddings)[0]
                pbar.update(1)
        else:
            similarities = cosine_similarity(query_embedding, stored_embeddings)[0]
        
        # Get top-k most similar chunks
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "chunk": self.embeddings_data["chunks"][idx],
                "metadata": self.embeddings_data["metadata"][idx],
                "similarity": float(similarities[idx]),
                "method": "traditional"
            })
        
        return results
    
    def _fallback_text_search(self, query, top_k=5):
        """Fallback text-based search when embedding generation fails."""
        print(f"🔍 Performing text-based search for: '{query}'")
        
        if not self.embeddings_data or not self.embeddings_data.get("chunks"):
            print("❌ No embeddings data available for fallback search")
            return []
        
        query_lower = query.lower()
        results = []
        
        # Simple keyword matching
        for i, chunk in enumerate(self.embeddings_data["chunks"]):
            chunk_lower = chunk.lower()
            
            # Calculate simple similarity based on keyword overlap
            query_words = set(query_lower.split())
            chunk_words = set(chunk_lower.split())
            
            if query_words & chunk_words:  # If there's any word overlap
                overlap = len(query_words & chunk_words)
                total_unique = len(query_words | chunk_words)
                similarity = overlap / total_unique if total_unique > 0 else 0
                
                if similarity > 0.1:  # Only include if there's meaningful overlap
                    results.append({
                        "chunk": chunk,
                        "metadata": self.embeddings_data["metadata"][i],
                        "similarity": similarity,
                        "method": "text_fallback"
                    })
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]
    
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
        
        # Format results to match traditional format
        formatted_results = []
        for result in results:
            formatted_results.append({
                "chunk": result["document"],
                "metadata": result["metadata"],
                "similarity": 1.0 - result["distance"],  # Convert distance to similarity
                "method": "chromadb"
            })
        
        return formatted_results
    
    def search_hybrid(self, query, top_k=20, filter_dict=None):
        """Hybrid search combining both methods."""
        from tqdm.auto import tqdm
        
        print(f"🔍 Performing hybrid search...")
        with tqdm(total=3, desc="Hybrid search", unit="step") as pbar:
            # Step 1: Traditional search - search for more results to get better coverage
            pbar.set_description("Traditional search")
            try:
                traditional_results = self.search_traditional(query, top_k * 2, show_progress=False)  # Get more results
                print(f"   Traditional search found {len(traditional_results)} results")
            except Exception as e:
                print(f"   Traditional search failed: {e}")
                traditional_results = []
            pbar.update(1)
            
            # Step 2: ChromaDB search - search for more results to get better coverage
            pbar.set_description("ChromaDB search")
            try:
                chromadb_results = self.search_chromadb(query, top_k * 2, filter_dict, show_progress=False) if self.use_chromadb else []
                print(f"   ChromaDB search found {len(chromadb_results)} results")
            except Exception as e:
                print(f"   ChromaDB search failed: {e}")
                chromadb_results = []
            pbar.update(1)
            
            # Step 3: Combine and deduplicate
            pbar.set_description("Combining results")
            all_results = traditional_results + chromadb_results
            
            # Simple deduplication based on metadata
            seen_titles = set()
            unique_results = []
            
            for result in all_results:
                title = result["metadata"].get("title", "")
                if title not in seen_titles:
                    seen_titles.add(title)
                    unique_results.append(result)
            
            # Sort by similarity and return top_k
            unique_results.sort(key=lambda x: x["similarity"], reverse=True)
            pbar.update(1)
        
        print(f"✅ Combined search found {len(unique_results)} unique results")
        return unique_results[:top_k]
    
    def display_results(self, results, show_method=True):
        """Display search results in a nice format."""
        if not results:
            print("❌ No relevant results found.")
            return
        
        print(f"\n🔍 Found {len(results)} relevant chunks:")
        
        # Show summary statistics instead of listing every chunk
        if len(results) > 0:
            # Count by source
            source_counts = {}
            method_counts = {}
            total_chars = 0
            
            for result in results:
                metadata = result.get('metadata', {})
                source = metadata.get('source', 'Unknown')
                method = result.get('method', 'Unknown')
                
                source_counts[source] = source_counts.get(source, 0) + 1
                method_counts[method] = method_counts.get(method, 0) + 1
                total_chars += len(result.get('chunk', ''))
            
            print(f"📊 Summary:")
            print(f"   📚 Total chunks: {len(results)}")
            print(f"   📝 Total characters: {total_chars:,}")
            print(f"   🔍 Search methods: {', '.join([f'{k} ({v})' for k, v in method_counts.items()])}")
            print(f"   📖 Sources: {', '.join([f'{k} ({v})' for k, v in source_counts.items()])}")
            
            # Show top 3 results as examples
            if len(results) > 3:
                print(f"\n📄 Top 3 results (showing examples):")
                for i, result in enumerate(results[:3], 1):
                    metadata = result.get('metadata', {})
                    title = metadata.get('title', 'No title available')
                    similarity = result.get('similarity', 0)
                    print(f"   {i}. {title[:60]}... (similarity: {similarity:.3f})")
                print(f"   ... and {len(results) - 3} more chunks")
            else:
                print(f"\n📄 All {len(results)} results:")
                for i, result in enumerate(results, 1):
                    metadata = result.get('metadata', {})
                    title = metadata.get('title', 'No title available')
                    similarity = result.get('similarity', 0)
                    print(f"   {i}. {title[:60]}... (similarity: {similarity:.3f})")
    
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
        hypotheses = self.hypothesis_generator.generate(context_chunks, n=n)
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
                    critique_result = self.hypothesis_critic.critique(hypotheses[i], context_chunks)
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
                        hypotheses[i] = self.hypothesis_generator.generate(context_chunks, n=1)[0]
                    
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

    def _display_commands(self):
        """Display available commands."""
        print(f"\n💡 Search method:")
        print(f"   - 'hybrid': Combined search using both traditional and ChromaDB methods (default)")
        
        print(f"\n💡 Additional commands:")
        print(f"   - 'quit': Exit the system")
        print(f"   - 'stats': Show database statistics")
        print(f"   - 'filter <source>': Filter by source (e.g., 'filter pubmed')")
        print(f"   - 'clear filter': Clear current filter")
        print(f"   - 'clear context': Clear stored context chunks")
        print(f"   - 'search <N> <query>': Search with N results (e.g., 'search 50 cancer')")
        print(f"   - 'hypothesize: <question>': Generate hypotheses for a question")
        print(f"   - 'critique: <hypothesis>': Critique a hypothesis using last context")
        print(f"   - 'searchgen <question>': Run full search/generator/critic feedback loop")
        print(f"   - 'generate': Generate hypotheses using last search results")
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
            elif query.lower() == 'clear context':
                self.last_context_chunks = None
                print("🧠 Stored context chunks cleared")
            elif query.lower().startswith('search '):
                # Search with custom number of results
                try:
                    parts = query.split(' ', 2)  # Split into ['search', 'N', 'query']
                    if len(parts) < 3:
                        print("❌ Usage: search <N> <query> (e.g., 'search 50 cancer')")
                        continue
                    
                    num_results = int(parts[1])
                    search_query = parts[2]
                    
                    if num_results <= 0 or num_results > 1000:
                        print("❌ Number of results must be between 1 and 1000")
                        continue
                    
                    print(f"\n🔍 Searching for: '{search_query}' (requesting {num_results} results)")
                    try:
                        results = self.search_hybrid(search_query, top_k=num_results, filter_dict=current_filter)
                        if results:
                            self.display_results(results)
                            
                            # Store context chunks for hypothesis generation
                            context_chunks = [result["document"] for result in results]
                            self.last_context_chunks = context_chunks
                            
                            # Ask user if they want to generate hypotheses
                            print(f"\n🧠 Found {len(context_chunks)} relevant chunks for hypothesis generation.")
                            print("Would you like to generate and critique hypotheses based on these results?")
                            response = input("Enter 'y' to start hypothesis generation, or any other key to skip: ").strip().lower()
                            
                            if response == 'y':
                                print(f"\n🚀 Starting hypothesis generation and critique process...")
                                self.iterative_hypothesis_generation(search_query)
                            else:
                                print("⏭️ Skipping hypothesis generation.")
                        else:
                            print("❌ No results found. Try a different search term or check if data is loaded.")
                    except Exception as e:
                        print(f"❌ Search failed: {e}")
                        print("💡 Try running 'stats' to check data availability")
                    
                    # Display commands after search
                    self._display_commands()
                except ValueError:
                    print("❌ Invalid number format. Usage: search <N> <query> (e.g., 'search 50 cancer')")
            elif query.lower() == 'generate':
                # Generate hypotheses using last search results
                if not hasattr(self, 'last_context_chunks') or not self.last_context_chunks:
                    print("❌ No context available. Run a search first to get context chunks.")
                    continue
                print(f"\n🚀 Generating hypotheses using {len(self.last_context_chunks)} stored context chunks...")
                self.iterative_hypothesis_generation("Using stored context")
            elif query.lower().startswith('hypothesize:'):
                # Generate hypotheses for a question using the full feedback loop
                search_query = query[len('hypothesize:'):].strip()
                self.iterative_hypothesis_generation(search_query)
            elif query.lower().startswith('critique:'):
                # Critique a hypothesis using last context
                hypothesis = query[len('critique:'):].strip()
                if not hasattr(self, 'last_context_chunks') or not self.last_context_chunks:
                    print("❌ No context available. Run a search first to get context chunks.")
                    continue
                print(f"\n🧑‍🔬 Critiquing hypothesis: {hypothesis}")
                critique = self.hypothesis_critic.critique(hypothesis, self.last_context_chunks)
                print(critique)
            elif query.lower().startswith('searchgen'):
                # Full search/generator/critic feedback loop
                # Handle both "searchgen cancer" and "searchgen: cancer"
                if query.lower().startswith('searchgen:'):
                    user_prompt = query[len('searchgen:'):].strip()
                else:
                    user_prompt = query[len('searchgen'):].strip()
                
                if not user_prompt:
                    print("❌ Usage: searchgen <question> (e.g., 'searchgen cancer' or 'searchgen: cancer')")
                    continue
                
                print(f"\n🔍 Searching for: '{user_prompt}'")
                try:
                    # Get maximum possible results for comprehensive search
                    results = self.search_hybrid(user_prompt, top_k=1000, filter_dict=current_filter)
                    if results:
                        # Store context chunks for hypothesis generation
                        context_chunks = [result["document"] for result in results]
                        self.last_context_chunks = context_chunks
                        
                        print(f"\n🧠 Collected {len(context_chunks)} relevant chunks for hypothesis generation.")
                        print(f"🚀 Starting hypothesis generation and critique process...")
                        self.iterative_hypothesis_generation(user_prompt)
                    else:
                        print("❌ No results found. Try a different search term or check if data is loaded.")
                except Exception as e:
                    print(f"❌ Search failed: {e}")
                    print("💡 Try running 'stats' to check data availability")
                
                # Display commands after search
                self._display_commands()
            elif query:
                print(f"\n🔍 Searching for: '{query}'")
                try:
                    # Increase top_k for more comprehensive results
                    results = self.search_hybrid(query, top_k=20, filter_dict=current_filter)
                    if results:
                        self.display_results(results)
                        
                        # Store context chunks for hypothesis generation
                        context_chunks = [result["document"] for result in results]
                        self.last_context_chunks = context_chunks
                        
                        # Ask user if they want to generate hypotheses
                        print(f"\n🧠 Found {len(context_chunks)} relevant chunks for hypothesis generation.")
                        print("Would you like to generate and critique hypotheses based on these results?")
                        response = input("Enter 'y' to start hypothesis generation, or any other key to skip: ").strip().lower()
                        
                        if response == 'y':
                            print(f"\n🚀 Starting hypothesis generation and critique process...")
                            self.iterative_hypothesis_generation(query)
                        else:
                            print("⏭️ Skipping hypothesis generation.")
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