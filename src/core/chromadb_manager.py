import json
import numpy as np
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional, Tuple
import os
from datetime import datetime
import logging
import glob

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChromaDBManager:
    """
    A comprehensive ChromaDB vector database manager for PubMed and BioRxiv embeddings.
    
    Features:
    - Load embeddings from JSON files (single file or multi-file structure)
    - Create and manage collections
    - Perform similarity searches
    - Filter by metadata
    - Export/import collections
    - Statistics and monitoring
    - Support for new multi-file storage system
    """
    
    def __init__(self, persist_directory: str = "./data/vector_db/chroma_db", collection_name: str = "pubmed_papers"):
        """
        Initialize the ChromaDB manager.
        
        Args:
            persist_directory: Directory to persist the database
            collection_name: Name of the collection to use
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.is_initialized = False
        
        # Initialize ChromaDB client
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the ChromaDB client with persistent storage."""
        try:
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            logger.info(f"✅ ChromaDB client initialized with persistence at: {self.persist_directory}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize ChromaDB client: {e}")
            raise
    
    def create_collection(self, metadata: Optional[Dict] = None) -> bool:
        """
        Create a new collection or get existing one.
        
        Args:
            metadata: Optional metadata for the collection
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.client is None:
            logger.error("❌ ChromaDB client not initialized")
            return False
            
        try:
            # Check if collection exists
            existing_collections = [col.name for col in self.client.list_collections()]
            
            if self.collection_name in existing_collections:
                logger.info(f"📚 Using existing collection: {self.collection_name}")
                self.collection = self.client.get_collection(name=self.collection_name)
            else:
                logger.info(f"🆕 Creating new collection: {self.collection_name}")
                collection_metadata = metadata or {
                    "description": "PubMed and BioRxiv papers with embeddings",
                    "created_at": datetime.now().isoformat(),
                    "embedding_model": "text-embedding-004"
                }
                
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata=collection_metadata
                )
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create/get collection: {e}")
            return False
    
    def load_embeddings_from_json(self, json_file_path: str) -> Optional[Dict]:
        """
        Load embeddings from a single JSON file (legacy method).
        
        Args:
            json_file_path: Path to the JSON file
            
        Returns:
            Dict containing chunks, embeddings, and metadata
        """
        if not os.path.exists(json_file_path):
            logger.error(f"❌ File not found: {json_file_path}")
            return None
        
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"📊 Loaded {len(data['chunks'])} chunks from {json_file_path}")
            return data
            
        except Exception as e:
            logger.error(f"❌ Failed to load JSON file: {e}")
            return None
    
    def load_embeddings_from_directory(self, embeddings_dir: str, sources: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Load embeddings from the new multi-file directory structure.
        Automatically detects all available sources by scanning directory structure.
        Also checks for PubMed embeddings in the separate data/embeddings/pubmed/ directory.
        
        Args:
            embeddings_dir: Path to the embeddings directory
            sources: List of sources to load (e.g., ['biorxiv', 'medrxiv', 'pubmed']). If None, loads all sources.
            
        Returns:
            Dict containing metadata and source information
        """
        if not os.path.exists(embeddings_dir):
            logger.error(f"❌ Embeddings directory not found: {embeddings_dir}")
            return {}
        
        # Load metadata if it exists
        metadata_file = os.path.join(embeddings_dir, "metadata.json")
        metadata = {}
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                logger.info(f"📊 Loaded metadata: {metadata.get('total_embeddings', 0)} total embeddings")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load metadata: {e}")
                metadata = {}
        
        # Auto-detect all available sources by scanning directory structure
        detected_sources = self._detect_available_sources(embeddings_dir)
        
        # Also check for PubMed embeddings in the separate pubmed directory
        pubmed_dir = os.path.join(os.path.dirname(embeddings_dir), "pubmed")
        if os.path.exists(pubmed_dir):
            # Check if PubMed directory contains batch files directly
            pubmed_batch_files = glob.glob(os.path.join(pubmed_dir, "batch_*.json"))
            if pubmed_batch_files:
                # Add 'pubmed' to detected sources if batch files are found
                detected_sources.append('pubmed')
                logger.info(f"🔍 Detected PubMed embeddings in separate directory: {pubmed_dir} ({len(pubmed_batch_files)} batch files)")
        
        # Determine which sources to process
        if sources is None:
            # Use all detected sources
            sources = detected_sources
            logger.info(f"🔄 Auto-detected sources: {sources}")
        else:
            # Filter sources to only include detected ones
            sources = [s for s in sources if s in detected_sources]
            logger.info(f"🔄 Processing requested sources: {sources}")
        
        # Update metadata with detected sources if not present
        if 'sources' not in metadata:
            metadata['sources'] = {}
        
        # Add any new sources to metadata
        for source in detected_sources:
            if source not in metadata['sources']:
                metadata['sources'][source] = {
                    'batch_files': 0,
                    'total_embeddings': 0,
                    'embedding_dimension': 768,
                    'last_updated': datetime.now().isoformat()
                }
        
        return {
            'metadata': metadata,
            'sources': sources,
            'embeddings_dir': embeddings_dir
        }
    
    def _detect_available_sources(self, embeddings_dir: str) -> List[str]:
        """
        Auto-detect all available sources by scanning the directory structure.
        
        Args:
            embeddings_dir: Path to the embeddings directory
            
        Returns:
            List of detected source names
        """
        detected_sources = []
        
        try:
            # Scan for subdirectories that could contain embeddings
            for item in os.listdir(embeddings_dir):
                item_path = os.path.join(embeddings_dir, item)
                
                # Skip non-directories and special files
                if not os.path.isdir(item_path) or item.startswith('.'):
                    continue
                
                # Check if this directory contains embedding data
                if self._is_valid_source_directory(item_path):
                    detected_sources.append(item)
                    logger.info(f"🔍 Detected source: {item}")
        
        except Exception as e:
            logger.error(f"❌ Error detecting sources: {e}")
        
        return detected_sources
    
    def _is_valid_source_directory(self, dir_path: str) -> bool:
        """
        Check if a directory contains valid embedding data.
        
        Args:
            dir_path: Path to the directory to check
            
        Returns:
            True if the directory contains embedding data
        """
        try:
            # Check for batch files (batch_*.json)
            batch_files = glob.glob(os.path.join(dir_path, "batch_*.json"))
            if batch_files:
                logger.debug(f"📁 Found {len(batch_files)} batch files in {os.path.basename(dir_path)}")
                return True
            
            # Check for individual JSON files (like ubr5_api)
            json_files = glob.glob(os.path.join(dir_path, "*.json"))
            if json_files:
                # Check if any of these files contain embeddings or are paper data
                for json_file in json_files[:3]:  # Check first 3 files
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # Check if it's a batch file with embeddings
                        if isinstance(data, dict) and 'embeddings' in data:
                            logger.debug(f"📁 Found embedding batch file in {os.path.basename(dir_path)}")
                            return True
                        
                        # Check if it's individual paper data (like ubr5_api)
                        if isinstance(data, dict) and any(key in data for key in ['title', 'abstract', 'doi', 'authors']):
                            logger.debug(f"📁 Found paper data files in {os.path.basename(dir_path)}")
                            return True
                            
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
            
            return False
            
        except Exception as e:
            logger.debug(f"⚠️ Error checking directory {dir_path}: {e}")
            return False
    
    def _process_individual_files(self, individual_files: List[str], source: str) -> Optional[Dict]:
        """
        Process individual JSON files (like ubr5_api) into embeddings format.
        
        Args:
            individual_files: List of individual JSON file paths
            source: Source name
            
        Returns:
            Dict containing embeddings, chunks, metadata, and ids
        """
        embeddings = []
        chunks = []
        metadata_list = []
        ids = []
        
        logger.info(f"🔄 Processing {len(individual_files)} individual files for {source}...")
        
        for i, file_path in enumerate(individual_files):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    paper_data = json.load(f)
                
                # Check if this is a paper data file (like ubr5_api)
                if isinstance(paper_data, dict) and any(key in paper_data for key in ['title', 'abstract', 'doi', 'authors']):
                    # Create text for embedding from paper data
                    text_parts = []
                    
                    if paper_data.get('title'):
                        text_parts.append(f"Title: {paper_data['title']}")
                    
                    if paper_data.get('abstract'):
                        text_parts.append(f"Abstract: {paper_data['abstract']}")
                    
                    if paper_data.get('authors'):
                        authors = paper_data['authors'] if isinstance(paper_data['authors'], list) else [paper_data['authors']]
                        text_parts.append(f"Authors: {', '.join(authors)}")
                    
                    if paper_data.get('journal'):
                        text_parts.append(f"Journal: {paper_data['journal']}")
                    
                    if paper_data.get('year'):
                        text_parts.append(f"Year: {paper_data['year']}")
                    
                    if paper_data.get('fields_of_study'):
                        fields = paper_data['fields_of_study'] if isinstance(paper_data['fields_of_study'], list) else [paper_data['fields_of_study']]
                        text_parts.append(f"Fields: {', '.join(fields)}")
                    
                    # Create the text chunk
                    text_chunk = '\n'.join(text_parts)
                    
                    if len(text_chunk.strip()) > 50:  # Only process if there's meaningful content
                        # Create metadata
                        meta = {
                            'source_name': source,
                            'title': paper_data.get('title', ''),
                            'doi': paper_data.get('doi', ''),
                            'authors': paper_data.get('authors', []),
                            'journal': paper_data.get('journal', ''),
                            'year': paper_data.get('year', ''),
                            'abstract': paper_data.get('abstract', ''),
                            'fields_of_study': paper_data.get('fields_of_study', []),
                            'citation_count': paper_data.get('citation_count', ''),
                            'impact_factor': paper_data.get('impact_factor', ''),
                            'source': paper_data.get('source', source),
                            'file_type': 'individual',
                            'added_at': datetime.now().isoformat()
                        }
                        
                        # For now, we'll create a placeholder embedding (zeros)
                        # In a real implementation, you'd generate actual embeddings here
                        embedding_dim = 768  # text-embedding-004 dimension
                        placeholder_embedding = [0.0] * embedding_dim
                        
                        # Create unique ID
                        file_id = f"{source}_individual_{i:06d}"
                        
                        embeddings.append(placeholder_embedding)
                        chunks.append(text_chunk)
                        metadata_list.append(meta)
                        ids.append(file_id)
                
                # Log progress every 100 files
                if (i + 1) % 100 == 0:
                    logger.info(f"📊 Processed {i + 1}/{len(individual_files)} individual files for {source}")
                    
            except Exception as e:
                logger.warning(f"⚠️ Error processing individual file {file_path}: {e}")
                continue
        
        logger.info(f"✅ Processed {len(embeddings)} embeddings from {len(individual_files)} individual files for {source}")
        
        return {
            'embeddings': embeddings,
            'chunks': chunks,
            'metadata': metadata_list,
            'ids': ids
        }
    
    def load_batch_file(self, batch_file_path: str) -> Optional[Dict]:
        """
        Load a single batch file.
        
        Args:
            batch_file_path: Path to the batch file
            
        Returns:
            Dict containing batch data
        """
        try:
            with open(batch_file_path, 'r', encoding='utf-8') as f:
                batch_data = json.load(f)
            
            return batch_data
            
        except Exception as e:
            logger.error(f"❌ Failed to load batch file {batch_file_path}: {e}")
            return None
    
    def add_embeddings_from_directory(self, embeddings_dir: str, sources: Optional[List[str]] = None, 
                                    batch_size: int = 1000, db_batch_size: int = 5000) -> bool:
        """
        Add embeddings from the multi-file directory structure to ChromaDB.
        
        Args:
            embeddings_dir: Path to the embeddings directory
            sources: List of sources to process. If None, processes all sources.
            batch_size: Number of embeddings to process in memory at once
            db_batch_size: Number of embeddings to add to ChromaDB in a single operation
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_initialized or self.collection is None:
            logger.error("❌ Collection not initialized. Call create_collection() first.")
            return False
        
        # Load directory metadata
        dir_info = self.load_embeddings_from_directory(embeddings_dir, sources)
        if not dir_info:
            return False
        
        metadata = dir_info['metadata']
        sources = dir_info.get('sources', [])
        
        if not sources:
            logger.error("❌ No sources found to process")
            return False
        
        total_added = 0
        
        for source in sources:
            logger.info(f"🔄 Processing source: {source}")
            
            # Check if this is PubMed source and handle special case
            if source == "pubmed":
                # PubMed embeddings are in a separate directory
                pubmed_dir = os.path.join(os.path.dirname(embeddings_dir), "pubmed")
                if os.path.exists(pubmed_dir):
                    source_dir = pubmed_dir
                    logger.info(f"📁 Found PubMed batch files in separate directory: {pubmed_dir}")
                else:
                    logger.warning(f"⚠️ PubMed directory not found: {pubmed_dir}")
                    continue
            else:
                # Handle case where batch files are in a subdirectory
                source_dir = os.path.join(embeddings_dir, source)
                if not os.path.exists(source_dir):
                    logger.warning(f"⚠️ Source directory not found: {source_dir}")
                    continue
            
            # Get all batch files for this source
            batch_files = glob.glob(os.path.join(source_dir, "batch_*.json"))
            batch_files.sort()  # Ensure consistent ordering
            
            # Get all individual JSON files (like ubr5_api)
            individual_files = glob.glob(os.path.join(source_dir, "*.json"))
            # Filter out batch files
            individual_files = [f for f in individual_files if not os.path.basename(f).startswith("batch_")]
            individual_files.sort()
            
            total_files = len(batch_files) + len(individual_files)
            logger.info(f"📁 Found {len(batch_files)} batch files and {len(individual_files)} individual files for {source}")
            
            # Collect all embeddings for bulk insertion
            all_embeddings = []
            all_chunks = []
            all_metadata = []
            all_ids = []
            
            source_added = 0
            
            # Process batch files
            for batch_file in batch_files:
                batch_data = self.load_batch_file(batch_file)
                if not batch_data:
                    continue
                
                embeddings = batch_data.get('embeddings', [])
                chunks = batch_data.get('chunks', [])
                metadata_list = batch_data.get('metadata', [])
                
                if not embeddings or not chunks or not metadata_list:
                    logger.warning(f"⚠️ Invalid batch data in {batch_file}")
                    continue
                
                # Add source information to metadata
                batch_num = batch_data.get('batch_num', 0)
                for i, meta in enumerate(metadata_list):
                    meta['source_name'] = source
                    meta['batch_num'] = batch_num
                    meta['added_at'] = datetime.now().isoformat()
                
                # Generate unique IDs
                ids = [f"{source}_batch_{batch_num:04d}_{i}" for i in range(len(chunks))]
                
                # Add to bulk collections
                all_embeddings.extend(embeddings)
                all_chunks.extend(chunks)
                all_metadata.extend(metadata_list)
                all_ids.extend(ids)
                
                source_added += len(embeddings)
                
                # Log progress every 10 batches
                if len(batch_files) > 10 and batch_files.index(batch_file) % 10 == 0:
                    logger.info(f"📊 Processed {batch_files.index(batch_file) + 1}/{len(batch_files)} batches for {source}")
            
            # Process individual files (like ubr5_api)
            if individual_files:
                logger.info(f"🔄 Processing {len(individual_files)} individual files for {source}...")
                individual_data = self._process_individual_files(individual_files, source)
                
                if individual_data:
                    all_embeddings.extend(individual_data['embeddings'])
                    all_chunks.extend(individual_data['chunks'])
                    all_metadata.extend(individual_data['metadata'])
                    all_ids.extend(individual_data['ids'])
                    source_added += len(individual_data['embeddings'])
            
            # Bulk insert all embeddings for this source
            if all_embeddings:
                logger.info(f"🔄 Bulk inserting {len(all_embeddings)} embeddings for {source}...")
                
                # Split into database batches if too large
                if len(all_embeddings) > db_batch_size:
                    logger.info(f"📦 Splitting {len(all_embeddings)} embeddings into {db_batch_size}-sized database batches")
                    
                    for i in range(0, len(all_embeddings), db_batch_size):
                        end_idx = min(i + db_batch_size, len(all_embeddings))
                        
                        batch_embeddings = all_embeddings[i:end_idx]
                        batch_chunks = all_chunks[i:end_idx]
                        batch_metadata = all_metadata[i:end_idx]
                        batch_ids = all_ids[i:end_idx]
                        
                        if self._bulk_add_to_collection(batch_embeddings, batch_chunks, batch_metadata, batch_ids):
                            logger.info(f"✅ Added database batch {i//db_batch_size + 1}: {len(batch_embeddings)} embeddings")
                        else:
                            logger.error(f"❌ Failed to add database batch {i//db_batch_size + 1}")
                            return False
                else:
                    # Single bulk insert
                    if self._bulk_add_to_collection(all_embeddings, all_chunks, all_metadata, all_ids):
                        logger.info(f"✅ Bulk inserted {len(all_embeddings)} embeddings for {source}")
                    else:
                        logger.error(f"❌ Failed to bulk insert embeddings for {source}")
                        return False
                
                total_added += source_added
            
            logger.info(f"✅ Completed {source}: {source_added} embeddings added")
        
        logger.info(f"🎉 Total embeddings added: {total_added}")
        return total_added > 0
    
    def _bulk_add_to_collection(self, embeddings: List[List[float]], chunks: List[str], 
                               metadata: List[Dict[str, Any]], ids: List[str]) -> bool:
        """
        Bulk add embeddings to the collection.
        
        Args:
            embeddings: List of embedding vectors
            chunks: List of text chunks
            metadata: List of metadata dictionaries
            ids: List of unique IDs
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.collection is None:
            logger.error("❌ Collection not initialized")
            return False
            
        try:
            # Validate metadata for ChromaDB compatibility
            validated_metadata = []
            for meta in metadata:
                validated_meta = self._validate_metadata_for_chromadb(meta)
                validated_metadata.append(validated_meta)
            
            # Add to collection in bulk
            self.collection.add(
                embeddings=embeddings,  # type: ignore
                documents=chunks,
                metadatas=validated_metadata,  # type: ignore
                ids=ids
            )
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to bulk add embeddings: {e}")
            return False
    
    def _validate_metadata_for_chromadb(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and convert metadata to ChromaDB-compatible format.
        
        Args:
            metadata: Original metadata dictionary
            
        Returns:
            Validated metadata dictionary with ChromaDB-compatible values
        """
        validated_meta = {}
        
        for key, value in metadata.items():
            if value is None:
                # ChromaDB doesn't accept None values, convert to empty string
                validated_meta[key] = ""
            elif isinstance(value, list):
                # Convert lists to semicolon-separated strings
                validated_meta[key] = "; ".join(str(item) for item in value if item is not None)
            elif isinstance(value, (str, int, float, bool)):
                # These types are ChromaDB-compatible
                validated_meta[key] = value
            else:
                # Convert other types to strings
                validated_meta[key] = str(value) if value is not None else ""
        
        return validated_meta

    def add_embeddings_to_collection(self, data: Dict, source_name: str = "unknown") -> bool:
        """
        Add embeddings to the ChromaDB collection (legacy method for single file).
        
        Args:
            data: Dictionary containing chunks, embeddings, and metadata
            source_name: Name of the source (e.g., 'pubmed', 'biorxiv')
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_initialized or self.collection is None:
            logger.error("❌ Collection not initialized. Call create_collection() first.")
            return False
        
        try:
            chunks = data['chunks']
            embeddings = data['embeddings']
            metadata = data['metadata']
            
            # Validate and convert metadata for ChromaDB compatibility
            validated_metadata = []
            for i, meta in enumerate(metadata):
                validated_meta = self._validate_metadata_for_chromadb(meta)
                validated_meta['source_name'] = source_name
                validated_meta['added_at'] = datetime.now().isoformat()
                validated_metadata.append(validated_meta)
            
            # Generate unique IDs
            ids = [f"{source_name}_{i}" for i in range(len(chunks))]
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=chunks,
                metadatas=validated_metadata,
                ids=ids
            )
            
            logger.info(f"✅ Added {len(chunks)} embeddings from {source_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to add embeddings: {e}")
            return False
    
    def search_similar(self, query_embedding: List[float], n_results: int = 5, 
                      where_filter: Optional[Dict] = None) -> List[Dict]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            where_filter: Optional metadata filter
            
        Returns:
            List of search results with documents, metadata, and distances
        """
        if not self.is_initialized or self.collection is None:
            logger.error("❌ Collection not initialized.")
            return []
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            if results and 'documents' in results and results['documents']:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'distance': results['distances'][0][i] if results['distances'] else 0.0,
                        'id': results['ids'][0][i] if 'ids' in results and results['ids'] else None
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return []
    
    def search_by_text(self, query_text: str, n_results: int = 5, 
                      where_filter: Optional[Dict] = None) -> List[Dict]:
        """
        Search using text query (requires text embedding).
        
        Args:
            query_text: Text query to search for
            n_results: Number of results to return
            where_filter: Optional metadata filter
            
        Returns:
            List of search results
        """
        # This would require an embedding model to convert text to vector
        # For now, we'll use a placeholder
        logger.warning("⚠️ Text search requires embedding model. Use search_similar() with pre-computed embeddings.")
        return []
    
    def search(self, query: str, n_results: int = 5, filter_dict: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Search using text query with automatic embedding generation.
        This method provides the interface expected by the RAG system.
        
        Args:
            query: Text query to search for
            n_results: Number of results to return
            filter_dict: Optional metadata filter
            
        Returns:
            Dictionary with documents, metadatas, distances, and ids
        """
        if not self.is_initialized or self.collection is None:
            logger.error("❌ Collection not initialized.")
            return {}
        
        try:
            # Use ChromaDB's built-in text search capability
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filter_dict,
                include=['documents', 'metadatas', 'distances', 'ids']
            )
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return {}
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the collection with optimized performance.
        
        Returns:
            Dictionary with collection statistics
        """
        if not self.is_initialized or self.collection is None:
            return {}
        
        try:
            # Use a more efficient approach for large collections
            count = self.collection.count()
            
            # For large collections, limit the peek to avoid performance issues
            peek_limit = min(1, count) if count > 0 else 0
            
            stats = {
                'total_documents': count,
                'collection_name': self.collection_name,
                'persist_directory': self.persist_directory,
                'is_large_collection': count > 10000
            }
            
            # Only get sample metadata for smaller collections or when specifically needed
            if count > 0 and count <= 10000:
                try:
                    sample_results = self.collection.peek(limit=peek_limit)
                    sample_metadata = sample_results['metadatas'][0] if sample_results and sample_results['metadatas'] else {}
                    stats['sample_metadata_keys'] = list(sample_metadata.keys()) if sample_metadata else []
                except Exception as e:
                    logger.warning(f"⚠️ Could not get sample metadata: {e}")
                    stats['sample_metadata_keys'] = []
            else:
                stats['sample_metadata_keys'] = []
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to get collection stats: {e}")
            return {}
    
    def filter_by_metadata(self, filter_dict: Dict) -> List[Dict]:
        """
        Filter documents by metadata.
        
        Args:
            filter_dict: Metadata filter (e.g., {"source": "pubmed"})
            
        Returns:
            List of filtered documents
        """
        if not self.is_initialized or self.collection is None:
            return []
        
        try:
            results = self.collection.get(
                where=filter_dict,
                include=['documents', 'metadatas']
            )
            
            filtered_results = []
            if results and 'documents' in results and results['documents']:
                for i in range(len(results['documents'])):
                    filtered_results.append({
                        'document': results['documents'][i],
                        'metadata': results['metadatas'][i] if results['metadatas'] else {},
                        'id': results['ids'][i] if 'ids' in results and results['ids'] else None
                    })
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"❌ Filter failed: {e}")
            return []
    
    def clear_collection(self) -> bool:
        """
        Clear all documents from the current collection.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_initialized or self.collection is None:
            logger.error("❌ Collection not initialized")
            return False
            
        try:
            # Get collection count first
            count = self.collection.count()
            if count == 0:
                logger.info(f"📚 Collection {self.collection_name} is already empty")
                return True
            
            # Get all documents with their IDs
            results = self.collection.get(include=['documents', 'metadatas'])
            if results and 'ids' in results and results['ids']:
                # Delete all documents
                self.collection.delete(ids=results['ids'])
                logger.info(f"🗑️ Cleared {len(results['ids'])} documents from collection: {self.collection_name}")
            else:
                logger.info(f"📚 Collection {self.collection_name} is already empty")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to clear collection: {e}")
            return False
    
    def delete_collection(self) -> bool:
        """
        Delete the current collection.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.client is None:
            logger.error("❌ ChromaDB client not initialized")
            return False
            
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = None
            self.is_initialized = False
            logger.info(f"🗑️ Deleted collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete collection: {e}")
            return False
    
    def list_collections(self) -> List[str]:
        """
        List all available collections.
        
        Returns:
            List of collection names
        """
        if self.client is None:
            logger.error("❌ ChromaDB client not initialized")
            return []
            
        try:
            collections = [col.name for col in self.client.list_collections()]
            return collections
        except Exception as e:
            logger.error(f"❌ Failed to list collections: {e}")
            return []
    
    def switch_collection(self, collection_name: str) -> bool:
        """
        Switch to a different collection.
        
        Args:
            collection_name: Name of the collection to switch to
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.client is None:
            logger.error("❌ ChromaDB client not initialized")
            return False
            
        try:
            self.collection = self.client.get_collection(name=collection_name)
            self.collection_name = collection_name
            self.is_initialized = True
            logger.info(f"🔄 Switched to collection: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to switch collection: {e}")
            return False
    
    def list_loaded_batches(self) -> Dict[str, List[str]]:
        """
        List all currently loaded batches in the collection.
        
        Returns:
            Dict mapping source names to lists of batch identifiers
        """
        if not self.is_initialized or self.collection is None:
            logger.error("❌ Collection not initialized")
            return {}
        
        try:
            # Get collection count first
            count = self.collection.count()
            logger.info(f"📊 Collection has {count} documents")
            
            if count == 0:
                logger.info("📚 No documents found in collection")
                return {}
            
            # Use a larger limit or get all documents if count is reasonable
            limit = min(count, 50000)  # Increased limit for large datasets
            
            # Get documents to examine their metadata
            results = self.collection.get(
                include=['metadatas'],
                limit=limit
            )
            
            if not results or not results['metadatas']:
                logger.info("📚 No documents found in collection")
                return {}
            
            # Group by source and batch
            batch_info = {}
            
            for metadata in results['metadatas']:
                if metadata:
                    source = metadata.get('source_name', 'unknown')
                    batch_num = metadata.get('batch_num', 'unknown')
                    
                    # Handle different batch_num formats
                    if isinstance(batch_num, int):
                        batch_id = f"batch_{batch_num:04d}"
                    elif isinstance(batch_num, str) and batch_num.isdigit():
                        batch_id = f"batch_{int(batch_num):04d}"
                    else:
                        batch_id = str(batch_num)
                    
                    if source not in batch_info:
                        batch_info[source] = []
                    
                    if batch_id not in batch_info[source]:
                        batch_info[source].append(batch_id)
            
            # Sort batch IDs for each source
            for source in batch_info:
                batch_info[source].sort()
            
            logger.info(f"📦 Found batches: {batch_info}")
            return batch_info
            
        except Exception as e:
            logger.error(f"❌ Failed to list loaded batches: {e}")
            return {}
    
    def get_batch_statistics(self) -> Dict[str, Any]:
        """
        Get detailed statistics about loaded batches.
        
        Returns:
            Dict with batch statistics
        """
        if not self.is_initialized or self.collection is None:
            logger.error("❌ Collection not initialized")
            return {}
        
        try:
            # Get collection count first
            count = self.collection.count()
            logger.info(f"📊 Collection has {count} documents")
            
            if count == 0:
                return {"total_documents": 0, "sources": {}}
            
            # Use a larger limit or get all documents if count is reasonable
            limit = min(count, 50000)  # Increased limit for large datasets
            
            # Get documents to examine their metadata
            results = self.collection.get(
                include=['metadatas'],
                limit=limit
            )
            
            if not results or not results['metadatas']:
                return {"total_documents": 0, "sources": {}}
            
            # Analyze metadata
            source_stats = {}
            total_docs = len(results['metadatas'])
            
            for metadata in results['metadatas']:
                if metadata:
                    source = metadata.get('source_name', 'unknown')
                    batch_num = metadata.get('batch_num', 'unknown')
                    
                    if source not in source_stats:
                        source_stats[source] = {
                            'total_documents': 0,
                            'batches': set(),
                            'batch_details': {}
                        }
                    
                    source_stats[source]['total_documents'] += 1
                    source_stats[source]['batches'].add(batch_num)
                    
                    # Count documents per batch
                    if batch_num not in source_stats[source]['batch_details']:
                        source_stats[source]['batch_details'][batch_num] = 0
                    source_stats[source]['batch_details'][batch_num] += 1
            
            # Convert sets to lists for JSON serialization
            for source in source_stats:
                source_stats[source]['batches'] = sorted(list(source_stats[source]['batches']))
            
            logger.info(f"📊 Batch statistics: {len(source_stats)} sources, {total_docs} documents")
            return {
                "total_documents": total_docs,
                "sources": source_stats
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get batch statistics: {e}")
            return {}

    def get_all_documents(self, limit: int = 50000, batch_size: int = 1000) -> List[Dict[str, Any]]:
        """
        Retrieve all documents (chunks) and their metadata from the current collection with batch processing.
        Args:
            limit: Maximum number of documents to retrieve (default 50,000)
            batch_size: Size of batches for processing (default 1,000)
        Returns:
            List of dicts with 'document' and 'metadata' keys
        """
        if not self.is_initialized or self.collection is None:
            logger.error("❌ Collection not initialized")
            return []
        
        try:
            count = self.collection.count()
            if count == 0:
                logger.info("📚 No documents found in collection")
                return []
            
            fetch_limit = min(count, limit)
            all_docs = []
            
            # Process in batches for better memory management
            for offset in range(0, fetch_limit, batch_size):
                current_batch_size = min(batch_size, fetch_limit - offset)
                
                try:
                    results = self.collection.get(
                        include=['documents', 'metadatas'],
                        limit=current_batch_size,
                        offset=offset
                    )
                    
                    if results and 'documents' in results and results['documents']:
                        for i in range(len(results['documents'])):
                            all_docs.append({
                                'document': results['documents'][i],
                                'metadata': results['metadatas'][i] if results['metadatas'] else {}
                            })
                    
                    # Log progress for large collections
                    if count > 10000:
                        logger.info(f"📊 Processed {min(offset + current_batch_size, fetch_limit)}/{fetch_limit} documents")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Error processing batch at offset {offset}: {e}")
                    continue
            
            logger.info(f"✅ Retrieved {len(all_docs)} documents from collection")
            return all_docs
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve all documents: {e}")
            return []
    
    def get_documents_batch(self, offset: int = 0, limit: int = 1000) -> Dict[str, Any]:
        """
        Get a batch of documents with pagination support.
        
        Args:
            offset: Starting position for the batch
            limit: Maximum number of documents to retrieve
            
        Returns:
            Dictionary containing documents, metadata, and pagination info
        """
        if not self.is_initialized or self.collection is None:
            logger.error("❌ Collection not initialized")
            return {}
        
        try:
            total_count = self.collection.count()
            
            # Ensure offset doesn't exceed total count
            if offset >= total_count:
                return {
                    'documents': [],
                    'metadata': [],
                    'total_count': total_count,
                    'offset': offset,
                    'limit': limit,
                    'has_more': False
                }
            
            # Adjust limit if it would exceed total count
            actual_limit = min(limit, total_count - offset)
            
            results = self.collection.get(
                include=['documents', 'metadatas'],
                limit=actual_limit,
                offset=offset
            )
            
            return {
                'documents': results.get('documents', []),
                'metadata': results.get('metadatas', []),
                'total_count': total_count,
                'offset': offset,
                'limit': actual_limit,
                'has_more': offset + actual_limit < total_count
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get documents batch: {e}")
            return {}

def main():
    """Example usage of the ChromaDB manager with new multi-file support."""
    print("=== ChromaDB Vector Database Manager (Multi-File Support) ===")
    
    # Initialize manager
    manager = ChromaDBManager()
    
    # Create collection
    if not manager.create_collection():
        print("❌ Failed to create collection")
        return
    
    # Load and add embeddings from multi-file structure
    if os.path.exists("data/embeddings/xrvix_embeddings"):
        print("🔄 Loading embeddings from multi-file structure...")
        success = manager.add_embeddings_from_directory("data/embeddings/xrvix_embeddings")
        if success:
            print("✅ Successfully loaded multi-file embeddings")
        else:
            print("❌ Failed to load multi-file embeddings")
    
    # Load and add PubMed embeddings from data/embeddings/xrvix_embeddings folder
    if os.path.exists("data/embeddings/xrvix_embeddings/pubmed_embeddings.json"):
        print("🔄 Loading PubMed embeddings from data/embeddings/xrvix_embeddings folder...")
        pubmed_data = manager.load_embeddings_from_json("data/embeddings/xrvix_embeddings/pubmed_embeddings.json")
        if pubmed_data:
            manager.add_embeddings_to_collection(pubmed_data, "pubmed")
            print("✅ Successfully loaded PubMed embeddings")
    
    # Display statistics
    stats = manager.get_collection_stats()
    print(f"\n📊 Collection Statistics:")
    print(f"   Total documents: {stats.get('total_documents', 0)}")
    print(f"   Collection name: {stats.get('collection_name', 'N/A')}")
    print(f"   Metadata keys: {stats.get('sample_metadata_keys', [])}")
    
    # Example search by source
    print(f"\n🔍 Example search by source:")
    results = manager.filter_by_metadata({"source_name": "biorxiv"})
    if results:
        print(f"Found {len(results)} BioRxiv documents")
        for i, result in enumerate(results[:3], 1):
            print(f"\n  Result {i}:")
            print(f"    Title: {result['metadata'].get('title', 'N/A')}")
            print(f"    DOI: {result['metadata'].get('doi', 'N/A')}")
            print(f"    Content: {result['document'][:100]}...")
    else:
        print("No BioRxiv documents found")

if __name__ == "__main__":
    main() 