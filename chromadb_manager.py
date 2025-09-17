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
        
        Args:
            embeddings_dir: Path to the embeddings directory
            sources: List of sources to load (e.g., ['biorxiv', 'medrxiv']). If None, loads all sources.
            
        Returns:
            Dict containing metadata and source information
        """
        if not os.path.exists(embeddings_dir):
            logger.error(f"❌ Embeddings directory not found: {embeddings_dir}")
            return {}
        
        # Load metadata
        metadata_file = os.path.join(embeddings_dir, "metadata.json")
        if not os.path.exists(metadata_file):
            logger.error(f"❌ Metadata file not found: {metadata_file}")
            return {}
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            logger.info(f"📊 Loaded metadata: {metadata['total_embeddings']} total embeddings")
            
            # Determine which sources to process
            available_sources = list(metadata['sources'].keys())
            if sources is None:
                sources = available_sources
            else:
                sources = [s for s in sources if s in available_sources]
            
            logger.info(f"🔄 Processing sources: {sources}")
            
            return {
                'metadata': metadata,
                'sources': sources,
                'embeddings_dir': embeddings_dir
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to load metadata: {e}")
            return {}
    
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
            source_dir = os.path.join(embeddings_dir, source)
            
            if not os.path.exists(source_dir):
                logger.warning(f"⚠️ Source directory not found: {source_dir}")
                continue
            
            # Get all batch files for this source
            batch_files = glob.glob(os.path.join(source_dir, "batch_*.json"))
            batch_files.sort()  # Ensure consistent ordering
            
            logger.info(f"📁 Found {len(batch_files)} batch files for {source}")
            
            # Collect all embeddings for bulk insertion
            all_embeddings = []
            all_chunks = []
            all_metadata = []
            all_ids = []
            
            source_added = 0
            
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
            # Add to collection in bulk
            self.collection.add(
                embeddings=embeddings,  # type: ignore
                documents=chunks,
                metadatas=metadata,  # type: ignore
                ids=ids
            )
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to bulk add embeddings: {e}")
            return False
    
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
            
            # Add source information to metadata
            for i, meta in enumerate(metadata):
                meta['source_name'] = source_name
                meta['added_at'] = datetime.now().isoformat()
            
            # Generate unique IDs
            ids = [f"{source_name}_{i}" for i in range(len(chunks))]
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadata,
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
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        if not self.is_initialized or self.collection is None:
            return {}
        
        try:
            count = self.collection.count()
            
            # Get sample metadata to understand structure
            sample_results = self.collection.peek(limit=1)
            sample_metadata = sample_results['metadatas'][0] if sample_results and sample_results['metadatas'] else {}
            
            stats = {
                'total_documents': count,
                'collection_name': self.collection_name,
                'sample_metadata_keys': list(sample_metadata.keys()) if sample_metadata else [],
                'persist_directory': self.persist_directory
            }
            
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

    def get_all_documents(self, limit: int = 50000) -> List[Dict[str, Any]]:
        """
        Retrieve all documents (chunks) and their metadata from the current collection.
        Args:
            limit: Maximum number of documents to retrieve (default 50,000)
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
            results = self.collection.get(
                include=['documents', 'metadatas'],
                limit=fetch_limit
            )
            all_docs = []
            if results and 'documents' in results and results['documents']:
                for i in range(len(results['documents'])):
                    all_docs.append({
                        'document': results['documents'][i],
                        'metadata': results['metadatas'][i] if results['metadatas'] else {}
                    })
            return all_docs
        except Exception as e:
            logger.error(f"❌ Failed to retrieve all documents: {e}")
            return []

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