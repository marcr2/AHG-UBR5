"""
Robust xrvix loader for large datasets.
Handles memory issues, timeouts, and provides progress tracking.
"""

import os
import json
import glob
import gc
import time
from tqdm import tqdm
from chromadb_manager import ChromaDBManager
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustXrvixLoader:
    """
    Robust loader for xrvix embeddings that handles large datasets.
    """
    
    def __init__(self, chroma_manager=None):
        self.chroma_manager = chroma_manager or ChromaDBManager()
        self.processed_batches = set()
        self.failed_batches = set()
        self.retry_attempts = 5  # Increased retry attempts
        self.retry_delay = 3  # Increased base delay
        self.compaction_error_count = 0  # Track compaction errors
        
    def load_with_progress(self, embeddings_dir="xrvix_embeddings", 
                          sources=None, batch_size=25, db_batch_size=100,
                          max_batches_per_run=None):
        """
        Load xrvix embeddings with robust error handling and progress tracking.
        
        Args:
            embeddings_dir: Directory containing embeddings
            sources: List of sources to process (None = all)
            batch_size: Number of batches to process in memory at once
            db_batch_size: Number of embeddings to add to ChromaDB at once (reduced for stability)
            max_batches_per_run: Limit batches per run (for testing/debugging)
        """
        print("🚀 Starting robust xrvix loading...")
        print(f"🔧 Settings: batch_size={batch_size}, db_batch_size={db_batch_size}")
        
        # Initialize ChromaDB
        if not self.chroma_manager.create_collection():
            print("❌ Failed to create ChromaDB collection")
            return False
        
        # Load directory info
        dir_info = self.chroma_manager.load_embeddings_from_directory(embeddings_dir, sources)
        if not dir_info:
            print("❌ Failed to load directory info")
            return False
        
        metadata = dir_info['metadata']
        sources = dir_info.get('sources', [])
        
        print(f"📊 Total embeddings in metadata: {metadata.get('total_embeddings', 0)}")
        print(f"📁 Sources to process: {sources}")
        
        total_loaded = 0
        
        for source in sources:
            print(f"\n🔄 Processing source: {source}")
            # Reset compaction error count for each source
            self.compaction_error_count = 0
            source_loaded = self._process_source(
                embeddings_dir, source, batch_size, db_batch_size, max_batches_per_run
            )
            total_loaded += source_loaded
            
            # Force garbage collection after each source
            gc.collect()
            
            # Small delay to let ChromaDB settle
            time.sleep(2)  # Increased delay
        
        print(f"\n🎉 Loading complete! Total embeddings loaded: {total_loaded}")
        print(f"✅ Successful batches: {len(self.processed_batches)}")
        print(f"❌ Failed batches: {len(self.failed_batches)}")
        
        if self.failed_batches:
            print(f"⚠️  Failed batch files: {sorted(list(self.failed_batches))[:10]}...")
        
        return total_loaded > 0
    
    def _process_source(self, embeddings_dir, source, batch_size, db_batch_size, max_batches_per_run):
        """Process a single source with robust error handling."""
        source_dir = os.path.join(embeddings_dir, source)
        
        if not os.path.exists(source_dir):
            print(f"❌ Source directory not found: {source_dir}")
            return 0
        
        # Get all batch files
        batch_files = glob.glob(os.path.join(source_dir, "batch_*.json"))
        batch_files.sort()
        
        if max_batches_per_run:
            batch_files = batch_files[:max_batches_per_run]
            print(f"🔧 Limited to {max_batches_per_run} batches for testing")
        
        print(f"📁 Found {len(batch_files)} batch files")
        
        if len(batch_files) == 0:
            return 0
        
        # Process in smaller chunks to manage memory
        total_loaded = 0
        chunk_size = batch_size
        
        # Create a single progress bar for all batch files
        with tqdm(total=len(batch_files), desc=f"Processing {source}", unit="batch") as pbar:
            for chunk_start in range(0, len(batch_files), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(batch_files))
                chunk_files = batch_files[chunk_start:chunk_end]
                
                chunk_loaded = self._process_batch_chunk_with_progress(
                    chunk_files, source, db_batch_size, pbar
                )
                total_loaded += chunk_loaded
                
                # Force garbage collection after each chunk
                gc.collect()
                
                # Small delay between chunks to prevent overwhelming ChromaDB
                time.sleep(0.5)
        
        print(f"✅ Completed {source}: {total_loaded} embeddings loaded")
        return total_loaded
    
    def _process_batch_chunk_with_progress(self, batch_files, source, db_batch_size, pbar):
        """Process a chunk of batch files with progress bar update."""
        all_embeddings = []
        all_chunks = []
        all_metadata = []
        all_ids = []
        
        # Process each batch file with error handling
        for batch_file in batch_files:
            try:
                batch_data = self._load_batch_file_safe(batch_file)
                if not batch_data:
                    pbar.update(1)
                    continue
                
                embeddings = batch_data.get('embeddings', [])
                chunks = batch_data.get('chunks', [])
                metadata_list = batch_data.get('metadata', [])
                
                if not embeddings or not chunks or not metadata_list:
                    logger.warning(f"⚠️  Invalid batch data in {os.path.basename(batch_file)}")
                    self.failed_batches.add(os.path.basename(batch_file))
                    pbar.update(1)
                    continue
                
                # Add source information to metadata
                batch_num = batch_data.get('batch_num', 0)
                for i, meta in enumerate(metadata_list):
                    meta['source_name'] = source
                    meta['batch_num'] = batch_num
                    meta['added_at'] = self._get_timestamp()
                
                # Generate unique IDs
                ids = [f"{source}_batch_{batch_num:04d}_{i}" for i in range(len(chunks))]
                
                # Add to collections
                all_embeddings.extend(embeddings)
                all_chunks.extend(chunks)
                all_metadata.extend(metadata_list)
                all_ids.extend(ids)
                
                self.processed_batches.add(os.path.basename(batch_file))
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'embeddings': len(all_embeddings),
                    'failed': len(self.failed_batches)
                })
                
            except Exception as e:
                logger.error(f"❌ Error processing {os.path.basename(batch_file)}: {e}")
                self.failed_batches.add(os.path.basename(batch_file))
                pbar.update(1)
                continue
        
        # Bulk insert to ChromaDB
        if all_embeddings:
            return self._bulk_insert_embeddings(
                all_embeddings, all_chunks, all_metadata, all_ids, db_batch_size
            )
        
        return 0
    
    def _process_batch_chunk(self, batch_files, source, db_batch_size):
        """Process a chunk of batch files (legacy method for backward compatibility)."""
        all_embeddings = []
        all_chunks = []
        all_metadata = []
        all_ids = []
        
        # Process each batch file with error handling
        for batch_file in tqdm(batch_files, desc="Loading batch files", unit="file"):
            try:
                batch_data = self._load_batch_file_safe(batch_file)
                if not batch_data:
                    continue
                
                embeddings = batch_data.get('embeddings', [])
                chunks = batch_data.get('chunks', [])
                metadata_list = batch_data.get('metadata', [])
                
                if not embeddings or not chunks or not metadata_list:
                    logger.warning(f"⚠️  Invalid batch data in {os.path.basename(batch_file)}")
                    self.failed_batches.add(os.path.basename(batch_file))
                    continue
                
                # Add source information to metadata
                batch_num = batch_data.get('batch_num', 0)
                for i, meta in enumerate(metadata_list):
                    meta['source_name'] = source
                    meta['batch_num'] = batch_num
                    meta['added_at'] = self._get_timestamp()
                
                # Generate unique IDs
                ids = [f"{source}_batch_{batch_num:04d}_{i}" for i in range(len(chunks))]
                
                # Add to collections
                all_embeddings.extend(embeddings)
                all_chunks.extend(chunks)
                all_metadata.extend(metadata_list)
                all_ids.extend(ids)
                
                self.processed_batches.add(os.path.basename(batch_file))
                
            except Exception as e:
                logger.error(f"❌ Error processing {os.path.basename(batch_file)}: {e}")
                self.failed_batches.add(os.path.basename(batch_file))
                continue
        
        # Bulk insert to ChromaDB
        if all_embeddings:
            return self._bulk_insert_embeddings(
                all_embeddings, all_chunks, all_metadata, all_ids, db_batch_size
            )
        
        return 0
    
    def _load_batch_file_safe(self, batch_file_path):
        """Safely load a batch file with error handling."""
        try:
            with open(batch_file_path, 'r', encoding='utf-8') as f:
                batch_data = json.load(f)
            return batch_data
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON decode error in {os.path.basename(batch_file_path)}: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Error loading {os.path.basename(batch_file_path)}: {e}")
            return None
    
    def _bulk_insert_embeddings(self, embeddings, chunks, metadata, ids, db_batch_size):
        """Bulk insert embeddings to ChromaDB with batching and retry logic."""
        total_inserted = 0
        
        # Adjust batch size based on compaction error count
        adjusted_batch_size = db_batch_size
        if self.compaction_error_count > 5:
            adjusted_batch_size = max(50, db_batch_size // 2)
            logger.info(f"🔄 Reducing batch size to {adjusted_batch_size} due to compaction errors")
        elif self.compaction_error_count > 10:
            adjusted_batch_size = max(25, db_batch_size // 4)
            logger.info(f"🔄 Further reducing batch size to {adjusted_batch_size} due to many compaction errors")
        
        # Split into database batches if too large
        if len(embeddings) > adjusted_batch_size:
            for i in range(0, len(embeddings), adjusted_batch_size):
                end_idx = min(i + adjusted_batch_size, len(embeddings))
                
                batch_embeddings = embeddings[i:end_idx]
                batch_chunks = chunks[i:end_idx]
                batch_metadata = metadata[i:end_idx]
                batch_ids = ids[i:end_idx]
                
                batch_num = i//adjusted_batch_size + 1
                if self._add_to_collection_with_retry(batch_embeddings, batch_chunks, batch_metadata, batch_ids, batch_num):
                    total_inserted += len(batch_embeddings)
                else:
                    print(f"   ❌ Failed database batch {batch_num} after all retries")
        else:
            # Single bulk insert
            if self._add_to_collection_with_retry(embeddings, chunks, metadata, ids, 1):
                total_inserted = len(embeddings)
            else:
                print(f"   ❌ Failed bulk insert after all retries")
        
        return total_inserted
    
    def _add_to_collection_with_retry(self, embeddings, chunks, metadata, ids, batch_num):
        """Add embeddings to ChromaDB collection with retry logic."""
        for attempt in range(self.retry_attempts):
            try:
                if self.chroma_manager.collection is None:
                    logger.error("❌ No ChromaDB collection available")
                    return False
                
                self.chroma_manager.collection.add(
                    embeddings=embeddings,
                    documents=chunks,
                    metadatas=metadata,
                    ids=ids
                )
                return True
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"❌ Attempt {attempt + 1}/{self.retry_attempts} failed for batch {batch_num}: {error_msg}")
                
                # Check if it's a compaction error
                if "compaction" in error_msg.lower() or "hnsw" in error_msg.lower():
                    self.compaction_error_count += 1
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"⚠️  Compaction error detected (count: {self.compaction_error_count}). Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    
                    # Force garbage collection
                    gc.collect()
                    
                    # Try to reset the collection connection
                    try:
                        if self.chroma_manager.collection and hasattr(self.chroma_manager.collection, 'name'):
                            self.chroma_manager.switch_collection(self.chroma_manager.collection.name)
                    except:
                        pass
                    
                    # For compaction errors, also try to reduce batch size on next attempt
                    if attempt > 1:
                        logger.info(f"🔄 Reducing batch size for next attempt...")
                else:
                    # For other errors, shorter delay
                    time.sleep(2)
                
                if attempt == self.retry_attempts - 1:
                    logger.error(f"❌ All {self.retry_attempts} attempts failed for batch {batch_num}")
                    return False
        
        return False
    
    def _get_timestamp(self):
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_stats(self):
        """Get loading statistics."""
        try:
            stats = self.chroma_manager.get_collection_stats()
            return {
                'total_documents': stats.get('total_documents', 0),
                'processed_batches': len(self.processed_batches),
                'failed_batches': len(self.failed_batches),
                'failed_files': list(self.failed_batches)
            }
        except Exception as e:
            logger.error(f"❌ Error getting stats: {e}")
            return {}

def main():
    """Main function to run robust xrvix loading."""
    print("=== Robust Xrvix Loader ===")
    print("This loader handles large datasets with memory management and error recovery.")
    print("📊 Progress: Single progress bar for entire process")
    
    # Initialize loader
    loader = RobustXrvixLoader()
    
    # Load with conservative settings to prevent compaction errors
    success = loader.load_with_progress(
        embeddings_dir="xrvix_embeddings",
        sources=None,  # All sources
        batch_size=25,  # Process 25 batch files at a time (reduced)
        db_batch_size=250,  # Add 250 embeddings to ChromaDB at once (reduced)
        max_batches_per_run=None  # Process all batches
    )
    
    if success:
        stats = loader.get_stats()
        print(f"\n📊 Final Statistics:")
        print(f"   Total documents: {stats.get('total_documents', 0)}")
        print(f"   Processed batches: {stats.get('processed_batches', 0)}")
        print(f"   Failed batches: {stats.get('failed_batches', 0)}")
        
        if stats.get('failed_batches', 0) > 0:
            print(f"   Failed files: {stats.get('failed_files', [])[:5]}...")
    else:
        print("❌ Loading failed")

if __name__ == "__main__":
    main() 