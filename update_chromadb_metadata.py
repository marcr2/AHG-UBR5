#!/usr/bin/env python3
"""
Script to update existing ChromaDB data with new metadata fields.

This script adds the following fields to existing ChromaDB metadata:
- authors: Author information (extracted from existing data or inferred)
- publication_date: Publication date in YYYY-MM-DD format
- citation_count: Number of citations ("not found" if not available)
- journal: Journal or source name
"""

import json
import re
from datetime import datetime
from typing import Dict, Any
from tqdm import tqdm

def extract_authors_from_existing(metadata: Dict[str, Any]) -> str:
    """Extract author information from existing metadata."""
    # Check if authors field already exists
    if 'authors' in metadata and metadata['authors']:
        return metadata['authors']
    
    # Try to extract from title or other fields
    title = metadata.get('title', '')
    if title:
        # Look for common author patterns in titles
        # This is a fallback - ideally authors should be in the raw data
        author_patterns = [
            r'by\s+([^,]+(?:\s+et\s+al)?)',  # "by Author Name" or "by Author Name et al"
            r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+et\s+al',  # "Author et al"
            r'(Dr\.\s+[A-Z][a-z]+\s+[A-Z][a-z]+)\s+et\s+al',  # "Dr. Author et al"
            r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+and\s+others',  # "Author and others"
            r'(Dr\.\s+[A-Z][a-z]+\s+[A-Z][a-z]+)\s+and\s+others',  # "Dr. Author and others"
        ]
        
        for pattern in author_patterns:
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                return match.group(1)
    
    return "Unknown authors"

def extract_publication_date_from_existing(metadata: Dict[str, Any]) -> str:
    """Extract publication date from existing metadata."""
    def is_valid_year(year_str):
        """Check if a year string represents a valid scientific publication year."""
        try:
            year = int(year_str)
            # Valid years should be between 1900 and current year + 1
            current_year = datetime.now().year
            return 1900 <= year <= current_year + 1
        except (ValueError, TypeError):
            return False
    
    # Try to extract from DOI
    doi = metadata.get('doi', '')
    if doi:
        year_match = re.search(r'(\d{4})', doi)
        if year_match and is_valid_year(year_match.group(1)):
            return f"{year_match.group(1)}-01-01"
    
    # Try to extract from title or other fields
    title = metadata.get('title', '')
    if title:
        # Try DD-MM-YYYY format first (as specified by user)
        if re.match(r'\d{2}-\d{2}-\d{4}', title):
            year = title[6:10]
            if is_valid_year(year):
                # Convert DD-MM-YYYY to YYYY-MM-DD
                return f"{year}-{title[3:5]}-{title[0:2]}"
        # Try YYYY-MM-DD format
        elif re.match(r'\d{4}-\d{2}-\d{2}', title):
            year = title[:4]
            if is_valid_year(year):
                return title
        # Try YYYY format
        elif re.match(r'\d{4}', title):
            if is_valid_year(title):
                return f"{title}-01-01"
        else:
            # Fallback to finding any 4-digit year
            year_match = re.search(r'(\d{4})', title)
            if year_match and is_valid_year(year_match.group(1)):
                return f"{year_match.group(1)}-01-01"
    
    return f"{datetime.now().year}-01-01"

def extract_citation_count_from_existing(metadata: Dict[str, Any]) -> str:
    """Extract citation count from existing metadata."""
    # Return "not found" for existing data
    return "not found"

def extract_journal_info_from_existing(metadata: Dict[str, Any]) -> str:
    """Extract journal info from existing metadata."""
    source = metadata.get('source', '')
    if source in ['biorxiv', 'medrxiv']:
        return f"{source.upper()}"
    elif source == 'pubmed':
        return "PubMed"
    else:
        return "Unknown journal"

def update_chromadb_metadata():
    """
    Update existing ChromaDB data with new metadata fields.
    """
    print("🔄 Updating ChromaDB metadata...")
    
    try:
        from chromadb_manager import ChromaDBManager
        
        # Initialize ChromaDB manager
        db_manager = ChromaDBManager()
        
        # Get all collections
        collections = db_manager.list_collections()
        
        if not collections:
            print("📊 No collections found in ChromaDB")
            return
        
        total_updated = 0
        total_papers = 0
        
        # First, count total papers across all collections
        print("📊 Counting total papers across all collections...")
        for collection_name in collections:
            if db_manager.switch_collection(collection_name):
                results = db_manager.collection.get()
                if results['ids']:
                    total_papers += len(results['ids'])
        
        print(f"🔄 Processing {total_papers} total papers across {len(collections)} collection(s)...")
        
        # Create main progress bar for papers
        with tqdm(total=total_papers, desc="Updating papers", unit="paper") as pbar:
            for collection_name in collections:
                print(f"📊 Processing collection: {collection_name}")
                
                # Switch to the collection
                if not db_manager.switch_collection(collection_name):
                    print(f"   ❌ Failed to switch to collection {collection_name}")
                    continue
                
                # Get all data from the collection
                results = db_manager.collection.get()
                
                if not results['ids']:
                    print(f"   No data found in collection {collection_name}")
                    continue
                
                total_entries = len(results['ids'])
                print(f"   📝 Processing {total_entries} papers...")
                
                # Update metadata for each entry
                updated_metadata = []
                updated_count = 0
                
                for i, metadata in enumerate(results['metadatas']):
                    if metadata is None:
                        metadata = {}
                    
                    original_metadata = metadata.copy()
                    
                    # Add authors field if it doesn't exist
                    if 'authors' not in metadata or not metadata['authors']:
                        metadata['authors'] = extract_authors_from_existing(metadata)
                        updated_count += 1
                    
                    # Add new fields if they don't exist
                    if 'publication_date' not in metadata:
                        metadata['publication_date'] = extract_publication_date_from_existing(metadata)
                        updated_count += 1
                    
                    if 'citation_count' not in metadata:
                        metadata['citation_count'] = extract_citation_count_from_existing(metadata)
                        updated_count += 1
                    
                    if 'journal' not in metadata:
                        metadata['journal'] = extract_journal_info_from_existing(metadata)
                        updated_count += 1
                    
                    updated_metadata.append(metadata)
                    
                    # Update progress bar
                    pbar.update(1)
            
            if updated_count > 0:
                # Update the collection with new metadata in batches
                print(f"   💾 Saving updates to ChromaDB in batches...")
                
                # ChromaDB has a max batch size limit, so we need to process in chunks
                batch_size = 5000  # Safe batch size well under the 5461 limit
                total_batches = (len(updated_metadata) + batch_size - 1) // batch_size
                
                for batch_idx in range(total_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, len(updated_metadata))
                    
                    batch_ids = results['ids'][start_idx:end_idx]
                    batch_metadata = updated_metadata[start_idx:end_idx]
                    
                    print(f"      📦 Processing batch {batch_idx + 1}/{total_batches} ({len(batch_ids)} papers)...")
                    
                    db_manager.collection.update(
                        ids=batch_ids,
                        metadatas=batch_metadata
                    )
                
                print(f"   ✅ Updated {len(updated_metadata)} entries in {collection_name}")
                print(f"   📝 Added {updated_count} new metadata fields per entry")
                total_updated += len(updated_metadata)
            else:
                print(f"   ℹ️  No updates needed for {collection_name}")
        
        print(f"\n✅ ChromaDB metadata update complete!")
        print(f"📊 Total entries updated: {total_updated}")
        
    except Exception as e:
        print(f"❌ Error updating ChromaDB: {e}")
        print("Please ensure ChromaDB is running and accessible.")

def main():
    """Main function to update ChromaDB metadata."""
    print("=== ChromaDB Metadata Update ===")
    print("Adding authors, publication_date, citation_count, and journal fields")
    print()
    
    # Update ChromaDB metadata
    update_chromadb_metadata()
    
    print("\n🎉 Metadata update complete!")
    print("\nNew metadata fields added:")
    print("  - authors: Author information (extracted from existing data or inferred)")
    print("  - publication_date: Publication date in YYYY-MM-DD format")
    print("  - citation_count: Number of citations (\"not found\" if not available)")
    print("  - journal: Journal or source name")
    print("\nThese fields will now be available for filtering and display in your RAG system.")

if __name__ == "__main__":
    main() 