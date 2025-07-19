import os
import json
import pandas as pd
from paperscraper.pubmed import get_and_dump_pubmed_papers
import requests
from tqdm.auto import tqdm
import numpy as np
from processing_config import get_config, print_config_info
from chromadb_manager import ChromaDBManager

# --- CONFIG ---
# Load configuration from environment variable or use default
CONFIG_PROFILE = os.environ.get("PROCESSING_PROFILE", "balanced")
config = get_config(CONFIG_PROFILE)

# --- UTILS ---
def get_google_embedding(text, api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {"model": "models/text-embedding-004", "content": {"parts": [{"text": text}]}}
    
    # Add rate limiting delay
    if config.get("rate_limit_delay", 0) > 0:
        import time
        time.sleep(config.get("rate_limit_delay", 0))
    
    response = requests.post(url, headers=headers, json=data, timeout=config.get("request_timeout", 30))
    response.raise_for_status()
    return response.json()["embedding"]["values"]

def chunk_paragraphs(text):
    if not isinstance(text, str):
        return []
    return [p.strip() for p in text.split("\n") if p.strip()]

def save_embeddings_to_json(embeddings_data, filename="embeddings.json"):
    """Save embeddings and metadata to JSON file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(embeddings_data, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved embeddings to {filename}")

def load_embeddings_from_json(filename="embeddings.json"):
    """Load embeddings and metadata from JSON file"""
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"chunks": [], "embeddings": [], "metadata": []}

DUMP_FILE = "pubmed_dump.jsonl"
EMBEDDINGS_FILE = "pubmed_embeddings.json"

def get_search_terms():
    print("=== PubMed Search Terms Manager ===")
    print("Commands:")
    print("  - Type a search term to add it")
    print("  - 'list' to see current terms")
    print("  - 'remove <index>' to remove a term (e.g., 'remove 2')")
    print("  - 'clear' to remove all terms")
    print("  - 'done' when finished")
    print()
    
    terms = ["UBR-5","E3 ubiquitin-protein ligase UBR5","E3 ubiquitin-protein ligase","E3 ubiquitin-protein"]
    
    while True:
        user_input = input("Enter command or search term: ").strip()
        
        if user_input.lower() == 'done':
            break
        elif user_input.lower() == 'list':
            if not terms:
                print("📝 No search terms added yet.")
            else:
                print("📝 Current search terms:")
                for i, term in enumerate(terms, 1):
                    print(f"  {i}. {term}")
            print()
        elif user_input.lower() == 'clear':
            terms.clear()
            print("🗑️  All search terms cleared.")
            print()
        elif user_input.lower().startswith('remove '):
            try:
                # Extract index from "remove <index>"
                index_str = user_input[7:].strip()  # Remove "remove " prefix
                index = int(index_str) - 1  # Convert to 0-based index
                
                if 0 <= index < len(terms):
                    removed_term = terms.pop(index)
                    print(f"🗑️  Removed: {removed_term}")
                else:
                    print(f"❌ Invalid index. Use 'list' to see current terms.")
            except ValueError:
                print("❌ Invalid format. Use 'remove <number>' (e.g., 'remove 2')")
            print()
        elif user_input:
            # Add new search term
            if user_input in terms:
                print(f"⚠️  '{user_input}' is already in the list.")
            else:
                terms.append(user_input)
                print(f"✅ Added: {user_input}")
            print()
        else:
            print("❌ Please enter a search term or command.")
            print()
    
    if not terms:
        print("No search terms entered. Exiting.")
        exit(0)
    
    print("📝 Final search terms:")
    for i, term in enumerate(terms, 1):
        print(f"  {i}. {term}")
    print()
    
    return list(terms)

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

def main():
    print("=== Starting PubMed Scraping and Processing (JSON Storage) ===")
    print(f"🔧 Using configuration profile: {CONFIG_PROFILE}")
    print_config_info()
    
    # Get search terms from user
    search_terms = get_search_terms()
    print(f"🔍 Search terms: {', '.join(search_terms)}")

    # Load API key
    with open("keys.json") as f:
        api_key = json.load(f)["GOOGLE_API_KEY"]

    # Step 1: Fetch and dump PubMed papers
    print(f"\n🔄 Fetching PubMed papers for {len(search_terms)} search terms...")
    print("⏱️  This may take several minutes...")
    
    with tqdm(total=len(search_terms), desc="Fetching papers", unit="term") as pbar:
        # Convert to the expected format for the function
        keywords = [[term] for term in search_terms]
        get_and_dump_pubmed_papers(keywords, DUMP_FILE)
        pbar.update(len(search_terms))
    
    print(f"✅ Saved PubMed results to {DUMP_FILE}")

    # Step 2: Load dump and process
    print(f"\n📊 Loading PubMed dump...")
    df = pd.read_json(DUMP_FILE, lines=True)
    print(f"📊 Found {len(df)} papers")

    # Initialize embeddings storage
    embeddings_data = {
        "chunks": [],
        "embeddings": [],
        "metadata": [],
        "stats": {
            "total_papers": len(df),
            "total_chunks": 0,
            "total_embeddings": 0
        }
    }

    # Process papers with progress bar
    total_chunks = 0
    total_embeddings = 0
    
    print(f"\n🔄 Processing papers and creating embeddings...")
    with tqdm(total=len(df), desc="Processing papers", unit="paper") as pbar:
        for idx, row in df.iterrows():
            title = row.get("title", "")
            doi = row.get("doi", "")
            abstract = row.get("abstract", "")
            paragraphs = chunk_paragraphs(abstract)
            
            # Process paragraphs for this paper
            paper_embeddings = 0
            for i, para in enumerate(paragraphs):
                if not para.strip():  # Skip empty paragraphs
                    continue
                    
                try:
                    embedding = get_google_embedding(para, api_key)
                    
                    # Store chunk, embedding, and metadata
                    embeddings_data["chunks"].append(para)
                    embeddings_data["embeddings"].append(embedding)
                    embeddings_data["metadata"].append({
                        "title": title,
                        "doi": doi,
                        "source": "pubmed",
                        "paper_index": idx,
                        "para_idx": i,
                        "chunk_length": len(para)
                    })
                    
                    paper_embeddings += 1
                    total_embeddings += 1
                except Exception as e:
                    print(f"\n⚠️  Embedding error for PubMed paper {idx}: {e}")
                    continue
            
            total_chunks += len(paragraphs)
            pbar.update(1)
            pbar.set_postfix({"chunks": total_chunks, "embeddings": total_embeddings})
    
    # Update final statistics
    embeddings_data["stats"]["total_chunks"] = total_chunks
    embeddings_data["stats"]["total_embeddings"] = total_embeddings
    
    # Save embeddings to JSON
    save_embeddings_to_json(embeddings_data, EMBEDDINGS_FILE)
    
    # Step 3: Automatically load embeddings into ChromaDB
    load_embeddings_to_chromadb(embeddings_data, "pubmed")
    
    # Print final statistics
    print(f"\n🎉 PubMed processing complete!")
    print(f"📊 Total chunks processed: {total_chunks}")
    print(f"📊 Total embeddings created: {total_embeddings}")
    print(f"📁 Embeddings file: {os.path.abspath(EMBEDDINGS_FILE)}")
    print(f"📊 File size: {os.path.getsize(EMBEDDINGS_FILE) / 1024:.1f} KB")
    print(f"🗄️  Embeddings loaded into ChromaDB for RAG queries")

if __name__ == "__main__":
    main() 