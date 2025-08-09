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
from processing_config import get_config, print_config_info
from chromadb_manager import ChromaDBManager
import warnings

# Suppress paperscraper warnings
warnings.filterwarnings("ignore", message="Could not find paper")
warnings.filterwarnings("ignore", category=UserWarning, module="paperscraper")

# Global counter for citation processing errors
citation_error_counter = {
    'doi_lookup_failures': 0,
    'title_lookup_failures': 0,
    'total_failures': 0
}

# Global list to collect papers for batch citation processing
papers_for_citation_processing = []

def reset_citation_error_counter():
    """Reset the global citation error counter."""
    global citation_error_counter
    citation_error_counter = {
        'doi_lookup_failures': 0,
        'title_lookup_failures': 0,
        'total_failures': 0
    }

def get_citation_error_summary():
    """Get a summary of citation processing errors."""
    global citation_error_counter
    return citation_error_counter.copy()

def get_papers_for_citation_processing():
    """Get the global papers_for_citation_processing list."""
    global papers_for_citation_processing
    return papers_for_citation_processing

def reset_papers_for_citation_processing():
    """Reset the global papers collection for batch citation processing."""
    global papers_for_citation_processing
    papers_for_citation_processing = []

def add_paper_for_citation_processing(paper_data):
    """Add a paper to the global collection for batch citation processing."""
    global papers_for_citation_processing
    papers_for_citation_processing.append(paper_data)

def process_citations_batch(papers_batch):
    """Process citations, journal info, and impact factors for a batch of papers."""
    global citation_error_counter
    
    if not papers_batch:
        return {}
    
    print(f"📊 Processing citations for {len(papers_batch)} papers...")
    
    results = {}
    processed_count = 0
    failed_count = 0
    
    for paper_data in papers_batch:
        idx, row, source = paper_data
        paper_key = f"{source}_{idx}"
        
        try:
            # Process citation count
            citation_count = extract_citation_count(row)
            
            # Process journal info
            journal = extract_journal_info(row)
            
            # Process impact factor
            impact_factor = extract_impact_factor(row, journal)
            
            results[paper_key] = {
                'citation_count': citation_count,
                'journal': journal,
                'impact_factor': impact_factor
            }
            processed_count += 1
            
        except Exception as e:
            failed_count += 1
            citation_error_counter['total_failures'] += 1
            results[paper_key] = {
                'citation_count': 'not found',
                'journal': 'Unknown journal',
                'impact_factor': 'not found'
            }
    
    print(f"✅ Citation processing complete: {processed_count} successful, {failed_count} failed")
    return results

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
    """Extract citation count from paper data using paperscraper with robust error handling."""
    global citation_error_counter
    
    # First try to get citation count from DOI if available
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
        for attempt in range(3):  # Try up to 3 times
            try:
                # Wrap the entire paperscraper call in a try-except to handle internal library errors
                citations = get_citations_by_doi(doi)
                if citations is not None and citations > 0:
                    return str(citations)
                elif citations == 0:
                    # Explicitly return 0 if paperscraper found the paper but it has 0 citations
                    return "0"
            except Exception as e:
                # Silently handle errors, just increment counter
                if attempt == 2:  # Only count on final attempt
                    citation_error_counter['doi_lookup_failures'] += 1
                    citation_error_counter['total_failures'] += 1
                time.sleep(2 + attempt * 2)  # Progressive delay (2s, 4s, 6s)
                continue
    
    # If no DOI or DOI lookup failed, try title
    title = paper_data.get('title', '')
    if title:
        # Clean and validate title
        title = str(title).strip()
        if not title or title.lower() in ['nan', 'none', 'null', '']:
            title = ''
    
    if title:
        for attempt in range(3):  # Try up to 3 times
            try:
                # Wrap the entire paperscraper call in a try-except to handle internal library errors
                citations = get_citations_from_title(title)
                if citations is not None and citations > 0:
                    return str(citations)
                elif citations == 0:
                    # Explicitly return 0 if paperscraper found the paper but it has 0 citations
                    return "0"
            except Exception as e:
                # Silently handle errors, just increment counter
                if attempt == 2:  # Only count on final attempt
                    citation_error_counter['title_lookup_failures'] += 1
                    citation_error_counter['total_failures'] += 1
                time.sleep(2 + attempt * 2)  # Progressive delay (2s, 4s, 6s)
                continue
    
    # Fallback to checking if citation data is already in the paper_data
    citation_fields = [
        'citations', 'citation_count', 'cited_by_count', 'times_cited',
        'reference_count', 'cited_count', 'num_citations', 'citedByCount'
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
    """Extract journal information from paper data using paperscraper with robust error handling."""
    # First check if journal information is already in the paper_data
    journal_fields = [
        'journal', 'journal_name', 'publication', 'source', 'venue',
        'journal_title', 'publication_venue', 'journal_ref', 'journal-ref'
    ]
    
    for field in journal_fields:
        if field in paper_data and paper_data[field]:
            journal_name = str(paper_data[field])
            # Try to get the full journal name using paperscraper
            for attempt in range(2):  # Try up to 2 times
                try:
                    impactor = Impactor()
                    results = impactor.search(journal_name, threshold=85)
                    if results:
                        # Return the most relevant journal name
                        return results[0]['journal']
                except Exception as e:
                    # Silently handle errors
                    time.sleep(2 + attempt * 2)  # Progressive delay (2s, 4s)
                    continue
            # If paperscraper failed, return the original journal name
            return journal_name
    
    # If no journal found in paper_data, check source
    source = paper_data.get('source', '')
    if source in ['biorxiv', 'medrxiv']:
        return f"{source.upper()}"
    
    return "Unknown journal"

def extract_impact_factor(paper_data, journal_name=None):
    """Extract or calculate impact factor from paper data using paperscraper with robust error handling."""
    # First, try to get impact factor directly from the data
    impact_fields = [
        'impact_factor', 'journal_impact_factor', 'if', 'jif',
        'impact', 'journal_if', 'journal_impact'
    ]
    
    for field in impact_fields:
        if field in paper_data and paper_data[field] is not None:
            try:
                impact = float(paper_data[field])
                return str(max(0, impact))
            except (ValueError, TypeError):
                continue
    
    # If no direct impact factor, try to get it using paperscraper
    if not journal_name:
        journal_name = extract_journal_info(paper_data)
    
    if journal_name and journal_name != "Unknown journal":
        for attempt in range(2):  # Try up to 2 times
            try:
                impactor = Impactor()
                results = impactor.search(journal_name, threshold=85)
                if results:
                    # Return the impact factor of the most relevant match
                    impact_factor = results[0].get('factor', 0)
                    if impact_factor > 0:
                        return str(impact_factor)
            except Exception as e:
                # Silently handle errors
                time.sleep(2 + attempt * 2)  # Progressive delay (2s, 4s)
                continue
    
    # Fallback to the old estimation method
    return estimate_impact_factor(journal_name)

def estimate_impact_factor(journal_name):
    """Estimate impact factor based on journal name."""
    if not journal_name or journal_name == "Unknown journal":
        return "not found"
    
    # Comprehensive impact factor mapping based on recent data
    impact_factors = {
        'nature': 49.962,
        'science': 56.9,
        'cell': 66.85,
        'nature medicine': 87.241,
        'nature biotechnology': 68.164,
        'nature genetics': 41.307,
        'nature cell biology': 28.213,
        'nature immunology': 31.25,
        'nature reviews immunology': 108.555,
        'immunity': 43.474,
        'journal of immunology': 5.422,
        'journal of experimental medicine': 17.579,
        'proceedings of the national academy of sciences': 12.779,
        'pnas': 12.779,
        'plos one': 3.752,
        'bioinformatics': 6.937,
        'nucleic acids research': 19.16,
        'genome research': 11.093,
        'genome biology': 17.906,
        'cell reports': 9.995,
        'molecular cell': 19.328,
        'developmental cell': 13.417,
        'current biology': 10.834,
        'elife': 8.713,
        'plos biology': 9.593,
        'plos genetics': 6.02,
        'plos computational biology': 4.7,
        'bmc genomics': 4.317,
        'bmc bioinformatics': 3.169,
        'nature communications': 17.694,
        'nature methods': 47.99,
        'nature neuroscience': 25.0,
        'neuron': 16.2,
        'the lancet': 202.731,
        'new england journal of medicine': 176.079,
        'jama': 157.335,
        'biorxiv': 0.0,  # Preprint servers have no impact factor
        'medrxiv': 0.0,
        'arxiv': 0.0,
        'chemrxiv': 0.0
    }
    
    journal_lower = journal_name.lower().strip()
    
    # Direct match
    if journal_lower in impact_factors:
        return str(impact_factors[journal_lower])
    
    # Partial match
    for key, impact in impact_factors.items():
        if key in journal_lower or journal_lower in key:
            return str(impact)
    
    # Default for unknown journals
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
    
    # Reset citation error counter at the start
    reset_citation_error_counter()
    reset_papers_for_citation_processing()  # Reset citation processing collection
    
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
    
    # Track papers processed for batch citation processing
    papers_processed_since_last_citation_batch = 0
    
    print(f"\n🔄 Processing papers and creating embeddings...")
    with tqdm(total=len(df), desc="Processing papers", unit="paper") as pbar:
        for idx, row in df.iterrows():
            title = row.get("title", "")
            doi = row.get("doi", "")
            abstract = row.get("abstract", "")
            authors_raw = row.get("authors", "")
            # Convert author list to string for ChromaDB compatibility
            if isinstance(authors_raw, list):
                author = "; ".join(authors_raw)  # Join authors with semicolon separator
            else:
                author = str(authors_raw) if authors_raw else ""
            
            # Extract basic metadata fields (without citations/journal/impact factor)
            publication_date = extract_publication_date(row)
            year = publication_date[:4] if publication_date else str(datetime.now().year)
            
            # Add paper to batch citation processing collection
            add_paper_for_citation_processing((idx, row, "pubmed"))
            
            # Extract additional metadata fields (without citations/journal/impact factor)
            additional_metadata = extract_additional_metadata(row)
            
            paragraphs = chunk_paragraphs(abstract)
            
            # Process paragraphs for this paper
            paper_embeddings = 0
            for i, para in enumerate(paragraphs):
                if not para.strip():  # Skip empty paragraphs
                    continue
                    
                try:
                    embedding = get_google_embedding(para, api_key)
                    
                    # Create comprehensive metadata object (without citations/journal/impact factor)
                    metadata = {
                        "title": title,
                        "doi": doi,
                        "author": author,  # Use author field name for consistency
                        "publication_date": publication_date,
                        "citation_count": "pending",  # Will be updated in batch processing
                        "journal": "pending",  # Will be updated in batch processing
                        "impact_factor": "pending",  # Will be updated in batch processing
                        "source": "pubmed",
                        "paper_index": idx,
                        "para_idx": i,
                        "chunk_length": len(para),
                        "year": year  # Add explicit year field
                    }
                    
                    # Add additional metadata fields if available
                    metadata.update(additional_metadata)
                    
                    # Store chunk, embedding, and metadata
                    embeddings_data["chunks"].append(para)
                    embeddings_data["embeddings"].append(embedding)
                    embeddings_data["metadata"].append(metadata)
                    
                    paper_embeddings += 1
                    total_embeddings += 1
                except Exception as e:
                    print(f"\n⚠️  Embedding error for PubMed paper {idx}: {e}")
                    continue
            
            total_chunks += len(paragraphs)
            
            # Increment papers processed counter
            papers_processed_since_last_citation_batch += 1
            
            # Check if we need to process citations in batch (every 1000 papers)
            if papers_processed_since_last_citation_batch >= 1000:
                # Process citations for the collected papers
                citation_results = process_citations_batch(get_papers_for_citation_processing())
                
                # Update metadata for all papers in the current batch
                for result in embeddings_data["metadata"]:
                    paper_key = f"{result['source']}_{result['paper_index']}"
                    if paper_key in citation_results:
                        result['citation_count'] = citation_results[paper_key]['citation_count']
                        result['journal'] = citation_results[paper_key]['journal']
                        result['impact_factor'] = citation_results[paper_key]['impact_factor']
                
                # Reset the collection
                reset_papers_for_citation_processing()
                papers_processed_since_last_citation_batch = 0
            
            pbar.update(1)
            pbar.set_postfix({"chunks": total_chunks, "embeddings": total_embeddings})
    
    # Process any remaining papers for citations
    citation_results = process_citations_batch(get_papers_for_citation_processing())
    
    # Update metadata for all papers in the current batch
    for result in embeddings_data["metadata"]:
        paper_key = f"{result['source']}_{result['paper_index']}"
        if paper_key in citation_results:
            result['citation_count'] = citation_results[paper_key]['citation_count']
            result['journal'] = citation_results[paper_key]['journal']
            result['impact_factor'] = citation_results[paper_key]['impact_factor']
    
    # Reset the collection
    reset_papers_for_citation_processing()
    
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
    
    # Print citation processing summary
    citation_summary = get_citation_error_summary()
    if citation_summary['total_failures'] > 0:
        print(f"\n📊 Citation Processing Summary:")
        print(f"   DOI lookup failures: {citation_summary['doi_lookup_failures']}")
        print(f"   Title lookup failures: {citation_summary['title_lookup_failures']}")
        print(f"   Total citation processing failures: {citation_summary['total_failures']}")
    else:
        print(f"\n✅ All citation lookups completed successfully!")

if __name__ == "__main__":
    main() 