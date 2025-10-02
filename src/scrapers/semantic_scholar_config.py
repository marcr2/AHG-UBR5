"""
Configuration file for the Semantic Scholar scraper.
Contains all configurable parameters for paper collection and processing.
"""

# --- DEFAULT SEARCH CONFIGURATION ---
DEFAULT_SEARCH_TERMS = [
    # Direct UBR5 searches
    "UBR5",
    "ubr5", 
    "Ubr5",
    "ubiquitin protein ligase E3 component n-recognin 5",
    "EDD1",
    "edd1",
    "Edd1",
    "E3 ubiquitin-protein ligase UBR5",
    "ubiquitin ligase UBR5",
    
    # Functional searches
    "UBR5 function",
    "UBR5 regulation", 
    "UBR5 expression",
    "UBR5 mutation",
    "UBR5 activity",
    "UBR5 mechanism",
    
    # Disease-related searches
    "UBR5 cancer",
    "UBR5 disease",
    "UBR5 pathway",
    "UBR5 oncology",
    "UBR5 tumor",
    "UBR5 metastasis",
    
    # Interaction searches
    "UBR5 interaction",
    "UBR5 binding",
    "UBR5 complex",
    "UBR5 protein-protein",
    "UBR5 substrate",
    
    # Biological process searches
    "UBR5 ubiquitination",
    "UBR5 degradation",
    "UBR5 signaling",
    "UBR5 transcription",
    "UBR5 DNA repair",
    
    # Tissue and cell type searches
    "UBR5 tissue",
    "UBR5 cell line",
    "UBR5 organ",
    "UBR5 development"
]

# --- API CONFIGURATION ---
# Semantic Scholar API
SEMANTIC_SCHOLAR_CONFIG = {
    "base_url": "https://api.semanticscholar.org/graph/v1",
    "search_endpoint": "/paper/search",
    "paper_endpoint": "/paper",
    "batch_size": 100,
    "max_papers_per_query": 50,
    "fields": [
        "paperId", "title", "abstract", "venue", "year", "authors",
        "referenceCount", "citationCount", "openAccessPdf", 
        "publicationDate", "publicationTypes", "fieldsOfStudy",
        "publicationVenue", "externalIds", "url", "isOpenAccess"
    ]
}

# Google Scholar (via scholarly)
SCHOLARLY_CONFIG = {
    "max_papers_per_query": 30,
    "search_keywords": [
        "ubr5",
        "UBR5", 
        "ubr-5",
        "UBR-5"
    ]
}

# --- RATE LIMITING ---
RATE_LIMITING = {
    "semantic_scholar_delay": 3.0,  # 3 seconds between requests (conservative)
    "scholarly_delay": 0.2,         # 200ms between requests
    "keyword_delay": 5.0,           # 5 seconds between search keywords
    "embedding_delay": 0.1,         # 100ms between embedding requests
    "max_retries": 3,
    "timeout": 30,
    "rate_limit_wait": 60           # Wait 60 seconds on rate limit
}

# --- EMBEDDING CONFIGURATION ---
EMBEDDING_CONFIG = {
    "model": "text-embedding-004",
    "api_endpoint": "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedText",
    "max_text_length": 8000,        # Maximum text length for embedding
    "text_components": [
        "title", "abstract", "authors", "journal", "year", "fields_of_study"
    ]
}

# --- PAPER PROCESSING ---
PAPER_PROCESSING = {
    "min_title_length": 10,         # Minimum title length to consider valid
    "min_abstract_length": 50,      # Minimum abstract length to consider valid
    "title_similarity_threshold": 0.8,  # Similarity threshold for duplicate detection
    "max_abstract_length": 5000,    # Maximum abstract length to process
    "required_fields": ["title"],   # Fields that must be present
    "optional_fields": ["abstract", "authors", "journal", "year", "doi"]
}

# --- STORAGE CONFIGURATION ---
STORAGE_CONFIG = {
    "embeddings_dir": "xrvix_embeddings",
    "source_name": "semantic_scholar",
    "file_format": "json",
    "metadata_file": "metadata.json",
    "max_filename_length": 100,
    "save_individual_files": True,
    "save_metadata": True
}

# --- CHROMADB INTEGRATION ---
CHROMADB_CONFIG = {
    "collection_name": "ubr5_papers",
    "persist_directory": "./chroma_db",
    "metadata_fields": [
        "title", "doi", "authors", "journal", "year", "citation_count",
        "source", "is_preprint", "publication_date", "fields_of_study",
        "publication_types", "abstract"
    ],
    "max_abstract_length": 1000,    # Limit abstract length in ChromaDB
    "id_prefix": "ubr5_api"
}

# --- SEARCH CONFIGURATION ---
SEARCH_CONFIG = {
    "comprehensive": {
        "max_papers": 1000,
        "use_semantic_scholar": True,
        "use_scholarly": True,
        "search_all_keywords": True,
        "deduplication": True
    },
    "focused": {
        "max_papers": 500,
        "use_semantic_scholar": True,
        "use_scholarly": False,
        "search_all_keywords": False,
        "deduplication": True
    },
    "quick": {
        "max_papers": 200,
        "use_semantic_scholar": True,
        "use_scholarly": False,
        "search_all_keywords": False,
        "deduplication": False
    }
}

# --- LOGGING CONFIGURATION ---
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(levelname)s - %(message)s",
    "file": "ubr5_api_scraping.log",
    "console": True,
    "file_handler": True
}

# --- PERFORMANCE PROFILES ---
PERFORMANCE_PROFILES = {
    "conservative": {
        "rate_limit_delay": 0.2,
        "strategy_delay": 2.0,
        "max_workers": 1,
        "batch_size": 50
    },
    "balanced": {
        "rate_limit_delay": 0.1,
        "strategy_delay": 1.0,
        "max_workers": 2,
        "batch_size": 100
    },
    "aggressive": {
        "rate_limit_delay": 0.05,
        "strategy_delay": 0.5,
        "max_workers": 4,
        "batch_size": 200
    }
}

def get_config(profile: str = "balanced") -> dict:
    """
    Get configuration for a specific performance profile.
    
    Args:
        profile: Performance profile name
        
    Returns:
        Configuration dictionary
    """
    if profile not in PERFORMANCE_PROFILES:
        profile = "balanced"
    
    config = PERFORMANCE_PROFILES[profile].copy()
    config.update({
        "search_keywords": SCHOLARLY_CONFIG["search_keywords"],
        "semantic_scholar": SEMANTIC_SCHOLAR_CONFIG,
        "scholarly": SCHOLARLY_CONFIG,
        "rate_limiting": RATE_LIMITING,
        "embedding": EMBEDDING_CONFIG,
        "paper_processing": PAPER_PROCESSING,
        "storage": STORAGE_CONFIG,
        "chromadb": CHROMADB_CONFIG,
        "logging": LOGGING_CONFIG
    })
    
    return config

def print_config_info(profile: str = "balanced"):
    """Print current configuration information."""
    config = get_config(profile)
    
    print("🔧 UBR5 Scraper Configuration:")
    print(f"   Performance profile: {profile}")
    print(f"   Search keywords: {len(config['search_keywords'])} keywords")
    print(f"   Rate limit delay: {config['rate_limiting']['semantic_scholar_delay']}s")
    print(f"   Keyword delay: {config['rate_limiting']['keyword_delay']}s")
    print(f"   Embedding model: {config['embedding']['model']}")
    print(f"   Storage directory: {config['storage']['embeddings_dir']}")
    print(f"   ChromaDB collection: {config['chromadb']['collection_name']}")
    print()
    print("💡 Performance Tips:")
    print("   - Use 'conservative' profile for most reliable scraping")
    print("   - Use 'balanced' profile for good performance/reliability")
    print("   - Use 'aggressive' profile for fastest processing (may hit rate limits)")
    print("   - Adjust rate limiting if hitting API limits")

if __name__ == "__main__":
    print_config_info()
