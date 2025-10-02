import os
import json
import requests
import time
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import logging
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import pickle
from pathlib import Path
import urllib.parse
import signal
from src.core.processing_config import get_config, print_config_info
from src.core.chromadb_manager import ChromaDBManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data/logs/semantic_scholar_scraping.log')
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings for external libraries
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Could not find paper.*")

class SemanticScholarScraper:
    """
    Comprehensive paper scraper using Semantic Scholar API.
    Collects the same data and metadata as xrvix and PubMed scrapers.
    Note: Google Scholar removed due to CAPTCHA/rate limiting issues.
    """
    
    def __init__(self, api_keys: Dict[str, str] = None):
        """
        Initialize the Semantic Scholar scraper.
        
        Args:
            api_keys: Dictionary containing API keys for different services
        """
        self.api_keys = api_keys or {}
        self.embeddings_dir = "data/embeddings/semantic_scholar"
        self.scraped_data_dir = "data/scraped_data/semantic_scholar"
        self.papers_data = []
        self.processed_dois = set()
        
        # Load API keys from keys.json if not provided
        if not self.api_keys:
            self._load_api_keys()
        
        # Get Semantic Scholar API key
        self.semantic_scholar_api_key = self.api_keys.get("SEMANTIC_SCHOLAR_API_KEY")
        if self.semantic_scholar_api_key:
            logger.info("✅ Semantic Scholar API key found - using authenticated requests")
        else:
            logger.info("⚠️ No Semantic Scholar API key found - using unauthenticated requests (lower rate limits)")
        
        # API endpoints and rate limiting
        self.semantic_scholar_base = "https://api.semanticscholar.org/v1"
        self.semantic_scholar_v2_base = "https://api.semanticscholar.org/graph/v1"
        self.scholarly_base = "https://scholar.google.com"
        
        # Rate limiting configuration - Adjust based on API key availability
        if self.semantic_scholar_api_key:
            # More aggressive rate limiting with API key
            self.rate_limit_delay = 1.0  # 1 second between requests (with API key)
            self.semantic_scholar_rate_limit = 1.0  # 1 second between Semantic Scholar requests
            self.keyword_delay = 2.0  # 2 seconds between different keywords
            logger.info("🚀 Using faster rate limiting with API key")
        else:
            # Conservative rate limiting without API key
            self.rate_limit_delay = 3.0  # 3 seconds between requests (conservative)
            self.semantic_scholar_rate_limit = 3.0  # 3 seconds between Semantic Scholar requests
            self.keyword_delay = 5.0  # 5 seconds between different keywords
            logger.info("🐌 Using conservative rate limiting without API key")
        
        self.max_retries = 3
        self.timeout = 30
        
        # Note: Google Scholar removed due to CAPTCHA/rate limiting issues
        # Using Semantic Scholar as primary source (reliable, no CAPTCHA)
        
        # Default search terms (can be customized)
        self.default_search_terms = [
            "UBR5", "ubr5", "Ubr5",
            "ubiquitin protein ligase E3 component n-recognin 5",
            "EDD1", "edd1", "Edd1",
            "E3 ubiquitin-protein ligase UBR5",
            "ubiquitin ligase UBR5",
            "UBR5 gene", "UBR5 protein",
            "UBR5 mutation", "UBR5 expression",
            "UBR5 function", "UBR5 regulation"
        ]
        
        # Ensure embeddings directory exists
        os.makedirs(self.embeddings_dir, exist_ok=True)
        
        # Initialize ChromaDB manager
        try:
            self.chromadb_manager = ChromaDBManager()
            if not self.chromadb_manager.create_collection():
                logger.warning("⚠️ Failed to create ChromaDB collection, will retry during integration")
        except Exception as e:
            logger.error(f"❌ Failed to initialize ChromaDB manager: {e}")
            self.chromadb_manager = None
    
    def _load_keywords_from_config(self):
        """Load keywords from configuration file."""
        import json
        import os
        
        config_file = "config/search_keywords_config.json"
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                semantic_keywords = config.get("semantic_keywords", "")
                
                if semantic_keywords:
                    # Parse comma-separated keywords and clean them
                    keywords = [keyword.strip() for keyword in semantic_keywords.split(',') if keyword.strip()]
                    return keywords
            except Exception as e:
                logger.warning(f"Could not load keywords from config: {e}")
        
        return None
        
    
    def check_api_key_availability(self) -> Dict[str, bool]:
        """
        Check which API keys are available.
        
        Returns:
            Dictionary indicating which API keys are available
        """
        availability = {
            "google_api": bool(self.api_keys.get("GOOGLE_API_KEY")),
            "semantic_scholar": True,  # No API key required for basic usage
        }
        
        logger.info("🔑 API Key Availability:")
        logger.info(f"   Google API (embeddings): {'✅' if availability['google_api'] else '❌'}")
        logger.info(f"   Semantic Scholar: {'✅' if availability['semantic_scholar'] else '❌'}")
        logger.info("   Google Scholar: ❌ Removed (CAPTCHA/rate limiting issues)")
        
        return availability
    
    def test_api_connectivity(self) -> Dict[str, bool]:
        """
        Test connectivity to various APIs.
        
        Returns:
            Dictionary indicating which APIs are accessible
        """
        logger.info("🔍 Testing API connectivity...")
        
        connectivity = {
            "semantic_scholar": False,
            "google_api": False
        }
        
        # Test Semantic Scholar API
        try:
            test_url = f"{self.semantic_scholar_v2_base}/paper/search"
            test_params = {"query": "test", "limit": 1}
            response = requests.get(test_url, params=test_params, timeout=10)
            
            if response.status_code == 200:
                connectivity["semantic_scholar"] = True
                logger.info("✅ Semantic Scholar API: Accessible")
            else:
                logger.warning(f"⚠️ Semantic Scholar API: HTTP {response.status_code}")
        except Exception as e:
            logger.warning(f"⚠️ Semantic Scholar API: Connection failed - {e}")
        
        # Test Google API (if key available)
        google_api_key = self.api_keys.get("GOOGLE_API_KEY")
        if google_api_key:
            try:
                test_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"
                test_data = {"content": {"parts": [{"text": "test"}]}}
                test_params = {"key": google_api_key}
                response = requests.post(test_url, json=test_data, params=test_params, timeout=10)
                
                if response.status_code == 200:
                    connectivity["google_api"] = True
                    logger.info("✅ Google API: Accessible")
                else:
                    logger.warning(f"⚠️ Google API: HTTP {response.status_code}")
            except Exception as e:
                logger.warning(f"⚠️ Google API: Connection failed - {e}")
        else:
            logger.info("ℹ️ Google API: No API key provided")
        
        # Google Scholar removed due to CAPTCHA/rate limiting issues
        logger.info("ℹ️ Google Scholar: Removed (CAPTCHA/rate limiting issues)")
        
        return connectivity
    
    
    def _load_api_keys(self):
        """Load API keys from keys.json file."""
        try:
            with open("config/keys.json", 'r') as f:
                keys_data = json.load(f)
                self.api_keys.update(keys_data)
                logger.info("✅ Loaded API keys from config/keys.json")
                
                # Log available API keys (without exposing the actual keys)
                available_keys = list(keys_data.keys())
                logger.info(f"📋 Available API keys: {', '.join(available_keys)}")
                
        except FileNotFoundError:
            logger.warning("⚠️ config/keys.json not found, using default configuration")
        except json.JSONDecodeError:
            logger.error("❌ Invalid JSON in config/keys.json")
        except Exception as e:
            logger.error(f"❌ Error loading API keys: {e}")
    
    def search_semantic_scholar(self, query: str, limit: int = 100) -> List[Dict]:
        """
        Search for papers using Semantic Scholar API with improved error handling and retry mechanism.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of paper dictionaries
        """
        logger.info(f"🔍 Searching Semantic Scholar for: {query}")
        
        papers = []
        offset = 0
        batch_size = 100
        consecutive_empty_responses = 0
        max_consecutive_empty = 3
        
        # Ensure papers is properly initialized
        if papers is None:
            papers = []
        
        while len(papers) < limit:
            try:
                # Use v2 API for better results
                url = f"{self.semantic_scholar_v2_base}/paper/search"
                params = {
                    "query": query,
                    "limit": min(batch_size, limit - len(papers)),
                    "offset": offset,
                    "fields": "paperId,title,abstract,venue,year,authors,referenceCount,citationCount,openAccessPdf,publicationDate,publicationTypes,fieldsOfStudy,publicationVenue,externalIds"
                }
                
                # Prepare headers with API key if available
                headers = {}
                if self.semantic_scholar_api_key:
                    headers["x-api-key"] = self.semantic_scholar_api_key
                
                # Add retry mechanism with exponential backoff
                response = None
                for attempt in range(self.max_retries):
                    try:
                        response = requests.get(url, params=params, headers=headers, timeout=self.timeout)
                        
                        # Check if response is valid
                        if response and response.content:
                            consecutive_empty_responses = 0  # Reset counter on successful response
                            break
                        else:
                            consecutive_empty_responses += 1
                            logger.warning(f"⚠️ Empty response from Semantic Scholar API (attempt {attempt + 1}/{self.max_retries})")
                            
                            if consecutive_empty_responses >= max_consecutive_empty:
                                logger.error(f"❌ Too many consecutive empty responses ({consecutive_empty_responses}), stopping search")
                                return papers
                            
                            # Exponential backoff
                            wait_time = (2 ** attempt) * self.rate_limit_delay
                            logger.info(f"⏳ Waiting {wait_time:.1f}s before retry...")
                            time.sleep(wait_time)
                            
                    except requests.exceptions.RequestException as e:
                        logger.warning(f"⚠️ Request error (attempt {attempt + 1}/{self.max_retries}): {e}")
                        if attempt < self.max_retries - 1:
                            wait_time = (2 ** attempt) * self.rate_limit_delay
                            time.sleep(wait_time)
                        else:
                            logger.error(f"❌ All retry attempts failed for Semantic Scholar API")
                            return papers
                
                if not response or not response.content:
                    logger.warning("⚠️ Empty response from Semantic Scholar API after all retries")
                    break
                    
                if response.status_code == 200:
                    try:
                        # Check if response content is valid JSON
                        if not response.text or response.text.strip() == "":
                            logger.warning("⚠️ Empty response content from Semantic Scholar API")
                            break
                            
                        try:
                            data = response.json()
                        except json.JSONDecodeError as e:
                            logger.warning(f"⚠️ Invalid JSON response from Semantic Scholar API: {e}")
                            break
                            
                        if not data or not isinstance(data, dict):
                            logger.warning("⚠️ Invalid response format from Semantic Scholar API")
                            break
                            
                        batch_papers = data.get("data", [])
                        if not batch_papers or not isinstance(batch_papers, list):
                            logger.warning("⚠️ No valid papers data in Semantic Scholar API response")
                            break
                        
                        # Additional validation: filter out None values from batch_papers
                        valid_papers = [p for p in batch_papers if p is not None and isinstance(p, dict)]
                        if len(valid_papers) != len(batch_papers):
                            logger.warning(f"⚠️ Filtered out {len(batch_papers) - len(valid_papers)} invalid papers from batch")
                        
                        # Process each paper
                        for paper in valid_papers:
                            try:
                                processed_paper = self._process_semantic_scholar_paper(paper)
                                if processed_paper:
                                    papers.append(processed_paper)
                            except Exception as e:
                                logger.warning(f"⚠️ Skipping malformed paper data: {e}")
                                continue
                        
                        # Update offset safely
                        if valid_papers and isinstance(valid_papers, list):
                            offset += len(valid_papers)
                        
                        # Rate limiting - use Semantic Scholar specific delay
                        time.sleep(self.semantic_scholar_rate_limit)
                    except (KeyError, TypeError) as e:
                        logger.error(f"❌ Error parsing Semantic Scholar API response: {e}")
                        break
                    
                elif response.status_code == 429:
                    logger.warning("⚠️ Rate limit hit, waiting 60 seconds...")
                    logger.info("💡 Consider using fewer keywords or longer delays to avoid rate limiting")
                    time.sleep(60)
                    continue
                elif response.status_code == 403:
                    logger.error("❌ Semantic Scholar API access forbidden - check API key or quota")
                    break
                elif response.status_code == 500:
                    logger.warning("⚠️ Semantic Scholar API server error, retrying...")
                    time.sleep(5)
                    continue
                else:
                    logger.error(f"❌ Semantic Scholar API error: {response.status_code} - {response.text[:200]}")
                    break
                    
            except Exception as e:
                logger.error(f"❌ Error searching Semantic Scholar: {e}")
                break
        
        # Final validation of collected papers
        if papers is None:
            papers = []
        elif not isinstance(papers, list):
            logger.warning("⚠️ Papers variable is not a list, resetting to empty list")
            papers = []
        
        logger.info(f"✅ Found {len(papers)} papers from Semantic Scholar")
        # Ensure we return a valid list even if empty
        return papers
    
    def _process_semantic_scholar_paper(self, paper: Dict) -> Optional[Dict]:
        """
        Process a paper from Semantic Scholar API into standard format.
        
        Args:
            paper: Raw paper data from Semantic Scholar
            
        Returns:
            Processed paper dictionary or None if invalid
        """
        try:
            # Validate input
            if not paper or not isinstance(paper, dict):
                logger.warning("⚠️ Invalid paper data received from Semantic Scholar API")
                return None
            
            # Check for None values in critical fields
            if paper.get("title") is None:
                logger.warning("⚠️ Paper title is None, skipping")
                return None
                
            # Extract basic information
            title = paper.get("title", "")
            if not title or len(title) < 10:
                return None
            
            # Extract DOI
            doi = None
            external_ids = paper.get("externalIds")
            if external_ids and isinstance(external_ids, dict):
                doi = external_ids.get("DOI") or external_ids.get("doi")
            
            # Extract authors
            authors = []
            authors_data = paper.get("authors")
            if authors_data and isinstance(authors_data, list):
                for author in authors_data:
                    if author and isinstance(author, dict) and "name" in author:
                        author_name = author.get("name")
                        if author_name and isinstance(author_name, str):
                            authors.append(author_name)
            
            # Extract journal/venue
            venue = paper.get("venue", "")
            if not venue:
                publication_venue = paper.get("publicationVenue")
                if publication_venue and isinstance(publication_venue, dict):
                    venue = publication_venue.get("name", "")
            
            # Extract year
            year = paper.get("year")
            if not year:
                pub_date = paper.get("publicationDate")
                if pub_date and isinstance(pub_date, str) and len(pub_date) >= 4:
                    try:
                        year = int(pub_date[:4])
                    except (ValueError, TypeError):
                        year = None
            
            # Extract abstract
            abstract = paper.get("abstract", "")
            
            # Extract citation counts
            citation_count = paper.get("citationCount")
            if citation_count is None or not isinstance(citation_count, (int, float)):
                citation_count = 0
                
            reference_count = paper.get("referenceCount")
            if reference_count is None or not isinstance(reference_count, (int, float)):
                reference_count = 0
            
            # Extract fields of study
            fields_of_study = paper.get("fieldsOfStudy", [])
            if not isinstance(fields_of_study, list):
                fields_of_study = []
            
            # Extract publication types
            publication_types = paper.get("publicationTypes", [])
            if not isinstance(publication_types, list):
                publication_types = []
            
            # Check if it's a preprint
            is_preprint = False
            if publication_types:
                try:
                    is_preprint = any(pt and isinstance(pt, str) and pt.lower() in ["preprint", "workingpaper"] for pt in publication_types)
                except Exception:
                    is_preprint = False
            
            # Extract impact factor
            impact_factor = self._get_impact_factor(venue)
            
            # Create processed paper
            processed_paper = {
                "title": title,
                "doi": doi,
                "authors": authors,
                "journal": venue,
                "year": year,
                "abstract": abstract,
                "citation_count": str(citation_count) if citation_count else "0",
                "reference_count": str(reference_count) if reference_count else "0",
                "impact_factor": impact_factor,
                "fields_of_study": fields_of_study,
                "publication_types": publication_types,
                "is_preprint": is_preprint,
                "source": "semantic_scholar",
                "paper_id": paper.get("paperId") if paper.get("paperId") else None,
                "open_access_pdf": None,  # Will be set below if valid
                "publication_date": paper.get("publicationDate") if paper.get("publicationDate") else None,
                "raw_data": paper  # Keep original data for reference
            }
            
            # Safely extract open access PDF URL
            try:
                open_access_pdf = paper.get("openAccessPdf")
                if open_access_pdf and isinstance(open_access_pdf, dict):
                    pdf_url = open_access_pdf.get("url")
                    if pdf_url and isinstance(pdf_url, str):
                        processed_paper["open_access_pdf"] = pdf_url
            except Exception:
                processed_paper["open_access_pdf"] = None
            
            return processed_paper
            
        except Exception as e:
            logger.error(f"❌ Error processing Semantic Scholar paper: {e}")
            return None
    
    
    def _get_impact_factor(self, journal_name: str) -> str:
        """
        Get impact factor for a journal using the same comprehensive database as PubMed scraper.
        
        Args:
            journal_name: Name of the journal
            
        Returns:
            Impact factor as string or "not found"
        """
        if not journal_name or journal_name == "Unknown journal":
            return "not found"
        
        # Comprehensive impact factor mapping based on recent data (2023-2024)
        impact_factors = {
            # Top-tier journals
            'nature': 49.962,
            'science': 56.9,
            'cell': 66.85,
            'nature medicine': 87.241,
            'nature biotechnology': 68.164,
            'nature genetics': 41.307,
            'nature cell biology': 28.213,
            'nature immunology': 31.25,
            'nature reviews immunology': 108.555,
            'nature reviews molecular cell biology': 81.3,
            'nature reviews genetics': 42.7,
            'nature reviews cancer': 75.4,
            'nature reviews drug discovery': 120.1,
            
            # Immunology journals
            'immunity': 43.474,
            'journal of immunology': 5.422,
            'journal of experimental medicine': 17.579,
            'nature immunology': 31.25,
            'immunological reviews': 13.0,
            'trends in immunology': 13.1,
            'european journal of immunology': 5.4,
            'journal of allergy and clinical immunology': 14.2,
            
            # General science journals
            'proceedings of the national academy of sciences': 12.779,
            'pnas': 12.779,
            'plos one': 3.752,
            'plos biology': 9.593,
            'plos genetics': 6.02,
            'plos computational biology': 4.7,
            'elife': 8.713,
            
            # Bioinformatics and computational biology
            'bioinformatics': 6.937,
            'nucleic acids research': 19.16,
            'genome research': 11.093,
            'genome biology': 17.906,
            'bmc genomics': 4.317,
            'bmc bioinformatics': 3.169,
            'briefings in bioinformatics': 13.9,
            'bioinformatics and biology insights': 2.1,
            
            # Cell biology journals
            'cell reports': 9.995,
            'molecular cell': 19.328,
            'developmental cell': 13.417,
            'current biology': 10.834,
            'cell metabolism': 29.0,
            'cell stem cell': 25.3,
            'cancer cell': 50.3,
            'molecular biology of the cell': 3.9,
            
            # Nature family journals
            'nature communications': 17.694,
            'nature methods': 47.99,
            'nature neuroscience': 25.0,
            'nature structural & molecular biology': 15.8,
            'nature chemical biology': 15.0,
            'nature materials': 41.2,
            'nature physics': 20.5,
            'nature chemistry': 24.4,
            
            # Neuroscience journals
            'neuron': 16.2,
            'journal of neuroscience': 6.7,
            'nature neuroscience': 25.0,
            'trends in neurosciences': 16.2,
            'cerebral cortex': 4.9,
            'neuroimage': 7.4,
            
            # Medical journals
            'the lancet': 202.731,
            'new england journal of medicine': 176.079,
            'jama': 157.335,
            'bmj': 105.7,
            'nature medicine': 87.241,
            'cell metabolism': 29.0,
            'diabetes': 8.0,
            'circulation': 37.8,
            
            # Preprint servers (no impact factor)
            'biorxiv': 0.0,
            'medrxiv': 0.0,
            'arxiv': 0.0,
            'chemrxiv': 0.0,
            'bioarxiv': 0.0,
            
            # Biochemistry journals
            'journal of biological chemistry': 5.5,
            'biochemistry': 3.2,
            'protein science': 6.3,
            'journal of molecular biology': 5.0,
            'structure': 4.2,
            
            # Genetics journals
            'genetics': 4.4,
            'genome research': 11.093,
            'genome biology': 17.906,
            'human molecular genetics': 5.1,
            'american journal of human genetics': 11.0,
            
            # Cancer journals
            'cancer cell': 50.3,
            'cancer research': 13.3,
            'journal of clinical oncology': 50.7,
            'nature cancer': 23.0,
            'cancer discovery': 28.2,
            
            # Microbiology journals
            'cell host & microbe': 30.3,
            'nature microbiology': 20.5,
            'journal of bacteriology': 3.2,
            'applied and environmental microbiology': 4.4,
            
            # Plant biology journals
            'plant cell': 12.1,
            'plant journal': 7.0,
            'plant physiology': 8.0,
            'nature plants': 15.8,
            
            # Other specialized journals
            'journal of proteome research': 4.4,
            'proteomics': 4.0,
            'mass spectrometry reviews': 8.0,
            'analytical chemistry': 8.0,
            'journal of chromatography a': 4.1,
        }
        
        journal_lower = journal_name.lower().strip()
        
        # Direct match
        if journal_lower in impact_factors:
            return str(impact_factors[journal_lower])
        
        # Partial match - check if any key is contained in the journal name
        for key, impact in impact_factors.items():
            if key in journal_lower or journal_lower in key:
                return str(impact)
        
        # Fuzzy matching for common variations
        journal_variations = {
            'nature': ['nat ', 'nature '],
            'science': ['science ', 'sci '],
            'cell': ['cell ', 'cell:'],
            'journal': ['j ', 'journal of', 'j.'],
            'proceedings': ['proc ', 'proceedings of'],
            'plos': ['plos ', 'public library of science'],
            'bmc': ['bmc ', 'biomed central'],
            'pnas': ['proc natl acad sci', 'proceedings of the national academy'],
        }
        
        for base_name, variations in journal_variations.items():
            for variation in variations:
                if variation in journal_lower:
                    if base_name in impact_factors:
                        return str(impact_factors[base_name])
        
        # Default for unknown journals - estimate based on journal name patterns
        if any(word in journal_lower for word in ['nature', 'science', 'cell']):
            return "15.0"  # High-impact estimate
        elif any(word in journal_lower for word in ['journal', 'proceedings', 'plos']):
            return "5.0"   # Medium-impact estimate
        elif any(word in journal_lower for word in ['bmc', 'frontiers', 'molecules']):
            return "3.0"   # Lower-impact estimate
        else:
            return "not found"
    
    def search_papers(self, max_papers: int = None, search_terms: List[str] = None) -> List[Dict]:
        """
        Search for papers using specific keywords.
        Fetches as many papers as possible from each source.
        
        Args:
            max_papers: Optional maximum limit (None = no limit)
            search_terms: Optional list of search terms (uses default if None)
            
        Returns:
            List of unique papers
        """
        if search_terms is None:
            search_terms = self.default_search_terms
            
        if max_papers is None:
            logger.info("🔍 Starting paper search with specific keywords (no limit - fetching all available papers)")
        else:
            logger.info(f"🔍 Starting paper search with specific keywords (target: {max_papers} papers)")
        
        all_papers = []
        seen_titles = set()
        
        # Use custom keywords if set, otherwise try to load from config file
        if hasattr(self, 'search_keywords') and self.search_keywords:
            search_keywords = self.search_keywords
            logger.info(f"📋 Using custom search keywords: {', '.join(search_keywords)}")
        else:
            # Try to load from configuration file
            search_keywords = self._load_keywords_from_config()
            if search_keywords:
                logger.info(f"📋 Using keywords from config: {', '.join(search_keywords)}")
            else:
                # Fall back to default UBR5 keywords
                search_keywords = [
                    "ubr5",
                    "UBR5", 
                    "ubr-5",
                    "UBR-5"
                ]
                logger.info(f"📋 Using default UBR5 keywords: {', '.join(search_keywords)}")
        
        # Validate search keywords
        if not search_keywords or not isinstance(search_keywords, list) or len(search_keywords) == 0:
            logger.error("❌ Invalid or empty search keywords")
            return []
        
        # Use tqdm for search keywords progress
        with tqdm(total=len(search_keywords), desc="Search keywords", unit="keyword") as keyword_pbar:
            for keyword in search_keywords:
                try:
                    # Check if we've hit the limit (if one was set)
                    if max_papers is not None and len(all_papers) >= max_papers:
                        break
                        
                    keyword_pbar.set_description(f"Searching: {keyword}")
                    logger.info(f"🔍 Searching with keyword: {keyword}")
                    
                    # Search Semantic Scholar - fetch maximum available papers with timeout
                    try:
                        logger.info(f"🔍 Searching Semantic Scholar for keyword: {keyword}")
                        # Add timeout to prevent hanging
                        import threading
                        result = [None]
                        exception = [None]
                        
                        def search_with_timeout():
                            try:
                                result[0] = self.search_semantic_scholar(keyword, limit=1000)
                            except Exception as e:
                                exception[0] = e
                        
                        search_thread = threading.Thread(target=search_with_timeout)
                        search_thread.daemon = True
                        search_thread.start()
                        search_thread.join(timeout=120)  # 2 minute timeout
                        
                        if search_thread.is_alive():
                            logger.warning(f"⚠️ Search for '{keyword}' timed out after 2 minutes, skipping")
                            continue
                        
                        if exception[0]:
                            raise exception[0]
                        
                        semantic_papers = result[0]
                        if semantic_papers and isinstance(semantic_papers, list) and len(semantic_papers) > 0:
                            logger.info(f"📊 Found {len(semantic_papers)} papers for keyword '{keyword}'")
                            added_count = 0
                            for paper in semantic_papers:
                                if paper and isinstance(paper, dict) and self._is_unique_paper(paper, seen_titles):
                                    all_papers.append(paper)
                                    added_count += 1
                                    # Safe title access
                                    title = paper.get("title", "")
                                    if title:
                                        seen_titles.add(title.lower())
                                    # Check limit only if one was set
                                    if max_papers is not None and len(all_papers) >= max_papers:
                                        break
                            logger.info(f"✅ Added {added_count} new unique papers from keyword '{keyword}' (total: {len(all_papers)})")
                        else:
                            logger.warning(f"⚠️ No valid papers returned from Semantic Scholar for keyword: {keyword}")
                    except Exception as e:
                        logger.error(f"❌ Error searching Semantic Scholar for keyword '{keyword}': {e}")
                        continue
                    
                    # Update progress
                    keyword_pbar.update(1)
                    keyword_pbar.set_postfix({"papers_found": len(all_papers)})
                    
                    # Rate limiting between keywords - use conservative delay
                    time.sleep(self.keyword_delay)
                    
                except Exception as e:
                    logger.error(f"❌ Error processing keyword '{keyword}': {e}")
                    continue
        
        if not all_papers:
            logger.warning("⚠️ No papers were collected from any source")
            return []
            
        logger.info(f"✅ Collected {len(all_papers)} unique UBR5-related papers")
        logger.info(f"📊 Search Summary:")
        logger.info(f"   - Keywords searched: {len(search_keywords)}")
        logger.info(f"   - Total unique papers: {len(all_papers)}")
        logger.info(f"   - Source: Semantic Scholar only")
        # Ensure we return a valid list
        return all_papers
    
    def _is_unique_paper(self, paper: Dict, seen_titles: set) -> bool:
        """
        Check if a paper is unique based on title similarity.
        
        Args:
            paper: Paper dictionary
            seen_titles: Set of already seen titles
            
        Returns:
            True if paper is unique, False otherwise
        """
        # Validate input
        if not paper or not isinstance(paper, dict):
            return False
            
        title = paper.get("title", "").lower().strip()
        if not title:
            return False
        
        # Check for exact match
        if title in seen_titles:
            return False
        
        # Check for similar titles (fuzzy matching)
        try:
            for seen_title in seen_titles:
                if seen_title and isinstance(seen_title, str):
                    similarity = self._calculate_title_similarity(title, seen_title)
                    if similarity > 0.8:  # 80% similarity threshold
                        return False
        except Exception as e:
            logger.warning(f"⚠️ Error in title similarity calculation: {e}")
            return False
        
        return True
    
    def _calculate_title_similarity(self, title1: str, title2: str) -> float:
        """
        Calculate similarity between two titles using simple word overlap.
        
        Args:
            title1: First title
            title2: Second title
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Validate inputs
            if not title1 or not title2 or not isinstance(title1, str) or not isinstance(title2, str):
                return 0.0
                
            words1 = set(title1.split())
            words2 = set(title2.split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            if len(union) == 0:
                return 0.0
                
            return len(intersection) / len(union)
        except Exception as e:
            logger.warning(f"⚠️ Error calculating title similarity: {e}")
            return 0.0
    
    def generate_embeddings(self, papers: List[Dict], api_key: str) -> List[Dict]:
        """
        Generate embeddings for papers using Google's text-embedding-004 model.
        
        Args:
            papers: List of paper dictionaries
            api_key: Google API key
            
        Returns:
            List of papers with embeddings
        """
        logger.info(f"🔍 Generating embeddings for {len(papers)} papers")
        
        papers_with_embeddings = []
        current_batch = []
        batch_num = 0
        
        for i, paper in enumerate(tqdm(papers, desc="Generating embeddings")):
            try:
                # Create text for embedding
                text_for_embedding = self._create_embedding_text(paper)
                
                # Generate embedding
                embedding = self._get_google_embedding(text_for_embedding, api_key)
                
                if embedding:
                    paper["embedding"] = embedding
                    paper["embedding_text"] = text_for_embedding
                    papers_with_embeddings.append(paper)
                    
                    # Add to batch and save when full
                    current_batch, batch_num, batch_file = self.save_embedding_realtime(
                        paper, current_batch, batch_num, batch_size=10
                    )
                    
                    # Print progress every 10 papers
                    if len(papers_with_embeddings) % 10 == 0:
                        logger.info(f"💾 Processed {len(papers_with_embeddings)} papers so far...")
                
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.error(f"❌ Error generating embedding for paper {i}: {e}")
                continue
        
        # Save any remaining papers in the current batch
        if current_batch:
            batch_file = self.save_batch_semantic_scholar(current_batch, batch_num)
            logger.info(f"💾 Saved final Semantic Scholar batch {batch_num:04d} with {len(current_batch)} papers")
        
        logger.info(f"✅ Generated embeddings for {len(papers_with_embeddings)} papers")
        return papers_with_embeddings
    
    def _create_embedding_text(self, paper: Dict) -> str:
        """
        Create text for embedding generation.
        
        Args:
            paper: Paper dictionary
            
        Returns:
            Text string for embedding
        """
        text_parts = []
        
        # Title
        if paper.get("title"):
            text_parts.append(f"Title: {paper['title']}")
        
        # Abstract
        if paper.get("abstract"):
            text_parts.append(f"Abstract: {paper['abstract']}")
        
        # Authors
        if paper.get("authors"):
            authors_text = "; ".join(paper["authors"])
            text_parts.append(f"Authors: {authors_text}")
        
        # Journal
        if paper.get("journal"):
            text_parts.append(f"Journal: {paper['journal']}")
        
        # Year
        if paper.get("year"):
            text_parts.append(f"Year: {paper['year']}")
        
        # Fields of study
        if paper.get("fields_of_study"):
            fields_text = "; ".join(paper["fields_of_study"])
            text_parts.append(f"Fields: {fields_text}")
        
        return " | ".join(text_parts)
    
    def _get_google_embedding(self, text: str, api_key: str) -> Optional[List[float]]:
        """
        Get embedding from Google's text-embedding-004 model.
        
        Args:
            text: Text to embed
            api_key: Google API key
            
        Returns:
            Embedding vector or None if failed
        """
        try:
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"
            
            headers = {
                "Content-Type": "application/json"
            }
            
            data = {
                "content": {
                    "parts": [{"text": text}]
                }
            }
            
            params = {
                "key": api_key
            }
            
            response = requests.post(url, headers=headers, json=data, params=params, timeout=self.timeout)
            
            if response.status_code == 200:
                result = response.json()
                embedding = result.get("embedding", {}).get("values", [])
                return embedding
            else:
                logger.error(f"❌ Google API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error getting Google embedding: {e}")
            return None
    
    def save_embeddings(self, papers_with_embeddings: List[Dict], source: str = "semantic_scholar"):
        """
        Save embeddings to the data/embeddings/xrvix_embeddings folder.
        
        Args:
            papers_with_embeddings: List of papers with embeddings
            source: Source identifier for the embeddings
        """
        logger.info(f"💾 Saving {len(papers_with_embeddings)} embeddings to {self.embeddings_dir}")
        
        # Create source directory
        source_dir = os.path.join(self.embeddings_dir, source)
        os.makedirs(source_dir, exist_ok=True)
        
        # Save individual paper files
        for i, paper in enumerate(papers_with_embeddings):
            try:
                # Create filename based on title and DOI
                filename = self._create_filename(paper)
                filepath = os.path.join(source_dir, filename)
                
                # Save paper data
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(paper, f, indent=2, ensure_ascii=False)
                
            except Exception as e:
                logger.error(f"❌ Error saving paper {i}: {e}")
                continue
        
        # Save metadata file
        metadata_file = os.path.join(source_dir, "metadata.json")
        metadata = {
            "source": source,
            "total_papers": len(papers_with_embeddings),
            "created_at": datetime.now().isoformat(),
            "embedding_model": "text-embedding-004",
            "papers": [
                {
                    "title": paper.get("title", ""),
                    "doi": paper.get("doi", ""),
                    "source": paper.get("source", ""),
                    "year": paper.get("year", ""),
                    "filename": self._create_filename(paper)
                }
                for paper in papers_with_embeddings
            ]
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def save_batch_semantic_scholar(self, batch_data, batch_num):
        """Save a batch of Semantic Scholar embeddings to a file"""
        # Ensure directory exists
        os.makedirs(self.embeddings_dir, exist_ok=True)
        
        batch_file = os.path.join(self.embeddings_dir, f"batch_{batch_num:04d}.json")
        
        batch_content = {
            "source": "semantic_scholar",
            "batch_num": batch_num,
            "timestamp": datetime.now().isoformat(),
            "papers": batch_data,
            "stats": {
                "total_papers": len(batch_data)
            }
        }
        
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(batch_content, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Saved Semantic Scholar batch {batch_num:04d} with {len(batch_data)} papers")
        return batch_file

    def save_embedding_realtime(self, paper: Dict, current_batch: List, batch_num: int, batch_size: int = 10):
        """
        Add a paper to the current batch and save when full.
        
        Args:
            paper: Paper dictionary with embedding
            current_batch: Current batch list
            batch_num: Current batch number
            batch_size: Size of batch before saving
        """
        try:
            # Add paper to current batch
            current_batch.append(paper)
            
            # Save batch if it reaches the size limit
            if len(current_batch) >= batch_size:
                batch_file = self.save_batch_semantic_scholar(current_batch, batch_num)
                # Reset batch
                current_batch = []
                batch_num += 1
                return current_batch, batch_num, batch_file
            else:
                return current_batch, batch_num, None
            
        except Exception as e:
            logger.error(f"❌ Error saving paper in real-time: {e}")
            return current_batch, batch_num, None

    def _update_metadata_realtime(self, source_dir: str, paper: Dict, source: str):
        """Update metadata file in real-time"""
        metadata_file = os.path.join(source_dir, "metadata.json")
        
        # Load existing metadata or create new
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                metadata = {
                    "source": source,
                    "total_papers": 0,
                    "created_at": datetime.now().isoformat(),
                    "embedding_model": "text-embedding-004",
                    "papers": []
                }
        else:
            metadata = {
                "source": source,
                "total_papers": 0,
                "created_at": datetime.now().isoformat(),
                "embedding_model": "text-embedding-004",
                "papers": []
            }
        
        # Add new paper to metadata
        paper_metadata = {
            "title": paper.get("title", ""),
            "doi": paper.get("doi", ""),
            "source": paper.get("source", ""),
            "year": paper.get("year", ""),
            "filename": self._create_filename(paper)
        }
        
        # Check if paper already exists (avoid duplicates)
        existing_papers = [p for p in metadata["papers"] if p.get("doi") == paper_metadata.get("doi")]
        if not existing_papers:
            metadata["papers"].append(paper_metadata)
            metadata["total_papers"] = len(metadata["papers"])
        
        # Save updated metadata
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Saved embeddings to {source_dir}")
    
    def _create_filename(self, paper: Dict) -> str:
        """
        Create a filename for a paper.
        
        Args:
            paper: Paper dictionary
            
        Returns:
            Filename string
        """
        title = paper.get("title", "untitled")
        doi = paper.get("doi", "")
        
        # Clean title for filename - remove all non-alphanumeric characters except hyphens and underscores
        clean_title = re.sub(r'[^\w\s-]', '', title)
        clean_title = re.sub(r'[-\s]+', '-', clean_title)
        clean_title = clean_title[:100]  # Limit length
        
        if doi:
            # Clean DOI for filename - remove all invalid characters
            clean_doi = re.sub(r'[<>:"/\\|?*]', '_', doi)
            clean_doi = re.sub(r'[^\w\-_.]', '_', clean_doi)
            clean_doi = clean_doi[:50]  # Limit DOI length
            filename = f"{clean_title}_{clean_doi}.json"
        else:
            filename = f"{clean_title}_{abs(hash(title))}.json"
        
        return filename
    
    def integrate_with_chromadb(self, papers_with_embeddings: List[Dict]):
        """
        Integrate embeddings into ChromaDB database.
        
        Args:
            papers_with_embeddings: List of papers with embeddings
        """
        logger.info(f"🔗 Integrating {len(papers_with_embeddings)} embeddings into ChromaDB")
        
        try:
            # Check if ChromaDB manager is available
            if self.chromadb_manager is None:
                logger.error("❌ ChromaDB manager not initialized")
                return
            
            # Ensure ChromaDB collection is properly initialized
            if not self.chromadb_manager.is_initialized or self.chromadb_manager.collection is None:
                logger.info("🔧 Initializing ChromaDB collection...")
                if not self.chromadb_manager.create_collection():
                    logger.error("❌ Failed to initialize ChromaDB collection")
                    return
            
            # Prepare data for ChromaDB with progress bar
            documents = []
            metadatas = []
            ids = []
            embeddings = []
            
            logger.info("🔧 Preparing metadata for ChromaDB integration...")
            with tqdm(total=len(papers_with_embeddings), desc="Preparing metadata", unit="paper") as metadata_pbar:
                for i, paper in enumerate(papers_with_embeddings):
                    # Validate paper has embedding
                    if not paper.get("embedding"):
                        logger.warning(f"⚠️ Paper {i} missing embedding, skipping")
                        continue
                    
                    # Create document text
                    doc_text = self._create_embedding_text(paper)
                    
                    # Create metadata with proper string conversion for ChromaDB
                    def safe_join_list(data, separator="; "):
                        """Safely join list data into string for ChromaDB metadata."""
                        if data is None:
                            return ""
                        elif isinstance(data, list):
                            return separator.join(str(item) for item in data if item is not None)
                        elif isinstance(data, str):
                            return data
                        else:
                            return str(data) if data is not None else ""
                    
                    metadata = {
                        "title": str(paper.get("title", "") or ""),
                        "doi": str(paper.get("doi", "") or ""),
                        "authors": safe_join_list(paper.get("authors", [])),
                        "journal": str(paper.get("journal", "") or ""),
                        "year": str(paper.get("year", "") or ""),
                        "citation_count": str(paper.get("citation_count", "0") or "0"),
                        "source": str(paper.get("source", "") or ""),
                        "source_name": "semantic_scholar",  # Add source_name for ChromaDB manager
                        "is_preprint": str(paper.get("is_preprint", False) or False),
                        "publication_date": str(paper.get("publication_date", "") or ""),
                        "fields_of_study": safe_join_list(paper.get("fields_of_study", [])),
                        "publication_types": safe_join_list(paper.get("publication_types", [])),
                        "abstract": str(paper.get("abstract", "") or "")[:1000],  # Limit length
                        "added_at": str(datetime.now().isoformat())
                    }
                    
                    # Create unique ID
                    paper_id = f"ubr5_api_{i}_{abs(hash(paper.get('title', '')))}"
                    
                    documents.append(doc_text)
                    metadatas.append(metadata)
                    ids.append(paper_id)
                    embeddings.append(paper["embedding"])
                    
                    metadata_pbar.update(1)
                    metadata_pbar.set_postfix({"processed": i+1, "total": len(papers_with_embeddings)})
            
            # Validate we have data to add
            if not documents or not embeddings:
                logger.error("❌ No valid documents or embeddings to add to ChromaDB")
                return
            
            # Add to ChromaDB with embeddings using the bulk add method for proper metadata validation
            success = self.chromadb_manager._bulk_add_to_collection(
                embeddings=embeddings,
                chunks=documents,
                metadata=metadatas,
                ids=ids
            )
            
            if not success:
                logger.error("❌ Failed to add embeddings to ChromaDB")
                return
            
            logger.info(f"✅ Successfully integrated {len(documents)} embeddings into ChromaDB")
            
        except Exception as e:
            logger.error(f"❌ Error integrating with ChromaDB: {e}")
            import traceback
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
    
    def run_complete_scraping(self, max_papers: int = None):
        """
        Run the complete UBR5 scraping pipeline.
        Fetches as many papers as possible from all sources.
        
        Args:
            max_papers: Optional maximum number of papers to collect (None = no limit)
        """
        if max_papers is None:
            logger.info("🚀 Starting complete UBR5 scraping pipeline (no paper limit - fetching all available)")
        else:
            logger.info(f"🚀 Starting complete UBR5 scraping pipeline (target: {max_papers} papers)")
        
        try:
            # Check if we have the required Google API key for embeddings
            google_api_key = self.api_keys.get("GOOGLE_API_KEY")
            if not google_api_key:
                logger.error("❌ GOOGLE_API_KEY not found in keys.json. Cannot generate embeddings.")
                logger.info("💡 Please add your Google API key to keys.json to enable embedding generation.")
                return
            
            # Step 1: Search for papers
            papers = self.search_papers(max_papers=max_papers)
            
            if not papers:
                logger.warning("⚠️ No papers found, exiting")
                return
            
            # Step 2: Generate embeddings
            papers_with_embeddings = self.generate_embeddings(papers, google_api_key)
            
            if not papers_with_embeddings:
                logger.warning("⚠️ No embeddings generated, exiting")
                return
            
            # Step 3: Save embeddings
            self.save_embeddings(papers_with_embeddings, source="semantic_scholar")
            
            # Step 4: Integrate with ChromaDB
            self.integrate_with_chromadb(papers_with_embeddings)
            
            logger.info("🎉 UBR5 scraping pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Error in scraping pipeline: {e}")
            raise

def main():
    """Main function to run the UBR5 API scraper."""
    print("🔍 UBR5 API Scraper - Comprehensive Paper Collection")
    print("=" * 60)
    
    # Initialize scraper
    scraper = UBR5APIScraper()
    
    # Check API key availability
    availability = scraper.check_api_key_availability()
    
    # Test API connectivity
    connectivity = scraper.test_api_connectivity()
    
    # Show warnings if critical keys are missing
    if not availability["google_api"]:
        print("\n⚠️  WARNING: GOOGLE_API_KEY not found!")
        print("   This key is required for generating embeddings.")
        print("   Please add it to your keys.json file.")
        print("   Example keys.json structure:")
        print("   {")
        print('     "GOOGLE_API_KEY": "your_google_api_key_here"')
        print("   }")
        print()
    
    # Get user input
    print("\n📊 Paper Collection Options:")
    print("   1. Collect unlimited papers (recommended)")
    print("   2. Set a specific paper limit")
    print("   3. Use default limit (500 papers)")
    
    try:
        choice = input("\nEnter your choice (1-3, default 1): ").strip() or "1"
        
        if choice == "1":
            max_papers = None
            print("✅ Will collect ALL available UBR5 papers (no limit)")
        elif choice == "2":
            max_papers = int(input("Enter maximum number of papers to collect: "))
            print(f"✅ Will collect up to {max_papers} papers")
        elif choice == "3":
            max_papers = 500
            print("✅ Using default limit: 500 papers")
        else:
            max_papers = None
            print("✅ Invalid choice, defaulting to unlimited collection")
            
    except ValueError:
        max_papers = None
        print("✅ Invalid input, defaulting to unlimited collection")
    
    # Check if we can proceed
    if not availability["google_api"]:
        print("\n❌ Cannot proceed without Google API key for embeddings.")
        print("   Please add your GOOGLE_API_KEY to keys.json and try again.")
        return
    
    if max_papers is None:
        print("\n🚀 Starting unlimited UBR5 paper collection...")
        print("   This will fetch ALL available papers from Semantic Scholar")
    else:
        print(f"\n🚀 Starting UBR5 paper collection (target: {max_papers} papers)...")
    
    # Run scraping
    scraper.run_complete_scraping(max_papers=max_papers)

if __name__ == "__main__":
    main()
