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
from processing_config import get_config, print_config_info
from chromadb_manager import ChromaDBManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ubr5_api_scraping.log')
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings for external libraries
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Could not find paper.*")

class UBR5APIScraper:
    """
    Comprehensive UBR5 paper scraper using Scholarly and Semantic Scholar APIs.
    Collects the same data and metadata as xrvix and PubMed scrapers.
    """
    
    def __init__(self, api_keys: Dict[str, str] = None):
        """
        Initialize the UBR5 API scraper.
        
        Args:
            api_keys: Dictionary containing API keys for different services
        """
        self.api_keys = api_keys or {}
        self.embeddings_dir = "xrvix_embeddings"
        self.papers_data = []
        self.processed_dois = set()
        
        # Load API keys from keys.json if not provided
        if not self.api_keys:
            self._load_api_keys()
        
        # API endpoints and rate limiting
        self.semantic_scholar_base = "https://api.semanticscholar.org/v1"
        self.semantic_scholar_v2_base = "https://api.semanticscholar.org/graph/v1"
        self.scholarly_base = "https://scholar.google.com"
        
        # Rate limiting configuration
        self.rate_limit_delay = 0.1  # 100ms between requests
        self.max_retries = 3
        self.timeout = 30
        
        # UBR5 search terms
        self.ubr5_search_terms = [
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
        self.chromadb_manager = ChromaDBManager()
        self.chromadb_manager.create_collection()
        
        logger.info("🔍 UBR5 API Scraper initialized successfully")
    
    def check_api_key_availability(self) -> Dict[str, bool]:
        """
        Check which API keys are available.
        
        Returns:
            Dictionary indicating which API keys are available
        """
        availability = {
            "google_api": bool(self.api_keys.get("GOOGLE_API_KEY")),
            "scholar_api": bool(self.api_keys.get("SCHOLAR_API_KEY") or self.api_keys.get("GOOGLE_SCHOLAR_API_KEY")),
            "semantic_scholar": True,  # No API key required for basic usage
        }
        
        logger.info("🔑 API Key Availability:")
        logger.info(f"   Google API (embeddings): {'✅' if availability['google_api'] else '❌'}")
        logger.info(f"   Scholar API: {'✅' if availability['scholar_api'] else '❌'}")
        logger.info(f"   Semantic Scholar: {'✅' if availability['semantic_scholar'] else '❌'}")
        
        return availability
    
    def _load_api_keys(self):
        """Load API keys from keys.json file."""
        try:
            with open("keys.json", 'r') as f:
                keys_data = json.load(f)
                self.api_keys.update(keys_data)
                logger.info("✅ Loaded API keys from keys.json")
                
                # Log available API keys (without exposing the actual keys)
                available_keys = list(keys_data.keys())
                logger.info(f"📋 Available API keys: {', '.join(available_keys)}")
                
        except FileNotFoundError:
            logger.warning("⚠️ keys.json not found, using default configuration")
        except json.JSONDecodeError:
            logger.error("❌ Invalid JSON in keys.json")
        except Exception as e:
            logger.error(f"❌ Error loading API keys: {e}")
    
    def search_semantic_scholar(self, query: str, limit: int = 100) -> List[Dict]:
        """
        Search for papers using Semantic Scholar API.
        
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
                
                response = requests.get(url, params=params, timeout=self.timeout)
                
                # Check if response is valid
                if not response or not response.content:
                    logger.warning("⚠️ Empty response from Semantic Scholar API")
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
                        
                        # Rate limiting
                        time.sleep(self.rate_limit_delay)
                    except (KeyError, TypeError) as e:
                        logger.error(f"❌ Error parsing Semantic Scholar API response: {e}")
                        break
                    
                elif response.status_code == 429:
                    logger.warning("⚠️ Rate limit hit, waiting 60 seconds...")
                    time.sleep(60)
                    continue
                else:
                    logger.error(f"❌ Semantic Scholar API error: {response.status_code}")
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
    
    def search_scholarly(self, query: str, limit: int = 100) -> List[Dict]:
        """
        Search for papers using Google Scholar (via scholarly library).
        First tries to use API key if available, then falls back to non-API method.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of paper dictionaries
        """
        logger.info(f"🔍 Searching Google Scholar for: {query}")
        
        # Check if we have a Scholar API key
        scholar_api_key = self.api_keys.get("SCHOLAR_API_KEY") or self.api_keys.get("GOOGLE_SCHOLAR_API_KEY")
        
        if scholar_api_key:
            logger.info("🔑 Using Scholar API key for enhanced search")
            return self._search_scholarly_with_api(query, limit, scholar_api_key)
        else:
            logger.info("🔍 No Scholar API key found, using standard scholarly library")
            return self._search_scholarly_standard(query, limit)
    
    def _search_scholarly_with_api(self, query: str, limit: int, api_key: str) -> List[Dict]:
        """
        Search Google Scholar using API key for enhanced results.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            api_key: Scholar API key
            
        Returns:
            List of paper dictionaries
        """
        try:
            # Try to use the official Google Scholar API if available
            # Note: This is a placeholder for when Google Scholar API becomes available
            logger.info("🔑 Scholar API method not yet implemented, falling back to standard method")
            return self._search_scholarly_standard(query, limit)
            
        except Exception as e:
            logger.error(f"❌ Error with Scholar API method: {e}")
            logger.info("🔄 Falling back to standard scholarly method")
            return self._search_scholarly_standard(query, limit)
    
    def _search_scholarly_standard(self, query: str, limit: int) -> List[Dict]:
        """
        Search Google Scholar using the standard scholarly library (non-API method).
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of paper dictionaries
        """
        try:
            # Import scholarly here to avoid import issues
            from scholarly import scholarly
            
            papers = []
            search_query = scholarly.search_pubs(query)
            
            for i, pub in enumerate(search_query):
                if i >= limit:
                    break
                
                try:
                    processed_paper = self._process_scholarly_paper(pub)
                    if processed_paper:
                        papers.append(processed_paper)
                    
                    # Rate limiting
                    time.sleep(self.rate_limit_delay)
                    
                except Exception as e:
                    logger.error(f"❌ Error processing scholarly paper {i}: {e}")
                    continue
            
            logger.info(f"✅ Found {len(papers)} papers from Google Scholar (standard method)")
            return papers
            
        except ImportError:
            logger.warning("⚠️ scholarly library not available, skipping Google Scholar search")
            return []
        except Exception as e:
            logger.error(f"❌ Error searching Google Scholar: {e}")
            return []
    
    def _process_scholarly_paper(self, pub) -> Optional[Dict]:
        """
        Process a paper from Google Scholar into standard format.
        
        Args:
            pub: Raw paper data from scholarly
            
        Returns:
            Processed paper dictionary or None if invalid
        """
        try:
            # Validate input
            if not pub or not isinstance(pub, dict):
                logger.warning("⚠️ Invalid paper data received from Google Scholar")
                return None
                
            # Extract basic information
            title = pub.get("title", "")
            if not title or len(title) < 10:
                return None
            
            # Extract authors
            authors = []
            if "author" in pub:
                for author in pub["author"]:
                    if "name" in author:
                        authors.append(author["name"])
            
            # Extract journal/venue
            venue = pub.get("venue", "")
            
            # Extract year
            year = pub.get("year")
            
            # Extract abstract
            abstract = pub.get("abstract", "")
            
            # Extract citation count
            citation_count = pub.get("num_citations", 0)
            
            # Extract URL
            url = pub.get("url", "")
            
            # Extract publication type
            pub_type = pub.get("pub_type", "")
            
            # Check if it's a preprint
            is_preprint = "preprint" in pub_type.lower() or "working" in pub_type.lower()
            
            # Create processed paper
            processed_paper = {
                "title": title,
                "doi": None,  # Google Scholar doesn't provide DOI
                "authors": authors,
                "journal": venue,
                "year": year,
                "abstract": abstract,
                "citation_count": str(citation_count) if citation_count else "0",
                "reference_count": "0",  # Google Scholar doesn't provide this
                "fields_of_study": [],
                "publication_types": [pub_type] if pub_type else [],
                "is_preprint": is_preprint,
                "source": "google_scholar",
                "paper_id": pub.get("scholar_id"),
                "open_access_pdf": url if url else None,
                "publication_date": None,
                "raw_data": pub  # Keep original data for reference
            }
            
            return processed_paper
            
        except Exception as e:
            logger.error(f"❌ Error processing scholarly paper: {e}")
            return None
    
    def search_ubr5_papers(self, max_papers: int = None) -> List[Dict]:
        """
        Search for UBR5-related papers using specific keywords.
        Fetches as many papers as possible from each source.
        
        Args:
            max_papers: Optional maximum limit (None = no limit)
            
        Returns:
            List of unique papers
        """
        if max_papers is None:
            logger.info("🔍 Starting UBR5 paper search with specific keywords (no limit - fetching all available papers)")
        else:
            logger.info(f"🔍 Starting UBR5 paper search with specific keywords (target: {max_papers} papers)")
        
        all_papers = []
        seen_titles = set()
        
        # Use specific UBR5 keywords
        search_keywords = [
            "ubr5",
            "UBR5", 
            "ubr-5",
            "UBR-5"
        ]
        
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
                    
                    # Search Semantic Scholar - fetch maximum available papers
                    try:
                        semantic_papers = self.search_semantic_scholar(keyword, limit=1000)  # Increased limit
                        if semantic_papers and isinstance(semantic_papers, list) and len(semantic_papers) > 0:
                            for paper in semantic_papers:
                                if paper and isinstance(paper, dict) and self._is_unique_paper(paper, seen_titles):
                                    all_papers.append(paper)
                                    # Safe title access
                                    title = paper.get("title", "")
                                    if title:
                                        seen_titles.add(title.lower())
                                    # Check limit only if one was set
                                    if max_papers is not None and len(all_papers) >= max_papers:
                                        break
                        else:
                            logger.warning(f"⚠️ No valid papers returned from Semantic Scholar for keyword: {keyword}")
                    except Exception as e:
                        logger.error(f"❌ Error searching Semantic Scholar for keyword '{keyword}': {e}")
                        continue
                    
                    # Search Google Scholar - fetch maximum available papers
                    if max_papers is None or len(all_papers) < max_papers:
                        try:
                            scholarly_papers = self.search_scholarly(keyword, limit=1000)  # Increased limit
                            if scholarly_papers and isinstance(scholarly_papers, list) and len(scholarly_papers) > 0:
                                for paper in scholarly_papers:
                                    if paper and isinstance(paper, dict) and self._is_unique_paper(paper, seen_titles):
                                        all_papers.append(paper)
                                        # Safe title access
                                        title = paper.get("title", "")
                                        if title:
                                            seen_titles.add(title.lower())
                                        # Check limit only if one was set
                                        if max_papers is None or len(all_papers) >= max_papers:
                                            break
                            else:
                                logger.warning(f"⚠️ No valid papers returned from Google Scholar for keyword: {keyword}")
                        except Exception as e:
                            logger.error(f"❌ Error searching Google Scholar for keyword '{keyword}': {e}")
                            continue
                    
                    # Update progress
                    keyword_pbar.update(1)
                    keyword_pbar.set_postfix({"papers_found": len(all_papers)})
                    
                    # Rate limiting between keywords
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"❌ Error processing keyword '{keyword}': {e}")
                    continue
        
        if not all_papers:
            logger.warning("⚠️ No papers were collected from any source")
            return []
            
        logger.info(f"✅ Collected {len(all_papers)} unique UBR5-related papers")
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
                
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.error(f"❌ Error generating embedding for paper {i}: {e}")
                continue
        
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
            url = "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedText"
            
            headers = {
                "Content-Type": "application/json"
            }
            
            data = {
                "text": text
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
    
    def save_embeddings(self, papers_with_embeddings: List[Dict], source: str = "ubr5_api"):
        """
        Save embeddings to the xrvix_embeddings folder.
        
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
        
        # Clean title for filename
        clean_title = re.sub(r'[^\w\s-]', '', title)
        clean_title = re.sub(r'[-\s]+', '-', clean_title)
        clean_title = clean_title[:100]  # Limit length
        
        if doi:
            filename = f"{clean_title}_{doi.replace('/', '_')}.json"
        else:
            filename = f"{clean_title}_{hash(title)}.json"
        
        return filename
    
    def integrate_with_chromadb(self, papers_with_embeddings: List[Dict]):
        """
        Integrate embeddings into ChromaDB database.
        
        Args:
            papers_with_embeddings: List of papers with embeddings
        """
        logger.info(f"🔗 Integrating {len(papers_with_embeddings)} embeddings into ChromaDB")
        
        try:
            # Prepare data for ChromaDB with progress bar
            documents = []
            metadatas = []
            ids = []
            
            logger.info("🔧 Preparing metadata for ChromaDB integration...")
            with tqdm(total=len(papers_with_embeddings), desc="Preparing metadata", unit="paper") as metadata_pbar:
                for i, paper in enumerate(papers_with_embeddings):
                    # Create document text
                    doc_text = self._create_embedding_text(paper)
                    
                    # Create metadata
                    metadata = {
                        "title": paper.get("title", ""),
                        "doi": paper.get("doi", ""),
                        "authors": "; ".join(paper.get("authors", [])),
                        "journal": paper.get("journal", ""),
                        "year": str(paper.get("year", "")) if paper.get("year") else "",
                        "citation_count": paper.get("citation_count", "0"),
                        "source": paper.get("source", ""),
                        "is_preprint": str(paper.get("is_preprint", False)),
                        "publication_date": paper.get("publication_date", ""),
                        "fields_of_study": "; ".join(paper.get("fields_of_study", [])),
                        "publication_types": "; ".join(paper.get("publication_types", [])),
                        "abstract": paper.get("abstract", "")[:1000]  # Limit length
                    }
                    
                    # Create unique ID
                    paper_id = f"ubr5_api_{i}_{hash(paper.get('title', ''))}"
                    
                    documents.append(doc_text)
                    metadatas.append(metadata)
                    ids.append(paper_id)
                    
                    metadata_pbar.update(1)
                    metadata_pbar.set_postfix({"processed": i+1, "total": len(papers_with_embeddings)})
            
            # Add to ChromaDB
            self.chromadb_manager.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"✅ Successfully integrated {len(papers_with_embeddings)} embeddings into ChromaDB")
            
        except Exception as e:
            logger.error(f"❌ Error integrating with ChromaDB: {e}")
    
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
            papers = self.search_ubr5_papers(max_papers=max_papers)
            
            if not papers:
                logger.warning("⚠️ No papers found, exiting")
                return
            
            # Step 2: Generate embeddings
            papers_with_embeddings = self.generate_embeddings(papers, google_api_key)
            
            if not papers_with_embeddings:
                logger.warning("⚠️ No embeddings generated, exiting")
                return
            
            # Step 3: Save embeddings
            self.save_embeddings(papers_with_embeddings, source="ubr5_api")
            
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
    
    # Show warnings if critical keys are missing
    if not availability["google_api"]:
        print("\n⚠️  WARNING: GOOGLE_API_KEY not found!")
        print("   This key is required for generating embeddings.")
        print("   Please add it to your keys.json file.")
        print("   Example keys.json structure:")
        print("   {")
        print('     "GOOGLE_API_KEY": "your_google_api_key_here"')
        print('     "SCHOLAR_API_KEY": "optional_scholar_api_key"')
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
        print("   This will fetch ALL available papers from Semantic Scholar and Google Scholar")
    else:
        print(f"\n🚀 Starting UBR5 paper collection (target: {max_papers} papers)...")
    
    # Run scraping
    scraper.run_complete_scraping(max_papers=max_papers)

if __name__ == "__main__":
    main()
