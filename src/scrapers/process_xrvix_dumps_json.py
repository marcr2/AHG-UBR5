import glob
import os
import pkg_resources
import pandas as pd
# Optional imports for paperscraper (fallback if not available)
try:
    from paperscraper.xrxiv.xrxiv_query import XRXivQuery
    from paperscraper.citations import get_citations_from_title, get_citations_by_doi
    from paperscraper.impact import Impactor
    PAPERSCRAPER_AVAILABLE = True
except ImportError:
    PAPERSCRAPER_AVAILABLE = False
    print("⚠️  paperscraper not available - using alternative APIs only")
import requests
import json
import re
from tqdm.auto import tqdm
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random
from src.core.processing_config import get_config, print_config_info, CITATION_MAX_WORKERS, CITATION_TIMEOUT, CITATION_BATCH_SIZE
import warnings
import logging
import xml.etree.ElementTree as ET
from urllib.parse import quote
import threading
from collections import deque
# Optional import for matplotlib (used for plotting)
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib not available - plotting features disabled")

from tqdm import tqdm
import sys
import traceback
import hashlib
import pickle
from pathlib import Path

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data/logs/paper_processing.log')
    ]
)
logger = logging.getLogger(__name__)

# Suppress scholarly logging messages
logging.getLogger("scholarly").setLevel(logging.WARNING)
logging.getLogger("paperscraper").setLevel(logging.WARNING)

# Additional logging suppression for other potential sources
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

# More comprehensive logging suppression
logging.getLogger("paperscraper.citations").setLevel(logging.WARNING)
logging.getLogger("paperscraper.citations.citations").setLevel(logging.WARNING)
logging.getLogger("paperscraper.impact").setLevel(logging.WARNING)
logging.getLogger("scholarly.scholarly").setLevel(logging.WARNING)

# Suppress all logging from paperscraper and scholarly
for logger_name in ["paperscraper", "scholarly", "urllib3", "requests"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)
    logging.getLogger(logger_name).propagate = False

# More comprehensive warning suppression for paperscraper
warnings.filterwarnings("ignore", message="Could not find paper")
warnings.filterwarnings("ignore", category=UserWarning, module="paperscraper")
warnings.filterwarnings("ignore", category=UserWarning, module="scholarly")
warnings.filterwarnings("ignore", message=".*Could not find paper.*")
warnings.filterwarnings("ignore", message=".*assuming 0 citation.*")

# Suppress all warnings from paperscraper and scholarly modules
warnings.filterwarnings("ignore", module="paperscraper")
warnings.filterwarnings("ignore", module="scholarly")
warnings.filterwarnings("ignore", module="urllib3")
warnings.filterwarnings("ignore", module="requests")

# Additional suppression for specific warning patterns
warnings.filterwarnings(
    "ignore", message=".*Could not find paper.*", category=UserWarning
)
warnings.filterwarnings(
    "ignore", message=".*assuming 0 citation.*", category=UserWarning
)
warnings.filterwarnings(
    "ignore", message=".*Could not find paper.*", category=RuntimeWarning
)
warnings.filterwarnings(
    "ignore", message=".*assuming 0 citation.*", category=RuntimeWarning
)

# Note: The scholarly library (used by paperscraper for Google Scholar citations)
# generates INFO messages when encountering CAPTCHAs and anti-bot measures.
# These messages are now suppressed to reduce console noise.
#
# Citation strategy improvements:
# 1. Check existing citation data first (fastest)
# 2. Try DOI-based lookup (more reliable, fewer attempts)
# 3. Only try title-based lookup if DOI fails (less reliable, single attempt)
# 4. Increased delays between requests to be more respectful to Google Scholar
# 5. Reduced retry attempts to avoid overwhelming the service

# Optional import for matplotlib (used for plotting)
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
    MATPLOTLIB_DATES_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    MATPLOTLIB_DATES_AVAILABLE = False
    print("⚠️  matplotlib not available - plotting features disabled")





def extract_impact_factor(paper_data, journal_name=None):
    """Extract impact factor using arXiv API + hybrid fallback."""
    logger.info(f"🔍 DEBUG: Extracting impact factor for paper with DOI: {paper_data.get('doi', 'None')}")
    
    # Check if this is a preprint first - preprints have no impact factor
    # Quick check for bioRxiv DOIs (10.1101/) and medRxiv DOIs (10.1101/)
    doi = paper_data.get("doi", "")
    if doi.startswith("10.1101/"):
        logger.info(f"🔍 DEBUG: bioRxiv DOI detected, returning impact factor: 0.0")
        return "0.0"  # Preprints have no impact factor
    
    is_preprint, preprint_type, preprint_score = is_preprint_paper(paper_data)
    if is_preprint:
        logger.info(f"🔍 DEBUG: Preprint detected, returning impact factor: 0.0")
        return "0.0"  # Preprints have no impact factor
    
    # First, try to get impact factor directly from the data
    impact_fields = [
        "impact_factor", "journal_impact_factor", "if", "jif",
        "impact", "journal_if", "journal_impact"
    ]

    for field in impact_fields:
        if field in paper_data and paper_data[field] is not None:
            try:
                impact = float(paper_data[field])
                logger.info(f"🔍 DEBUG: Found impact factor in field '{field}': {impact}")
                return str(max(0, impact))
            except (ValueError, TypeError):
                continue

    # If no direct impact factor, try to get it from APIs
    if not journal_name:
        logger.info(f"🔍 DEBUG: No journal name provided, extracting it first")
        journal_name = extract_journal_info(paper_data)

    if journal_name and journal_name != "Unknown journal":
        logger.info(f"🔍 DEBUG: Using journal name: {journal_name}")
        
        # Try to get impact factor from arXiv API first (primary source)
        arxiv_id = extract_arxiv_id(paper_data)
        if arxiv_id:
            logger.info(f"🔍 DEBUG: Trying arXiv API for impact factor with ID: {arxiv_id}")
            impact_factor = get_impact_factor_arxiv(arxiv_id)
            if impact_factor is not None:
                logger.info(f"🔍 DEBUG: Got impact factor from arXiv API: {impact_factor}")
                return str(impact_factor)
            else:
                logger.info(f"🔍 DEBUG: arXiv API failed to return impact factor")
        
        # Try to get impact factor from OpenAlex API (fastest fallback)
        doi = paper_data.get("doi", "")
        if doi:
            logger.info(f"🔍 DEBUG: Trying OpenAlex API for impact factor with DOI: {doi}")
            impact_factor = get_impact_factor_openalex(doi)
            if impact_factor is not None:
                logger.info(f"🔍 DEBUG: Got impact factor from OpenAlex API: {impact_factor}")
                return str(impact_factor)
            
            # Try to get impact factor from Semantic Scholar API (second fallback)
            logger.info(f"🔍 DEBUG: Trying Semantic Scholar API for impact factor with DOI: {doi}")
            impact_factor = get_impact_factor_semantic_scholar(doi)
            if impact_factor is not None:
                logger.info(f"🔍 DEBUG: Got impact factor from Semantic Scholar API: {impact_factor}")
                return str(impact_factor)
            
            # Try to get impact factor from CrossRef API (third fallback)
            logger.info(f"🔍 DEBUG: Trying CrossRef API for impact factor with DOI: {doi}")
            impact_factor = get_impact_factor_crossref(doi)
            if impact_factor is not None:
                logger.info(f"🔍 DEBUG: Got impact factor from CrossRef API: {impact_factor}")
                return str(impact_factor)
            
            # Try to get impact factor from PubMed Central API (fourth fallback)
            logger.info(f"🔍 DEBUG: Trying PubMed Central API for impact factor with DOI: {doi}")
            impact_factor = get_impact_factor_pubmed_central(doi)
            if impact_factor is not None:
                logger.info(f"🔍 DEBUG: Got impact factor from PubMed Central API: {impact_factor}")
                return str(impact_factor)
            
            # Try to get impact factor from journal name using CrossRef (last fallback)
            logger.info(f"🔍 DEBUG: Trying CrossRef journal search for impact factor")
            impact_factor = get_impact_factor_by_journal_crossref(journal_name)
            if impact_factor is not None:
                logger.info(f"🔍 DEBUG: Got impact factor from CrossRef journal search: {impact_factor}")
                return str(impact_factor)

    # Fallback to the estimation method
    logger.info(f"🔍 DEBUG: Falling back to impact factor estimation for journal: {journal_name}")
    estimated_impact = estimate_impact_factor(journal_name)
    logger.info(f"🔍 DEBUG: Estimated impact factor: {estimated_impact}")
    return estimated_impact


def estimate_impact_factor(journal_name):
    """Estimate impact factor based on journal name."""
    logger.info(f"🔍 DEBUG: Estimating impact factor for journal: {journal_name}")
    
    if not journal_name or journal_name == "Unknown journal":
        logger.info(f"🔍 DEBUG: Invalid journal name, returning 'not found'")
        return "not found"

    # Comprehensive impact factor mapping based on recent data
    impact_factors = {
        "nature": 49.962,
        "science": 56.9,
        "cell": 66.85,
        "nature medicine": 87.241,
        "nature biotechnology": 68.164,
        "nature genetics": 41.307,
        "nature cell biology": 28.213,
        "nature immunology": 31.25,
        "nature reviews immunology": 108.555,
        "immunity": 43.474,
        "journal of immunology": 5.422,
        "journal of experimental medicine": 17.579,
        "proceedings of the national academy of sciences": 12.779,
        "pnas": 12.779,
        "plos one": 3.752,
        "bioinformatics": 6.937,
        "nucleic acids research": 19.16,
        "genome research": 11.093,
        "genome biology": 17.906,
        "cell reports": 9.995,
        "molecular cell": 19.328,
        "developmental cell": 13.417,
        "current biology": 10.834,
        "elife": 8.713,
        "plos biology": 9.593,
        "plos genetics": 6.02,
        "plos computational biology": 4.7,
        "bmc genomics": 4.317,
        "bmc bioinformatics": 3.169,
        "nature communications": 17.694,
        "nature methods": 47.99,
        "nature neuroscience": 25.0,
        "neuron": 16.2,
        "the lancet": 202.731,
        "new england journal of medicine": 176.079,
        "jama": 157.335,
        "biorxiv": 0.0,  # Preprint servers have no impact factor
        "medrxiv": 0.0,
        "arxiv": 0.0,
        "chemrxiv": 0.0,
    }

    journal_lower = journal_name.lower().strip()
    logger.info(f"🔍 DEBUG: Normalized journal name: '{journal_lower}'")

    # Direct match
    if journal_lower in impact_factors:
        impact = impact_factors[journal_lower]
        logger.info(f"🔍 DEBUG: Direct match found, impact factor: {impact}")
        return str(impact)

    # Partial match
    for key, impact in impact_factors.items():
        if key in journal_lower or journal_lower in key:
            logger.info(f"🔍 DEBUG: Partial match found with '{key}', impact factor: {impact}")
            return str(impact)

    logger.info(f"🔍 DEBUG: No match found, returning 'not found'")
    return "not found"


def extract_additional_metadata(paper_data):
    """Extract additional metadata fields that paperscraper might provide."""
    metadata = {}

    # Extract keywords
    keyword_fields = ["keywords", "keyword", "subject", "subjects", "tags"]
    for field in keyword_fields:
        if field in paper_data and paper_data[field]:
            keywords = paper_data[field]
            if isinstance(keywords, list):
                metadata["keywords"] = keywords
            elif isinstance(keywords, str):
                # Split on common delimiters
                metadata["keywords"] = [
                    k.strip()
                    for k in keywords.replace(";", ",").split(",")
                    if k.strip()
                ]
            break

    # Extract abstract
    abstract_fields = ["abstract", "summary", "description"]
    for field in abstract_fields:
        if field in paper_data and paper_data[field]:
            metadata["abstract_full"] = str(paper_data[field])
            break

    # Extract affiliation information
    affiliation_fields = ["affiliations", "affiliation", "institutions", "institution"]
    for field in affiliation_fields:
        if field in paper_data and paper_data[field]:
            affiliations = paper_data[field]
            if isinstance(affiliations, list):
                metadata["affiliations"] = affiliations
            elif isinstance(affiliations, str):
                metadata["affiliations"] = [affiliations]
            break

    # Extract funding information
    funding_fields = ["funding", "grants", "grant", "funding_source"]
    for field in funding_fields:
        if field in paper_data and paper_data[field]:
            metadata["funding"] = str(paper_data[field])
            break

    # Extract license information
    license_fields = ["license", "licence", "copyright", "rights"]
    for field in license_fields:
        if field in paper_data and paper_data[field]:
            metadata["license"] = str(paper_data[field])
            break

    # Extract URL information
    url_fields = ["url", "link", "pdf_url", "full_text_url"]
    for field in url_fields:
        if field in paper_data and paper_data[field]:
            metadata["url"] = str(paper_data[field])
            break

    # Extract category/subject classification
    category_fields = ["category", "categories", "classification", "subject_class"]
    for field in category_fields:
        if field in paper_data and paper_data[field]:
            categories = paper_data[field]
            if isinstance(categories, list):
                metadata["categories"] = categories
            elif isinstance(categories, str):
                metadata["categories"] = [categories]
            break

    # Extract version information (for preprints)
    version_fields = ["version", "revision", "v"]
    for field in version_fields:
        if field in paper_data and paper_data[field]:
            metadata["version"] = str(paper_data[field])
            break

    # Extract language
    language_fields = ["language", "lang", "locale"]
    for field in language_fields:
        if field in paper_data and paper_data[field]:
            metadata["language"] = str(paper_data[field])
            break

    return metadata


# Global counter for citation processing errors
citation_error_counter = {
    "doi_lookup_failures": 0,
    "title_lookup_failures": 0,
    "total_failures": 0,
}

# Global list to collect papers for batch citation processing
papers_for_citation_processing = []

# Simple citation cache to avoid re-processing
CITATION_CACHE = {}


def reset_citation_error_counter():
    """Reset the global citation error counter."""
    global citation_error_counter
    citation_error_counter = {
        "doi_lookup_failures": 0,
        "title_lookup_failures": 0,
        "total_failures": 0,
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


def get_cached_citation(paper_key):
    """Get citation from cache if available."""
    global CITATION_CACHE
    return CITATION_CACHE.get(paper_key)


def cache_citation(paper_key, citation_data):
    """Cache citation data for reuse."""
    global CITATION_CACHE
    CITATION_CACHE[paper_key] = citation_data


def clear_citation_cache():
    """Clear the citation cache."""
    global CITATION_CACHE
    CITATION_CACHE.clear()


def process_citations_batch(papers_batch):
    """Process citations in parallel with smart fallbacks for maximum speed and reliability."""
    global citation_error_counter

    if not papers_batch:
        return {}

    # Clean progress message - no verbose printing
    results = {}
    processed_count = 0
    failed_count = 0

    def process_single_paper(paper_data):
        """Process a single paper's citations with smart fallback strategy."""
        idx, row, source = paper_data
        paper_key = f"{source}_{idx}"
        
        # Check cache first (fastest)
        cached_result = get_cached_citation(paper_key)
        if cached_result:
            return paper_key, cached_result
        
        # Check if this is a preprint - skip expensive API calls for preprints
        is_preprint, preprint_type, preprint_score = is_preprint_paper(row)
        if is_preprint:
            result = {
                "citation_count": "0",  # Preprints typically have minimal citations
                "journal": preprint_type.upper() if preprint_type in ["biorxiv", "medrxiv", "arxiv", "chemrxiv"] else "Preprint",
                "impact_factor": "0.0",  # Preprints have no impact factor
            }
            cache_citation(paper_key, result)
            return paper_key, result
        
        try:
            # Strategy 1: Check if citation data is already in the paper_data (fastest)
            citation_fields = [
                "citations", "citation_count", "cited_by_count", "times_cited",
                "reference_count", "cited_count", "num_citations", "citedByCount"
            ]
            
            for field in citation_fields:
                if field in row and row[field] is not None:
                    try:
                        count = int(row[field])
                        if count >= 0:
                            result = {
                                "citation_count": str(count),
                                "journal": extract_journal_info_fast(row),
                                "impact_factor": extract_impact_factor_fast(row),
                            }
                            # Cache the result for future use
                            cache_citation(paper_key, result)
                            return paper_key, result
                    except (ValueError, TypeError):
                        continue

            # Strategy 2: Try DOI-based lookup (most reliable)
            doi = row.get("doi", "")
            if doi and str(doi).strip().startswith("10."):
                # Try OpenCitations API first (fastest)
                try:
                    citations = get_citations_opencitations(doi)
                    if citations is not None:
                        result = {
                            "citation_count": str(citations),
                            "journal": extract_journal_info_fast(row),
                            "impact_factor": extract_impact_factor_fast(row),
                        }
                        # Cache the result for future use
                        cache_citation(paper_key, result)
                        return paper_key, result
                except Exception:
                    pass
                
                # Try PubMed Central API as fallback
                try:
                    citations = get_citations_pubmed_central(doi)
                    if citations is not None:
                        result = {
                            "citation_count": str(citations),
                            "journal": extract_journal_info_fast(row),
                            "impact_factor": extract_impact_factor_fast(row),
                        }
                        # Cache the result for future use
                        cache_citation(paper_key, result)
                        return paper_key, result
                except Exception:
                    pass

            # Strategy 3: Title-based lookup (least reliable but catches edge cases)
            title = row.get("title", "")
            if title and len(str(title).strip()) >= 10:
                try:
                    citations = get_citations_pubmed_central_by_title(title)
                    if citations is not None:
                        result = {
                            "citation_count": str(citations),
                            "journal": extract_journal_info_fast(row),
                            "impact_factor": extract_impact_factor_fast(row),
                        }
                        # Cache the result for future use
                        cache_citation(paper_key, result)
                        return paper_key, result
                except Exception:
                    pass

            # If all strategies fail, return default values
            result = {
                "citation_count": "not found",
                "journal": extract_journal_info_fast(row),
                "impact_factor": extract_impact_factor_fast(row),
            }
            # Cache the result for future use
            cache_citation(paper_key, result)
            return paper_key, result

        except Exception as e:
            # Increment error counter silently
            citation_error_counter["total_failures"] += 1
            return paper_key, {
                "citation_count": "error",
                "journal": "error",
                "impact_factor": "error",
            }

    # Process papers in parallel with a single progress bar for metadata extrapolation
    with ThreadPoolExecutor(max_workers=CITATION_MAX_WORKERS) as executor:
        future_to_paper = {
            executor.submit(process_single_paper, paper_data): paper_data
            for paper_data in papers_batch
        }

        # Single progress bar for metadata extrapolation (aggregate of all metadata fields)
        with tqdm(
            total=len(papers_batch),
            desc="Extrapolating metadata (citations, journals, impact factors, authors, dates)",
            unit="paper"
        ) as pbar:
            for future in as_completed(future_to_paper):
                try:
                    paper_key, result = future.result()
                    results[paper_key] = result
                    processed_count += 1
                except Exception:
                    failed_count += 1
                
                pbar.update(1)

    return results


def extract_journal_info_fast(paper_data):
    """Fast journal info extraction with minimal API calls."""
    # Check if this is a preprint first - skip expensive operations for preprints
    is_preprint, preprint_type, preprint_score = is_preprint_paper(paper_data)
    if is_preprint:
        # Return appropriate preprint server name
        source = paper_data.get("source", "").lower()
        if source in ["biorxiv", "medrxiv", "arxiv", "chemrxiv"]:
            return source.upper()
        elif preprint_type in ["biorxiv", "medrxiv", "arxiv", "chemrxiv"]:
            return preprint_type.upper()
        else:
            return "Preprint"
    
    # First check if journal info is already in the data
    journal_fields = ["journal", "journal_name", "publication_venue", "venue", "container_title"]
    
    for field in journal_fields:
        if field in paper_data and paper_data[field]:
            journal = str(paper_data[field]).strip()
            if journal and journal.lower() not in ["nan", "none", "null", ""]:
                return journal
    
    # If no journal info found, return default
    return "Unknown journal"

def extract_impact_factor_fast(paper_data):
    """Fast impact factor extraction with minimal API calls."""
    # Check if this is a preprint first - preprints have no impact factor
    is_preprint, preprint_type, preprint_score = is_preprint_paper(paper_data)
    if is_preprint:
        return "0.0"  # Preprints have no impact factor
    
    # First check if impact factor is already in the data
    impact_fields = ["impact_factor", "jcr_impact_factor", "sjr_impact_factor", "h_index"]
    
    for field in impact_fields:
        if field in paper_data and paper_data[field] is not None:
            try:
                impact = float(paper_data[field])
                if impact >= 0:
                    return str(impact)
            except (ValueError, TypeError):
                continue
    
    # If no impact factor found, return default
    return "not found"

def extract_arxiv_id(paper_data):
    """Extract arXiv ID from paper data."""
    logger.info(f"🔍 DEBUG: Extracting arXiv ID from paper data")
    
    # Check various fields for arXiv ID
    arxiv_fields = [
        "arxiv_id", "arxiv", "arxiv_id_v1", "arxiv_id_v2", "id", "identifier"
    ]
    
    for field in arxiv_fields:
        if field in paper_data and paper_data[field]:
            value = str(paper_data[field]).strip()
            logger.info(f"🔍 DEBUG: Found field '{field}' with value: {value}")
            
            if "arxiv.org" in value or value.startswith("arXiv:"):
                # Extract arXiv ID from URL or "arXiv:" prefix
                if "arxiv.org" in value:
                    arxiv_id = value.split("/")[-1]
                    logger.info(f"🔍 DEBUG: Extracted arXiv ID from URL: {arxiv_id}")
                    return arxiv_id
                elif value.startswith("arXiv:"):
                    arxiv_id = value[6:]  # Remove "arXiv:" prefix
                    logger.info(f"🔍 DEBUG: Extracted arXiv ID from prefix: {arxiv_id}")
                    return arxiv_id
                logger.info(f"🔍 DEBUG: Returning raw value as arXiv ID: {value}")
                return value
    
    # Check if source is arXiv
    source = paper_data.get("source", "")
    logger.info(f"🔍 DEBUG: Paper source: {source}")
    
    if source == "arxiv":
        # Try to extract from title or other fields
        title = paper_data.get("title", "")
        logger.info(f"🔍 DEBUG: Paper title: {title}")
        
        if title and "arXiv:" in title:
            match = re.search(r'arXiv:(\d+\.\d+)', title)
            if match:
                arxiv_id = match.group(1)
                logger.info(f"🔍 DEBUG: Extracted arXiv ID from title: {arxiv_id}")
                return arxiv_id
    
    logger.info(f"🔍 DEBUG: No arXiv ID found")
    return None

def get_journal_info_arxiv(arxiv_id):
    """Get journal information from arXiv API using arXiv ID."""
    logger.info(f"🔍 DEBUG: Calling arXiv API for journal info with ID: {arxiv_id}")
    
    try:
        # Clean arXiv ID
        arxiv_id = str(arxiv_id).strip()
        if not arxiv_id:
            logger.info(f"🔍 DEBUG: Invalid arXiv ID: {arxiv_id}")
            return None
            
        # Extract arXiv ID if it's a full URL
        if "arxiv.org" in arxiv_id:
            arxiv_id = arxiv_id.split("/")[-1]
            logger.info(f"🔍 DEBUG: Cleaned arXiv ID from URL: {arxiv_id}")
        
        # Remove version suffix if present (e.g., v1, v2)
        original_arxiv_id = arxiv_id
        arxiv_id = re.sub(r'v\d+$', '', arxiv_id)
        if original_arxiv_id != arxiv_id:
            logger.info(f"🔍 DEBUG: Removed version suffix: {original_arxiv_id} -> {arxiv_id}")
        
        # arXiv API endpoint
        url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
        logger.info(f"🔍 DEBUG: Calling arXiv API URL: {url}")
        
        response = requests.get(url, timeout=10)
        logger.info(f"🔍 DEBUG: arXiv API response status: {response.status_code}")
        
        if response.status_code == 200:
            # Parse XML response
            root = ET.fromstring(response.content)
            logger.info(f"🔍 DEBUG: Successfully parsed arXiv API XML response")
            
            # Extract journal information
            journal_ref = root.find('.//{http://arxiv.org/schemas/atom}journal_ref')
            if journal_ref is not None and journal_ref.text:
                journal_info = journal_ref.text.strip()
                logger.info(f"🔍 DEBUG: Found journal_ref in arXiv API: {journal_info}")
                return journal_info
            else:
                logger.info(f"🔍 DEBUG: No journal_ref found in arXiv API response")
            
            # Extract primary category for preprint identification
            primary_category = root.find('.//{http://arxiv.org/schemas/atom}primary_category')
            if primary_category is not None:
                category = primary_category.get('term', '')
                if category:
                    journal_info = f"arXiv ({category})"
                    logger.info(f"🔍 DEBUG: Using primary category as journal: {journal_info}")
                    return journal_info
                else:
                    logger.info(f"🔍 DEBUG: No category term found in primary_category")
            else:
                logger.info(f"🔍 DEBUG: No primary_category found in arXiv API response")
            
            logger.info(f"🔍 DEBUG: Returning default 'arXiv preprint'")
            return "arXiv preprint"
        else:
            logger.info(f"🔍 DEBUG: arXiv API returned error status: {response.status_code}")
            logger.info(f"🔍 DEBUG: Response content: {response.text[:200]}...")
                
        return None
        
    except Exception as e:
        logger.error(f"🔍 DEBUG: Error calling arXiv API: {e}")
        return None

def get_journal_info_openalex(doi):
    """Get journal information from OpenAlex API using DOI."""
    logger.info(f"🔍 DEBUG: Calling OpenAlex API for journal info with DOI: {doi}")
    
    try:
        # Clean DOI
        doi = str(doi).strip()
        if not doi or not doi.startswith("10."):
            logger.info(f"🔍 DEBUG: Invalid DOI: {doi}")
            return None
            
        # OpenAlex API endpoint
        url = f"https://api.openalex.org/works/doi:{doi}"
        logger.info(f"🔍 DEBUG: Calling OpenAlex API URL: {url}")
        
        response = requests.get(url, timeout=10)
        logger.info(f"🔍 DEBUG: OpenAlex API response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"🔍 DEBUG: Successfully parsed OpenAlex API response")
            
            # Get journal information
            if 'host_venue' in data and data['host_venue']:
                venue = data['host_venue']
                if 'display_name' in venue:
                    journal_name = venue['display_name']
                    logger.info(f"🔍 DEBUG: Found journal display_name in OpenAlex: {journal_name}")
                    return journal_name
                elif 'title' in venue:
                    journal_name = venue['title']
                    logger.info(f"🔍 DEBUG: Found journal title in OpenAlex: {journal_name}")
                    return journal_name
                else:
                    logger.info(f"�� DEBUG: No display_name or title found in OpenAlex host_venue")
            else:
                logger.info(f"🔍 DEBUG: No host_venue found in OpenAlex response")
                    
        else:
            logger.info(f"🔍 DEBUG: OpenAlex API returned error status: {response.status_code}")
            logger.info(f"🔍 DEBUG: Response content: {response.text[:200]}...")
                    
        return None
        
    except Exception as e:
        logger.error(f"🔍 DEBUG: Error calling OpenAlex API: {e}")
        return None

def get_journal_info_semantic_scholar(doi):
    """Get journal information from Semantic Scholar API using DOI."""
    logger.info(f"🔍 DEBUG: Calling Semantic Scholar API for journal info with DOI: {doi}")
    
    try:
        # Clean DOI
        doi = str(doi).strip()
        if not doi or not doi.startswith("10."):
            logger.info(f"🔍 DEBUG: Invalid DOI: {doi}")
            return None
            
        # Semantic Scholar API endpoint
        url = f"https://api.semanticscholar.org/v1/paper/{doi}"
        logger.info(f"🔍 DEBUG: Calling Semantic Scholar API URL: {url}")
        
        response = requests.get(url, timeout=10)
        logger.info(f"🔍 DEBUG: Semantic Scholar API response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"🔍 DEBUG: Successfully parsed Semantic Scholar API response")
            
            # Get journal information
            if 'venue' in data and data['venue']:
                journal_name = data['venue']
                logger.info(f"🔍 DEBUG: Found venue in Semantic Scholar: {journal_name}")
                logger.info(f"🔍 DEBUG: Estimating impact factor for journal")
                return estimate_impact_factor(journal_name)
            else:
                logger.info(f"🔍 DEBUG: No venue found in Semantic Scholar response")
                
        else:
            logger.info(f"🔍 DEBUG: Semantic Scholar API returned error status: {response.status_code}")
            logger.info(f"🔍 DEBUG: Response content: {response.text[:200]}...")
                
        return None
        
    except Exception as e:
        logger.error(f"🔍 DEBUG: Error calling Semantic Scholar API: {e}")
        return None

def get_impact_factor_crossref(doi):
    """Get impact factor from CrossRef API metadata."""
    try:
        # Clean DOI
        doi = str(doi).strip()
        if not doi or not doi.startswith("10."):
            return None
            
        # CrossRef API endpoint
        url = f"https://api.crossref.org/works/{doi}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            work = data.get("message", {})
            
            # Look for impact factor in various metadata fields
            # Note: CrossRef doesn't directly provide impact factors, but we can look for
            # journal information that might help with estimation
            journal_title = work.get("container-title", [None])[0]
            if journal_title:
                # Try to get impact factor from our estimation database
                return estimate_impact_factor(journal_title)
                
        return None
        
    except Exception:
        return None

def get_impact_factor_pubmed_central(doi):
    """Get impact factor from PubMed Central API metadata."""
    try:
        # Clean DOI
        doi = str(doi).strip()
        if not doi or not doi.startswith("10."):
            return None
            
        # PubMed Central API endpoint
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        params = {
            "db": "pmc",
            "id": doi,
            "retmode": "json"
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "result" in data and doi in data["result"]:
                paper_data = data["result"][doi]
                journal = paper_data.get("fulljournalname", paper_data.get("journal", None))
                if journal:
                    # Try to get impact factor from our estimation database
                    return estimate_impact_factor(journal)
                    
        return None
        
    except Exception:
        return None

def get_impact_factor_by_journal_crossref(journal_name):
    """Get impact factor by searching CrossRef for journal information."""
    try:
        # Clean journal name
        journal_name = str(journal_name).strip()
        if not journal_name or journal_name == "Unknown journal":
            return None
            
        # CrossRef API endpoint for journal search
        url = f"https://api.crossref.org/journals"
        params = {
            "query": journal_name,
            "rows": 1
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            items = data.get("message", {}).get("items", [])
            if items:
                # Try to get impact factor from our estimation database
                return estimate_impact_factor(journal_name)
                
        return None
        
    except Exception:
        return None

def extract_authors_enhanced(paper_data):
    """Extract enhanced author information using multiple APIs."""
    # First check if author information is already in the paper_data
    author_fields = [
        "authors", "author", "creator", "contributor", "author_list",
        "author_names", "author_name", "authors_list"
    ]

    for field in author_fields:
        if field in paper_data and paper_data[field]:
            authors = paper_data[field]
            
            # Handle different author formats
            if isinstance(authors, str):
                # If it's a string, try to parse it
                if ";" in authors:
                    # Split by semicolon
                    author_list = [author.strip() for author in authors.split(";") if author.strip()]
                elif "," in authors:
                    # Split by comma
                    author_list = [author.strip() for author in authors.split(",") if author.strip()]
                else:
                    # Single author
                    author_list = [authors.strip()] if authors.strip() else []
            elif isinstance(authors, (list, tuple)):
                # If it's already a list/tuple
                author_list = [str(author).strip() for author in authors if author]
            else:
                # Convert to string and try to parse
                author_str = str(authors)
                if ";" in author_str:
                    author_list = [author.strip() for author in author_str.split(";") if author.strip()]
                elif "," in author_str:
                    author_list = [author.strip() for author in author_str.split(",") if author.strip()]
                else:
                    author_list = [author_str.strip()] if author_str.strip() else []
            
            # Filter out empty or invalid authors
            author_list = [author for author in author_list if author and author.lower() not in ["nan", "none", "null", ""]]
            
            if author_list:
                return ", ".join(author_list)

    # If no authors found, return default
    return "Unknown author"

def get_authors_crossref(doi):
    """Get author information from CrossRef API using DOI."""
    try:
        # Clean DOI
        doi = str(doi).strip()
        if not doi or not doi.startswith("10."):
            return None
            
        # CrossRef API endpoint
        url = f"https://api.crossref.org/works/{doi}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            work = data.get("message", {})
            
            # Get authors
            authors = work.get("author", [])
            if authors:
                author_names = []
                for author in authors:
                    if "given" in author and "family" in author:
                        author_names.append(f"{author['given']} {author['family']}")
                    elif "family" in author:
                        author_names.append(author["family"])
                    elif "name" in author:
                        author_names.append(author["name"])
                
                if author_names:
                    return "; ".join(author_names)
                    
        return None
        
    except Exception:
        return None

def get_authors_pubmed_central(doi):
    """Get author information from PubMed Central API using DOI."""
    try:
        # Clean DOI
        doi = str(doi).strip()
        if not doi or not doi.startswith("10."):
            return None
            
        # PubMed Central API endpoint
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        params = {
            "db": "pmc",
            "id": doi,
            "retmode": "json"
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "result" in data and doi in data["result"]:
                paper_data = data["result"][doi]
                authors = paper_data.get("authors", [])
                if authors:
                    return "; ".join(authors)
                    
        return None
        
    except Exception:
        return None

def extract_citation_count(paper_data):
    """Extract citation count using OpenCitations API and PubMed Central API for better reliability."""
    global citation_error_counter

    # First, check if citation data is already in the paper_data (fastest)
    citation_fields = [
        "citations", "citation_count", "cited_by_count", "times_cited",
        "reference_count", "cited_count", "num_citations", "citedByCount"
    ]
    
    for field in citation_fields:
        if field in paper_data and paper_data[field] is not None:
            try:
                count = int(paper_data[field])
                if count >= 0:
                    return str(count)
            except (ValueError, TypeError):
                continue

    # Try to get citation count from DOI if available
    doi = paper_data.get("doi", "")
    if doi:
        # Clean and validate DOI
        doi = str(doi).strip()
        if not doi or doi.lower() in ["nan", "none", "null", ""]:
            doi = ""
        elif not doi.startswith("10."):
            doi = ""

    if doi:
        # Try OpenCitations API first (fast and reliable)
        try:
            citations = get_citations_opencitations(doi)
            if citations is not None:
                return str(citations)
        except Exception:
            pass
        
        # Try PubMed Central API as fallback
        try:
            citations = get_citations_pubmed_central(doi)
            if citations is not None:
                return str(citations)
        except Exception:
            pass
        
        # Final fallback to paperscraper if other APIs fail and paperscraper is available
        if PAPERSCRAPER_AVAILABLE:
            try:
                citations = get_citations_by_doi(doi)
                if citations is not None and citations >= 0:
                    return str(citations)
            except Exception as e:
                citation_error_counter["doi_lookup_failures"] += 1
                citation_error_counter["total_failures"] += 1

    # Only try title-based lookup if DOI failed and we have a good title
    title = paper_data.get("title", "")
    if title:
        title = str(title).strip()
        if not title or title.lower() in ["nan", "none", "null", ""]:
            title = ""
        elif len(title) < 10:
            title = ""

    if title:
        # Try PubMed Central API for title-based search
        try:
            citations = get_citations_pubmed_central_by_title(title)
            if citations is not None:
                return str(citations)
        except Exception:
            pass
        
        # Fallback to paperscraper if available
        if PAPERSCRAPER_AVAILABLE:
            try:
                citations = get_citations_from_title(title)
                if citations is not None and citations >= 0:
                    return str(citations)
            except Exception as e:
                citation_error_counter["title_lookup_failures"] += 1
                citation_error_counter["total_failures"] += 1

    # Final fallback: check for any other citation-related fields
    for field in paper_data.keys():
        if "citation" in field.lower() or "cited" in field.lower():
            try:
                value = paper_data[field]
                if isinstance(value, (int, float)) and value >= 0:
                    return str(int(value))
                elif isinstance(value, str):
                    import re
                    numbers = re.findall(r"\d+", value)
                    if numbers:
                        return str(int(numbers[0]))
            except (ValueError, TypeError):
                continue

    return "not found"


def extract_journal_info(paper_data):
    """Extract journal information using arXiv API + hybrid fallback."""
    logger.info(f"🔍 DEBUG: Extracting journal info for paper with DOI: {paper_data.get('doi', 'None')}")
    
    # Check if this is a preprint first - skip expensive API calls for preprints
    # Quick check for bioRxiv DOIs (10.1101/) and medRxiv DOIs (10.1101/)
    doi = paper_data.get("doi", "")
    if doi.startswith("10.1101/"):
        logger.info(f"🔍 DEBUG: bioRxiv DOI detected, returning: BIORXIV")
        return "BIORXIV"
    
    is_preprint, preprint_type, preprint_score = is_preprint_paper(paper_data)
    logger.info(f"🔍 DEBUG: Preprint detection result - is_preprint: {is_preprint}, type: {preprint_type}, score: {preprint_score}")
    logger.info(f"🔍 DEBUG: Paper data keys: {list(paper_data.keys())}")
    logger.info(f"🔍 DEBUG: Source field: '{paper_data.get('source', 'NOT_FOUND')}'")
    logger.info(f"🔍 DEBUG: Journal field: '{paper_data.get('journal', 'NOT_FOUND')}'")
    
    if is_preprint:
        # Return appropriate preprint server name
        source = paper_data.get("source", "").lower()
        if source in ["biorxiv", "medrxiv", "arxiv", "chemrxiv"]:
            logger.info(f"🔍 DEBUG: Preprint detected, returning: {source.upper()}")
            return source.upper()
        elif preprint_type in ["biorxiv", "medrxiv", "arxiv", "chemrxiv"]:
            logger.info(f"🔍 DEBUG: Preprint detected, returning: {preprint_type.upper()}")
            return preprint_type.upper()
        else:
            logger.info(f"🔍 DEBUG: Preprint detected, returning: Preprint")
            return "Preprint"
    
    # First check if journal information is already in the paper_data
    journal_fields = [
        "journal", "journal_name", "publication", "source", "venue",
        "journal_title", "publication_venue", "journal_ref", "journal-ref"
    ]

    for field in journal_fields:
        if field in paper_data and paper_data[field]:
            journal_name = str(paper_data[field])
            logger.info(f"🔍 DEBUG: Found journal in field '{field}': {journal_name}")
            
            # Try to get enhanced journal info from arXiv API first
            arxiv_id = extract_arxiv_id(paper_data)
            if arxiv_id:
                logger.info(f"🔍 DEBUG: Found arXiv ID: {arxiv_id}")
                enhanced_journal = get_journal_info_arxiv(arxiv_id)
                if enhanced_journal:
                    logger.info(f"🔍 DEBUG: Enhanced journal from arXiv API: {enhanced_journal}")
                    return enhanced_journal
                else:
                    logger.info(f"🔍 DEBUG: arXiv API returned no journal info")
            
            # Try to get enhanced journal info from CrossRef API
            enhanced_journal = get_journal_info_crossref(paper_data.get("doi", ""))
            if enhanced_journal:
                logger.info(f"🔍 DEBUG: Enhanced journal from CrossRef API: {enhanced_journal}")
                return enhanced_journal
            
            # Try to get enhanced journal info from PubMed Central API
            enhanced_journal = get_journal_info_pubmed_central(paper_data.get("doi", ""))
            if enhanced_journal:
                logger.info(f"🔍 DEBUG: Enhanced journal from PubMed Central API: {enhanced_journal}")
                return enhanced_journal
            
            # Return the original journal name if no enhancement found
            logger.info(f"🔍 DEBUG: Returning original journal name: {journal_name}")
            return journal_name

    # If no journal found in paper_data, check source
    source = paper_data.get("source", "")
    if source in ["biorxiv", "medrxiv"]:
        logger.info(f"🔍 DEBUG: Source is preprint server: {source.upper()}")
        return f"{source.upper()}"

    # Try to get journal info from arXiv ID first (primary source)
    arxiv_id = extract_arxiv_id(paper_data)
    if arxiv_id:
        logger.info(f"🔍 DEBUG: Trying arXiv API with ID: {arxiv_id}")
        enhanced_journal = get_journal_info_arxiv(arxiv_id)
        if enhanced_journal:
            logger.info(f"�� DEBUG: Got journal from arXiv API: {enhanced_journal}")
            return enhanced_journal
        else:
            logger.info(f"🔍 DEBUG: arXiv API failed to return journal info")

    # Try to get journal info from DOI with hybrid fallback
    doi = paper_data.get("doi", "")
    if doi:
        logger.info(f"🔍 DEBUG: Trying DOI-based journal lookup: {doi}")
        # Try OpenAlex first (fastest fallback)
        enhanced_journal = get_journal_info_openalex(doi)
        if enhanced_journal:
            logger.info(f"�� DEBUG: Got journal from OpenAlex API: {enhanced_journal}")
            return enhanced_journal
        
        # Try CrossRef as second fallback
        enhanced_journal = get_journal_info_crossref(doi)
        if enhanced_journal:
            logger.info(f"🔍 DEBUG: Got journal from CrossRef API: {enhanced_journal}")
            return enhanced_journal
        
        # Try PubMed Central as third fallback
        enhanced_journal = get_journal_info_pubmed_central(doi)
        if enhanced_journal:
            logger.info(f"🔍 DEBUG: Got journal from PubMed Central API: {enhanced_journal}")
            return enhanced_journal

    logger.info(f"🔍 DEBUG: No journal info found, returning 'Unknown journal'")
    return "Unknown journal"


# Optional import for matplotlib.dates (used for plotting)
try:
    import matplotlib.dates as mdates
    MATPLOTLIB_DATES_AVAILABLE = True
except ImportError:
    MATPLOTLIB_DATES_AVAILABLE = False
    print("⚠️  matplotlib.dates not available - plotting features disabled")

from collections import deque
import threading

# --- REQUEST RATE TRACKING ---
request_count_lock = threading.Lock()
request_count = 0
request_start_time = None


def reset_request_counter():
    global request_count, request_start_time
    with request_count_lock:
        request_count = 0
        request_start_time = time.time()


def increment_request_counter():
    global request_count
    with request_count_lock:
        request_count += 1


def get_requests_per_sec():
    global request_count, request_start_time
    with request_count_lock:
        elapsed = time.time() - request_start_time if request_start_time else 1
        return request_count / elapsed if elapsed > 0 else 0


class LiveRatePlot:
    def __init__(self, window_seconds=300, title="Requests/sec (rolling, last 5 min)"):
        if not MATPLOTLIB_AVAILABLE:
            print("⚠️  matplotlib not available - plotting disabled")
            self.plt_available = False
            return
        
        try:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(10, 4))
            self.ax.set_title(title)
            self.ax.set_xlabel("Time")
            self.ax.set_ylabel("Requests/sec")
            self.plt_available = True
        except Exception as e:
            print(f"⚠️  Error initializing matplotlib: {e}")
            self.plt_available = False
        
        self.rate_times = deque()
        self.rate_values = deque()
        self.error_times = deque()
        self.error_values = deque()
        self.lock = threading.Lock()
        self.window_seconds = window_seconds
        self.last_plot_update = 0

    def update(self, rate, error=False):
        if not hasattr(self, 'plt_available') or not self.plt_available:
            return
            
        now = datetime.now()
        with self.lock:
            self.rate_times.append(now)
            self.rate_values.append(rate)
            if error:
                self.error_times.append(now)
                self.error_values.append(float(rate))

    def refresh(self):
        if not hasattr(self, 'plt_available') or not self.plt_available:
            return
            
        with self.lock:
            now = datetime.now()
            # Remove old points
            while (
                self.rate_times
                and (now - self.rate_times[0]).total_seconds() > self.window_seconds
            ):
                self.rate_times.popleft()
                self.rate_values.popleft()
            while (
                self.error_times
                and (now - self.error_times[0]).total_seconds() > self.window_seconds
            ):
                self.error_times.popleft()
                self.error_values.popleft()
            self.ax.clear()
            self.ax.set_title("Requests/sec (rolling, last 5 min)")
            self.ax.set_xlabel("Time")
            self.ax.set_ylabel("Requests/sec")
            if self.rate_times:
                self.ax.plot(
                    self.rate_times,
                    self.rate_values,
                    label="Requests/sec",
                    color="blue",
                )
            if self.error_times:
                self.ax.scatter(
                    self.error_times,
                    self.error_values,
                    color="red",
                    label="429 error",
                    marker="x",
                    s=60,
                )
            self.ax.legend()
            self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            self.fig.autofmt_xdate()
            plt.tight_layout()
            plt.pause(0.01)

    def close(self):
        if not hasattr(self, 'plt_available') or not self.plt_available:
            return
            
        plt.ioff()
        plt.show()


# --- CONFIG ---
# Import optimized configuration
try:
    from src.core.processing_config import (
        MAX_WORKERS,
        BATCH_SIZE,
        RATE_LIMIT_DELAY,
        REQUEST_TIMEOUT,
        MIN_CHUNK_LENGTH,
        MAX_CHUNK_LENGTH,
        SAVE_INTERVAL,
        DUMPS,
    )

    print("✅ Using optimized configuration from processing_config.py")
except ImportError:
    print("⚠️  processing_config.py not found, using fallback configuration")
    # Fallback configuration
    MAX_WORKERS = 3
    BATCH_SIZE = 200
    RATE_LIMIT_DELAY = 0.02
    REQUEST_TIMEOUT = 60
    MIN_CHUNK_LENGTH = 50
    MAX_CHUNK_LENGTH = 8000
    SAVE_INTERVAL = 1000
    DUMPS = ["biorxiv", "medrxiv"]

# Set dump root to use our local directory for downloaded dumps
DUMP_ROOT = "data/scraped_data/paperscraper_dumps/server_dumps"  # Use our local directory
EMBEDDINGS_DIR = "data/embeddings/xrvix_embeddings"
SCRAPED_DATA_DIR = "data/scraped_data/xrvix"

# Rate limiting settings
# Different limits for different API endpoints
EMBEDDING_MAX_REQUESTS_PER_MINUTE = 1500  # Embedding API: 1500 requests per minute
GEMINI_MAX_REQUESTS_PER_MINUTE = 1000  # Gemini API: 1000 requests per minute
MAX_REQUESTS_PER_MINUTE = (
    EMBEDDING_MAX_REQUESTS_PER_MINUTE  # Use embedding limit for processing
)
MAX_BATCH_ENQUEUED_TOKENS = 3_000_000  # Max batch enqueued tokens
RATE_LIMIT_WINDOW = 60  # seconds
MAX_RETRIES = 3
BASE_BACKOFF_DELAY = 1  # seconds

# Global rate limiting
request_times = []


def is_rate_limited():
    """Check if we're currently rate limited"""
    global request_times
    current_time = time.time()

    # Remove old requests outside the window
    request_times = [t for t in request_times if current_time - t < RATE_LIMIT_WINDOW]

    # Check if we've made too many requests recently
    return len(request_times) >= MAX_REQUESTS_PER_MINUTE


def wait_for_rate_limit():
    """Wait until we're no longer rate limited"""
    from tqdm.auto import tqdm

    while is_rate_limited():
        # With 1500 req/min limit, we need to wait at least 1 minute after hitting the limit
        sleep_time = 60 + random.uniform(0, 10)  # 60-70 seconds to be safe
        # tqdm progress bar for rate limit wait
        with tqdm(
            total=sleep_time,
            desc="Rate limited: waiting to resume",
            bar_format="{l_bar}{bar}| {remaining} left",
            colour="#FFA500",
            leave=False,
        ) as pbar:
            waited = 0.0
            while waited < sleep_time:
                time.sleep(0.1)
                waited += 0.1
                pbar.update(0.1)
        # Clear old request times after waiting
        global request_times
        current_time = time.time()
        request_times = [
            t for t in request_times if current_time - t < RATE_LIMIT_WINDOW
        ]


def record_request():
    """Record that we made a request"""
    global request_times
    request_times.append(time.time())



# --- CITATION CACHE (for end-of-process enrichment and offline backfill) ---
CITATION_CACHE = {}


def merge_citation_results_into_cache(citation_results):
    """Merge results from process_citations_batch into in-memory cache."""
    global CITATION_CACHE
    if not citation_results:
        return
    for paper_key, stats in citation_results.items():
        # Store only expected fields as strings
        if not isinstance(stats, dict):
            continue
        entry = CITATION_CACHE.get(paper_key, {})
        for k in ("citation_count", "journal", "impact_factor"):
            v = stats.get(k)
            if v is not None:
                entry[k] = str(v) if not isinstance(v, str) else v
        CITATION_CACHE[paper_key] = entry


def save_citation_cache(embeddings_dir: str):
    """Persist the in-memory citation cache to embeddings_dir/citation_cache.json."""
    try:
        os.makedirs(embeddings_dir, exist_ok=True)
        cache_path = os.path.join(embeddings_dir, "citation_cache.json")
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(CITATION_CACHE, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️  Failed to save citation cache: {e}")


def load_citation_cache(embeddings_dir: str):
    """Load citation cache from disk into memory (merging into existing cache)."""
    global CITATION_CACHE
    cache_path = os.path.join(embeddings_dir, "citation_cache.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(v, dict):
                            CITATION_CACHE[k] = {**CITATION_CACHE.get(k, {}), **v}
        except Exception as e:
            print(f"⚠️  Failed to load citation cache: {e}")


def enrich_all_batches_metadata_from_citations(
    embeddings_dir: str, citation_cache=None
) -> dict:
    """
    Sweep all batch_*.json files in embeddings_dir/* and update only 'pending' fields
    (citation_count, journal, impact_factor) from the provided citation_cache or from
    embeddings_dir/citation_cache.json if None.

    Returns a stats dict.
    """
    if citation_cache is None:
        # Load from disk if not provided
        cache_path = os.path.join(embeddings_dir, "citation_cache.json")
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                citation_cache = json.load(f)
        except Exception:
            citation_cache = {}

    stats = {"files_scanned": 0, "files_modified": 0, "records_updated": 0}
    if not citation_cache:
        print("No citation cache available for enrichment.")
        return stats

    sources = [
        d
        for d in os.listdir(embeddings_dir)
        if os.path.isdir(os.path.join(embeddings_dir, d))
    ]
    for source in sources:
        source_dir = os.path.join(embeddings_dir, source)
        batch_files = glob.glob(os.path.join(source_dir, "batch_*.json"))
        for batch_path in batch_files:
            stats["files_scanned"] += 1
            try:
                with open(batch_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                metas = data.get("metadata", [])
                modified = False
                
                # Add progress bar for metadata processing
                with tqdm(total=len(metas), desc=f"Processing metadata in {os.path.basename(batch_path)}", unit="entry") as meta_pbar:
                    for meta in metas:
                        paper_key = f"{meta.get('source')}_{meta.get('paper_index')}"
                        cache_entry = citation_cache.get(paper_key)
                        if not cache_entry:
                            logger.info(f"No cache entry for {paper_key}")
                            meta_pbar.update(1)
                            continue
                        if (
                            meta.get("citation_count") == "pending"
                            and cache_entry.get("citation_count") is not None
                        ):
                            meta["citation_count"] = cache_entry["citation_count"]
                            modified = True
                            stats["records_updated"] += 1
                        if meta.get("journal") == "pending" and cache_entry.get("journal"):
                            meta["journal"] = cache_entry["journal"]
                            modified = True
                            stats["records_updated"] += 1
                        if meta.get("impact_factor") == "pending" and cache_entry.get(
                            "impact_factor"
                        ):
                            meta["impact_factor"] = cache_entry["impact_factor"]
                            modified = True
                            stats["records_updated"] += 1
                        meta_pbar.update(1)
                if modified:
                    with open(batch_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    stats["files_modified"] += 1
            except Exception as e:
                logger.error(f"⚠️  Failed to enrich {batch_path}: {e}")
                continue
    logger.info(
        f"Enrichment sweep complete. Scanned: {stats['files_scanned']}, Modified: {stats['files_modified']}, Records updated: {stats['records_updated']}"
    )
    return stats


def get_google_embedding(text, api_key, retry_count=0, skip_base_delay=False):
    """Get embedding with global rate limiting and exponential backoff"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "models/text-embedding-004",
        "content": {"parts": [{"text": text}]},
    }
    try:
        # Strict global paper/sec rate limit before every API call
        # wait_for_paper_rate_limit() # Removed global rate limiting
        # Add base rate limiting delay (ensures we stay under 25 req/sec)
        if RATE_LIMIT_DELAY > 0 and not skip_base_delay:
            time.sleep(RATE_LIMIT_DELAY)
        response = requests.post(
            url, headers=headers, json=data, timeout=REQUEST_TIMEOUT
        )
        increment_request_counter()

        if response.status_code == 429:
            if retry_count < MAX_RETRIES:
                logger.info(
                    f"⚠️  Rate limited (attempt {retry_count + 1}/{MAX_RETRIES}). Waiting 60s..."
                )
                time.sleep(60)
                return get_google_embedding(text, api_key, retry_count + 1)
            else:
                logger.error(f"❌ Max retries reached for rate limit")
                return None
        response.raise_for_status()
        return response.json()["embedding"]["values"]
    except requests.exceptions.Timeout:
        logger.warning(f"⚠️  Timeout for text: {text[:50]}...")
        return None
    except requests.exceptions.RequestException as e:
        if "429" in str(e):
            if retry_count < MAX_RETRIES:
                logger.info(
                    f"⚠️  Rate limited (attempt {retry_count + 1}/{MAX_RETRIES}). Waiting 60s..."
                )
                time.sleep(60)
                return get_google_embedding(text, api_key, retry_count + 1)
            else:
                logger.error(f"❌ Max retries reached for rate limit")
                return None
        else:
            logger.error(f"⚠️  API error: {e}")
            return None
    except Exception as e:
        logger.error(f"⚠️  Unexpected error: {e}")
        return None


def chunk_paragraphs(text):
    if not isinstance(text, str):
        return []

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    # Filter by length
    filtered_paragraphs = []
    for para in paragraphs:
        if MIN_CHUNK_LENGTH <= len(para) <= MAX_CHUNK_LENGTH:
            filtered_paragraphs.append(para)

    return filtered_paragraphs


def process_paper_embeddings(paper_data, api_key):
    """
    Process a single paper and generate embeddings for its paragraphs.

    Args:
        paper_data: Tuple of (idx, row, source)
        api_key: Google API key

    Returns:
        List of embedding results or None if failed
    """
    idx, row, source = paper_data
    title = row.get("title", "")
    doi = row.get("doi", "")
    abstract = row.get("abstract", "")
    author = extract_authors_enhanced(row)  # Use enhanced author extraction

    # Extract basic metadata fields
    publication_date = extract_publication_date(row)
    year = publication_date[:4] if publication_date else str(datetime.now().year)

    # Extract journal and impact factor information immediately
    journal_info = extract_journal_info(row)
    impact_factor_info = extract_impact_factor(row, journal_info)

    # Extract additional metadata fields
    additional_metadata = extract_additional_metadata(row)

    paragraphs = chunk_paragraphs(abstract)

    # Skip papers with no meaningful content
    if not paragraphs:
        return []

    results = []

    for i, para in enumerate(paragraphs):
        if not para.strip():
            continue

        embedding = get_google_embedding(para, api_key)
        if embedding is None:
            continue

        # Create comprehensive metadata object with all information
        metadata = {
            "title": title,
            "doi": doi,
            "author": author,
            "publication_date": publication_date,
            "citation_count": "pending",  # Will be updated in batch processing
            "journal": journal_info,  # Already extracted
            "impact_factor": impact_factor_info,  # Already extracted
            "source": source,
            "paper_index": idx,
            "para_idx": i,
            "chunk_length": len(para),
            "year": year,
        }

        # Add additional metadata fields if available
        metadata.update(additional_metadata)

        results.append({"embedding": embedding, "chunk": para, "metadata": metadata})

    return results


def ensure_directory(path):
    """Ensure directory exists"""
    os.makedirs(path, exist_ok=True)


def save_batch(batch_data, source, batch_num, embeddings_dir):
    """Save a batch of embeddings to a file"""
    batch_file = os.path.join(embeddings_dir, source, f"batch_{batch_num:04d}.json")

    batch_content = {
        "source": source,
        "batch_num": batch_num,
        "timestamp": datetime.now().isoformat(),
        "embeddings": batch_data["embeddings"],
        "chunks": batch_data["chunks"],
        "metadata": batch_data["metadata"],
    }

    with open(batch_file, "w", encoding="utf-8") as f:
        json.dump(batch_content, f, indent=2, ensure_ascii=False)

    return batch_file


def load_metadata(embeddings_dir):
    """Load or create metadata index"""
    metadata_file = os.path.join(embeddings_dir, "metadata.json")

    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"⚠️  Error loading metadata: {e}")

    # Return default structure if no existing file
    return {
        "created": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "total_embeddings": 0,
        "total_chunks": 0,
        "total_papers": 0,
        "sources": {},
        "batches": {},
        "processed_papers": {},  # Track processed papers by source
    }


def save_metadata(metadata, embeddings_dir):
    """Save metadata index"""
    metadata["last_updated"] = datetime.now().isoformat()
    # Convert sets to lists for JSON serialization
    if "processed_papers" in metadata:
        for source, papers in metadata["processed_papers"].items():
            if isinstance(papers, set):
                metadata["processed_papers"][source] = list(papers)
    metadata_file = os.path.join(embeddings_dir, "metadata.json")
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def get_next_batch_num(source, embeddings_dir):
    """Get the next batch number for a source"""
    source_dir = os.path.join(embeddings_dir, source)
    if not os.path.exists(source_dir):
        return 0

    existing_batches = [
        f
        for f in os.listdir(source_dir)
        if f.startswith("batch_") and f.endswith(".json")
    ]
    if not existing_batches:
        return 0

    batch_nums = [int(f.split("_")[1].split(".")[0]) for f in existing_batches]
    return max(batch_nums) + 1


def get_processed_papers(metadata, source):
    """Get set of already processed paper indices for a source"""
    processed = metadata.get("processed_papers", {}).get(source, [])
    # Convert to set (JSON saves sets as lists)
    if isinstance(processed, list):
        processed = set(processed)
    elif not isinstance(processed, set):
        processed = set()
    return processed


def mark_paper_processed(metadata, source, paper_index):
    """Mark a paper as processed"""
    if "processed_papers" not in metadata:
        metadata["processed_papers"] = {}
    if source not in metadata["processed_papers"]:
        metadata["processed_papers"][source] = set()

    # Ensure it's a set (JSON might have saved it as a list)
    if isinstance(metadata["processed_papers"][source], list):
        metadata["processed_papers"][source] = set(metadata["processed_papers"][source])

    metadata["processed_papers"][source].add(paper_index)


# Remove wait_for_paper_rate_limit and all references to paper_rate_lock, paper_timestamps, paper_rate_event, and related logic

def enrich_batch_metadata_from_citations(current_batch, citation_results):
    """Update current_batch metadata entries with citation/journal/impact values when available.

    This fills only entries that are still marked as 'pending' to avoid overwriting any pre-filled values.
    """
    try:
        if not current_batch or not citation_results:
            return
        meta_list = current_batch.get("metadata", [])
        
        # Add progress bar for metadata enrichment
        with tqdm(total=len(meta_list), desc="Enriching metadata with citations", unit="entry") as pbar:
            for meta in meta_list:
                paper_key = f"{meta.get('source')}_{meta.get('paper_index')}"
                paper_stats = citation_results.get(paper_key)
                if not paper_stats:
                    pbar.update(1)
                    continue
                # Only fill if still pending
                if meta.get("citation_count") == "pending":
                    meta["citation_count"] = paper_stats.get(
                        "citation_count", meta.get("citation_count")
                    )
                if meta.get("journal") == "pending":
                    meta["journal"] = paper_stats.get("journal", meta.get("journal"))
                if meta.get("impact_factor") == "pending":
                    meta["impact_factor"] = paper_stats.get(
                        "impact_factor", meta.get("impact_factor")
                    )
                pbar.update(1)
    except Exception as e:
        logger.error(f"⚠️  enrich_batch_metadata_from_citations error: {e}")


def optimized_parallel_process_papers(
    unprocessed_papers,
    api_key,
    current_batch,
    metadata,
    db,
    batch_num,
    embeddings_dir,
    effective_workers,
    start_time,
):
    """Optimized parallel processing that can achieve 5 papers/second while respecting rate limits"""
    db_embeddings = 0
    db_chunks = 0
    worker_delay = RATE_LIMIT_DELAY  # RATE_LIMIT_DELAY / effective_workers if effective_workers > 1 else RATE_LIMIT_DELAY
    logger.info(f"🚀 Optimized parallel processing:")
    logger.info(f"   Workers: {effective_workers}")
    logger.info(f"   Base delay: {RATE_LIMIT_DELAY}s")
    logger.info(f"   Worker delay: {worker_delay:.2f}s")
    logger.info(f"   Target: 25 papers/second")
    
    # Remove live rate plotting for cleaner output
    reset_request_counter()
    reset_papers_for_citation_processing()  # Reset citation processing collection

    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        future_to_paper = {
            executor.submit(
                process_paper_embeddings_optimized,
                paper_data,
                api_key,
                effective_workers,
                worker_delay,
            ): paper_data
            for paper_data in unprocessed_papers
        }

        # Single progress bar for embedding generation
        with tqdm(
            total=len(unprocessed_papers),
            desc=f"Generating embeddings for {db}",
            unit="paper",
        ) as pbar:
            for future in as_completed(future_to_paper):
                paper_data = future_to_paper[future]
                idx, row, source = paper_data
                error_429 = False
                try:
                    results = future.result()
                    if results:
                        # Add paper to citation processing only when it's going to be processed
                        add_paper_for_citation_processing(paper_data)
                        
                        for result in results:
                            current_batch["embeddings"].append(result["embedding"])
                            current_batch["chunks"].append(result["chunk"])
                            current_batch["metadata"].append(result["metadata"])
                            db_embeddings += 1
                            metadata["total_embeddings"] += 1
                        
                        # When batch is full, process citations immediately and save
                        if len(current_batch["embeddings"]) >= BATCH_SIZE:
                            # Process citations for the current batch only
                            citation_results = process_citations_batch(
                                get_papers_for_citation_processing()
                            )
                            
                            # Enrich current batch metadata with citation results
                            enrich_batch_metadata_from_citations(
                                current_batch, citation_results
                            )
                            
                            # Reset the collection after processing this batch
                            reset_papers_for_citation_processing()

                            # Save the enriched batch
                            batch_file = save_batch(
                                current_batch, db, batch_num, embeddings_dir
                            )
                            metadata["batches"][f"{db}_batch_{batch_num:04d}"] = {
                                "file": batch_file,
                                "embeddings": len(current_batch["embeddings"]),
                                "created": datetime.now().isoformat(),
                            }
                            current_batch = {
                                "embeddings": [],
                                "chunks": [],
                                "metadata": [],
                            }
                            batch_num += 1
                except Exception as e:
                    if "429" in str(e):
                        error_429 = True
                    # Remove verbose error printing for cleaner output
                    pass
                mark_paper_processed(metadata, db, idx)
                db_chunks += len(chunk_paragraphs(row.get("abstract", "")))
                metadata["total_chunks"] += len(
                    chunk_paragraphs(row.get("abstract", ""))
                )

                if pbar.n % SAVE_INTERVAL == 0 and pbar.n > 0:
                    save_metadata(metadata, embeddings_dir)
                
                # Clean progress bar update
                pbar.set_postfix({
                    "embeddings": db_embeddings,
                    "chunks": db_chunks,
                    "batches": batch_num
                })
                pbar.update(1)

    return db_embeddings, db_chunks, batch_num


def process_paper_embeddings_optimized(paper_data, api_key, num_workers, worker_delay):
    """
    Optimized version that processes paragraphs in parallel within a paper.

    Args:
        paper_data: Tuple of (idx, row, source)
        api_key: Google API key
        num_workers: Number of workers for parallel processing
        worker_delay: Delay for this specific worker

    Returns:
        List of embedding results or None if failed
    """
    idx, row, source = paper_data
    title = row.get("title", "")
    doi = row.get("doi", "")
    abstract = row.get("abstract", "")
    author = extract_authors_enhanced(row)  # Use enhanced author extraction

    # Extract basic metadata fields
    publication_date = extract_publication_date(row)
    year = publication_date[:4] if publication_date else str(datetime.now().year)

    # Extract journal and impact factor information immediately
    journal_info = extract_journal_info(row)
    impact_factor_info = extract_impact_factor(row, journal_info)

    # Extract additional metadata fields
    additional_metadata = extract_additional_metadata(row)

    paragraphs = chunk_paragraphs(abstract)

    # Skip papers with no meaningful content
    if not paragraphs:
        return []

    # For papers with few paragraphs, process sequentially with proper delays
    if len(paragraphs) <= 2:
        results = []
        for i, para in enumerate(paragraphs):
            if para.strip():
                # Add worker-specific delay to stagger requests
                if i > 0:
                    time.sleep(worker_delay)
                # Don't skip base delay - use proper rate limiting
                embedding = get_google_embedding(para, api_key, skip_base_delay=False)
                if embedding is not None:
                    # Create comprehensive metadata object with all information
                    metadata = {
                        "title": title,
                        "doi": doi,
                        "author": author,
                        "publication_date": publication_date,
                        "citation_count": "pending",  # Will be updated in batch processing
                        "journal": journal_info,  # Already extracted
                        "impact_factor": impact_factor_info,  # Already extracted
                        "source": source,
                        "paper_index": idx,
                        "para_idx": i,
                        "chunk_length": len(para),
                        "year": year,
                    }

                    # Add additional metadata fields if available
                    metadata.update(additional_metadata)

                    results.append(
                        {"embedding": embedding, "chunk": para, "metadata": metadata}
                    )
        return results

    # For papers with many paragraphs, process in parallel
    results = []

    # Calculate optimal delay between requests for this worker
    worker_id = hash(f"{idx}_{source}") % num_workers
    para_delay = worker_delay / len(paragraphs) if len(paragraphs) > 1 else worker_delay

    with ThreadPoolExecutor(max_workers=min(5, len(paragraphs))) as executor:
        # Submit paragraph processing tasks
        future_to_para = {}
        for i, para in enumerate(paragraphs):
            if para.strip():
                future = executor.submit(
                    get_google_embedding_with_delay,
                    para,
                    api_key,
                    para_delay * i,  # Stagger paragraph requests
                )
                future_to_para[future] = (i, para)

        # Collect results
        for future in as_completed(future_to_para):
            i, para = future_to_para[future]
            try:
                embedding = future.result()
                if embedding is not None:
                    # Create comprehensive metadata object with all information
                    metadata = {
                        "title": title,
                        "doi": doi,
                        "author": author,
                        "publication_date": publication_date,
                        "citation_count": "pending",  # Will be updated in batch processing
                        "journal": journal_info,  # Already extracted
                        "impact_factor": impact_factor_info,  # Already extracted
                        "source": source,
                        "paper_index": idx,
                        "para_idx": i,
                        "chunk_length": len(para),
                        "year": year,
                    }

                    # Add additional metadata fields if available
                    metadata.update(additional_metadata)

                    results.append(
                        {"embedding": embedding, "chunk": para, "metadata": metadata}
                    )
            except Exception as e:
                logger.error(f"⚠️  Error processing paragraph {i} in paper {idx}: {e}")

    return results


def get_google_embedding_with_delay(text, api_key, delay_offset=0):
    """Get embedding with a delay offset for staggered requests"""
    if delay_offset > 0:
        time.sleep(delay_offset)
    return get_google_embedding(text, api_key)


def legacy_sequential_process_xrvix():
    """
    Legacy sequential xrvix processing: processes all papers and paragraphs strictly sequentially,
    with optional rate limit safe mode (enforces delay between requests).
    Shows a live matplotlib line graph of papers/sec over time (rolling 5 minutes), with red markers for 429 errors.
    """
    logger.info("\n=== Legacy Sequential xrvix Processing (No Parallel, No Rate Limit) ===")
    logger.info(
        "⚠️  WARNING: This mode ignores all rate limits and processes as fast as possible by default!"
    )
    logger.info("   You may hit 429 errors if you exceed the API quota.")

    # Check if matplotlib is available for plotting
    if not MATPLOTLIB_AVAILABLE or not MATPLOTLIB_DATES_AVAILABLE:
        logger.warning("⚠️  matplotlib not available - plotting features disabled")
        logger.warning("   Running without live plotting...")
        plt = None
        mdates = None
    else:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    
    from collections import deque
    import threading
    import time
    from src.core.processing_config import PERFORMANCE_PROFILES

    # Ask user if they want rate limit safe mode
    rate_limit_safe = False
    delay = 0.0
    logger.info("\n💡 Option: Run in rate limit safe mode?")
    logger.info("   This will add a delay after each API request to avoid 429 errors.")
    while True:
        ans = input("Run in rate limit safe mode? (y/n): ").strip().lower()
        if ans in ["y", "yes"]:
            rate_limit_safe = True
            delay = PERFORMANCE_PROFILES["rate_limit_safe"]["rate_limit_delay"]
            logger.info(
                f"✅ Rate limit safe mode enabled. Delay: {delay} seconds between requests."
            )
            break
        elif ans in ["n", "no"]:
            logger.info("⚠️  Running in burst mode (no delay, may hit rate limits)")
            break
        else:
            logger.info("Please enter 'y' or 'n'.")

    # Live plot setup
    if MATPLOTLIB_AVAILABLE and MATPLOTLIB_DATES_AVAILABLE:
        try:
            plt.ion()
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.set_title("Requests/sec (rolling, last 5 min)")
            ax.set_xlabel("Time")
            ax.set_ylabel("Requests/sec")
            plotting_enabled = True
        except Exception as e:
            logger.warning(f"⚠️  Error initializing matplotlib: {e}")
            plotting_enabled = False
    else:
        plotting_enabled = False
    
    rate_times = deque()
    rate_values = deque()
    error_times = deque()
    error_values = deque()
    lock = threading.Lock()
    window_seconds = 300  # 5 minutes
    last_plot_update = [0]

    def update_plot():
        if not plotting_enabled:
            return
            
        with lock:
            now = datetime.now()
            # Remove old points
            while rate_times and (now - rate_times[0]).total_seconds() > window_seconds:
                rate_times.popleft()
                rate_values.popleft()
            while (
                error_times and (now - error_times[0]).total_seconds() > window_seconds
            ):
                error_times.popleft()
                error_values.popleft()
            ax.clear()
            ax.set_title("Requests/sec (rolling, last 5 min)")
            ax.set_xlabel("Time")
            ax.set_ylabel("Requests/sec")
            if rate_times:
                ax.plot(rate_times, rate_values, label="Requests/sec", color="blue")
            if error_times:
                ax.scatter(
                    error_times,
                    error_values,
                    color="red",
                    label="429 error",
                    marker="x",
                    s=60,
                )
            ax.legend()
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            fig.autofmt_xdate()
            plt.tight_layout()
            plt.pause(0.01)

    # Load API key
    with open("config/keys.json") as f:
        api_key = json.load(f)["GOOGLE_API_KEY"]

    # Setup directory structure
    ensure_directory(EMBEDDINGS_DIR)

    # Load or create metadata
    metadata = load_metadata(EMBEDDINGS_DIR)
    logger.info(f"📊 Loaded metadata: {metadata['total_embeddings']} total embeddings")

    for db in DUMPS:
        logger.info(f"\n🔄 Processing {db} (legacy sequential)...")
        source_dir = os.path.join(EMBEDDINGS_DIR, db)
        ensure_directory(source_dir)
        dump_paths = [
            os.path.join(DUMP_ROOT, f)
            for f in os.listdir(DUMP_ROOT)
            if f.startswith(db)
        ]
        if not dump_paths:
            logger.error(f"❌ No dump found for {db}, skipping.")
            continue
        path = sorted(dump_paths, reverse=True)[0]
        logger.info(f"📁 Using dump file: {os.path.basename(path)}")
        querier = XRXivQuery(path)
        if querier.errored:
            logger.error(f"❌ Error loading {db} dump, skipping.")
            continue
        logger.info(f"📊 Found {len(querier.df)} papers in {db}")
        processed_papers = get_processed_papers(metadata, db)
        logger.info(f"📊 Already processed: {len(processed_papers)} papers")
        unprocessed_papers = []
        for idx, row in querier.df.iterrows():
            if idx not in processed_papers:
                unprocessed_papers.append((idx, row, db))
        logger.info(f"📊 Remaining to process: {len(unprocessed_papers)} papers")
        if not unprocessed_papers:
            logger.info(f"✅ All papers for {db} already processed!")
            continue
        current_batch = {"embeddings": [], "chunks": [], "metadata": []}
        batch_num = get_next_batch_num(db, EMBEDDINGS_DIR)
        db_embeddings = 0
        db_chunks = 0
        start_time = time.time()
        from tqdm.auto import tqdm

        last_rate_update = time.time()
        papers_processed = 0
        error_flag = [False]

        def get_embedding_with_429_flag(*args, **kwargs):
            try:
                return get_google_embedding(*args, **kwargs)
            except Exception as e:
                if "429" in str(e):
                    error_flag[0] = True
                raise

        reset_request_counter()
        with tqdm(
            total=len(unprocessed_papers),
            desc=f"Processing {db} (legacy)",
            unit="paper",
        ) as pbar:
            for n, (idx, row, source) in enumerate(unprocessed_papers, 1):
                title = row.get("title", "")
                doi = row.get("doi", "")
                abstract = row.get("abstract", "")
                paragraphs = chunk_paragraphs(abstract)
                paper_had_429 = False
                for i, para in enumerate(paragraphs):
                    if not para.strip():
                        continue
                    # No explicit delay or rate limit unless rate_limit_safe is enabled!
                    try:
                        embedding = get_google_embedding(
                            para, api_key, skip_base_delay=True
                        )
                        if rate_limit_safe and delay > 0:
                            time.sleep(delay)
                    except Exception as e:
                        if "429" in str(e):
                            paper_had_429 = True
                            # Mark error for plotting
                            with lock:
                                error_times.append(datetime.now())
                                error_values.append(
                                    float(0.0)
                                )  # Always append float for consistency
                        continue
                    if embedding is None:
                        continue
                    current_batch["embeddings"].append(embedding)
                    current_batch["chunks"].append(para)
                    current_batch["metadata"].append(
                        {
                            "title": title,
                            "doi": doi,
                            "source": source,
                            "paper_index": idx,
                            "para_idx": i,
                            "chunk_length": len(para),
                        }
                    )
                    db_embeddings += 1
                    db_chunks += 1
                    metadata["total_embeddings"] += 1
                    metadata["total_chunks"] += 1
                    # Save batch if it reaches the size limit
                    if len(current_batch["embeddings"]) >= BATCH_SIZE:
                        batch_file = save_batch(
                            current_batch, db, batch_num, EMBEDDINGS_DIR
                        )
                        metadata["batches"][f"{db}_batch_{batch_num:04d}"] = {
                            "file": batch_file,
                            "embeddings": len(current_batch["embeddings"]),
                            "created": datetime.now().isoformat(),
                        }
                        current_batch = {"embeddings": [], "chunks": [], "metadata": []}
                        batch_num += 1
                # Mark paper as processed
                mark_paper_processed(metadata, db, idx)
                papers_processed += 1
                # Save metadata periodically
                if n % SAVE_INTERVAL == 0:
                    save_metadata(metadata, EMBEDDINGS_DIR)
                # Progress bar update and live plot update
                elapsed = time.time() - start_time
                rate = papers_processed / elapsed if elapsed > 0 else 0
                now = datetime.now()
                with lock:
                    rate_times.append(now)
                    rate_values.append(get_requests_per_sec())
                if paper_had_429:
                    with lock:
                        error_times.append(now)
                        error_values.append(
                            float(rate)
                        )  # Always append float for consistency
                pbar.set_postfix(
                    {
                        "req_rate": f"{get_requests_per_sec():.1f} reqs/s",
                        "delay": f"{delay:.2f}s" if rate_limit_safe else "burst",
                    }
                )
                pbar.update(1)
                # Update plot every 2 seconds
                if time.time() - last_plot_update[0] > 2:
                    # Use requests/sec for plotting
                    now = datetime.now()
                    with lock:
                        rate_times.append(now)
                        rate_values.append(get_requests_per_sec())
                    update_plot()
                    last_plot_update[0] = time.time()
        # Final plot update
        update_plot()
        # Save any remaining embeddings in the current batch
        if current_batch["embeddings"]:
            batch_file = save_batch(current_batch, db, batch_num, EMBEDDINGS_DIR)
            metadata["batches"][f"{db}_batch_{batch_num:04d}"] = {
                "file": batch_file,
                "embeddings": len(current_batch["embeddings"]),
                "created": datetime.now().isoformat(),
            }
        total_processed = len(get_processed_papers(metadata, db))
        metadata["sources"][db] = {
            "papers": len(querier.df),
            "processed_papers": total_processed,
            "chunks": db_chunks,
            "embeddings": db_embeddings,
            "batches": batch_num + 1,
        }
        metadata["total_papers"] += len(querier.df)
        save_metadata(metadata, EMBEDDINGS_DIR)
        processing_time = time.time() - start_time
        papers_per_second = (
            len(unprocessed_papers) / processing_time if processing_time > 0 else 0
        )
        logger.info(
            f"✅ Finished {db}: {db_chunks} chunks processed, {db_embeddings} embeddings created in {batch_num + 1} batches"
        )
        logger.info(f"⏱️  Processing time: {processing_time/3600:.1f} hours")
        logger.info(f"🚀 Processing rate: {papers_per_second:.1f} papers/second")
    logger.info("\n🎉 All dumps processed (legacy sequential mode)!")
    if plotting_enabled:
        logger.info("\nClose the plot window to finish.")
        plt.ioff()
        plt.show()
    else:
        logger.info("\nPlotting was disabled - processing complete.")


def sequential_process_papers(
    unprocessed_papers,
    api_key,
    current_batch,
    metadata,
    db,
    batch_num,
    embeddings_dir,
    start_time,
):
    """Sequential processing for when parallel processing fails or is disabled."""
    db_embeddings = 0
    db_chunks = 0
    error_429 = False

    # Single progress bar for embedding generation
    with tqdm(
        total=len(unprocessed_papers),
        desc=f"Generating embeddings for {db} (sequential)",
        unit="paper",
    ) as pbar:
        for paper_data in unprocessed_papers:
            idx, row, source = paper_data
            
            try:
                results = process_paper_embeddings_optimized(
                    paper_data, api_key, 1, RATE_LIMIT_DELAY
                )
                if results:
                    # Add paper to citation processing
                    add_paper_for_citation_processing(paper_data)
                    
                    for result in results:
                        current_batch["embeddings"].append(result["embedding"])
                        current_batch["chunks"].append(result["chunk"])
                        current_batch["metadata"].append(result["metadata"])
                        db_embeddings += 1
                        metadata["total_embeddings"] += 1
                    
                    # When batch is full, process citations immediately and save
                    if len(current_batch["embeddings"]) >= BATCH_SIZE:
                        # Process citations for the current batch only
                        citation_results = process_citations_batch(
                            get_papers_for_citation_processing()
                        )
                        
                        # Enrich current batch metadata with citation results
                        enrich_batch_metadata_from_citations(
                            current_batch, citation_results
                        )
                        
                        # Reset the collection after processing this batch
                        reset_papers_for_citation_processing()

                        # Save the enriched batch
                        batch_file = save_batch(
                            current_batch, db, batch_num, embeddings_dir
                        )
                        metadata["batches"][f"{db}_batch_{batch_num:04d}"] = {
                            "file": batch_file,
                            "embeddings": len(current_batch["embeddings"]),
                            "created": datetime.now().isoformat(),
                        }
                        current_batch = {"embeddings": [], "chunks": [], "metadata": []}
                        batch_num += 1
                        
            except Exception as e:
                if "429" in str(e):
                    error_429 = True
                # Remove verbose error printing for cleaner output
                pass

            mark_paper_processed(metadata, db, idx)
            db_chunks += len(chunk_paragraphs(row.get("abstract", "")))
            metadata["total_chunks"] += len(chunk_paragraphs(row.get("abstract", "")))

            if pbar.n % SAVE_INTERVAL == 0 and pbar.n > 0:
                save_metadata(metadata, embeddings_dir)
            
            # Clean progress bar update
            pbar.set_postfix({
                "embeddings": db_embeddings,
                "chunks": db_chunks,
                "batches": batch_num
            })
            pbar.update(1)

    # Process any remaining papers for citations if there are papers in the current batch
    if current_batch["embeddings"]:
        citation_results = process_citations_batch(get_papers_for_citation_processing())
        enrich_batch_metadata_from_citations(current_batch, citation_results)
        reset_papers_for_citation_processing()

    return db_embeddings, db_chunks, batch_num


def get_citations_opencitations(doi):
    """Get citation count from OpenCitations API (fast and reliable)."""
    try:
        # Clean DOI
        doi = str(doi).strip()
        if not doi or not doi.startswith("10."):
            return None
            
        # OpenCitations API endpoint for citations
        url = f"https://opencitations.net/index/coci/api/v1/citations/{doi}"
        response = requests.get(url, timeout=3)  # Reduced from 10s to 3s
        
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list):
                return len(data)  # Count of papers that cite this one
            else:
                return 0
        else:
            return None
            
    except Exception:
        return None

def get_citations_pubmed_central(doi):
    """Get citation count from PubMed Central API."""
    try:
        # Clean DOI
        doi = str(doi).strip()
        if not doi or not doi.startswith("10."):
            return None
            
        # PubMed Central API endpoint
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            "db": "pmc",
            "term": doi,
            "retmode": "json",
            "retmax": 1
        }
        
        response = requests.get(url, params=params, timeout=3)  # Reduced from 10s to 3s
        if response.status_code == 200:
            data = response.json()
            if "esearchresult" in data and "count" in data["esearchresult"]:
                count = int(data["esearchresult"]["count"])
                return count
        return None
        
    except Exception:
        return None

def get_citations_pubmed_central_by_title(title):
    """Get citation count from PubMed Central API using title search."""
    try:
        # Clean title
        title = str(title).strip()
        if not title or len(title) < 10:
            return None
            
        # PubMed Central API endpoint for title search
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            "db": "pmc",
            "term": f'"{title}"[Title]',
            "retmode": "json",
            "retmax": 1
        }
        
        response = requests.get(url, params=params, timeout=3)  # Reduced from 10s to 3s
        if response.status_code == 200:
            data = response.json()
            if "esearchresult" in data and "count" in data["esearchresult"]:
                count = int(data["esearchresult"]["count"])
                return count
        return None
        
    except Exception:
        return None

def get_journal_info_crossref(doi):
    """Get enhanced journal information from CrossRef API."""
    try:
        # Clean DOI
        doi = str(doi).strip()
        if not doi or not doi.startswith("10."):
            return None
            
        # CrossRef API endpoint
        url = f"https://api.crossref.org/works/{doi}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            work = data.get("message", {})
            
            # Get journal title
            journal_title = work.get("container-title", [None])[0]
            if journal_title:
                return journal_title
                
            # Get publication venue
            venue = work.get("venue", [None])[0]
            if venue:
                return venue
                
        return None
        
    except Exception:
        return None

def get_journal_info_pubmed_central(doi):
    """Get enhanced journal information from PubMed Central API."""
    try:
        # Clean DOI
        doi = str(doi).strip()
        if not doi or not doi.startswith("10."):
            return None
            
        # PubMed Central API endpoint
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        params = {
            "db": "pmc",
            "id": doi,
            "retmode": "json"
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "result" in data and doi in data["result"]:
                paper_data = data["result"][doi]
                journal = paper_data.get("fulljournalname", paper_data.get("journal", None))
                if journal:
                    return journal
                    
        return None
        
    except Exception:
        return None

def get_impact_factor_crossref(doi):
    """Get impact factor from CrossRef API metadata."""
    try:
        # Clean DOI
        doi = str(doi).strip()
        if not doi or not doi.startswith("10."):
            return None
            
        # CrossRef API endpoint
        url = f"https://api.crossref.org/works/{doi}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            work = data.get("message", {})
            
            # Look for impact factor in various metadata fields
            # Note: CrossRef doesn't directly provide impact factors, but we can look for
            # journal information that might help with estimation
            journal_title = work.get("container-title", [None])[0]
            if journal_title:
                # Try to get impact factor from our estimation database
                return estimate_impact_factor(journal_title)
                
        return None
        
    except Exception:
        return None

def get_impact_factor_pubmed_central(doi):
    """Get impact factor from PubMed Central API metadata."""
    try:
        # Clean DOI
        doi = str(doi).strip()
        if not doi or not doi.startswith("10."):
            return None
            
        # PubMed Central API endpoint
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        params = {
            "db": "pmc",
            "id": doi,
            "retmode": "json"
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "result" in data and doi in data["result"]:
                paper_data = data["result"][doi]
                journal = paper_data.get("fulljournalname", paper_data.get("journal", None))
                if journal:
                    # Try to get impact factor from our estimation database
                    return estimate_impact_factor(journal)
                    
        return None
        
    except Exception:
        return None

def get_impact_factor_by_journal_crossref(journal_name):
    """Get impact factor by searching CrossRef for journal information."""
    try:
        # Clean journal name
        journal_name = str(journal_name).strip()
        if not journal_name or journal_name == "Unknown journal":
            return None
            
        # CrossRef API endpoint for journal search
        url = f"https://api.crossref.org/journals"
        params = {
            "query": journal_name,
            "rows": 1
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            items = data.get("message", {}).get("items", [])
            if items:
                # Try to get impact factor from our estimation database
                return estimate_impact_factor(journal_name)
                
        return None
        
    except Exception:
        return None

def extract_additional_metadata(paper_data):
    """Extract additional metadata fields from paper data."""
    additional_metadata = {}
    
    # Extract various metadata fields
    fields_to_extract = [
        "keywords", "subject", "category", "license", "language", "publisher",
        "issn", "isbn", "volume", "issue", "pages", "series"
    ]
    
    for field in fields_to_extract:
        if field in paper_data and paper_data[field]:
            value = paper_data[field]
            if isinstance(value, (list, tuple)):
                additional_metadata[field] = ", ".join(str(v) for v in value)
            else:
                additional_metadata[field] = str(value)
    
    return additional_metadata

def reset_citation_error_counter():
    """Reset the citation error counter."""
    global citation_error_counter
    citation_error_counter = {
        "doi_lookup_failures": 0,
        "title_lookup_failures": 0,
        "total_failures": 0
    }

def get_citation_error_summary():
    """Get a summary of citation processing errors."""
    global citation_error_counter
    return citation_error_counter.copy()

def reset_papers_for_citation_processing():
    """Reset the global papers collection for batch citation processing."""
    global papers_for_citation_processing
    papers_for_citation_processing = []

def add_paper_for_citation_processing(paper_data):
    """Add a paper to the global collection for batch citation processing."""
    global papers_for_citation_processing
    papers_for_citation_processing.append(paper_data)

def get_papers_for_citation_processing():
    """Get the current collection of papers for citation processing."""
    global papers_for_citation_processing
    return papers_for_citation_processing.copy()

# Initialize global variables
citation_error_counter = {
    "doi_lookup_failures": 0,
    "title_lookup_failures": 0,
    "total_failures": 0
}

papers_for_citation_processing = []

def main():
    logger.info(
        "=== Starting xrvix Dumps Processing (Multi-File Storage with Adaptive Parallel Processing) ==="
    )

    # Reset citation error counter at the start
    reset_citation_error_counter()

    logger.info(f"🔧 Static Processing Configuration:")
    logger.info(f"   Parallel workers: {MAX_WORKERS}")
    logger.info(f"   Batch size: {BATCH_SIZE}")
    logger.info(f"   Rate limit delay: {RATE_LIMIT_DELAY}s")
    logger.info(f"   Request timeout: {REQUEST_TIMEOUT}s")

    logger.info(f"   Sources: {', '.join(DUMPS)}")
    logger.info(
        f"🚦 Rate limiting: {MAX_REQUESTS_PER_MINUTE} requests per {RATE_LIMIT_WINDOW}s"
    )
    logger.info(f"   - Embedding API: {EMBEDDING_MAX_REQUESTS_PER_MINUTE} requests/minute")
    logger.info(f"   - Gemini API: {GEMINI_MAX_REQUESTS_PER_MINUTE} requests/minute")
    logger.info("")

    # Load API key
    with open("config/keys.json") as f:
        api_key = json.load(f)["GOOGLE_API_KEY"]

    # Setup directory structure
    ensure_directory(EMBEDDINGS_DIR)

    # Load or create metadata
    metadata = load_metadata(EMBEDDINGS_DIR)
    logger.info(f"📊 Loaded metadata: {metadata['total_embeddings']} total embeddings")

    try:
        for db in DUMPS:
            logger.info(f"\n🔄 Processing {db}...")

            # Create source directory
            source_dir = os.path.join(EMBEDDINGS_DIR, db)
            ensure_directory(source_dir)

            # Find dump file - look for files that start with the database name
            dump_paths = []
            if os.path.exists(DUMP_ROOT):
                dump_paths = [
                    os.path.join(DUMP_ROOT, f)
                    for f in os.listdir(DUMP_ROOT)
                    if f.startswith(db) and f.endswith('.jsonl')
                ]
            
            if not dump_paths:
                logger.error(f"❌ No dump found for {db} in {DUMP_ROOT}, skipping.")
                logger.info(f"💡 Available files: {os.listdir(DUMP_ROOT) if os.path.exists(DUMP_ROOT) else 'Directory does not exist'}")
                continue

            path = sorted(dump_paths, reverse=True)[0]
            logger.info(f"�� Using dump file: {os.path.basename(path)}")

            # Load dump
            querier = XRXivQuery(path)
            if querier.errored:
                logger.error(f"❌ Error loading {db} dump, skipping.")
                continue

            logger.info(f"📊 Found {len(querier.df)} papers in {db}")

            # Get already processed papers
            processed_papers = get_processed_papers(metadata, db)
            logger.info(f"📊 Already processed: {len(processed_papers)} papers")

            # Filter out already processed papers
            unprocessed_papers = []
            for idx, row in querier.df.iterrows():
                if idx not in processed_papers:
                    unprocessed_papers.append((idx, row, db))

            logger.info(f"📊 Remaining to process: {len(unprocessed_papers)} papers")

            if not unprocessed_papers:
                logger.info(f"✅ All papers for {db} already processed!")
                continue

            # Initialize batch tracking
            current_batch = {"embeddings": [], "chunks": [], "metadata": []}
            batch_num = get_next_batch_num(db, EMBEDDINGS_DIR)
            db_embeddings = 0
            db_chunks = 0

            start_time = time.time()

            # Use static configuration
            effective_workers = MAX_WORKERS
            logger.info(
                f"🚀 Starting parallel processing with {effective_workers} workers..."
            )

            # Check if user wants to force parallel processing (bypass automatic fallback)
            force_parallel = os.environ.get("FORCE_PARALLEL", "false").lower() == "true"

            # Only apply automatic fallback if not forcing parallel
            if not force_parallel:
                # Conservative fallback for very high rate limiting
                if RATE_LIMIT_DELAY > 2.0:  # Only for extremely conservative profiles
                    logger.info(
                        "🐌 Using sequential processing due to very high rate limiting (delay > 2s)"
                    )
                    db_embeddings, db_chunks, batch_num = sequential_process_papers(
                        unprocessed_papers,
                        api_key,
                        current_batch,
                        metadata,
                        db,
                        batch_num,
                        EMBEDDINGS_DIR,
                        start_time,
                    )
                else:
                    logger.info(
                        f"🚀 Using parallel processing with {effective_workers} workers (delay: {RATE_LIMIT_DELAY}s)"
                    )
                    db_embeddings, db_chunks, batch_num = (
                        optimized_parallel_process_papers(
                            unprocessed_papers,
                            api_key,
                            current_batch,
                            metadata,
                            db,
                            batch_num,
                            EMBEDDINGS_DIR,
                            effective_workers,
                            start_time,
                        )
                    )
            else:
                logger.info(
                    f"🚀 FORCING parallel processing with {effective_workers} workers (bypassing automatic fallback)"
                )
                db_embeddings, db_chunks, batch_num = optimized_parallel_process_papers(
                    unprocessed_papers,
                    api_key,
                    current_batch,
                    metadata,
                    db,
                    batch_num,
                    EMBEDDINGS_DIR,
                    effective_workers,
                    start_time,
                )

            # Save any remaining embeddings in the current batch
            if current_batch["embeddings"]:
                # Since papers are now only added to citation processing when they're actually processed,
                # we don't need to process citations here - they should have been processed already
                # when the batch was full, or will be processed in the next batch
                batch_file = save_batch(current_batch, db, batch_num, EMBEDDINGS_DIR)
                metadata["batches"][f"{db}_batch_{batch_num:04d}"] = {
                    "file": batch_file,
                    "embeddings": len(current_batch["embeddings"]),
                    "created": datetime.now().isoformat(),
                }

            # Update source statistics
            total_processed = len(get_processed_papers(metadata, db))
            metadata["sources"][db] = {
                "papers": len(querier.df),
                "processed_papers": total_processed,
                "chunks": db_chunks,
                "embeddings": db_embeddings,
                "batches": batch_num + 1,
            }
            metadata["total_papers"] += len(querier.df)

            # Save final metadata for this source
            save_metadata(metadata, EMBEDDINGS_DIR)

            processing_time = time.time() - start_time
            papers_per_second = (
                len(unprocessed_papers) / processing_time if processing_time > 0 else 0
            )

            logger.info(
                f"✅ Finished {db}: {db_chunks} chunks processed, {db_embeddings} embeddings created in {batch_num + 1} batches"
            )
            logger.info(f"⏱️  Processing time: {processing_time/3600:.1f} hours")
            logger.info(f"🚀 Processing rate: {papers_per_second:.1f} papers/second")

        # Persist citation cache for backfill/enrichment reuse
        save_citation_cache(EMBEDDINGS_DIR)

        # End-of-process enrichment sweep to ensure all batches have final data
        enrich_all_batches_metadata_from_citations(
            EMBEDDINGS_DIR, citation_cache=CITATION_CACHE
        )

        # Print final statistics
        logger.info(f"\n🎉 All dumps processed!")
        logger.info(f"📊 Total embeddings: {metadata['total_embeddings']}")
        logger.info(f"📊 Total chunks: {metadata['total_chunks']}")
        logger.info(f"📊 Total papers: {metadata['total_papers']}")
        logger.info(f"📁 Embeddings directory: {os.path.abspath(EMBEDDINGS_DIR)}")

        # Calculate total file size
        total_size = 0
        for root, dirs, files in os.walk(EMBEDDINGS_DIR):
            for file in files:
                total_size += os.path.getsize(os.path.join(root, file))
        logger.info(f"📊 Total size: {total_size / 1024 / 1024:.1f} MB")

        # Print source breakdown
        logger.info(f"\n📊 Source Breakdown:")
        for source, stats in metadata["sources"].items():
            logger.info(
                f"   {source}: {stats['papers']} papers, {stats.get('processed_papers', 0)} processed, {stats['embeddings']} embeddings, {stats['batches']} batches"
            )

        # Print citation processing summary
        citation_summary = get_citation_error_summary()
        if citation_summary["total_failures"] > 0:
            logger.info(f"\n📊 Citation Processing Summary:")
            logger.info(f"   DOI lookup failures: {citation_summary['doi_lookup_failures']}")
            logger.info(
                f"   Title lookup failures: {citation_summary['title_lookup_failures']}"
            )
            logger.info(
                f"   Total citation processing failures: {citation_summary['total_failures']}"
            )
        else:
            logger.info(f"\n✅ All citation lookups completed successfully!")

    except KeyboardInterrupt:
        logger.warning(f"\n⚠️  Processing interrupted by user.")
        # Print citation processing summary even on interruption
        citation_summary = get_citation_error_summary()
        if citation_summary["total_failures"] > 0:
            logger.info(f"\n📊 Citation Processing Summary (at interruption):")
            logger.info(f"   DOI lookup failures: {citation_summary['doi_lookup_failures']}")
            logger.info(
                f"   Title lookup failures: {citation_summary['title_lookup_failures']}"
            )
            logger.info(
                f"   Total citation processing failures: {citation_summary['total_failures']}"
            )
        else:
            logger.info(f"\n✅ All citation lookups completed successfully!")

    finally:
        # Clean up any resources if needed
        pass


if __name__ == "__main__":
    main()

def get_impact_factor_openalex(doi):
    """Get impact factor information from OpenAlex API using DOI."""
    logger.info(f"🔍 DEBUG: Calling OpenAlex API for impact factor with DOI: {doi}")
    
    try:
        # Clean DOI
        doi = str(doi).strip()
        if not doi or not doi.startswith("10."):
            logger.info(f"🔍 DEBUG: Invalid DOI: {doi}")
            return None
            
        # OpenAlex API endpoint
        url = f"https://api.openalex.org/works/doi:{doi}"
        logger.info(f"🔍 DEBUG: Calling OpenAlex API URL: {url}")
        
        response = requests.get(url, timeout=10)
        logger.info(f"🔍 DEBUG: OpenAlex API response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"🔍 DEBUG: Successfully parsed OpenAlex API response")
            
            # Get journal information for impact factor estimation
            if 'host_venue' in data and data['host_venue']:
                venue = data['host_venue']
                journal_name = venue.get('display_name') or venue.get('title')
                if journal_name:
                    logger.info(f"🔍 DEBUG: Found journal in OpenAlex: {journal_name}")
                    logger.info(f"🔍 DEBUG: Estimating impact factor for journal")
                    return estimate_impact_factor(journal_name)
                else:
                    logger.info(f"🔍 DEBUG: No journal name found in OpenAlex host_venue")
            else:
                logger.info(f"🔍 DEBUG: No host_venue found in OpenAlex response")
                    
        else:
            logger.info(f"🔍 DEBUG: OpenAlex API returned error status: {response.status_code}")
            logger.info(f"🔍 DEBUG: Response content: {response.text[:200]}...")
                    
        return None
        
    except Exception as e:
        logger.error(f"🔍 DEBUG: Error calling OpenAlex API: {e}")
        return None

def get_impact_factor_semantic_scholar(doi):
    """Get impact factor information from Semantic Scholar API using DOI."""
    logger.info(f"🔍 DEBUG: Calling Semantic Scholar API for impact factor with DOI: {doi}")
    
    try:
        # Clean DOI
        doi = str(doi).strip()
        if not doi or not doi.startswith("10."):
            logger.info(f"🔍 DEBUG: Invalid DOI: {doi}")
            return None
            
        # Semantic Scholar API endpoint
        url = f"https://api.semanticscholar.org/v1/paper/{doi}"
        logger.info(f"🔍 DEBUG: Calling Semantic Scholar API URL: {url}")
        
        response = requests.get(url, timeout=10)
        logger.info(f"🔍 DEBUG: Semantic Scholar API response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"🔍 DEBUG: Successfully parsed Semantic Scholar API response")
            
            # Get journal information for impact factor estimation
            if 'venue' in data and data['venue']:
                journal_name = data['venue']
                logger.info(f"🔍 DEBUG: Found venue in Semantic Scholar: {journal_name}")
                logger.info(f"🔍 DEBUG: Estimating impact factor for journal")
                return estimate_impact_factor(journal_name)
            else:
                logger.info(f"🔍 DEBUG: No venue found in Semantic Scholar response")
                
        else:
            logger.info(f"🔍 DEBUG: Semantic Scholar API returned error status: {response.status_code}")
            logger.info(f"🔍 DEBUG: Response content: {response.text[:200]}...")
                
        return None
        
    except Exception as e:
        logger.error(f"🔍 DEBUG: Error calling Semantic Scholar API: {e}")
        return None

def is_preprint_paper(paper_data):
    """
    Determine if a paper is a preprint based on various indicators.
    
    Args:
        paper_data: Dictionary containing paper metadata
        
    Returns:
        tuple: (is_preprint: bool, preprint_type: str, preprint_score: float)
    """
    preprint_score = 0.0
    preprint_type = "unknown"
    
    # Check source field first (most reliable)
    source = paper_data.get("source", "").lower()
    if source in ["biorxiv", "medrxiv", "arxiv", "chemrxiv", "preprints"]:
        preprint_score += 0.8
        preprint_type = source
    
    # Check journal name for preprint indicators
    journal = paper_data.get("journal", "").lower()
    if journal:
        preprint_indicators = [
            "arxiv", "biorxiv", "medrxiv", "chemrxiv", "preprint", 
            "working paper", "manuscript", "draft"
        ]
        for indicator in preprint_indicators:
            if indicator in journal:
                preprint_score += 0.6
                preprint_type = "preprint_server"
                break
    
    # Check content type (from paperscraper dumps)
    content_type = paper_data.get("contentType", {}).get("name", "").lower()
    if content_type in ["working paper", "preprint", "manuscript"]:
        preprint_score += 0.7
        preprint_type = "working_paper"
    
    # Check if it's from a preprint server origin
    origin = paper_data.get("origin", "").lower()
    if origin in ["biorxiv", "medrxiv", "chemrxiv", "preprints"]:
        preprint_score += 0.9
        preprint_type = origin
    
    # Check version information
    version = paper_data.get("version", "")
    if version and version != "1":
        preprint_score += 0.3
        preprint_type = "versioned_preprint"
    
    # Check status (from paperscraper dumps)
    status = paper_data.get("status", "").lower()
    if status in ["submitted", "pending", "working"]:
        preprint_score += 0.4
        preprint_type = "submitted_preprint"
    
    # Determine final classification
    is_preprint = preprint_score >= 0.5
    
    return is_preprint, preprint_type, preprint_score

def load_paperscraper_dumps():
    """
    Load all paperscraper server dumps into memory for fast lookup.
    
    Returns:
        dict: Dictionary mapping DOI to paper metadata
    """
    dump_dir = os.path.join("AHG", "Lib", "site-packages", "paperscraper", "server_dumps")
    dump_cache = {}
    
    if not os.path.exists(dump_dir):
        logger.warning(f"Paperscraper dumps directory not found: {dump_dir}")
        return dump_cache
    
    try:
        dump_files = glob.glob(os.path.join(dump_dir, "*.json"))
        logger.info(f"Loading {len(dump_files)} paperscraper dump files...")
        
        for dump_file in tqdm(dump_files, desc="Loading paperscraper dumps"):
            try:
                with open(dump_file, 'r', encoding='utf-8') as f:
                    paper_data = json.load(f)
                    
                # Extract DOI as key
                doi = paper_data.get("doi")
                if doi:
                    dump_cache[doi] = paper_data
                    
            except Exception as e:
                logger.error(f"Error loading dump file {dump_file}: {e}")
                continue
        
        logger.info(f"Successfully loaded {len(dump_cache)} papers from dumps")
        return dump_cache
        
    except Exception as e:
        logger.error(f"Error loading paperscraper dumps: {e}")
        return dump_cache

def extract_from_paperscraper_dump(doi, dump_cache):
    """
    Extract metadata from paperscraper dumps using DOI.
    
    Args:
        doi: DOI string
        dump_cache: Dictionary of loaded dump data
        
    Returns:
        dict: Extracted metadata or None if not found
    """
    if not doi or not dump_cache:
        return None
    
    # Clean DOI
    doi = str(doi).strip()
    if not doi.startswith("10."):
        return None
    
    # Look up in dump cache
    paper_data = dump_cache.get(doi)
    if not paper_data:
        return None
    
    # Extract relevant metadata
    extracted_data = {}
    
    # Extract citation count
    metrics = paper_data.get("metrics", [])
    for metric in metrics:
        if metric.get("description") == "Citations":
            citation_count = metric.get("value", 0)
            if citation_count is not None:
                extracted_data["citation_count"] = str(citation_count)
            break
    
    # Extract publication date
    published_date = paper_data.get("publishedDate") or paper_data.get("submittedDate")
    if published_date:
        try:
            # Parse ISO date and extract year
            date_obj = datetime.fromisoformat(published_date.replace('Z', '+00:00'))
            extracted_data["publication_date"] = date_obj.strftime("%Y-%m-%d")
            extracted_data["publication_year"] = str(date_obj.year)
        except Exception:
            pass
    
    # Extract authors
    authors = paper_data.get("authors", [])
    if authors:
        author_names = []
        for author in authors:
            first_name = author.get("firstName", "").strip()
            last_name = author.get("lastName", "").strip()
            if first_name and last_name:
                author_names.append(f"{first_name} {last_name}")
            elif last_name:
                author_names.append(last_name)
        
        if author_names:
            extracted_data["authors"] = author_names
            extracted_data["author"] = "; ".join(author_names)
    
    # Extract journal/venue information
    content_type = paper_data.get("contentType", {}).get("name", "")
    if content_type:
        extracted_data["journal"] = content_type
    
    # Extract preprint status
    is_preprint, preprint_type, preprint_score = is_preprint_paper(paper_data)
    extracted_data["is_preprint"] = is_preprint
    extracted_data["preprint_type"] = preprint_type
    extracted_data["preprint_score"] = preprint_score
    
    # Extract categories/subjects
    categories = paper_data.get("categories", [])
    if categories:
        category_names = [cat.get("name", "") for cat in categories if cat.get("name")]
        if category_names:
            extracted_data["categories"] = category_names
    
    subject = paper_data.get("subject", {}).get("name", "")
    if subject:
        extracted_data["subject"] = subject
    
    return extracted_data

# Global cache for paperscraper dumps
PAPERSCRAPER_DUMP_CACHE = None

def get_paperscraper_dump_cache():
    """Get or create the paperscraper dump cache."""
    global PAPERSCRAPER_DUMP_CACHE
    if PAPERSCRAPER_DUMP_CACHE is None:
        PAPERSCRAPER_DUMP_CACHE = load_paperscraper_dumps()
    return PAPERSCRAPER_DUMP_CACHE

def extract_citation_count(paper_data):
    """Extract citation count using paperscraper dumps first, then API fallbacks."""
    global citation_error_counter

    # Check if this is a preprint first - skip expensive API calls for preprints
    # Quick check for bioRxiv DOIs (10.1101/) and medRxiv DOIs (10.1101/)
    doi = paper_data.get("doi", "")
    if doi.startswith("10.1101/"):
        logger.info(f"🔍 DEBUG: bioRxiv DOI detected, returning citation count: 0")
        return "0"  # Preprints typically have minimal citations
    
    is_preprint, preprint_type, preprint_score = is_preprint_paper(paper_data)
    if is_preprint:
        logger.info(f"🔍 DEBUG: Preprint detected, returning citation count: 0")
        return "0"  # Preprints typically have minimal citations

    # First, check if citation data is already in the paper_data (fastest)
    citation_fields = [
        "citations", "citation_count", "cited_by_count", "times_cited",
        "reference_count", "cited_count", "num_citations", "citedByCount"
    ]
    
    for field in citation_fields:
        if field in paper_data and paper_data[field] is not None:
            try:
                count = int(paper_data[field])
                if count >= 0:
                    return str(count)
            except (ValueError, TypeError):
                continue

    # Try to get citation count from paperscraper dumps first
    doi = paper_data.get("doi", "")
    if doi:
        # Clean and validate DOI
        doi = str(doi).strip()
        if not doi or doi.lower() in ["nan", "none", "null", ""]:
            doi = ""
        elif not doi.startswith("10."):
            doi = ""

    if doi:
        # Try paperscraper dumps first (fastest and most reliable)
        dump_cache = get_paperscraper_dump_cache()
        if dump_cache:
            extracted_data = extract_from_paperscraper_dump(doi, dump_cache)
            if extracted_data and "citation_count" in extracted_data:
                return extracted_data["citation_count"]
        
        # Fallback to OpenCitations API if not found in dumps
        try:
            citations = get_citations_opencitations(doi)
            if citations is not None:
                return str(citations)
        except Exception:
            pass
        
        # Try PubMed Central API as second fallback
        try:
            citations = get_citations_pubmed_central(doi)
            if citations is not None:
                return str(citations)
        except Exception:
            pass
        
        # Final fallback to paperscraper API if other methods fail
        if PAPERSCRAPER_AVAILABLE:
            try:
                citations = get_citations_by_doi(doi)
                if citations is not None and citations >= 0:
                    return str(citations)
            except Exception as e:
                citation_error_counter["doi_lookup_failures"] += 1
                citation_error_counter["total_failures"] += 1

    # Only try title-based lookup if DOI failed and we have a good title
    title = paper_data.get("title", "")
    if title:
        title = str(title).strip()
        if not title or title.lower() in ["nan", "none", "null", ""]:
            title = ""
        elif len(title) < 10:
            title = ""

    if title:
        # Try PubMed Central API for title-based search
        try:
            citations = get_citations_pubmed_central_by_title(title)
            if citations is not None:
                return str(citations)
        except Exception:
            pass
        
        # Fallback to paperscraper if available
        if PAPERSCRAPER_AVAILABLE:
            try:
                citations = get_citations_from_title(title)
                if citations is not None and citations >= 0:
                    return str(citations)
            except Exception as e:
                citation_error_counter["title_lookup_failures"] += 1
                citation_error_counter["total_failures"] += 1

    # Final fallback: check for any other citation-related fields
    for field in paper_data.keys():
        if "citation" in field.lower() or "cited" in field.lower():
            try:
                value = paper_data[field]
                if isinstance(value, (int, float)) and value >= 0:
                    return str(int(value))
                elif isinstance(value, str):
                    import re
                    numbers = re.findall(r"\d+", value)
                    if numbers:
                        return str(int(numbers[0]))
            except (ValueError, TypeError):
                continue

    return "not found"

def extract_publication_date(paper_data):
    """
    Extract publication date using paperscraper dumps first, then fallbacks.
    
    Args:
        paper_data: Dictionary containing paper metadata
        
    Returns:
        str: Publication date in YYYY-MM-DD format or "not found"
    """
    # First check if date is already in the paper_data
    date_fields = [
        "publication_date", "published_date", "date", "pub_date", 
        "publicationDate", "publishedDate", "submittedDate"
    ]
    
    for field in date_fields:
        if field in paper_data and paper_data[field]:
            date_value = str(paper_data[field]).strip()
            if date_value and date_value.lower() not in ["nan", "none", "null", ""]:
                try:
                    # Try to parse and format the date
                    if len(date_value) >= 4:  # At least year
                        if len(date_value) == 4:  # Just year
                            return f"{date_value}-01-01"
                        elif len(date_value) >= 7:  # Year-month or more
                            # Try to parse ISO format or other common formats
                            try:
                                date_obj = datetime.fromisoformat(date_value.replace('Z', '+00:00'))
                                return date_obj.strftime("%Y-%m-%d")
                            except:
                                # Try to extract year-month-day
                                import re
                                date_parts = re.findall(r'\d{4}-\d{2}-\d{2}', date_value)
                                if date_parts:
                                    return date_parts[0]
                                # Try to extract just year
                                year_match = re.search(r'\d{4}', date_value)
                                if year_match:
                                    return f"{year_match.group()}-01-01"
                except Exception:
                    continue
    
    # Try to get date from paperscraper dumps using DOI
    doi = paper_data.get("doi", "")
    if doi:
        dump_cache = get_paperscraper_dump_cache()
        if dump_cache:
            extracted_data = extract_from_paperscraper_dump(doi, dump_cache)
            if extracted_data and "publication_date" in extracted_data:
                return extracted_data["publication_date"]
    
    return "not found"

def extract_authors(paper_data):
    """
    Extract authors using paperscraper dumps first, then fallbacks.
    
    Args:
        paper_data: Dictionary containing paper metadata
        
    Returns:
        str: Semicolon-separated author names or "not found"
    """
    # First check if authors are already in the paper_data
    author_fields = [
        "authors", "author", "author_names", "author_list", 
        "creator", "creators", "contributors"
    ]
    
    for field in author_fields:
        if field in paper_data and paper_data[field]:
            authors = paper_data[field]
            if isinstance(authors, list):
                # Handle list of author dictionaries or strings
                author_names = []
                for author in authors:
                    if isinstance(author, dict):
                        # Extract from author dictionary
                        first_name = author.get("firstName", "").strip()
                        last_name = author.get("lastName", "").strip()
                        if first_name and last_name:
                            author_names.append(f"{first_name} {last_name}")
                        elif last_name:
                            author_names.append(last_name)
                        elif "name" in author:
                            author_names.append(str(author["name"]).strip())
                    elif isinstance(author, str):
                        author_names.append(author.strip())
                
                if author_names:
                    return "; ".join(author_names)
            elif isinstance(authors, str):
                # Handle string format
                authors_str = str(authors).strip()
                if authors_str and authors_str.lower() not in ["nan", "none", "null", ""]:
                    return authors_str
    
    # Try to get authors from paperscraper dumps using DOI
    doi = paper_data.get("doi", "")
    if doi:
        dump_cache = get_paperscraper_dump_cache()
        if dump_cache:
            extracted_data = extract_from_paperscraper_dump(doi, dump_cache)
            if extracted_data and "author" in extracted_data:
                return extracted_data["author"]
    
    return "not found"
