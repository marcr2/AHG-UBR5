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
import urllib.parse
import xml.etree.ElementTree as ET
import concurrent.futures
import threading
from functools import partial

# Suppress paperscraper warnings
warnings.filterwarnings("ignore", message="Could not find paper")
warnings.filterwarnings("ignore", category=UserWarning, module="paperscraper")

# Global counter for citation processing errors
citation_error_counter = {
    'doi_lookup_failures': 0,
    'title_lookup_failures': 0,
    'total_failures': 0,
    'successful_citations': 0,
    'failed_citations': 0
}

# Thread-local storage for rate limiting
thread_local = threading.local()

def get_rate_limiter():
    """Get or create a rate limiter for the current thread."""
    if not hasattr(thread_local, 'last_request_time'):
        thread_local.last_request_time = 0
    return thread_local.last_request_time

def set_rate_limiter(timestamp):
    """Set the last request time for the current thread."""
    thread_local.last_request_time = timestamp

def rate_limit_delay(min_delay=0.1):
    """Implement rate limiting to avoid overwhelming APIs."""
    current_time = time.time()
    last_request = get_rate_limiter()
    
    if current_time - last_request < min_delay:
        sleep_time = min_delay - (current_time - last_request)
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    set_rate_limiter(time.time())

def reset_citation_error_counter():
    """Reset the global citation error counter."""
    global citation_error_counter
    citation_error_counter = {
        'doi_lookup_failures': 0,
        'title_lookup_failures': 0,
        'total_failures': 0,
        'successful_citations': 0,
        'failed_citations': 0
    }

def get_citation_error_summary():
    """Get a summary of citation processing errors."""
    global citation_error_counter
    return citation_error_counter.copy()

def search_pubmed_comprehensive(search_terms, max_results=10000, date_from="1900", date_to=None):
    """
    Optimized PubMed search using efficient strategies to maximize paper discovery.
    
    Args:
        search_terms: List of search terms
        max_results: Maximum number of results to retrieve
        date_from: Start date for search (YYYY or YYYY-MM-DD)
        date_to: End date for search (YYYY or YYYY-MM-DD), defaults to current date
    
    Returns:
        List of paper dictionaries
    """
    if date_to is None:
        date_to = datetime.now().strftime("%Y-%m-%d")
    
    print(f"🔍 Starting optimized PubMed search...")
    print(f"   Date range: {date_from} to {date_to}")
    print(f"   Target results: {max_results}")
    
    all_papers = []
    
    # Strategy 1: Combined search for all UBR5 variants (most efficient)
    ubr5_variants = " OR ".join([f'"{term}"' for term in search_terms])
    combined_query = f"({ubr5_variants})[Title/Abstract]"
    
    # Strategy 2: Broader search including MeSH terms
    mesh_query = f"({ubr5_variants})[MeSH Terms]"
    
    # Strategy 3: Title-only search for precision
    title_query = f"({ubr5_variants})[Title]"
    
    search_strategies = [
        combined_query,      # Most comprehensive
        mesh_query,          # MeSH terms for broader coverage
        title_query          # Title-only for precision
    ]
    
    print(f"📋 Generated {len(search_strategies)} efficient search strategies")
    
    # Execute searches with progress tracking
    with tqdm(total=len(search_strategies), desc="Executing search strategies", unit="strategy") as pbar:
        for i, search_query in enumerate(search_strategies):
            try:
                # Add date filters to the search query
                date_filter = f" AND ({date_from}[Date - Publication]:{date_to}[Date - Publication])"
                full_query = search_query + date_filter
                
                print(f"\n🔍 Strategy {i+1}: {search_query}")
                
                # Use paperscraper's function but with better parameters
                papers = get_and_dump_pubmed_papers([[full_query]], f"temp_search_{i}.jsonl")
                
                # Load the temporary results
                if os.path.exists(f"temp_search_{i}.jsonl"):
                    with open(f"temp_search_{i}.jsonl", 'r', encoding='utf-8') as f:
                        temp_papers = [json.loads(line) for line in f if line.strip()]
                    
                    print(f"   📊 Found {len(temp_papers)} papers")
                    
                    # Validate paper data before adding
                    valid_papers = []
                    skipped_count = 0
                    for paper in temp_papers:
                        # Check if paper has required fields
                        if paper and isinstance(paper, dict):
                            title = paper.get('title', '')
                            abstract = paper.get('abstract', '')
                            
                            # Only add papers with both title and abstract
                            if title and abstract and title.strip() and abstract.strip():
                                valid_papers.append(paper)
                            else:
                                skipped_count += 1
                        else:
                            skipped_count += 1
                    
                    print(f"   ✅ Valid papers: {len(valid_papers)} out of {len(temp_papers)} (skipped {skipped_count})")
                    
                    # Add unique valid papers to our collection
                    for paper in valid_papers:
                        if paper not in all_papers:
                            all_papers.append(paper)
                    
                    # Clean up temporary file
                    os.remove(f"temp_search_{i}.jsonl")
                
                # Reduced delay - only 0.5 seconds between strategies
                time.sleep(0.5)
                
            except Exception as e:
                print(f"\n⚠️  Search strategy {i+1} failed: {e}")
                continue
            
            pbar.update(1)
            pbar.set_postfix({"papers_found": len(all_papers)})
            
            # Early exit if we have enough results
            if len(all_papers) >= max_results:
                print(f"\n✅ Reached target of {max_results} papers, stopping search")
                break
    
    # Remove duplicates based on DOI and title
    unique_papers = []
    seen_dois = set()
    seen_titles = set()
    
    for paper in all_papers:
        try:
            # Safely extract DOI and title with defensive programming
            doi = paper.get('doi', '')
            title = paper.get('title', '')
            
            # Handle None values safely
            if doi is None:
                doi = ""
            if title is None:
                title = ""
            
            # Convert to strings and strip
            doi = str(doi).strip()
            title = str(title).strip().lower()
            
            # Check if we've seen this paper before
            if doi and doi not in seen_dois:
                unique_papers.append(paper)
                seen_dois.add(doi)
            elif title and title not in seen_titles:
                unique_papers.append(paper)
                seen_titles.add(title)
        except Exception as e:
            print(f"⚠️  Error processing paper for deduplication: {e}")
            # Add the paper anyway to avoid losing data
            unique_papers.append(paper)
            continue
    
    print(f"🎯 Found {len(all_papers)} total papers, {len(unique_papers)} unique papers")
    return unique_papers

def search_pubmed_direct_api(search_terms, max_results=10000, date_from="1900", date_to=None):
    """
    Optimized fallback PubMed search using direct API calls for maximum paper discovery.
    
    Args:
        search_terms: List of search terms
        max_results: Maximum number of results to retrieve
        date_from: Start date for search (YYYY or YYYY-MM-DD)
        date_to: End date for search (YYYY or YYYY-MM-DD), defaults to current date
    
    Returns:
        List of paper dictionaries
    """
    if date_to is None:
        date_to = datetime.now().strftime("%Y-%m-%d")
    
    print(f"🔍 Starting optimized direct PubMed API search...")
    print(f"   Date range: {date_from} to {date_to}")
    print(f"   Target results: {max_results}")
    
    all_papers = []
    
    # PubMed E-utilities base URL
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    
    # Use the same efficient search strategies as the comprehensive search
    ubr5_variants = " OR ".join([f'"{term}"' for term in search_terms])
    
    # Strategy 1: Combined search for all UBR5 variants (most efficient)
    combined_query = f"({ubr5_variants})[Title/Abstract]"
    
    # Strategy 2: Broader search including MeSH terms
    mesh_query = f"({ubr5_variants})[MeSH Terms]"
    
    # Strategy 3: Title-only search for precision
    title_query = f"({ubr5_variants})[Title]"
    
    search_strategies = [
        combined_query,      # Most comprehensive
        mesh_query,          # MeSH terms for broader coverage
        title_query          # Title-only for precision
    ]
    
    print(f"📋 Generated {len(search_strategies)} efficient API search strategies")
    
    # Execute searches with progress tracking
    with tqdm(total=len(search_strategies), desc="Executing API searches", unit="strategy") as pbar:
        for i, search_query in enumerate(search_strategies):
            try:
                # Add date filters to the search query
                date_filter = f" AND ({date_from}[Date - Publication]:{date_to}[Date - Publication])"
                full_query = search_query + date_filter
                
                print(f"\n🔍 API Strategy {i+1}: {search_query}")
                
                # URL encode the query
                encoded_query = urllib.parse.quote(full_query)
                
                # Step 1: Search for IDs
                search_url = f"{base_url}esearch.fcgi?db=pubmed&term={encoded_query}&retmax={max_results}&retmode=json"
                response = requests.get(search_url, timeout=30)
                response.raise_for_status()
                
                search_data = response.json()
                if 'esearchresult' not in search_data:
                    continue
                
                id_list = search_data['esearchresult'].get('idlist', [])
                if not id_list:
                    continue
                
                print(f"   📊 Strategy {i+1} found {len(id_list)} papers")
                
                # Step 2: Fetch paper details in batches
                batch_size = 100  # PubMed allows up to 100 IDs per request
                for j in range(0, len(id_list), batch_size):
                    batch_ids = id_list[j:j+batch_size]
                    id_string = ','.join(batch_ids)
                    
                    # Fetch paper details
                    fetch_url = f"{base_url}efetch.fcgi?db=pubmed&id={id_string}&retmode=xml"
                    fetch_response = requests.get(fetch_url, timeout=30)
                    fetch_response.raise_for_status()
                    
                    # Parse XML response
                    try:
                        root = ET.fromstring(fetch_response.content)
                        for article in root.findall('.//PubmedArticle'):
                            paper_data = parse_pubmed_xml(article)
                            if paper_data and paper_data not in all_papers:
                                all_papers.append(paper_data)
                    except ET.ParseError:
                        continue
                    
                    # Reduced rate limiting delay
                    time.sleep(0.1)
                
                # Reduced delay between search strategies
                time.sleep(0.5)
                
            except Exception as e:
                print(f"\n⚠️  Search strategy {i+1} failed: {e}")
                continue
            
            pbar.update(1)
            pbar.set_postfix({"papers_found": len(all_papers)})
            
            # Early exit if we have enough results
            if len(all_papers) >= max_results:
                print(f"\n✅ Reached target of {max_results} papers, stopping search")
                break
    
    # Remove duplicates based on DOI and title
    unique_papers = []
    seen_dois = set()
    seen_titles = set()
    
    for paper in all_papers:
        doi = paper.get('doi', '').strip()
        title = paper.get('title', '').strip().lower()
        
        # Check if we've seen this paper before
        if doi and doi not in seen_dois:
            unique_papers.append(paper)
            seen_dois.add(doi)
        elif title and title not in seen_titles:
            unique_papers.append(paper)
            seen_titles.add(title)
    
    print(f"🎯 Found {len(all_papers)} total papers, {len(unique_papers)} unique papers")
    return unique_papers

def parse_pubmed_xml(article_element):
    """
    Parse PubMed XML article element into paper dictionary.
    
    Args:
        article_element: XML element representing a PubMed article
    
    Returns:
        Dictionary with paper data or None if parsing fails
    """
    try:
        paper_data = {}
        
        # Extract title
        title_elem = article_element.find('.//ArticleTitle')
        if title_elem is not None and title_elem.text:
            paper_data['title'] = title_elem.text.strip()
        
        # Extract abstract
        abstract_elem = article_element.find('.//Abstract/AbstractText')
        if abstract_elem is not None and abstract_elem.text:
            paper_data['abstract'] = abstract_elem.text.strip()
        
        # Extract journal
        journal_elem = article_element.find('.//Journal/Title')
        if journal_elem is not None and journal_elem.text:
            paper_data['journal'] = journal_elem.text.strip()
        
        # Extract publication date
        pub_date_elem = article_element.find('.//PubDate')
        if pub_date_elem is not None:
            year_elem = pub_date_elem.find('Year')
            month_elem = pub_date_elem.find('Month')
            day_elem = pub_date_elem.find('Day')
            
            if year_elem is not None and year_elem.text:
                year = year_elem.text.strip()
                month = month_elem.text.strip() if month_elem is not None else "01"
                day = day_elem.text.strip() if day_elem is not None else "01"
                
                # Ensure month and day are 2 digits
                month = month.zfill(2)
                day = day.zfill(2)
                
                paper_data['date'] = f"{year}-{month}-{day}"
        
        # Extract authors
        authors = []
        author_list = article_element.find('.//AuthorList')
        if author_list is not None:
            for author_elem in author_list.findall('Author'):
                last_name_elem = author_elem.find('LastName')
                fore_name_elem = author_elem.find('ForeName')
                
                if last_name_elem is not None and last_name_elem.text:
                    last_name = last_name_elem.text.strip()
                    fore_name = fore_name_elem.text.strip() if fore_name_elem is not None else ""
                    
                    if fore_name:
                        authors.append(f"{fore_name} {last_name}")
                    else:
                        authors.append(last_name)
        
        if authors:
            paper_data['authors'] = authors
        
        # Extract DOI
        doi_elem = article_element.find('.//ELocationID[@EIdType="doi"]')
        if doi_elem is not None and doi_elem.text:
            paper_data['doi'] = doi_elem.text.strip()
        
        # Extract PMID
        pmid_elem = article_element.find('.//PMID')
        if pmid_elem is not None and pmid_elem.text:
            paper_data['pmid'] = pmid_elem.text.strip()
        
        # Extract MeSH terms
        mesh_terms = []
        mesh_list = article_element.find('.//MeshHeadingList')
        if mesh_list is not None:
            for mesh_elem in mesh_list.findall('MeshHeading/DescriptorName'):
                if mesh_elem.text:
                    mesh_terms.append(mesh_elem.text.strip())
        
        if mesh_terms:
            paper_data['mesh_terms'] = mesh_terms
        
        # Only return if we have at least a title
        if paper_data.get('title'):
            return paper_data
        
    except Exception as e:
        print(f"⚠️  Error parsing PubMed XML: {e}")
        return None
    
    return None

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

def extract_citation_count_fast(paper_data):
    """Fast citation count extraction with better error handling and no Windows-incompatible signals."""
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
        try:
            # Rate limiting
            from processing_config import CITATION_RATE_LIMIT
            rate_limit_delay(CITATION_RATE_LIMIT)  # Use configured rate limit
            
            # Single attempt with shorter timeout
            citations = get_citations_by_doi(doi)
            
            if citations is not None and citations > 0:
                citation_error_counter['successful_citations'] += 1
                return str(citations)
            elif citations == 0:
                citation_error_counter['successful_citations'] += 1
                return "0"
            else:
                citation_error_counter['doi_lookup_failures'] += 1
                citation_error_counter['failed_citations'] += 1
                
        except Exception as e:
            citation_error_counter['doi_lookup_failures'] += 1
            citation_error_counter['failed_citations'] += 1
    
    # If no DOI or DOI lookup failed, try title
    title = paper_data.get('title', '')
    if title:
        # Clean and validate title
        title = str(title).strip()
        if not title or title.lower() in ['nan', 'none', 'null', '']:
            title = ''
    
    if title:
        try:
            # Rate limiting
            from processing_config import CITATION_RATE_LIMIT
            rate_limit_delay(CITATION_RATE_LIMIT)  # Use configured rate limit
            
            # Single attempt with shorter timeout
            citations = get_citations_from_title(title)
            
            if citations is not None and citations > 0:
                citation_error_counter['successful_citations'] += 1
                return str(citations)
            elif citations == 0:
                citation_error_counter['successful_citations'] += 1
                return "0"
            else:
                citation_error_counter['title_lookup_failures'] += 1
                citation_error_counter['failed_citations'] += 1
                
        except Exception as e:
            citation_error_counter['title_lookup_failures'] += 1
            citation_error_counter['failed_citations'] += 1
    
    # Fallback to checking if citation data is already in the paper_data
    citation_fields = [
        'citations', 'citation_count', 'cited_by_count', 'times_cited',
        'reference_count', 'cited_count', 'num_citations', 'citedByCount'
    ]
    
    for field in citation_fields:
        if field in paper_data and paper_data[field] is not None:
            try:
                count = int(paper_data[field])
                if count >= 0:
                    citation_error_counter['successful_citations'] += 1
                    return str(count)
            except (ValueError, TypeError):
                continue
    
    citation_error_counter['failed_citations'] += 1
    return "not found"

def extract_journal_info_fast(paper_data):
    """Fast journal info extraction with minimal API calls."""
    # First check if journal information is already in the paper_data
    journal_fields = [
        'journal', 'journal_name', 'publication', 'source', 'venue',
        'journal_title', 'publication_venue', 'journal_ref', 'journal-ref'
    ]
    
    for field in journal_fields:
        if field in paper_data and paper_data[field]:
            journal_name = str(paper_data[field])
            if journal_name and journal_name.lower() not in ['nan', 'none', 'null', '']:
                return journal_name
    
    # If no journal found in paper_data, check source
    source = paper_data.get('source', '')
    if source in ['biorxiv', 'medrxiv']:
        return f"{source.upper()}"
    
    return "Unknown journal"

def extract_impact_factor_fast(paper_data, journal_name=None):
    """Fast impact factor extraction with minimal API calls."""
    # First, try to get impact factor directly from the data
    impact_fields = [
        'impact_factor', 'journal_impact_factor', 'if', 'jif',
        'impact', 'journal_if', 'journal_impact'
    ]
    
    for field in impact_fields:
        if field in paper_data and paper_data[field] is not None:
            try:
                impact = float(paper_data[field])
                if impact > 0:
                    return str(impact)
            except (ValueError, TypeError):
                continue
    
    # Skip expensive API calls for now - return estimated value
    if not journal_name:
        journal_name = extract_journal_info_fast(paper_data)
    
    return estimate_impact_factor(journal_name)

def process_paper_citations_immediate(paper_data):
    """Process citations immediately for a single paper - simple and efficient."""
    try:
        # Process citation count
        citation_count = extract_citation_count_fast(paper_data)
        
        # Process journal info
        journal = extract_journal_info_fast(paper_data)
        
        # Process impact factor
        impact_factor = extract_impact_factor_fast(paper_data, journal)
        
        return {
            'citation_count': citation_count,
            'journal': journal,
            'impact_factor': impact_factor
        }
        
    except Exception as e:
        print(f"⚠️  Citation processing error: {e}")
        return {
            'citation_count': 'not found',
            'journal': 'Unknown journal',
            'impact_factor': 'not found'
        }

def estimate_impact_factor(journal_name):
    """Estimate impact factor based on journal name - fast local lookup without API calls."""
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

# Add citation processing configuration
CITATION_PROCESSING_ENABLED = os.environ.get("ENABLE_CITATIONS", "true").lower() == "true"
CITATION_TIMEOUT_SECONDS = int(os.environ.get("CITATION_TIMEOUT", "10"))

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
    """Safely chunk text into paragraphs, handling None and empty values."""
    if not text or not isinstance(text, str):
        return []
    
    # Split on newlines and filter out empty paragraphs
    paragraphs = [p.strip() for p in text.split("\n") if p and p.strip()]
    return paragraphs

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
EMBEDDINGS_FILE = "xrvix_embeddings/pubmed_embeddings.json"

def get_search_terms():
    # Simplified search terms for focused coverage
    terms = [
        "ubr5",
        "UBR5", 
        "ubr-5",
        "UBR-5"
    ]
    
    print(f"📋 Using {len(terms)} focused search terms: {', '.join(terms)}")
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

def ensure_directory_structure():
    """Ensure the xrvix_embeddings directory exists and handle migration from old paths."""
    # Create the main directory
    os.makedirs("xrvix_embeddings", exist_ok=True)
    
    # Check if old pubmed_embeddings.json exists and migrate it
    old_path = "pubmed_embeddings.json"
    new_path = "xrvix_embeddings/pubmed_embeddings.json"
    
    if os.path.exists(old_path) and not os.path.exists(new_path):
        print(f"🔄 Migrating {old_path} to {new_path}...")
        try:
            import shutil
            shutil.move(old_path, new_path)
            print(f"✅ Successfully migrated {old_path} to {new_path}")
        except Exception as e:
            print(f"⚠️  Failed to migrate file: {e}")
            print(f"   You may need to manually move {old_path} to {new_path}")
    
    # Create subdirectories for different sources if they don't exist
    subdirs = ["pubmed", "biorxiv", "medrxiv"]
    for subdir in subdirs:
        os.makedirs(os.path.join("xrvix_embeddings", subdir), exist_ok=True)

def check_old_embedding_files():
    """Check for old embedding files and provide guidance."""
    old_files = []
    
    # Check for old pubmed_embeddings.json in root
    if os.path.exists("pubmed_embeddings.json"):
        old_files.append("pubmed_embeddings.json")
    
    # Check for any other old embedding files
    for file in os.listdir("."):
        if file.endswith("_embeddings.json") and file != "pubmed_embeddings.json":
            old_files.append(file)
    
    if old_files:
        print("\n⚠️  Found old embedding files in root directory:")
        for file in old_files:
            print(f"   - {file}")
        print("\n💡 These files should be moved to the xrvix_embeddings/ folder for better organization.")
        print("   The system will attempt to migrate them automatically.")

def main():
    try:
        print("=== Starting PubMed Scraping and Processing (JSON Storage) ===")
        print(f"🔧 Using configuration profile: {CONFIG_PROFILE}")
        print_config_info()
        
        # Ensure directory structure and handle migration
        ensure_directory_structure()
        
        # Check for old embedding files
        check_old_embedding_files()
        
        # Reset citation error counter at the start
        reset_citation_error_counter()
        
        # Get search terms from user
        search_terms = get_search_terms()
        print(f"🔍 Search terms: {', '.join(search_terms)}")

        # Load API key
        with open("keys.json") as f:
            api_key = json.load(f)["GOOGLE_API_KEY"]

        # Step 1: Fetch and dump PubMed papers
        print(f"\n🔄 Fetching PubMed papers for {len(search_terms)} search terms...")
        print("⏱️  This may take several minutes...")
        
        # Use the new comprehensive search function
        papers_to_process = search_pubmed_comprehensive(search_terms, max_results=10000)
        
        if not papers_to_process:
            print("❌ No papers found. Exiting.")
            return
        
        print(f"✅ Found {len(papers_to_process)} papers from comprehensive search")
        
        # Save the comprehensive results directly
        with open(DUMP_FILE, 'w', encoding='utf-8') as f:
            for paper in papers_to_process:
                json.dump(paper, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"✅ Saved {len(papers_to_process)} papers to {DUMP_FILE}")
        print(f"📁 Embeddings will be saved to {EMBEDDINGS_FILE}")

        # Step 2: Process the papers directly
        print(f"\n📊 Processing {len(papers_to_process)} papers...")
        
        # Validate papers before creating DataFrame
        valid_papers = []
        for paper in papers_to_process:
            if paper and isinstance(paper, dict):
                # Ensure all required fields exist and are not None
                title = paper.get('title', '')
                abstract = paper.get('abstract', '')
                
                if title and abstract and title.strip() and abstract.strip():
                    # Convert any None values to empty strings
                    for key in paper:
                        if paper[key] is None:
                            paper[key] = ""
                    valid_papers.append(paper)
                else:
                    print(f"⚠️  Skipping paper: Missing title or abstract")
            else:
                print(f"⚠️  Skipping invalid paper data: {type(paper)}")
        
        if not valid_papers:
            print("❌ No valid papers to process. Exiting.")
            return
        
        print(f"📊 Processing {len(valid_papers)} valid papers out of {len(papers_to_process)} found")
        
        df = pd.DataFrame(valid_papers)
        print(f"📊 Loaded {len(df)} papers for processing")

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
        print(f"💡 Citations will be processed immediately for each paper")
        with tqdm(total=len(df), desc="Processing papers", unit="paper") as pbar:
            for idx, row in df.iterrows():
                try:
                    # Safely extract all fields with defensive programming
                    title = str(row.get("title", "") or "")
                    doi = str(row.get("doi", "") or "")
                    abstract = str(row.get("abstract", "") or "")
                    authors_raw = row.get("authors", "")
                    
                    # Handle None values safely
                    if title is None:
                        title = ""
                    if doi is None:
                        doi = ""
                    if abstract is None:
                        abstract = ""
                    if authors_raw is None:
                        authors_raw = ""
                    
                    # Convert author list to string for ChromaDB compatibility
                    if isinstance(authors_raw, list):
                        author = "; ".join([str(a) for a in authors_raw if a])  # Join authors with semicolon separator
                    else:
                        author = str(authors_raw) if authors_raw else ""
                    
                    # Extract basic metadata fields
                    try:
                        publication_date = extract_publication_date(row)
                        year = publication_date[:4] if publication_date else str(datetime.now().year)
                    except Exception as e:
                        print(f"⚠️  Error extracting publication date for paper {idx}: {e}")
                        publication_date = f"{datetime.now().year}-01-01"
                        year = str(datetime.now().year)
                    
                    # Process citations immediately for this paper
                    citation_data = process_paper_citations_immediate(row)
                    
                    # Extract additional metadata fields
                    try:
                        additional_metadata = extract_additional_metadata(row)
                    except Exception as e:
                        print(f"⚠️  Error extracting additional metadata for paper {idx}: {e}")
                        additional_metadata = {}
                    
                    # Update progress bar description to show citation processing
                    pbar.set_description(f"Processing paper {idx+1}/{len(df)} (citations: {citation_data['citation_count']})")
                    
                    # Only process papers that have abstracts
                    if not abstract or not abstract.strip():
                        print(f"⚠️  Skipping paper {idx}: No abstract available")
                        continue
                        
                    paragraphs = chunk_paragraphs(abstract)
                    
                    # Skip papers with no valid paragraphs
                    if not paragraphs:
                        print(f"⚠️  Skipping paper {idx}: No valid paragraphs found in abstract")
                        continue
                    
                    # Additional safety check for paper data
                    if not title or not title.strip():
                        print(f"⚠️  Skipping paper {idx}: No valid title")
                        continue
                    
                    # Process paragraphs for this paper
                    paper_embeddings = 0
                    for i, para in enumerate(paragraphs):
                        if not para or not para.strip():  # Skip empty paragraphs
                            continue
                            
                        try:
                            embedding = get_google_embedding(para, api_key)
                            
                            # Create comprehensive metadata object with immediate citation data
                            metadata = {
                                "title": title,
                                "doi": doi,
                                "author": author,  # Use author field name for consistency
                                "publication_date": publication_date,
                                "citation_count": citation_data['citation_count'],  # Immediate citation data
                                "journal": citation_data['journal'],  # Immediate journal data
                                "impact_factor": citation_data['impact_factor'],  # Immediate impact factor data
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
                    
                except Exception as e:
                    print(f"⚠️  Error processing paper {idx}: {e}")
                    continue
                
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
        
        # Print citation processing summary
        citation_summary = get_citation_error_summary()
        if citation_summary['total_failures'] > 0:
            print(f"\n📊 Citation Processing Summary:")
            print(f"   DOI lookup failures: {citation_summary['doi_lookup_failures']}")
            print(f"   Title lookup failures: {citation_summary['title_lookup_failures']}")
            print(f"   Total citation processing failures: {citation_summary['total_failures']}")
        else:
            print(f"\n✅ All citation lookups completed successfully!")
            
    except Exception as e:
        print(f"\n❌ Fatal error in main function: {e}")
        print("Stack trace:")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main() 