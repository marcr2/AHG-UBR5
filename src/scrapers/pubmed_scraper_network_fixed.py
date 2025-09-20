"""
Network-resilient PubMed scraper for UBR5 papers.
Handles DNS resolution failures and network timeouts gracefully.
"""

import os
import json
import pandas as pd
import requests
import re
from tqdm.auto import tqdm
import numpy as np
from datetime import datetime
import time
import logging
import urllib3
from urllib3.exceptions import NameResolutionError, ConnectTimeoutError
import concurrent.futures
import threading
from functools import partial
import sys
from typing import List, Dict, Optional, Tuple
import xml.etree.ElementTree as ET
import urllib.parse

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Import network fix
from src.core.network_fix import NetworkConnectivityFix

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", message="Could not find paper")
warnings.filterwarnings("ignore", category=UserWarning, module="paperscraper")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data/logs/pubmed_scraper_network_fixed.log')
    ]
)
logger = logging.getLogger(__name__)

class NetworkResilientPubMedScraper:
    """
    PubMed scraper with robust network error handling.
    """
    
    def __init__(self):
        self.network_fix = NetworkConnectivityFix()
        self.session = requests.Session()
        self.setup_session()
        
        # NCBI E-utilities endpoints (with fallbacks)
        self.ncbi_endpoints = [
            'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi',
            'https://www.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'
        ]
        
        self.current_endpoint = None
        self.find_working_endpoint()
        
    def setup_session(self):
        """Configure requests session with robust settings."""
        # Set longer timeouts
        self.session.timeout = (15, 45)  # (connect timeout, read timeout)
        
        # Add retry adapter
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        retry_strategy = Retry(
            total=5,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set user agent
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def find_working_endpoint(self):
        """Find a working NCBI endpoint."""
        for endpoint in self.ncbi_endpoints:
            success, message = self.network_fix.test_connectivity(endpoint)
            if success:
                self.current_endpoint = endpoint
                logger.info(f"✅ Using NCBI endpoint: {endpoint}")
                return
        
        logger.error("❌ No working NCBI endpoints found!")
        self.current_endpoint = self.ncbi_endpoints[0]  # Fallback to first
    
    def search_pubmed_with_retry(self, query: str, max_results: int = 100, 
                                date_from: str = "1900", date_to: str = None) -> List[Dict]:
        """
        Search PubMed with robust error handling and retry logic.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
            
        Returns:
            List of paper dictionaries
        """
        if date_to is None:
            date_to = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"🔍 Searching PubMed for: {query}")
        logger.info(f"📅 Date range: {date_from} to {date_to}")
        logger.info(f"🎯 Max results: {max_results}")
        
        papers = []
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                # Try to get paper IDs first
                paper_ids = self._get_paper_ids(query, max_results, date_from, date_to)
                
                if not paper_ids:
                    logger.warning("⚠️ No paper IDs found")
                    break
                
                logger.info(f"📊 Found {len(paper_ids)} paper IDs")
                
                # Get paper details in batches
                papers = self._get_paper_details_batch(paper_ids)
                
                if papers:
                    logger.info(f"✅ Successfully retrieved {len(papers)} papers")
                    break
                else:
                    logger.warning("⚠️ No paper details retrieved")
                    
            except (NameResolutionError, ConnectTimeoutError) as e:
                retry_count += 1
                logger.warning(f"⚠️ Network error (attempt {retry_count}/{max_retries}): {e}")
                
                if retry_count < max_retries:
                    # Try different endpoint
                    self.find_working_endpoint()
                    wait_time = 2 ** retry_count
                    logger.info(f"⏳ Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error("❌ All retry attempts failed")
                    break
                    
            except Exception as e:
                logger.error(f"❌ Unexpected error: {e}")
                break
        
        return papers
    
    def _get_paper_ids(self, query: str, max_results: int, 
                      date_from: str, date_to: str) -> List[str]:
        """Get paper IDs from PubMed search."""
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': min(max_results, 10000),  # NCBI limit
            'retmode': 'json',
            'sort': 'relevance',
            'mindate': date_from.replace('-', ''),
            'maxdate': date_to.replace('-', ''),
            'tool': 'UBR5_Scraper',
            'email': 'scraper@example.com'
        }
        
        try:
            response = self.session.get(self.current_endpoint, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            paper_ids = data.get('esearchresult', {}).get('idlist', [])
            
            return paper_ids
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Request failed: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON decode error: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            raise
    
    def _get_paper_details_batch(self, paper_ids: List[str]) -> List[Dict]:
        """Get paper details in batches to avoid overwhelming the API."""
        papers = []
        batch_size = 200  # NCBI recommended batch size
        
        for i in range(0, len(paper_ids), batch_size):
            batch_ids = paper_ids[i:i + batch_size]
            logger.info(f"📦 Processing batch {i//batch_size + 1}/{(len(paper_ids) + batch_size - 1)//batch_size}")
            
            try:
                batch_papers = self._get_paper_details(batch_ids)
                papers.extend(batch_papers)
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"❌ Error processing batch: {e}")
                continue
        
        return papers
    
    def _get_paper_details(self, paper_ids: List[str]) -> List[Dict]:
        """Get detailed information for a batch of paper IDs."""
        efetch_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
        
        params = {
            'db': 'pubmed',
            'id': ','.join(paper_ids),
            'retmode': 'xml',
            'rettype': 'abstract',
            'tool': 'UBR5_Scraper',
            'email': 'scraper@example.com'
        }
        
        try:
            response = self.session.get(efetch_url, params=params, timeout=45)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            papers = []
            
            for article in root.findall('.//PubmedArticle'):
                try:
                    paper = self._parse_pubmed_article(article)
                    if paper:
                        papers.append(paper)
                except Exception as e:
                    logger.warning(f"⚠️ Error parsing article: {e}")
                    continue
            
            return papers
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Request failed: {e}")
            raise
        except ET.ParseError as e:
            logger.error(f"❌ XML parse error: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            raise
    
    def _parse_pubmed_article(self, article) -> Optional[Dict]:
        """Parse a PubMed article XML element."""
        try:
            # Extract title
            title_elem = article.find('.//ArticleTitle')
            title = title_elem.text if title_elem is not None else ""
            
            if not title or len(title) < 10:
                return None
            
            # Extract abstract
            abstract_elem = article.find('.//AbstractText')
            abstract = abstract_elem.text if abstract_elem is not None else ""
            
            # Extract authors
            authors = []
            for author in article.findall('.//Author'):
                last_name = author.find('LastName')
                first_name = author.find('ForeName')
                if last_name is not None:
                    name = last_name.text
                    if first_name is not None:
                        name = f"{first_name.text} {name}"
                    authors.append(name)
            
            # Extract journal
            journal_elem = article.find('.//Journal/Title')
            journal = journal_elem.text if journal_elem is not None else ""
            
            # Extract publication date
            year_elem = article.find('.//PubDate/Year')
            year = int(year_elem.text) if year_elem is not None else None
            
            # Extract DOI
            doi_elem = article.find('.//ELocationID[@EIdType="doi"]')
            doi = doi_elem.text if doi_elem is not None else ""
            
            # Extract PMID
            pmid_elem = article.find('.//PMID')
            pmid = pmid_elem.text if pmid_elem is not None else ""
            
            # Create paper dictionary
            paper = {
                'title': title,
                'abstract': abstract,
                'authors': authors,
                'journal': journal,
                'year': year,
                'doi': doi,
                'pmid': pmid,
                'source': 'pubmed',
                'citation_count': '0',  # Will be updated later if needed
                'reference_count': '0',
                'impact_factor': 'not found',
                'fields_of_study': [],
                'publication_types': [],
                'is_preprint': False,
                'publication_date': f"{year}-01-01" if year else None,
                'raw_data': {
                    'pmid': pmid,
                    'doi': doi
                }
            }
            
            return paper
            
        except Exception as e:
            logger.warning(f"⚠️ Error parsing article: {e}")
            return None
    
    def search_ubr5_papers(self, max_papers: int = 1000) -> List[Dict]:
        """Search for UBR5-related papers using multiple search strategies."""
        logger.info("🔍 Starting UBR5 paper search with network-resilient PubMed scraper")
        
        # UBR5 search terms
        search_terms = [
            'UBR5[Title/Abstract]',
            'ubr-5[Title/Abstract]',
            'UBR5[Title]',
            'ubiquitin protein ligase E3 component n-recognin 5[Title/Abstract]',
            'EDD1[Title/Abstract]',
            'E3 ubiquitin-protein ligase UBR5[Title/Abstract]'
        ]
        
        all_papers = []
        seen_titles = set()
        
        for term in search_terms:
            try:
                logger.info(f"🔍 Searching for: {term}")
                
                # Search with current term
                papers = self.search_pubmed_with_retry(
                    query=term,
                    max_results=min(max_papers // len(search_terms), 500),
                    date_from="1900",
                    date_to=datetime.now().strftime("%Y-%m-%d")
                )
                
                # Add unique papers
                added_count = 0
                for paper in papers:
                    title = paper.get('title', '').lower().strip()
                    if title and title not in seen_titles:
                        all_papers.append(paper)
                        seen_titles.add(title)
                        added_count += 1
                        
                        if len(all_papers) >= max_papers:
                            break
                
                logger.info(f"✅ Added {added_count} new papers (total: {len(all_papers)})")
                
                # Rate limiting between searches
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Error searching for '{term}': {e}")
                continue
        
        logger.info(f"🎉 UBR5 search completed! Found {len(all_papers)} unique papers")
        return all_papers

def main():
    """Main function to run the network-resilient PubMed scraper."""
    print("🔍 Network-Resilient UBR5 PubMed Scraper")
    print("=" * 50)
    
    # Apply network fixes
    print("\n🔧 Applying network fixes...")
    network_fix = NetworkConnectivityFix()
    network_fix.disable_posthog()
    network_fix.patch_paperscraper()
    
    # Test connectivity
    print("\n📡 Testing connectivity...")
    connectivity_results = network_fix.test_all_endpoints()
    
    # Show connectivity status
    print("\n📊 Connectivity Status:")
    for service, (success, message) in connectivity_results.items():
        status = "✅" if success else "❌"
        print(f"   {status} {service}: {message}")
    
    # Initialize scraper
    scraper = NetworkResilientPubMedScraper()
    
    # Get user input
    try:
        max_papers = int(input("\nEnter maximum number of papers to collect (default 500): ") or "500")
    except ValueError:
        max_papers = 500
    
    print(f"\n🚀 Starting UBR5 paper collection (target: {max_papers} papers)...")
    
    # Run search
    papers = scraper.search_ubr5_papers(max_papers=max_papers)
    
    if papers:
        print(f"\n✅ Successfully collected {len(papers)} papers!")
        
        # Save results
        output_file = f"ubr5_pubmed_papers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {output_file}")
        
        # Show sample
        if papers:
            print(f"\n📄 Sample paper:")
            sample = papers[0]
            print(f"   Title: {sample.get('title', 'N/A')[:100]}...")
            print(f"   Journal: {sample.get('journal', 'N/A')}")
            print(f"   Year: {sample.get('year', 'N/A')}")
            print(f"   Authors: {len(sample.get('authors', []))} authors")
    else:
        print("\n❌ No papers were collected. Check network connectivity and try again.")

if __name__ == "__main__":
    main()
