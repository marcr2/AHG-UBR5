#!/usr/bin/env python3
"""
Utility functions for working with hypothesis citation mappings.
This script provides tools to easily map citations to hypotheses using the citation cache.
"""

import json
import os
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

# Set up logging
logger = logging.getLogger(__name__)

class CitationMappingUtils:
    """Utility class for working with hypothesis citation mappings."""
    
    def __init__(self, export_dir: str):
        """
        Initialize with the export directory path.
        
        Args:
            export_dir: Path to the hypothesis export directory
        """
        self.export_dir = export_dir
        self.sources_dir = os.path.join(export_dir, "sources")
        self.master_mapping_file = os.path.join(self.sources_dir, "master_hypothesis_citation_mapping.json")
        
    def load_master_mapping(self) -> Optional[Dict]:
        """Load the master hypothesis citation mapping."""
        if not os.path.exists(self.master_mapping_file):
            logger.error(f"Master mapping file not found: {self.master_mapping_file}")
            return None
            
        try:
            with open(self.master_mapping_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading master mapping: {e}")
            return None
    
    def get_hypothesis_citations(self, hypothesis_index: int) -> Optional[List[Dict]]:
        """
        Get all citations for a specific hypothesis.
        
        Args:
            hypothesis_index: Index of the hypothesis
            
        Returns:
            List of citation dictionaries or None if not found
        """
        master_mapping = self.load_master_mapping()
        if not master_mapping:
            return None
            
        for mapping in master_mapping.get('hypothesis_mappings', []):
            if mapping.get('hypothesis_index') == hypothesis_index:
                hypothesis_dir = os.path.join(self.sources_dir, f"hypothesis_{hypothesis_index}")
                citations = []
                
                for citation_file in mapping.get('citation_files', []):
                    citation_filepath = os.path.join(hypothesis_dir, citation_file['filename'])
                    if os.path.exists(citation_filepath):
                        try:
                            with open(citation_filepath, 'r', encoding='utf-8') as f:
                                citation_data = json.load(f)
                                citations.append(citation_data)
                        except Exception as e:
                            logger.warning(f"Error loading citation file {citation_filepath}: {e}")
                
                return citations
        
        logger.warning(f"Hypothesis {hypothesis_index} not found")
        return None
    
    def get_citation_cache_keys_for_hypothesis(self, hypothesis_index: int) -> List[str]:
        """
        Get citation cache keys for a specific hypothesis.
        
        Args:
            hypothesis_index: Index of the hypothesis
            
        Returns:
            List of citation cache keys
        """
        master_mapping = self.load_master_mapping()
        if not master_mapping:
            return []
            
        for mapping in master_mapping.get('hypothesis_mappings', []):
            if mapping.get('hypothesis_index') == hypothesis_index:
                hypothesis_dir = os.path.join(self.sources_dir, f"hypothesis_{hypothesis_index}")
                mapping_filepath = os.path.join(hypothesis_dir, "hypothesis_citation_mapping.json")
                
                if os.path.exists(mapping_filepath):
                    try:
                        with open(mapping_filepath, 'r', encoding='utf-8') as f:
                            mapping_data = json.load(f)
                            return mapping_data.get('citation_cache_keys', [])
                    except Exception as e:
                        logger.warning(f"Error loading mapping file {mapping_filepath}: {e}")
        
        return []
    
    def create_citation_summary_table(self) -> pd.DataFrame:
        """
        Create a summary table of all hypotheses and their citations.
        
        Returns:
            DataFrame with hypothesis and citation information
        """
        master_mapping = self.load_master_mapping()
        if not master_mapping:
            return pd.DataFrame()
        
        summary_data = []
        for mapping in master_mapping.get('hypothesis_mappings', []):
            hypothesis_index = mapping.get('hypothesis_index')
            hypothesis_text = mapping.get('hypothesis_text', '')
            citation_count = mapping.get('citation_count', 0)
            
            # Get citation cache keys
            cache_keys = self.get_citation_cache_keys_for_hypothesis(hypothesis_index)
            
            summary_data.append({
                'Hypothesis_Index': hypothesis_index,
                'Hypothesis_Text': hypothesis_text[:100] + '...' if len(hypothesis_text) > 100 else hypothesis_text,
                'Citation_Count': citation_count,
                'Citation_Cache_Keys': '; '.join(cache_keys),
                'Cache_Keys_Count': len(cache_keys)
            })
        
        return pd.DataFrame(summary_data)
    
    def export_citation_mapping_to_csv(self, output_file: str = None) -> str:
        """
        Export citation mapping to CSV file.
        
        Args:
            output_file: Output CSV file path (optional)
            
        Returns:
            Path to the created CSV file
        """
        if output_file is None:
            output_file = os.path.join(self.export_dir, "citation_mapping_summary.csv")
        
        df = self.create_citation_summary_table()
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        logger.info(f"Citation mapping summary exported to: {output_file}")
        return output_file
    
    def find_hypotheses_by_citation_cache_key(self, cache_key: str) -> List[int]:
        """
        Find all hypotheses that use a specific citation cache key.
        
        Args:
            cache_key: Citation cache key to search for
            
        Returns:
            List of hypothesis indices
        """
        master_mapping = self.load_master_mapping()
        if not master_mapping:
            return []
        
        matching_hypotheses = []
        for mapping in master_mapping.get('hypothesis_mappings', []):
            hypothesis_index = mapping.get('hypothesis_index')
            cache_keys = self.get_citation_cache_keys_for_hypothesis(hypothesis_index)
            
            if cache_key in cache_keys:
                matching_hypotheses.append(hypothesis_index)
        
        return matching_hypotheses
    
    def get_citation_content_for_hypothesis(self, hypothesis_index: int) -> List[str]:
        """
        Get the actual chunk content for all citations in a hypothesis.
        
        Args:
            hypothesis_index: Index of the hypothesis
            
        Returns:
            List of chunk content strings
        """
        citations = self.get_hypothesis_citations(hypothesis_index)
        if not citations:
            return []
        
        chunk_contents = []
        for citation in citations:
            chunk_content = citation.get('chunk_content', '')
            if chunk_content:
                chunk_contents.append(chunk_content)
        
        return chunk_contents

def main():
    """Example usage of the CitationMappingUtils."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python citation_mapping_utils.py <export_directory>")
        print("Example: python citation_mapping_utils.py hypothesis_export/Hypothesis_Export_09202025-2043")
        return
    
    export_dir = sys.argv[1]
    
    if not os.path.exists(export_dir):
        print(f"❌ Export directory not found: {export_dir}")
        return
    
    utils = CitationMappingUtils(export_dir)
    
    # Create summary table
    print("📊 Creating citation mapping summary...")
    summary_df = utils.create_citation_summary_table()
    
    if not summary_df.empty:
        print(f"\n📋 Summary of {len(summary_df)} hypotheses:")
        print(summary_df.to_string(index=False))
        
        # Export to CSV
        csv_file = utils.export_citation_mapping_to_csv()
        
        # Example: Get citations for first hypothesis
        if len(summary_df) > 0:
            first_hypothesis = summary_df.iloc[0]['Hypothesis_Index']
            print(f"\n🔍 Citations for hypothesis {first_hypothesis}:")
            citations = utils.get_hypothesis_citations(first_hypothesis)
            if citations:
                for i, citation in enumerate(citations[:3]):  # Show first 3 citations
                    citation_info = citation.get('citation_info', {})
                    print(f"  {i+1}. {citation_info.get('title', 'No title')}")
                    print(f"     Authors: {citation_info.get('authors', 'Unknown')}")
                    print(f"     Journal: {citation_info.get('journal', 'Unknown')}")
                    print(f"     DOI: {citation_info.get('doi', 'No DOI')}")
                    print()
    else:
        print("❌ No citation mappings found")

if __name__ == "__main__":
    main()
