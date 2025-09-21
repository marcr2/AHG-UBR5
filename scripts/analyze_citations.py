#!/usr/bin/env python3
"""
Command-line script for analyzing hypothesis citation mappings.
This provides easy access to the CitationMappingUtils functionality.
"""

import sys
import os
import glob
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.citation_mapping_utils import CitationMappingUtils

def find_latest_export():
    """Find the most recent hypothesis export directory."""
    export_pattern = "hypothesis_export/Hypothesis_Export_*"
    export_folders = sorted(glob.glob(export_pattern))
    
    if not export_folders:
        return None
    
    return export_folders[-1]

def main():
    """Main function for citation analysis."""
    if len(sys.argv) < 2:
        # Try to find the latest export automatically
        latest_export = find_latest_export()
        if latest_export:
            print(f"🔍 Using latest export: {latest_export}")
            export_dir = latest_export
        else:
            print("Usage: python scripts/analyze_citations.py <export_directory>")
            print("   or: python scripts/analyze_citations.py (uses latest export)")
            print("\nExample:")
            print("  python scripts/analyze_citations.py hypothesis_export/Hypothesis_Export_09202025-2043")
            return
    else:
        export_dir = sys.argv[1]
    
    if not os.path.exists(export_dir):
        print(f"❌ Export directory not found: {export_dir}")
        return
    
    print(f"📊 Analyzing citations from: {export_dir}")
    utils = CitationMappingUtils(export_dir)
    
    # Create summary table
    print("\n📋 Creating citation mapping summary...")
    summary_df = utils.create_citation_summary_table()
    
    if not summary_df.empty:
        print(f"\n📊 Summary of {len(summary_df)} hypotheses:")
        print("=" * 80)
        print(summary_df.to_string(index=False))
        
        # Export to CSV
        csv_file = utils.export_citation_mapping_to_csv()
        
        # Show some statistics
        print(f"\n📈 Statistics:")
        print(f"  Total hypotheses: {len(summary_df)}")
        print(f"  Average citations per hypothesis: {summary_df['Citation_Count'].mean():.1f}")
        print(f"  Max citations in a hypothesis: {summary_df['Citation_Count'].max()}")
        print(f"  Min citations in a hypothesis: {summary_df['Citation_Count'].min()}")
        
        # Example: Get citations for first hypothesis
        if len(summary_df) > 0:
            first_hypothesis = summary_df.iloc[0]['Hypothesis_Index']
            print(f"\n🔍 Example - Citations for hypothesis {first_hypothesis}:")
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
        print("❌ No citation mappings found in this export")

if __name__ == "__main__":
    main()
