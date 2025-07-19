#!/usr/bin/env python3
"""
Test script to verify the xrvix processing fix for the 'list' object has no attribute 'add' error.
"""

import json
import os
from process_xrvix_dumps_json import get_processed_papers, mark_paper_processed, load_metadata

def test_processed_papers_fix():
    """Test that the processed papers functions work correctly with JSON serialization."""
    
    print("🧪 Testing xrvix processing fix")
    print("=" * 40)
    
    # Create test metadata
    test_metadata = {
        "created": "2024-01-01T00:00:00",
        "total_embeddings": 0,
        "total_chunks": 0,
        "total_papers": 0,
        "sources": {},
        "batches": {},
        "processed_papers": {
            "biorxiv": [1, 2, 3, 4, 5],  # Simulate JSON-loaded list
            "medrxiv": set([10, 20, 30])  # Simulate set
        }
    }
    
    print("📊 Test metadata created with mixed list/set data")
    
    # Test get_processed_papers with list data
    print("\n🔍 Testing get_processed_papers...")
    
    biorxiv_processed = get_processed_papers(test_metadata, "biorxiv")
    print(f"   biorxiv (was list): {type(biorxiv_processed)} with {len(biorxiv_processed)} items")
    assert isinstance(biorxiv_processed, set), "biorxiv should be converted to set"
    
    medrxiv_processed = get_processed_papers(test_metadata, "medrxiv")
    print(f"   medrxiv (was set): {type(medrxiv_processed)} with {len(medrxiv_processed)} items")
    assert isinstance(medrxiv_processed, set), "medrxiv should remain a set"
    
    # Test mark_paper_processed
    print("\n🔍 Testing mark_paper_processed...")
    
    # Test with list data (should convert to set)
    mark_paper_processed(test_metadata, "biorxiv", 100)
    biorxiv_after = get_processed_papers(test_metadata, "biorxiv")
    print(f"   Added to biorxiv (was list): {type(biorxiv_after)} with {len(biorxiv_after)} items")
    assert isinstance(biorxiv_after, set), "biorxiv should be a set after marking"
    assert 100 in biorxiv_after, "New paper should be in processed set"
    
    # Test with set data (should remain set)
    mark_paper_processed(test_metadata, "medrxiv", 200)
    medrxiv_after = get_processed_papers(test_metadata, "medrxiv")
    print(f"   Added to medrxiv (was set): {type(medrxiv_after)} with {len(medrxiv_after)} items")
    assert isinstance(medrxiv_after, set), "medrxiv should remain a set"
    assert 200 in medrxiv_after, "New paper should be in processed set"
    
    # Test with new source
    mark_paper_processed(test_metadata, "new_source", 300)
    new_source_after = get_processed_papers(test_metadata, "new_source")
    print(f"   Added to new_source: {type(new_source_after)} with {len(new_source_after)} items")
    assert isinstance(new_source_after, set), "new_source should be a set"
    assert 300 in new_source_after, "New paper should be in processed set"
    
    # Test JSON serialization/deserialization
    print("\n🔍 Testing JSON serialization...")
    
    # Save to JSON (simulates what happens in real processing)
    test_file = "test_metadata.json"
    with open(test_file, 'w') as f:
        json.dump(test_metadata, f, indent=2)
    
    # Load from JSON (simulates what happens when restarting processing)
    with open(test_file, 'r') as f:
        loaded_metadata = json.load(f)
    
    # Test that loaded data works correctly
    loaded_biorxiv = get_processed_papers(loaded_metadata, "biorxiv")
    print(f"   Loaded biorxiv: {type(loaded_biorxiv)} with {len(loaded_biorxiv)} items")
    assert isinstance(loaded_biorxiv, set), "Loaded biorxiv should be a set"
    assert 100 in loaded_biorxiv, "Previously added paper should still be there"
    
    # Test adding to loaded data
    mark_paper_processed(loaded_metadata, "biorxiv", 400)
    final_biorxiv = get_processed_papers(loaded_metadata, "biorxiv")
    print(f"   Added to loaded biorxiv: {type(final_biorxiv)} with {len(final_biorxiv)} items")
    assert isinstance(final_biorxiv, set), "Final biorxiv should be a set"
    assert 400 in final_biorxiv, "New paper should be added successfully"
    
    # Clean up
    if os.path.exists(test_file):
        os.remove(test_file)
    
    print("\n✅ All tests passed! The fix should resolve the 'list' object has no attribute 'add' error.")
    print("\n💡 The issue was that JSON serialization converts Python sets to lists.")
    print("   The fix ensures that lists are properly converted back to sets before using .add()")

if __name__ == "__main__":
    test_processed_papers_fix() 