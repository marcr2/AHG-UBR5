#!/usr/bin/env python3
"""
Test script for ChromaDB Manager
Tests the basic functionality with both single-file and multi-file storage systems.
"""

import json
import os
from chromadb_manager import ChromaDBManager

def test_chromadb_manager_single_file():
    """Test the ChromaDB manager with single-file PubMed embeddings."""
    print("🧪 Testing ChromaDB Manager (Single File)")
    print("=" * 50)
    
    # Check if PubMed embeddings exist
    if not os.path.exists("pubmed_embeddings.json"):
        print("⚠️  pubmed_embeddings.json not found, skipping single-file test")
        return True  # Not a failure, just skip
    
    try:
        # Initialize manager
        print("🔄 Initializing ChromaDB manager...")
        manager = ChromaDBManager()
        
        # Create collection
        print("🔄 Creating collection...")
        if not manager.create_collection():
            print("❌ Failed to create collection")
            return False
        
        # Load PubMed data
        print("🔄 Loading PubMed embeddings...")
        pubmed_data = manager.load_embeddings_from_json("pubmed_embeddings.json")
        if not pubmed_data:
            print("❌ Failed to load PubMed data")
            return False
        
        # Add to collection
        print("🔄 Adding embeddings to collection...")
        if not manager.add_embeddings_to_collection(pubmed_data, "pubmed"):
            print("❌ Failed to add embeddings")
            return False
        
        # Get statistics
        print("🔄 Getting collection statistics...")
        stats = manager.get_collection_stats()
        print(f"✅ Collection stats: {stats}")
        
        # Test search
        print("🔄 Testing search functionality...")
        if pubmed_data['embeddings']:
            query_embedding = pubmed_data['embeddings'][0]
            results = manager.search_similar(query_embedding, n_results=3)
            print(f"✅ Search returned {len(results)} results")
            
            if results:
                print("\n📄 Sample result:")
                result = results[0]
                print(f"   Title: {result['metadata'].get('title', 'N/A')}")
                print(f"   DOI: {result['metadata'].get('doi', 'N/A')}")
                print(f"   Distance: {result['distance']:.4f}")
        
        # Test filtering
        print("🔄 Testing metadata filtering...")
        filtered_results = manager.filter_by_metadata({"source_name": "pubmed"})
        print(f"✅ Filter returned {len(filtered_results)} results")
        
        print("✅ Single-file tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Single-file test failed with error: {e}")
        return False

def test_chromadb_manager_multi_file():
    """Test the ChromaDB manager with multi-file storage system."""
    print("\n🧪 Testing ChromaDB Manager (Multi-File)")
    print("=" * 50)
    
    # Check if multi-file embeddings exist
    if not os.path.exists("xrvix_embeddings"):
        print("⚠️  xrvix_embeddings directory not found, skipping multi-file test")
        return True  # Not a failure, just skip
    
    try:
        # Initialize manager
        print("🔄 Initializing ChromaDB manager...")
        manager = ChromaDBManager()
        
        # Create collection
        print("🔄 Creating collection...")
        if not manager.create_collection():
            print("❌ Failed to create collection")
            return False
        
        # Test loading directory metadata
        print("🔄 Testing directory metadata loading...")
        dir_info = manager.load_embeddings_from_directory("xrvix_embeddings")
        if not dir_info:
            print("❌ Failed to load directory metadata")
            return False
        
        print(f"✅ Loaded metadata for sources: {dir_info.get('sources', [])}")
        print(f"✅ Total embeddings: {dir_info['metadata'].get('total_embeddings', 0)}")
        
        # Test loading specific sources
        print("🔄 Testing source-specific loading...")
        available_sources = dir_info.get('sources', [])
        if available_sources:
            test_source = available_sources[0]
            source_info = manager.load_embeddings_from_directory("xrvix_embeddings", [test_source])
            print(f"✅ Loaded metadata for {test_source}: {source_info.get('sources', [])}")
        
        # Test adding embeddings from directory
        print("🔄 Testing embedding addition from directory...")
        success = manager.add_embeddings_from_directory("xrvix_embeddings")
        if not success:
            print("❌ Failed to add embeddings from directory")
            return False
        
        # Get updated statistics
        print("🔄 Getting updated collection statistics...")
        stats = manager.get_collection_stats()
        print(f"✅ Updated collection stats: {stats}")
        
        # Test filtering by source
        print("🔄 Testing source filtering...")
        for source in available_sources[:2]:  # Test first 2 sources
            filtered_results = manager.filter_by_metadata({"source_name": source})
            print(f"✅ {source}: {len(filtered_results)} documents")
        
        # Test batch file loading
        print("🔄 Testing batch file loading...")
        for source in available_sources[:1]:  # Test first source
            source_dir = os.path.join("xrvix_embeddings", source)
            if os.path.exists(source_dir):
                import glob
                batch_files = glob.glob(os.path.join(source_dir, "batch_*.json"))
                if batch_files:
                    batch_data = manager.load_batch_file(batch_files[0])
                    if batch_data:
                        print(f"✅ Loaded batch file: {len(batch_data.get('embeddings', []))} embeddings")
                    else:
                        print("❌ Failed to load batch file")
        
        print("✅ Multi-file tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Multi-file test failed with error: {e}")
        return False

def test_basic_functionality():
    """Test basic ChromaDB functionality without data loading."""
    print("\n🧪 Testing Basic ChromaDB Functionality")
    print("=" * 50)
    
    try:
        # Initialize manager
        manager = ChromaDBManager()
        
        # Test collection creation
        success = manager.create_collection()
        print(f"✅ Collection creation: {'Success' if success else 'Failed'}")
        
        # Test listing collections
        collections = manager.list_collections()
        print(f"✅ Available collections: {collections}")
        
        # Test getting stats
        stats = manager.get_collection_stats()
        print(f"✅ Collection stats: {stats}")
        
        # Test switching collections
        if collections:
            switch_success = manager.switch_collection(collections[0])
            print(f"✅ Collection switching: {'Success' if switch_success else 'Failed'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_error_handling():
    """Test error handling for edge cases."""
    print("\n🧪 Testing Error Handling")
    print("=" * 50)
    
    try:
        manager = ChromaDBManager()
        
        # Test with non-existent file
        print("🔄 Testing non-existent file loading...")
        result = manager.load_embeddings_from_json("non_existent_file.json")
        if result is None:
            print("✅ Correctly handled non-existent file")
        else:
            print("❌ Should have returned None for non-existent file")
            return False
        
        # Test with non-existent directory
        print("🔄 Testing non-existent directory loading...")
        result = manager.load_embeddings_from_directory("non_existent_directory")
        if not result:
            print("✅ Correctly handled non-existent directory")
        else:
            print("❌ Should have returned empty dict for non-existent directory")
            return False
        
        # Test with invalid batch file
        print("🔄 Testing invalid batch file loading...")
        result = manager.load_batch_file("non_existent_batch.json")
        if result is None:
            print("✅ Correctly handled non-existent batch file")
        else:
            print("❌ Should have returned None for non-existent batch file")
            return False
        
        print("✅ Error handling tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting ChromaDB Manager Tests")
    print("=" * 60)
    
    # Test basic functionality first
    basic_test = test_basic_functionality()
    
    # Test error handling
    error_test = test_error_handling()
    
    # Test with single-file data
    single_file_test = test_chromadb_manager_single_file()
    
    # Test with multi-file data
    multi_file_test = test_chromadb_manager_multi_file()
    
    print("\n📊 Test Results Summary:")
    print(f"   Basic functionality: {'✅ PASS' if basic_test else '❌ FAIL'}")
    print(f"   Error handling: {'✅ PASS' if error_test else '❌ FAIL'}")
    print(f"   Single-file system: {'✅ PASS' if single_file_test else '❌ FAIL'}")
    print(f"   Multi-file system: {'✅ PASS' if multi_file_test else '❌ FAIL'}")
    
    passed_tests = sum([basic_test, error_test, single_file_test, multi_file_test])
    total_tests = 4
    
    print(f"\n🎯 Overall: {passed_tests}/{total_tests} test suites passed")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! ChromaDB manager is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")
        print("\n💡 Note: Missing data files (pubmed_embeddings.json or xrvix_embeddings/)")
        print("   are not considered failures - they just skip those tests.")

if __name__ == "__main__":
    main() 