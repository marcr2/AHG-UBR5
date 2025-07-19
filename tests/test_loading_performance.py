#!/usr/bin/env python3
"""
Test script to benchmark ChromaDB loading performance.
This will help you see the improvement from the optimizations.
"""

import time
import os
from chromadb_manager import ChromaDBManager

def benchmark_loading():
    """Benchmark the loading performance with different batch sizes."""
    
    print("🚀 ChromaDB Loading Performance Benchmark")
    print("=" * 50)
    
    # Check if we have data to load
    if not os.path.exists("xrvix_embeddings"):
        print("❌ No xrvix_embeddings directory found!")
        print("   Run the data processing first to generate embeddings.")
        return
    
    # Test different database batch sizes
    batch_sizes = [1000, 5000, 10000, 20000]
    
    for db_batch_size in batch_sizes:
        print(f"\n📦 Testing with database batch size: {db_batch_size}")
        print("-" * 40)
        
        # Initialize manager
        manager = ChromaDBManager(collection_name=f"test_batch_{db_batch_size}")
        
        # Create collection
        if not manager.create_collection():
            print("❌ Failed to create collection")
            continue
        
        # Time the loading
        start_time = time.time()
        
        try:
            success = manager.add_embeddings_from_directory(
                "xrvix_embeddings", 
                db_batch_size=db_batch_size
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if success:
                # Get stats
                stats = manager.get_collection_stats()
                total_docs = stats.get('total_documents', 0)
                
                # Calculate performance metrics
                docs_per_second = total_docs / duration if duration > 0 else 0
                
                print(f"✅ Success!")
                print(f"   Documents loaded: {total_docs:,}")
                print(f"   Time taken: {duration:.2f} seconds")
                print(f"   Rate: {docs_per_second:.1f} docs/second")
                print(f"   Batch size: {db_batch_size}")
                
                # Performance rating
                if docs_per_second >= 500:
                    rating = "🚀 Excellent"
                elif docs_per_second >= 200:
                    rating = "✅ Good"
                elif docs_per_second >= 50:
                    rating = "⚠️ Acceptable"
                else:
                    rating = "❌ Slow"
                
                print(f"   Performance: {rating}")
                
            else:
                print("❌ Loading failed")
                
        except Exception as e:
            print(f"❌ Error during loading: {e}")
        
        finally:
            # Clean up test collection
            try:
                manager.delete_collection()
            except:
                pass
    
    print(f"\n🎯 Performance Recommendations:")
    print(f"   - Use the batch size that gives the highest docs/second")
    print(f"   - Larger batch sizes generally perform better")
    print(f"   - If you get memory errors, reduce the batch size")
    print(f"   - Target: 200+ docs/second for good performance")

def quick_test():
    """Quick test with current settings."""
    print("⚡ Quick Performance Test")
    print("=" * 30)
    
    if not os.path.exists("xrvix_embeddings"):
        print("❌ No xrvix_embeddings directory found!")
        return
    
    manager = ChromaDBManager(collection_name="quick_test")
    
    if not manager.create_collection():
        print("❌ Failed to create collection")
        return
    
    start_time = time.time()
    
    try:
        success = manager.add_embeddings_from_directory("xrvix_embeddings", db_batch_size=10000)
        end_time = time.time()
        
        if success:
            stats = manager.get_collection_stats()
            total_docs = stats.get('total_documents', 0)
            duration = end_time - start_time
            rate = total_docs / duration if duration > 0 else 0
            
            print(f"✅ Loaded {total_docs:,} documents in {duration:.2f} seconds")
            print(f"📊 Rate: {rate:.1f} documents/second")
            
            if rate < 50:
                print("⚠️ Performance is slow. Consider:")
                print("   - Using larger batch sizes")
                print("   - Checking system resources")
                print("   - Running on SSD storage")
            elif rate < 200:
                print("✅ Performance is acceptable")
            else:
                print("🚀 Performance is excellent!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        try:
            manager.delete_collection()
        except:
            pass

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "full":
        benchmark_loading()
    else:
        quick_test() 