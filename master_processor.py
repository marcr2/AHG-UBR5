import os
import json
import sys
from pubmed_scraper_json import main as process_pubmed
from process_xrvix_dumps_json import main as process_xrvix
from chromadb_manager import ChromaDBManager
from processing_config import print_config_info, get_config, DB_BATCH_SIZE
import subprocess

def show_menu():
    """Show the main menu for data processing"""
    print("=== Master Data Processor ===")
    print("Choose which data sources to process:")
    print("1. PubMed papers only")
    print("2. xrvix dumps only (biorxiv, medrxiv)")
    print("3. Both PubMed and xrvix")
    print("4. Load data into vector database")
    print("5. Show data status")
    print("6. List loaded batches")
    print("7. Start Enhanced RAG System (with Hypothesis Generation & Critique)")
    print("8. Exit")
    print()

def get_user_choice():
    """Get user choice for processing"""
    while True:
        try:
            choice = input("Enter your choice (1-8): ").strip()
            if choice in ['1', '2', '3', '4', '5', '6', '7', '8']:
                return choice
            else:
                print("❌ Please enter a number between 1 and 8.")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)

def check_prerequisites():
    """Check if required files and dependencies are available"""
    print("🔍 Checking prerequisites...")
    
    # Check for API key
    if not os.path.exists("keys.json"):
        print("❌ keys.json not found! Please create it with your Google API key.")
        return False
    
    # Check for xrvix dumps
    try:
        import pkg_resources
        dump_root = pkg_resources.resource_filename("paperscraper", "server_dumps")
        if not os.path.exists(dump_root):
            print("⚠️  No xrvix dumps found. Run 'xrvix downloader.py' first.")
            return False
        
        dump_files = os.listdir(dump_root)
        if not any(f.startswith(('biorxiv', 'medrxiv')) for f in dump_files):
            print("⚠️  No xrvix dump files found. Run 'xrvix downloader.py' first.")
            return False
        
        print("✅ Prerequisites check passed!")
        return True
    except Exception as e:
        print(f"⚠️  Error checking prerequisites: {e}")
        return False

def load_data_into_vector_db():
    """Load processed data into ChromaDB vector database"""
    print("\n🗄️  Loading data into vector database...")
    
    try:
        # Initialize ChromaDB manager
        manager = ChromaDBManager()
        
        # Create collection
        if not manager.create_collection():
            print("❌ Failed to create ChromaDB collection")
            return False
        
        total_loaded = 0
        
        # Load PubMed embeddings (single file)
        if os.path.exists("pubmed_embeddings.json"):
            print("🔄 Loading PubMed embeddings...")
            pubmed_data = manager.load_embeddings_from_json("pubmed_embeddings.json")
            if pubmed_data:
                if manager.add_embeddings_to_collection(pubmed_data, "pubmed"):
                    total_loaded += len(pubmed_data.get('embeddings', []))
                    print(f"✅ Loaded {len(pubmed_data.get('embeddings', []))} PubMed embeddings")
                else:
                    print("❌ Failed to add PubMed embeddings to collection")
            else:
                print("❌ Failed to load PubMed data")
        
        # Load xrvix embeddings (multi-file) - using robust loader for large datasets
        if os.path.exists("xrvix_embeddings"):
            print("🔄 Loading xrvix embeddings (using robust loader)...")
            
            # Use robust loader for large datasets
            try:
                from robust_xrvix_loader import RobustXrvixLoader
                robust_loader = RobustXrvixLoader(manager)
                
                success = robust_loader.load_with_progress(
                    embeddings_dir="xrvix_embeddings",
                    sources=None,  # All sources
                    batch_size=20,  # Process 20 batch files at a time (very conservative)
                    db_batch_size=100,  # Add 100 embeddings to ChromaDB at once (very conservative)
                    max_batches_per_run=None  # Process all batches
                )
                
                if success:
                    stats = manager.get_collection_stats()
                    total_docs = stats.get('total_documents', 0)
                    print(f"✅ Loaded xrvix embeddings (total documents: {total_docs})")
                    total_loaded = total_docs
                else:
                    print("❌ Failed to load xrvix embeddings with robust loader")
                    
            except ImportError:
                # Fallback to original method if robust loader not available
                print("⚠️  Robust loader not available, using original method...")
                if manager.add_embeddings_from_directory("xrvix_embeddings", db_batch_size=DB_BATCH_SIZE):
                    stats = manager.get_collection_stats()
                    total_docs = stats.get('total_documents', 0)
                    print(f"✅ Loaded xrvix embeddings (total documents: {total_docs})")
                    total_loaded = total_docs
                else:
                    print("❌ Failed to load xrvix embeddings")
        
        if total_loaded == 0:
            print("⚠️  No data was loaded. Make sure you have processed some data first.")
            return False
        
        # Display final statistics
        stats = manager.get_collection_stats()
        print(f"\n📊 Vector Database Statistics:")
        print(f"   Total documents: {stats.get('total_documents', 0)}")
        print(f"   Collection name: {stats.get('collection_name', 'N/A')}")
        print(f"   Metadata keys: {stats.get('sample_metadata_keys', [])}")
        
        print(f"\n✅ Successfully loaded {total_loaded} embeddings into vector database!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading data into vector database: {e}")
        return False

def test_vector_database():
    """Test the vector database functionality"""
    print("\n🧪 Testing vector database...")
    
    try:
        # Initialize manager
        manager = ChromaDBManager()
        
        # Check if collection exists
        collections = manager.list_collections()
        if not collections:
            print("❌ No collections found. Load data first using option 4.")
            return False
        
        # Switch to the first collection
        if not manager.switch_collection(collections[0]):
            print("❌ Failed to switch to collection")
            return False
        
        # Get statistics
        stats = manager.get_collection_stats()
        print(f"📊 Collection: {stats.get('collection_name', 'N/A')}")
        print(f"📊 Total documents: {stats.get('total_documents', 0)}")
        
        if stats.get('total_documents', 0) == 0:
            print("❌ No documents in collection. Load data first using option 4.")
            return False
        
        # Test filtering by source
        print("\n🔍 Testing source filtering...")
        sources_to_test = ["pubmed", "biorxiv", "medrxiv"]
        
        for source in sources_to_test:
            results = manager.filter_by_metadata({"source_name": source})
            if results:
                print(f"   {source}: {len(results)} documents")
                # Show sample document
                sample = results[0]
                print(f"     Sample: {sample['metadata'].get('title', 'N/A')[:50]}...")
            else:
                print(f"   {source}: 0 documents")
        
        # Test similarity search (if we have embeddings)
        print("\n🔍 Testing similarity search...")
        try:
            # Get a sample document to use as query
            sample_results = manager.filter_by_metadata({})
            if sample_results:
                sample_doc = sample_results[0]
                print(f"   Using sample document: {sample_doc['metadata'].get('title', 'N/A')[:50]}...")
                
                # Note: This would require an embedding model to convert text to vector
                # For now, we'll just show that filtering works
                print("   ⚠️  Similarity search requires embedding model (not implemented)")
            else:
                print("   No documents available for search test")
        except Exception as e:
            print(f"   ⚠️  Search test error: {e}")
        
        print("\n✅ Vector database test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Vector database test failed: {e}")
        return False

def show_data_status():
    """Show the status of available data"""
    print("\n📊 Data Status:")
    
    # Check PubMed data
    if os.path.exists("pubmed_embeddings.json"):
        size = os.path.getsize("pubmed_embeddings.json") / 1024
        print(f"   PubMed: ✅ Available ({size:.1f} KB)")
    else:
        print("   PubMed: ❌ Not available")
    
    # Check xrvix data
    if os.path.exists("xrvix_embeddings"):
        metadata_path = os.path.join("xrvix_embeddings", "metadata.json")
        if os.path.exists(metadata_path):
            try:
                # Try to get file size first to avoid loading huge files
                file_size = os.path.getsize(metadata_path) / (1024 * 1024)  # Size in MB
                print(f"   xrvix: ✅ Available (metadata: {file_size:.1f} MB)")
                
                # Also check if there are batch files
                biorxiv_dir = os.path.join("xrvix_embeddings", "biorxiv")
                if os.path.exists(biorxiv_dir):
                    batch_files = [f for f in os.listdir(biorxiv_dir) if f.startswith("batch_") and f.endswith(".json")]
                    print(f"   xrvix: ✅ Batch files available ({len(batch_files)} batches in biorxiv/)")
                else:
                    print("   xrvix: ⚠️  No biorxiv directory found")
            except Exception as e:
                print(f"   xrvix: ✅ Available (metadata error: {e})")
        else:
            print("   xrvix: ⚠️  Directory exists but no metadata.json")
    else:
        print("   xrvix: ❌ Not available")
    
    # Check ChromaDB
    if os.path.exists("chroma_db"):
        try:
            manager = ChromaDBManager()
            collections = manager.list_collections()
            if collections:
                stats = manager.get_collection_stats()
                total_docs = stats.get('total_documents', 0)
                print(f"   ChromaDB: ✅ Available ({total_docs} documents)")
            else:
                print("   ChromaDB: ⚠️  Available but empty")
        except:
            print("   ChromaDB: ⚠️  Available but error accessing")
    else:
        print("   ChromaDB: ❌ Not available")

def quick_list_batches():
    """Quick function to list loaded batches without interactive menu."""
    print("\n📊 Quick Batch Listing:")
    try:
        manager = ChromaDBManager()
        collections = manager.list_collections()
        
        if not collections:
            print("❌ No collections found in the vector database.")
            return
        
        # Use the first collection
        collection_name = collections[0]
        if not manager.switch_collection(collection_name):
            print(f"❌ Failed to switch to collection: {collection_name}")
            return
        
        print(f"📚 Collection: {collection_name}")
        
        # Get basic stats
        stats = manager.get_collection_stats()
        print(f"   Total documents: {stats.get('total_documents', 0)}")
        
        # List batches
        batch_info = manager.list_loaded_batches()
        if batch_info:
            print("\n📦 Loaded Batches:")
            for source, batches in batch_info.items():
                print(f"   {source}: {len(batches)} batches")
                for batch in batches:
                    print(f"     - {batch}")
        else:
            print("\n📦 No batches found in this collection.")
        
        # Show detailed statistics
        batch_stats = manager.get_batch_statistics()
        if batch_stats and batch_stats.get('sources'):
            print("\n📊 Source Breakdown:")
            for source, stats in batch_stats['sources'].items():
                print(f"   {source}: {stats['total_documents']} documents, {len(stats['batches'])} batches")
        
    except Exception as e:
        print(f"❌ Error listing batches: {e}")

def list_loaded_batches():
    """List all loaded batches from the vector database."""
    print("\n📊 Listing Loaded Batches:")
    try:
        manager = ChromaDBManager()
        collections = manager.list_collections()
        
        if not collections:
            print("❌ No collections found in the vector database.")
            return
        
        print("Available collections:")
        for i, collection_name in enumerate(collections, 1):
            print(f"{i}. {collection_name}")
            
        while True:
            try:
                collection_choice = input("\nSelect a collection to view batches (1-" + str(len(collections)) + "), or 'q' to quit: ").strip()
                
                if collection_choice.lower() == 'q':
                    break
                
                if not collection_choice.isdigit() or int(collection_choice) < 1 or int(collection_choice) > len(collections):
                    print("❌ Please enter a valid number or 'q' to quit.")
                    continue
                
                collection_index = int(collection_choice) - 1
                selected_collection = collections[collection_index]
                
                if not manager.switch_collection(selected_collection):
                    print(f"❌ Failed to switch to collection: {selected_collection}")
                    continue
                
                print(f"\n📊 Collection: {selected_collection}")
                stats = manager.get_collection_stats()
                print(f"   Total documents: {stats.get('total_documents', 0)}")
                print(f"   Metadata keys: {stats.get('sample_metadata_keys', [])}")
                
                # List all batches (show all, even if very long)
                print("\n📦 Loaded Batches:")
                batch_info = manager.list_loaded_batches()
                if batch_info:
                    for source, batches in batch_info.items():
                        print(f"   {source}: {len(batches)} batches")
                        if len(batches) > 50:
                            print(f"     ⚠️  Large number of batches, listing all:")
                        for batch in batches:
                            print(f"     - {batch}")
                else:
                    print("   No batches found in this collection.")
                
                # Show detailed statistics
                print("\n📊 Detailed Statistics:")
                batch_stats = manager.get_batch_statistics()
                if batch_stats:
                    print(f"   Total documents: {batch_stats.get('total_documents', 0)}")
                    for source, stats in batch_stats.get('sources', {}).items():
                        print(f"   {source}: {stats['total_documents']} documents, {len(stats['batches'])} batches")
                        if stats['batch_details']:
                            print(f"     Batch details: {stats['batch_details']}")
                else:
                    print("   No statistics available.")
                
                print("\n💡 To load a specific batch, you would typically use the ChromaDBManager's load_embeddings_from_json or add_embeddings_from_directory methods.")
                print("   For example: manager.load_embeddings_from_json('path/to/batch_embeddings.json')")
                print("   Or: manager.add_embeddings_from_directory('path/to/batch_embeddings')")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error listing batches: {e}")
                break
    except Exception as e:
        print(f"❌ Error accessing vector database for batch listing: {e}")

def main():
    print("🎯 Welcome to the Master Data Processor!")
    print("This will process your scientific papers and create searchable embeddings.")
    print()
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above and try again.")
        return
    
    while True:
        show_menu()
        choice = get_user_choice()
        
        if choice == '1':
            print("\n🔄 Processing PubMed papers...")
            try:
                process_pubmed()
                print("✅ PubMed processing complete!")
            except Exception as e:
                print(f"❌ Error processing PubMed: {e}")
        
        elif choice == '2':
            print("\n🔄 Processing xrvix dumps...")
            try:
                process_xrvix()
                print("✅ xrvix processing complete!")
            except Exception as e:
                print(f"❌ Error processing xrvix: {e}")
        
        elif choice == '3':
            print("\n🔄 Processing both PubMed and xrvix...")
            try:
                print("\n📊 Step 1: Processing PubMed...")
                process_pubmed()
                print("✅ PubMed processing complete!")

                print("\n📊 Step 2: Processing xrvix...")
                process_xrvix()
                print("✅ xrvix processing complete!")

                print("\n🎉 All processing complete!")
            except Exception as e:
                print(f"❌ Error during processing: {e}")

        elif choice == '4':
            show_menu()

        elif choice == '5':
            show_data_status()

        elif choice == '6':
            list_loaded_batches()

        elif choice == '7':
            print("\n🚀 Launching Enhanced RAG System with Hypothesis Generation & Critique...\n")
            try:
                subprocess.run([sys.executable, "enhanced_rag_with_chromadb.py"])
            except Exception as e:
                print(f"❌ Failed to launch Enhanced RAG System: {e}")
        
        elif choice == '8':
            print("👋 Exiting. Goodbye!")
            break
        
        # Ask if user wants to continue
        if choice in ['1', '2', '3', '4', '5', '6', '7', '8']:
            print(f"\n💡 Next steps:")
            if choice in ['1', '2', '3']:
                print(f"   - Run option 4 to load data into vector database")
                print(f"   - Run option 5 to test the vector database")
            if choice == '6':
                print(f"   - Restart processing with new settings")
            if choice == '7':
                print(f"   - The Enhanced RAG System includes hypothesis generation and critique")
                print(f"   - Use 'hypothesis' command to generate scientific hypotheses")
                print(f"   - Use 'critique' command to evaluate hypotheses")
            print(f"   - Run 'enhanced_rag_with_chromadb.py' directly for advanced features")
            print()
            
            continue_choice = input("Continue with another operation? (y/n): ").strip().lower()
            if continue_choice != 'y':
                print("👋 Goodbye!")
                break

if __name__ == "__main__":
    main() 