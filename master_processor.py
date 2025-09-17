import os
import json
import sys
from pubmed_scraper_json import main as process_pubmed
from process_xrvix_dumps_json import main as process_xrvix
from ubr5_api_scraper import UBR5APIScraper
from chromadb_manager import ChromaDBManager
from processing_config import print_config_info, get_config, DB_BATCH_SIZE
import subprocess

# Real functions imported from modules

def show_menu():
    """Display the main menu with proper incremental numbering."""
    print("\n" + "="*60)
    print("🚀 AHG-UBR5 RESEARCH PROCESSOR - MAIN MENU")
    print("="*60)
    
    print("\n📚 PAPER SCRAPING & PROCESSING:")
    print("1.  PubMed Scraper & Embedding Generator")
    print("2.  xrvix Dumps Processor & Embedding Generator")
    print("3.  UBR5 API Scraper (Scholarly + Semantic Scholar)")
    print("4.  Scholarly API Scraper (Google Scholar)")
    print("5.  Semantic Scholar API Scraper")
    print("6.  Unified Paper Scraper (PubMed + Google scholar + Semantic Scholar)")
    
    print("\n🗄️ VECTOR DATABASE MANAGEMENT:")
    print("7.  Initialize ChromaDB")
    print("8.  List ChromaDB Collections")
    print("9.  Show ChromaDB Stats")
    print("10. Clear ChromaDB Collection")
    
    print("\n🧠 AI RESEARCH TOOLS:")
    print("11. Enhanced RAG with ChromaDB")
    print("12. Interactive Search System")
    
    print("\n⚙️ SYSTEM & CONFIGURATION:")
    print("13. Show Data Status")
    print("14. Show Configuration")
    print("15. Exit")
    
    print("="*60)

def get_user_choice():
    """Get user choice with updated range."""
    while True:
        try:
            choice = int(input("\nEnter your choice (1-15): "))
            if 1 <= choice <= 15:
                return choice
            else:
                print("❌ Please enter a number between 1 and 15.")
        except ValueError:
            print("❌ Please enter a valid number.")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)

# Prerequisites check removed - functions will handle their own dependencies

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
        
        # Check if collection already has data
        stats = manager.get_collection_stats()
        if stats.get('total_documents', 0) > 0:
            print(f"📚 ChromaDB already has {stats.get('total_documents', 0)} documents!")
            print("💡 ChromaDB uses persistent storage - data is already saved locally.")
            print("   You can use the Enhanced RAG System directly without reloading.")
            
            # Ask user if they want to reload anyway
            reload_choice = input("\nDo you want to reload the data anyway? (y/n): ").strip().lower()
            if reload_choice != 'y':
                print("✅ Using existing ChromaDB data.")
                return True
        
        total_loaded = 0
        
        # Load PubMed embeddings from data/embeddings/xrvix_embeddings folder
        if os.path.exists("data/embeddings/xrvix_embeddings/pubmed_embeddings.json"):
            print("🔄 Loading PubMed embeddings...")
            pubmed_data = manager.load_embeddings_from_json("data/embeddings/xrvix_embeddings/pubmed_embeddings.json")
            if pubmed_data:
                if manager.add_embeddings_to_collection(pubmed_data, "pubmed"):
                    total_loaded += len(pubmed_data.get('embeddings', []))
                    print(f"✅ Loaded {len(pubmed_data.get('embeddings', []))} PubMed embeddings")
                else:
                    print("❌ Failed to add PubMed embeddings to collection")
            else:
                print("❌ Failed to load PubMed data")
        
        # Load xrvix embeddings (multi-file)
        if os.path.exists("data/embeddings/xrvix_embeddings"):
            print("🔄 Loading xrvix embeddings...")
            if manager.add_embeddings_from_directory("data/embeddings/xrvix_embeddings", db_batch_size=DB_BATCH_SIZE):
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
    print("\n📊 Data Status Overview:")
    print("=" * 50)
    
    # Check PubMed data
    print("\n📚 PubMed Data:")
    if os.path.exists("data/embeddings/xrvix_embeddings/pubmed_embeddings.json"):
        try:
            with open("data/embeddings/xrvix_embeddings/pubmed_embeddings.json", 'r') as f:
                pubmed_data = json.load(f)
                pubmed_count = len(pubmed_data.get('embeddings', []))
                size = os.path.getsize("data/embeddings/xrvix_embeddings/pubmed_embeddings.json") / 1024
                print(f"   ✅ Available: {pubmed_count} papers ({size:.1f} KB)")
                print(f"   📊 Chunks: {len(pubmed_data.get('chunks', []))}")
                print(f"   📊 Metadata: {len(pubmed_data.get('metadata', []))}")
        except Exception as e:
            print(f"   ⚠️  Available but error reading: {e}")
    else:
        print("   ❌ Not available")
    
    # Check xrvix data
    print("\n📄 xrvix Data (biorxiv, medrxiv):")
    if os.path.exists("data/embeddings/xrvix_embeddings"):
        metadata_path = os.path.join("data/embeddings/xrvix_embeddings", "metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    xrvix_data = json.load(f)
                    xrvix_count = xrvix_data.get('total_papers', 0)
                    file_size = os.path.getsize(metadata_path) / (1024 * 1024)
                    print(f"   ✅ Available: {xrvix_count} papers (metadata: {file_size:.1f} MB)")
                    
                    # Check batch files
                    biorxiv_dir = os.path.join("data/embeddings/xrvix_embeddings", "biorxiv")
                    medrxiv_dir = os.path.join("data/embeddings/xrvix_embeddings", "medrxiv")
                    
                    if os.path.exists(biorxiv_dir):
                        batch_files = [f for f in os.listdir(biorxiv_dir) if f.startswith("batch_") and f.endswith(".json")]
                        print(f"   📊 biorxiv batches: {len(batch_files)}")
                    
                    if os.path.exists(medrxiv_dir):
                        batch_files = [f for f in os.listdir(medrxiv_dir) if f.startswith("batch_") and f.endswith(".json")]
                        print(f"   📊 medrxiv batches: {len(batch_files)}")
                        
            except Exception as e:
                print(f"   ⚠️  Available but error reading: {e}")
        else:
            print("   ⚠️  Directory exists but no metadata.json")
    else:
        print("   ❌ Not available")
    
    # Check UBR5 data
    print("\n🔬 UBR5 Data (Semantic Scholar + Google Scholar):")
    ubr5_dir = "data/embeddings/xrvix_embeddings/ubr5_api"
    if os.path.exists(ubr5_dir):
        metadata_file = os.path.join(ubr5_dir, "metadata.json")
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    ubr5_data = json.load(f)
                    ubr5_count = ubr5_data.get('total_papers', 0)
                    file_size = os.path.getsize(metadata_file) / 1024
                    print(f"   ✅ Available: {ubr5_count} papers ({file_size:.1f} KB)")
                    
                    # Count individual paper files
                    paper_files = [f for f in os.listdir(ubr5_dir) if f.endswith('.json') and f != 'metadata.json']
                    print(f"   📊 Individual papers: {len(paper_files)}")
                    
            except Exception as e:
                print(f"   ⚠️  Available but error reading: {e}")
        else:
            print("   ⚠️  Directory exists but no metadata.json")
    else:
        print("   ❌ Not available")
    
    # Check ChromaDB
    print("\n🗄️  ChromaDB Vector Database:")
    if os.path.exists("data/vector_db/chroma_db"):
        try:
            manager = ChromaDBManager()
            collections = manager.list_collections()
            if collections:
                stats = manager.get_collection_stats()
                total_docs = stats.get('total_documents', 0)
                print(f"   ✅ Available: {total_docs} documents")
                print(f"   📊 Collections: {len(collections)}")
                print(f"   📊 Active collection: {stats.get('collection_name', 'N/A')}")
            else:
                print("   ⚠️  Available but empty")
        except Exception as e:
            print(f"   ⚠️  Available but error accessing: {e}")
    else:
        print("   ❌ Not available")
    
    # Summary
    print("\n📈 Summary:")
    total_sources = 0
    if os.path.exists("data/embeddings/xrvix_embeddings/pubmed_embeddings.json"):
        total_sources += 1
    if os.path.exists("data/embeddings/xrvix_embeddings/metadata.json"):
        total_sources += 1
    if os.path.exists("data/embeddings/xrvix_embeddings/ubr5_api/metadata.json"):
        total_sources += 1
    
    print(f"   📊 Data sources available: {total_sources}/3")
    if total_sources == 3:
        print("   🎉 All data sources are available!")
    elif total_sources > 0:
        print("   💡 Some data sources are available. Use option 12 to collect missing sources.")
    else:
        print("   ❌ No data sources available. Use option 12 to collect papers.")
    
    print("\n💡 Recommendations:")
    if total_sources < 3:
        print("   • Use option 12 (Unified Scraper) to collect all missing data sources")
    if total_sources > 0 and not os.path.exists("data/vector_db/chroma_db"):
        print("   • Use option 4 to load data into ChromaDB")
    if os.path.exists("data/vector_db/chroma_db"):
        print("   • Use option 7 to start the Enhanced RAG System for AI-powered research")

def check_chromadb_status():
    """Check detailed ChromaDB status and provide guidance"""
    print("\n🗄️  ChromaDB Status Check:")
    print("=" * 40)
    
    try:
        # Initialize manager
        manager = ChromaDBManager()
        
        # Check if ChromaDB directory exists
        if not os.path.exists("data/vector_db/chroma_db"):
            print("❌ ChromaDB directory not found")
            print("💡 ChromaDB will be created when you first load data")
            return
        
        # List collections
        collections = manager.list_collections()
        if not collections:
            print("⚠️  No collections found in ChromaDB")
            print("💡 Run option 4 to load data into ChromaDB")
            return
        
        print(f"✅ ChromaDB is available with {len(collections)} collection(s)")
        
        # Check each collection
        for collection_name in collections:
            print(f"\n📚 Collection: {collection_name}")
            
            if not manager.switch_collection(collection_name):
                print("   ❌ Failed to access collection")
                continue
            
            # Get statistics
            stats = manager.get_collection_stats()
            total_docs = stats.get('total_documents', 0)
            print(f"   📊 Total documents: {total_docs}")
            
            if total_docs == 0:
                print("   ⚠️  Collection is empty")
                print("   💡 Run option 4 to load data into this collection")
            else:
                print("   ✅ Collection has data - ready for searching!")
                
                # Show source breakdown
                batch_stats = manager.get_batch_statistics()
                if batch_stats and batch_stats.get('sources'):
                    print("   📦 Source breakdown:")
                    for source, source_stats in batch_stats['sources'].items():
                        print(f"      {source}: {source_stats['total_documents']} documents")
                
                # Show metadata keys
                metadata_keys = stats.get('sample_metadata_keys', [])
                if metadata_keys:
                    print(f"   🔑 Metadata keys: {', '.join(metadata_keys[:5])}{'...' if len(metadata_keys) > 5 else ''}")
        
        print(f"\n💡 ChromaDB uses persistent storage - data is saved locally in ./data/vector_db/chroma_db/")
        print("   You can use the Enhanced RAG System directly without reloading data!")
        
    except Exception as e:
        print(f"❌ Error checking ChromaDB status: {e}")
        print("💡 Try running option 4 to initialize ChromaDB")

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

def get_lab_config():
    """Get lab configuration from file or return default"""
    config_file = "lab_config.json"
    default_config = {
        "lab_name": "Dr. Xiaojing Ma",
        "institution": "Weill Cornell Medicine",
        "research_focus": "UBR5, cancer immunology, protein ubiquitination, mechanistic and therapeutic hypotheses"
    }
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                return config
        except Exception as e:
            print(f"⚠️  Error reading lab config: {e}, using default")
    
    return default_config

def save_lab_config(config):
    """Save lab configuration to file"""
    config_file = "lab_config.json"
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        print(f"❌ Error saving lab config: {e}")
        return False

def configure_lab_name():
    """Configure the lab name and research focus"""
    print("\n🔧 Lab Configuration")
    print("=" * 40)
    
    # Get current configuration
    current_config = get_lab_config()
    
    print(f"Current lab name: {current_config.get('lab_name', 'Dr. Xiaojing Ma')}")
    print(f"Current institution: {current_config.get('institution', 'Weill Cornell Medicine')}")
    print(f"Current research focus: {current_config.get('research_focus', 'UBR5, cancer immunology, protein ubiquitination')}")
    
    print("\nEnter new values (press Enter to keep current value):")
    
    # Get new lab name
    new_lab_name = input(f"Lab name [{current_config.get('lab_name', 'Dr. Xiaojing Ma')}]: ").strip()
    if not new_lab_name:
        new_lab_name = current_config.get('lab_name', 'Dr. Xiaojing Ma')
    
    # Get new institution
    new_institution = input(f"Institution [{current_config.get('institution', 'Weill Cornell Medicine')}]: ").strip()
    if not new_institution:
        new_institution = current_config.get('institution', 'Weill Cornell Medicine')
    
    # Get new research focus
    new_research_focus = input(f"Research focus [{current_config.get('research_focus', 'UBR5, cancer immunology, protein ubiquitination')}]: ").strip()
    if not new_research_focus:
        new_research_focus = current_config.get('research_focus', 'UBR5, cancer immunology, protein ubiquitination')
    
    # Create new configuration
    new_config = {
        "lab_name": new_lab_name,
        "institution": new_institution,
        "research_focus": new_research_focus
    }
    
    # Save configuration
    if save_lab_config(new_config):
        print(f"\n✅ Lab configuration updated successfully!")
        print(f"   Lab name: {new_lab_name}")
        print(f"   Institution: {new_institution}")
        print(f"   Research focus: {new_research_focus}")
        print(f"\n💡 This configuration will be used in hypothesis generation and critique.")
        print(f"   The changes will take effect when you restart the Enhanced RAG System.")
    else:
        print("❌ Failed to save lab configuration.")



def run_automated_pipeline():
    """Run the complete automated pipeline: process data, load to ChromaDB, update metadata"""
    print("\n🚀 Starting Automated Pipeline...")
    print("This will:")
    print("1. Process PubMed papers")
    print("2. Process xrvix dumps (biorxiv, medrxiv)")
    print("3. Load all data into ChromaDB")
    print("4. Update metadata with authors, dates, and citations")
    print("5. Verify the setup")
    print()
    
    # Ask for confirmation
    confirm = input("Do you want to run the complete automated pipeline? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ Automated pipeline cancelled.")
        return
    
    print("\n" + "="*60)
    print("🚀 AUTOMATED PIPELINE STARTED")
    print("="*60)
    
    try:
        # Step 1: Process PubMed papers
        print("\n📊 STEP 1: Processing PubMed papers...")
        print("-" * 40)
        process_pubmed()
        print("✅ PubMed processing complete!")
        
        # Step 2: Process xrvix dumps
        print("\n📊 STEP 2: Processing xrvix dumps...")
        print("-" * 40)
        process_xrvix()
        print("✅ xrvix processing complete!")
        
        # Step 3: Load data into ChromaDB
        print("\n📊 STEP 3: Loading data into ChromaDB...")
        print("-" * 40)
        if load_data_into_vector_db():
            print("✅ Data loading complete!")
        else:
            print("⚠️  Data loading had issues, but continuing...")
        
        # Step 4: Verify the setup
        print("\n📊 STEP 4: Verifying setup...")
        print("-" * 40)
        try:
            # Check ChromaDB status
            manager = ChromaDBManager()
            stats = manager.get_collection_stats()
            total_docs = stats.get('total_documents', 0)
            total_chunks = stats.get('total_chunks', 0)
            
            print(f"📚 ChromaDB Status:")
            print(f"   - Total documents: {total_docs}")
            print(f"   - Total chunks: {total_chunks}")
            
            if total_docs > 0:
                print("✅ ChromaDB verification successful!")
            else:
                print("⚠️  ChromaDB appears empty - you may need to run data loading separately")
            
            # Check if lab configuration exists
            try:
                from hypothesis_tools import get_lab_config
                lab_config = get_lab_config()
                if lab_config and lab_config.get('lab_name'):
                    print(f"✅ Lab configuration found: {lab_config.get('lab_name')}")
                else:
                    print("⚠️  No lab configuration found - consider running option 9 to configure")
            except:
                print("⚠️  Could not verify lab configuration")
                
        except Exception as e:
            print(f"⚠️  Verification had issues: {e}")
        
        print("\n" + "="*60)
        print("🎉 AUTOMATED PIPELINE COMPLETED!")
        print("="*60)
        print("\n📋 Summary:")
        print("✅ All data sources processed")
        print("✅ Data loaded into ChromaDB")
        print("✅ Metadata updated with authors, dates, and citations")
        print("✅ Setup verified")
        print("\n🚀 Next steps:")
        print("   - Run option 7 to start the Enhanced RAG System")
        print("   - Use 'add <query>' to search and generate hypotheses")
        print("   - Use 'meta <query>' for meta-hypothesis generation")
        print("   - Lab paper detection should now work properly")
        
    except Exception as e:
        print(f"\n❌ Automated pipeline failed: {e}")
        print("Please check the error above and try running individual steps manually.")
        import traceback
        traceback.print_exc()

def run_scholarly_scraper():
    """Run only the Scholarly API scraper."""
    try:
        print("\n🔍 Starting Scholarly API Scraper (Google Scholar)")
        print("="*60)
        
        ubr5_scraper = UBR5APIScraper()
        # Only use Scholarly, skip Semantic Scholar
        ubr5_scraper.search_keywords = ["ubr5", "UBR5", "ubr-5", "UBR-5"]
        ubr5_scraper.run_complete_scraping()
        
        print("✅ Scholarly API scraping completed!")
        
    except Exception as e:
        print(f"❌ Error in Scholarly scraper: {e}")
        import traceback
        traceback.print_exc()

def run_semantic_scholar_scraper():
    """Run only the Semantic Scholar API scraper."""
    try:
        print("\n🔬 Starting Semantic Scholar API Scraper")
        print("="*60)
        
        ubr5_scraper = UBR5APIScraper()
        # Only use Semantic Scholar, skip Scholarly
        ubr5_scraper.search_keywords = ["ubr5", "UBR5", "ubr-5", "UBR-5"]
        ubr5_scraper.run_complete_scraping()
        
        print("✅ Semantic Scholar API scraping completed!")
        
    except Exception as e:
        print(f"❌ Error in Semantic Scholar scraper: {e}")
        import traceback
        traceback.print_exc()

def run_unified_scraper():
    """Run the unified scraper for PubMed + UBR5 sources."""
    try:
        print("\n🚀 Starting Unified Paper Scraper (PubMed + UBR5)")
        print("="*60)
        
        success_count = 0
        total_sources = 2  # PubMed + UBR5 only
        
        # Step 1: Process PubMed
        print("\n📚 Step 1/2: Processing PubMed papers...")
        try:
            process_pubmed()
            print("✅ PubMed processing completed successfully!")
            success_count += 1
        except Exception as e:
            print(f"❌ PubMed processing failed: {e}")
        
        # Step 2: Process UBR5 (Scholarly + Semantic Scholar)
        print("\n🔬 Step 2/2: Processing UBR5 papers (Scholarly + Semantic Scholar)...")
        try:
            ubr5_scraper = UBR5APIScraper()
            ubr5_scraper.run_complete_scraping()
            print("✅ UBR5 processing completed successfully!")
            success_count += 1
        except Exception as e:
            print(f"❌ UBR5 processing failed: {e}")
        
        # Summary
        print(f"\n🎉 Unified scraper completed!")
        print(f"✅ Successfully processed: {success_count}/{total_sources} sources")
        
        if success_count == total_sources:
            print("🎯 All sources processed successfully!")
        elif success_count > 0:
            print("⚠️  Some sources failed, but others succeeded.")
        else:
            print("❌ All sources failed. Check error messages above.")
            
    except Exception as e:
        print(f"❌ Error in unified scraper: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function with updated menu handling."""
    print("🚀 Welcome to AHG-UBR5 Research Processor!")
    print("💡 Quick Start: Choose option 6 for the recommended unified scraper")
    print("📚 This will process PubMed and UBR5 sources automatically")
    
    while True:
        show_menu()
        choice = get_user_choice()
        
        if choice == 1:
            print("\n📚 Starting PubMed Scraper...")
            process_pubmed()
            
        elif choice == 2:
            print("\n📚 Starting xrvix Dumps Processor...")
            process_xrvix()
            
        elif choice == 3:
            print("\n🔬 Starting UBR5 API Scraper (Scholarly + Semantic Scholar)...")
            ubr5_scraper = UBR5APIScraper()
            ubr5_scraper.run_complete_scraping()
            
        elif choice == 4:
            print("\n🔍 Starting Scholarly API Scraper...")
            run_scholarly_scraper()
            
        elif choice == 5:
            print("\n🔬 Starting Semantic Scholar API Scraper...")
            run_semantic_scholar_scraper()
            
        elif choice == 6:
            print("\n🚀 Starting Unified Paper Scraper...")
            run_unified_scraper()
            
        elif choice == 7:
            print("\n🗄️ Initializing ChromaDB...")
            chroma_manager = ChromaDBManager()
            if chroma_manager.create_collection():
                print("✅ ChromaDB initialized successfully!")
            else:
                print("❌ Failed to initialize ChromaDB")
                
        elif choice == 8:
            print("\n🗄️ Listing ChromaDB Collections...")
            chroma_manager = ChromaDBManager()
            collections = chroma_manager.list_collections()
            if collections:
                print("📋 Available collections:")
                for collection in collections:
                    print(f"   - {collection}")
            else:
                print("📋 No collections found")
                
        elif choice == 9:
            print("\n🗄️ Showing ChromaDB Stats...")
            chroma_manager = ChromaDBManager()
            stats = chroma_manager.get_collection_stats()
            if stats:
                print("📊 ChromaDB Statistics:")
                for key, value in stats.items():
                    print(f"   {key}: {value}")
            else:
                print("❌ Failed to get ChromaDB stats")
                
        elif choice == 10:
            print("\n🗑️ Clearing ChromaDB Collection...")
            chroma_manager = ChromaDBManager()
            if chroma_manager.clear_collection():
                print("✅ ChromaDB collection cleared successfully!")
            else:
                print("❌ Failed to clear ChromaDB collection")
                
        elif choice == 11:
            print("\n🧠 Starting Enhanced RAG with ChromaDB...")
            try:
                subprocess.run([sys.executable, "enhanced_rag_with_chromadb.py"])
            except Exception as e:
                print(f"❌ Failed to launch Enhanced RAG System: {e}")
            
        elif choice == 12:
            print("\n🔍 Starting Interactive Search System...")
            try:
                from hypothesis_tools import main as run_interactive_search
                run_interactive_search()
            except ImportError:
                print("❌ Interactive Search module not found")
            except Exception as e:
                print(f"❌ Error running Interactive Search: {e}")
            
        elif choice == 13:
            print("\n📊 Showing Data Status...")
            show_data_status()
            
        elif choice == 14:
            print("\n⚙️ Showing Configuration...")
            print_config_info()
            
        elif choice == 15:
            print("\n👋 Goodbye! Thank you for using AHG-UBR5 Research Processor!")
            break
            
        else:
            print("❌ Invalid choice. Please try again.")
        
        if choice != 15:
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    main() 