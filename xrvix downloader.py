from paperscraper.get_dumps import biorxiv, medrxiv
import pkg_resources
import os
from tqdm.auto import tqdm
import time

# Define custom download directory within AHG-UBR5
CUSTOM_DUMP_DIR = "paperscraper_dumps"

def ensure_dump_directory():
    """Ensure the custom dump directory exists"""
    if not os.path.exists(CUSTOM_DUMP_DIR):
        os.makedirs(CUSTOM_DUMP_DIR)
        print(f"📁 Created custom dump directory: {CUSTOM_DUMP_DIR}")
    else:
        print(f"📁 Using existing custom dump directory: {CUSTOM_DUMP_DIR}")

def download_with_progress(download_func, name, expected_time_minutes):
    """Download with progress bar simulation"""
    print(f"\nStarting {name} download (expected time: ~{expected_time_minutes} minutes)")
    
    # Create progress bar for the download process
    with tqdm(total=100, desc=f"Downloading {name}", unit="%") as pbar:
        # Simulate progress updates (since we can't track actual download progress)
        for i in range(10):
            time.sleep(expected_time_minutes * 6)  # Convert minutes to seconds, divided by 10 updates
            pbar.update(10)
        
        # Execute the actual download
        download_func()
        pbar.update(100 - pbar.n)  # Complete the progress bar
    
    # Find and print the downloaded file location
    # First check the custom directory
    custom_dump_files = [f for f in os.listdir(CUSTOM_DUMP_DIR) if f.startswith(name)]
    
    if custom_dump_files:
        # Files were moved to custom directory
        latest_file = sorted(custom_dump_files, reverse=True)[0]
        file_path = os.path.join(CUSTOM_DUMP_DIR, latest_file)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
        print(f"✅ {name} download complete!")
        print(f"📁 File location: {file_path}")
        print(f"📊 File size: {file_size:.1f} MB")
    else:
        # Check if files are in default location and move them
        default_dump_root = pkg_resources.resource_filename("paperscraper", "server_dumps")
        if os.path.exists(default_dump_root):
            default_dump_files = [f for f in os.listdir(default_dump_root) if f.startswith(name)]
            if default_dump_files:
                latest_file = sorted(default_dump_files, reverse=True)[0]
                source_path = os.path.join(default_dump_root, latest_file)
                dest_path = os.path.join(CUSTOM_DUMP_DIR, latest_file)
                
                # Move file to custom directory
                import shutil
                shutil.move(source_path, dest_path)
                
                file_size = os.path.getsize(dest_path) / (1024 * 1024)  # Convert to MB
                print(f"✅ {name} download complete!")
                print(f"📁 File moved to: {dest_path}")
                print(f"📊 File size: {file_size:.1f} MB")
            else:
                print(f"❌ {name} download failed - no file found")
        else:
            print(f"❌ {name} download failed - no file found")

def move_existing_dumps():
    """Move any existing dumps from default location to custom directory"""
    default_dump_root = pkg_resources.resource_filename("paperscraper", "server_dumps")
    
    if os.path.exists(default_dump_root):
        existing_files = os.listdir(default_dump_root)
        if existing_files:
            print(f"🔄 Moving {len(existing_files)} existing dump files to custom directory...")
            import shutil
            
            for filename in existing_files:
                source_path = os.path.join(default_dump_root, filename)
                dest_path = os.path.join(CUSTOM_DUMP_DIR, filename)
                
                if not os.path.exists(dest_path):  # Don't overwrite existing files
                    shutil.move(source_path, dest_path)
                    print(f"   📁 Moved: {filename}")
                else:
                    print(f"   ⚠️  Skipped: {filename} (already exists in custom directory)")
            
            print(f"✅ All existing dumps moved to: {CUSTOM_DUMP_DIR}")

### downloading xiv dumps
print("=== Starting xrvix Downloads ===")

# Ensure custom directory exists
ensure_dump_directory()

# Move any existing dumps to custom directory
move_existing_dumps()

# Download medrxiv
download_with_progress(medrxiv, "medrxiv", 30)

# Download biorxiv  
download_with_progress(biorxiv, "biorxiv", 60)

print("\n=== All Downloads Complete ===")
print(f"📂 Dump files are now stored in: {os.path.abspath(CUSTOM_DUMP_DIR)}")

# List all files in custom directory
if os.path.exists(CUSTOM_DUMP_DIR):
    custom_files = os.listdir(CUSTOM_DUMP_DIR)
    if custom_files:
        print(f"\n📋 Files in custom dump directory:")
        for filename in sorted(custom_files):
            file_path = os.path.join(CUSTOM_DUMP_DIR, filename)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
            print(f"   📄 {filename} ({file_size:.1f} MB)")
    else:
        print(f"\n📋 Custom dump directory is empty")
else:
    print(f"\n❌ Custom dump directory not found")