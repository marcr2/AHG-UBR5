from paperscraper.get_dumps import biorxiv, medrxiv
import pkg_resources
import os
from tqdm.auto import tqdm
import time

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
    dump_root = pkg_resources.resource_filename("paperscraper", "server_dumps")
    dump_files = [f for f in os.listdir(dump_root) if f.startswith(name)]
    if dump_files:
        latest_file = sorted(dump_files, reverse=True)[0]
        file_path = os.path.join(dump_root, latest_file)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
        print(f"✅ {name} download complete!")
        print(f"📁 File location: {file_path}")
        print(f"📊 File size: {file_size:.1f} MB")
    else:
        print(f"❌ {name} download failed - no file found")

### downloading xiv dumps
print("=== Starting xrvix Downloads ===")

# Download medrxiv
download_with_progress(medrxiv, "medrxiv", 30)

# Download biorxiv  
download_with_progress(biorxiv, "biorxiv", 60)

print("\n=== All Downloads Complete ===")
print("📂 Dump files are stored in:", pkg_resources.resource_filename("paperscraper", "server_dumps"))