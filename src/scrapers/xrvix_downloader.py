#!/usr/bin/env python3
"""
XRXiv Downloader for AHG-UBR5 Research Processor
Downloads preprint dumps from Biorxiv and Medrxiv using paperscraper package
"""

import os
import json
from datetime import datetime
import logging
import subprocess
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class XRXivDownloader:
    """Downloads preprint dumps from Biorxiv and Medrxiv servers using paperscraper."""
    
    def __init__(self):
        self.dump_dir = "data/scraped_data/paperscraper_dumps"
        self.server_dumps_dir = os.path.join(self.dump_dir, "server_dumps")
        self.ensure_dump_directory()
    
    def ensure_dump_directory(self):
        """Ensure the dump directory exists."""
        os.makedirs(self.dump_dir, exist_ok=True)
        os.makedirs(self.server_dumps_dir, exist_ok=True)
        logger.info(f"📁 Dump directory: {self.dump_dir}")
        logger.info(f"📁 Server dumps directory: {self.server_dumps_dir}")
    
    def install_paperscraper(self):
        """Install paperscraper package if not available."""
        try:
            import paperscraper
            logger.info("✅ paperscraper package is available")
            return True
        except ImportError:
            logger.info("📦 Installing paperscraper package...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "paperscraper"])
                logger.info("✅ paperscraper package installed successfully")
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed to install paperscraper: {e}")
                return False
    
    def download_biorxiv_dump(self):
        """Download Biorxiv preprint dump using paperscraper."""
        logger.info("🔄 Downloading Biorxiv preprint dump...")
        
        try:
            # Import paperscraper after installation
            from paperscraper.get_dumps import biorxiv
            
            logger.info("📥 Starting Biorxiv dump download (this may take ~1 hour)...")
            biorxiv()
            
            # Check if dump was created
            biorxiv_files = [f for f in os.listdir(self.server_dumps_dir) if 'biorxiv' in f.lower()]
            if biorxiv_files:
                logger.info(f"✅ Biorxiv dump downloaded: {biorxiv_files[0]}")
                return os.path.join(self.server_dumps_dir, biorxiv_files[0])
            else:
                logger.error("❌ Biorxiv dump file not found after download")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to download Biorxiv dump: {e}")
            return None
    
    def download_medrxiv_dump(self):
        """Download Medrxiv preprint dump using paperscraper."""
        logger.info("🔄 Downloading Medrxiv preprint dump...")
        
        try:
            # Import paperscraper after installation
            from paperscraper.get_dumps import medrxiv
            
            logger.info("📥 Starting Medrxiv dump download (this may take ~30 minutes)...")
            medrxiv()
            
            # Check if dump was created
            medrxiv_files = [f for f in os.listdir(self.server_dumps_dir) if 'medrxiv' in f.lower()]
            if medrxiv_files:
                logger.info(f"✅ Medrxiv dump downloaded: {medrxiv_files[0]}")
                return os.path.join(self.server_dumps_dir, medrxiv_files[0])
            else:
                logger.error("❌ Medrxiv dump file not found after download")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to download Medrxiv dump: {e}")
            return None
    
    def download_all_dumps(self):
        """Download all available preprint dumps."""
        logger.info("🚀 Starting XRXiv dump download...")
        print("="*60)
        
        # First, ensure paperscraper is installed
        if not self.install_paperscraper():
            logger.error("❌ Cannot proceed without paperscraper package")
            return False
        
        downloaded_files = []
        
        # Download Biorxiv
        logger.info("📥 Starting Biorxiv download...")
        biorxiv_file = self.download_biorxiv_dump()
        if biorxiv_file:
            downloaded_files.append(biorxiv_file)
        
        # Download Medrxiv
        logger.info("📥 Starting Medrxiv download...")
        medrxiv_file = self.download_medrxiv_dump()
        if medrxiv_file:
            downloaded_files.append(medrxiv_file)
        
        # Summary
        print("\n" + "="*60)
        print("📋 Download Summary:")
        print("="*60)
        
        if downloaded_files:
            print(f"✅ Successfully downloaded {len(downloaded_files)} dump files:")
            for file in downloaded_files:
                filename = os.path.basename(file)
                size = os.path.getsize(file) / (1024 * 1024)  # Size in MB
                print(f"   📄 {filename} ({size:.1f} MB)")
            
            print(f"\n💡 Files saved to: {self.server_dumps_dir}")
            print("💡 You can now run option 3 (Preprints only) to process these dumps")
        else:
            print("❌ No dump files were downloaded")
            print("💡 Check your internet connection and try again")
        
        return len(downloaded_files) > 0

def main():
    """Main function for standalone execution."""
    print("🚀 XRXiv Downloader - AHG-UBR5 Research Processor")
    print("="*60)
    
    downloader = XRXivDownloader()
    success = downloader.download_all_dumps()
    
    if success:
        print("\n✅ XRXiv download completed successfully!")
    else:
        print("\n❌ XRXiv download failed!")
    
    return success

if __name__ == "__main__":
    main()
