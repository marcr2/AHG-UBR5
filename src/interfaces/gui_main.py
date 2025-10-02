#!/usr/bin/env python3
"""
AHG-UBR5 Research Processor - GUI Main Application
AI-Powered Scientific Hypothesis Generator for UBR5 Protein Research
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import threading
import os
import sys
import json
import subprocess
import queue
import time
from datetime import datetime

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.scrapers.pubmed_scraper_json import main as process_pubmed
from src.scrapers.process_xrvix_dumps_json import main as process_xrvix
from src.scrapers.semantic_scholar_scraper import SemanticScholarScraper
from src.core.chromadb_manager import ChromaDBManager
from src.core.processing_config import print_config_info, get_config, DB_BATCH_SIZE

def parse_unified_keywords(keywords_str):
    """
    Parse keywords from GUI input or config file into a unified format.
    Returns a list of cleaned keywords that both PubMed and Semantic Scholar can use.
    """
    if not keywords_str or not keywords_str.strip():
        # Fall back to config file if no GUI input
        try:
            with open("config/search_keywords_config.json", "r", encoding="utf-8") as f:
                config = json.load(f)
            keywords_str = config.get("pubmed_keywords", "")
        except Exception:
            keywords_str = ""
    
    if not keywords_str or not keywords_str.strip():
        # Final fallback to default keywords
        return ["UBR5", "ubr-5", "ubr5", "tumor immunology", "protein degradation"]
    
    # Parse comma-separated keywords and clean them
    keywords = [term.strip() for term in keywords_str.split(',') if term.strip()]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for keyword in keywords:
        if keyword.lower() not in seen:
            seen.add(keyword.lower())
            unique_keywords.append(keyword)
    
    return unique_keywords

class ProgressMonitor:
    """Monitor progress from scraper processes."""
    
    def __init__(self, gui_instance):
        self.gui = gui_instance
        self.progress_queue = queue.Queue()
        self.monitoring = False
        
    def start_monitoring(self, output_widget, scraper_type):
        """Start monitoring progress for a specific scraper."""
        self.monitoring = True
        self.output_widget = output_widget
        self.scraper_type = scraper_type
        self.monitor_thread = threading.Thread(target=self._monitor_progress)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop monitoring progress."""
        self.monitoring = False
        
    def _monitor_progress(self):
        """Monitor progress in a separate thread."""
        while self.monitoring:
            try:
                # Check for progress updates every 100ms
                time.sleep(0.1)
                
                # Update GUI with any new progress information
                self.gui.root.after(0, self._update_progress_display)
                
            except Exception as e:
                print(f"Progress monitoring error: {e}")
                break
                
    def _update_progress_display(self):
        """Update the progress display in the GUI."""
        try:
            # This will be called from the main thread
            # We can add specific progress parsing here if needed
            pass
        except Exception as e:
            print(f"Progress update error: {e}")

class RealTimeProgressWrapper:
    """Custom stdout wrapper to capture and forward progress to GUI in real-time."""
    
    def __init__(self, gui_instance, progress_widget, output_widget):
        self.gui = gui_instance
        self.progress_widget = progress_widget
        self.output_widget = output_widget
        self.original_stdout = sys.stdout
        self.buffer = ""
        
    def write(self, text):
        """Capture stdout and forward meaningful progress to GUI."""
        # Write to original stdout (terminal)
        self.original_stdout.write(text)
        self.original_stdout.flush()
        
        # Add to buffer
        self.buffer += text
        
        # Check for meaningful progress updates
        if '\n' in self.buffer:
            lines = self.buffer.split('\n')
            # Process complete lines
            for line in lines[:-1]:  # All but the last (incomplete) line
                self._process_line(line.strip())
            # Keep the incomplete line in buffer
            self.buffer = lines[-1]
    
    def flush(self):
        """Flush the original stdout."""
        self.original_stdout.flush()
    
    def _process_line(self, line):
        """Process a complete line and update GUI if it contains meaningful progress."""
        if not line:
            return
            
        # Look for meaningful progress patterns
        meaningful_patterns = [
            "Executing strategies:",
            "Found",
            "papers of which",
            "unique",
            "Requests per second:",
            "ETA:",
            "Starting optimized PubMed search",
            "Generated",
            "efficient search strategies",
            "Cleaned search terms:",
            "SEARCH SUMMARY",
            "Success rate:",
            "Total papers collected:",
            # UBR5/Semantic Scholar patterns
            "Searching Semantic Scholar for keyword:",
            "Added",
            "new unique papers",
            "Collected",
            "unique UBR5-related papers",
            "Processing keyword:",
            "Semantic Scholar API",
            "Rate limit:"
        ]
        
        # Check if line contains any meaningful pattern
        if any(pattern in line for pattern in meaningful_patterns):
            # Update GUI progress
            self.gui.root.after(0, lambda: self._update_gui_progress(line))
    
    def _update_gui_progress(self, line):
        """Update GUI progress from main thread."""
        try:
            # Update progress widget
            self.progress_widget.config(text=line)
            # Log to output widget
            self.gui.log_message(self.output_widget, line)
        except Exception as e:
            print(f"GUI progress update error: {e}")

class ScraperProgressWrapper:
    """Wrapper to capture progress from scrapers and send to GUI."""
    
    def __init__(self, gui_instance, progress_widget, output_widget):
        self.gui = gui_instance
        self.progress_widget = progress_widget
        self.output_widget = output_widget
        self.current_progress = "Ready to start"
        
    def update_progress(self, message):
        """Update progress message."""
        self.current_progress = message
        self.gui.root.after(0, self._update_gui)
        
    def _update_gui(self):
        """Update GUI from main thread."""
        try:
            self.progress_widget.config(text=self.current_progress)
            self.gui.log_message(self.output_widget, self.current_progress)
        except Exception as e:
            print(f"GUI update error: {e}")
            
    def run_pubmed_with_progress(self, max_results):
        """Run PubMed scraper with real-time progress monitoring."""
        try:
            self.update_progress("🔍 Starting PubMed search...")
            
            # Import and run PubMed scraper with progress monitoring
            from src.scrapers.pubmed_scraper_json import search_pubmed_comprehensive
            
            # Get keywords from GUI using unified parsing
            keywords_str = self.gui.journal_keywords_var.get()
            search_terms = parse_unified_keywords(keywords_str)
            
            # Set up real-time progress monitoring
            progress_wrapper = RealTimeProgressWrapper(
                self.gui, 
                self.progress_widget, 
                self.output_widget
            )
            
            # Redirect stdout to our wrapper (this captures output and forwards to GUI)
            old_stdout = sys.stdout
            sys.stdout = progress_wrapper
            
            try:
                # Run the scraper with real-time progress updates (no max_results limit)
                papers = search_pubmed_comprehensive(
                    search_terms=search_terms
                )
                
                self.update_progress("✅ PubMed search completed successfully!")
                return papers
                
            finally:
                # Restore stdout
                sys.stdout = old_stdout
                
        except Exception as e:
            self.update_progress(f"❌ PubMed search failed: {e}")
            return []
            
    def run_ubr5_with_progress(self, keywords):
        """Run UBR5 scraper with real-time progress monitoring."""
        try:
            self.update_progress("🔍 Starting UBR5 Semantic Scholar search...")
            
            # Create UBR5 scraper
            semantic_scraper = SemanticScholarScraper()
            # Use unified keyword parsing to ensure consistency with PubMed
            search_keywords = parse_unified_keywords(keywords)
            semantic_scraper.search_keywords = search_keywords
            
            # Set up real-time progress monitoring
            progress_wrapper = RealTimeProgressWrapper(
                self.gui, 
                self.progress_widget, 
                self.output_widget
            )
            
            # Redirect stdout to our wrapper (this captures output and forwards to GUI)
            old_stdout = sys.stdout
            sys.stdout = progress_wrapper
            
            try:
                # Run the scraper with real-time progress updates
                semantic_scraper.run_complete_scraping()
                
                self.update_progress("✅ UBR5 Semantic Scholar search completed successfully!")
                
            finally:
                # Restore stdout
                sys.stdout = old_stdout
                
        except Exception as e:
            self.update_progress(f"❌ UBR5 search failed: {e}")
            
    def run_xrvix_with_progress(self):
        """Run xrvix scraper with progress monitoring."""
        try:
            self.update_progress("📥 Starting preprint download and processing...")
            
            # Check if dump files exist
            dump_dir = "data/scraped_data/paperscraper_dumps"
            server_dumps_dir = os.path.join(dump_dir, "server_dumps")
            
            # Check for dump files in the correct location
            biorxiv_dumps = []
            medrxiv_dumps = []
            
            if os.path.exists(server_dumps_dir):
                biorxiv_dumps = [f for f in os.listdir(server_dumps_dir) if f.startswith('biorxiv') and f.endswith('.jsonl')]
                medrxiv_dumps = [f for f in os.listdir(server_dumps_dir) if f.startswith('medrxiv') and f.endswith('.jsonl')]
            
            if not biorxiv_dumps and not medrxiv_dumps:
                self.update_progress("📥 No preprint dumps found! Downloading...")
                
                # Download dumps
                from src.scrapers.xrvix_downloader import XRXivDownloader
                downloader = XRXivDownloader()
                
                # Monitor download progress
                self.update_progress("📥 Downloading Biorxiv dump (this may take ~1 hour)...")
                biorxiv_file = downloader.download_biorxiv_dump()
                if biorxiv_file:
                    self.update_progress("✅ Biorxiv dump downloaded successfully!")
                else:
                    self.update_progress("❌ Failed to download Biorxiv dump")
                    
                self.update_progress("📥 Downloading Medrxiv dump (this may take ~30 minutes)...")
                medrxiv_file = downloader.download_medrxiv_dump()
                if medrxiv_file:
                    self.update_progress("✅ Medrxiv dump downloaded successfully!")
                else:
                    self.update_progress("❌ Failed to download Medrxiv dump")
            else:
                self.update_progress(f"✅ Found {len(biorxiv_dumps)} Biorxiv dumps and {len(medrxiv_dumps)} Medrxiv dumps")
            
            # Process dumps
            self.update_progress("🔄 Processing preprint dumps...")
            process_xrvix()
            self.update_progress("✅ Preprint processing completed successfully!")
            
        except Exception as e:
            self.update_progress(f"❌ Preprint processing failed: {e}")

class AIResearchProcessorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Research Processor")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 600)
        
        # Initialize progress monitor
        self.progress_monitor = ProgressMonitor(self)
        
        # Initialize configuration variables
        self.init_config_variables()
        
        # Configure style
        self.setup_styles()
        
        # Create main notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_tabs()
        
        # Status bar
        self.create_status_bar()
        
    def setup_styles(self):
        """Configure ttk styles for better appearance."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure notebook style
        style.configure('TNotebook.Tab', padding=[20, 10])
        
        # Configure button styles
        style.configure('Action.TButton', padding=[10, 5])
        style.configure('Danger.TButton', padding=[10, 5])
        
    def create_tabs(self):
        """Create all main tabs."""
        # Paper Scraping & Processing Tab
        self.paper_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.paper_tab, text="📚 Paper Scraping & Processing")
        self.create_paper_scraping_tab()
        
        # Vector Database Management Tab
        self.vector_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.vector_tab, text="🗄️ Vector Database Management")
        self.create_vector_db_tab()
        
        # Settings & Config Tab
        self.settings_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_tab, text="⚙️ Settings & Config")
        self.create_settings_tab()
        
        # Hypothesis Generation Tab
        self.hypothesis_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.hypothesis_tab, text="🧠 Hypothesis Generation")
        self.create_hypothesis_tab()
        
        # Tutorial Tab
        self.tutorial_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.tutorial_tab, text="📖 Tutorial")
        self.create_tutorial_tab()
        
    def create_status_bar(self):
        """Create status bar at bottom of window."""
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_label = ttk.Label(self.status_frame, text="Ready")
        self.status_label.pack(side=tk.LEFT, padx=5, pady=2)
        
        self.progress_bar = ttk.Progressbar(self.status_frame, mode='indeterminate')
        self.progress_bar.pack(side=tk.RIGHT, padx=5, pady=2)
        
    def update_status(self, message):
        """Update status bar message."""
        self.status_label.config(text=message)
        self.root.update_idletasks()
        
    def start_progress(self):
        """Start progress bar animation."""
        self.progress_bar.start()
        
    def stop_progress(self):
        """Stop progress bar animation."""
        self.progress_bar.stop()
        
    def log_message(self, text_widget, message):
        """Add message to text widget with timestamp."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        text_widget.insert(tk.END, f"[{timestamp}] {message}\n")
        text_widget.see(tk.END)  # Auto-scroll to bottom
        text_widget.update_idletasks()  # Ensure the widget updates immediately
        self.root.update_idletasks()
        
    def create_paper_scraping_tab(self):
        """Create Paper Scraping & Processing tab with subtabs."""
        # Create scrollable frame
        canvas = tk.Canvas(self.paper_tab)
        scrollbar = ttk.Scrollbar(self.paper_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Create notebook for subtabs
        paper_notebook = ttk.Notebook(scrollable_frame)
        paper_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Bind mousewheel to canvas
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Full Scraper subtab
        full_scraper_frame = ttk.Frame(paper_notebook)
        paper_notebook.add(full_scraper_frame, text="Full Scraper")
        self.create_full_scraper_subtab(full_scraper_frame)
        
        # Journal Articles subtab
        journal_frame = ttk.Frame(paper_notebook)
        paper_notebook.add(journal_frame, text="Journal Articles Only")
        self.create_journal_articles_subtab(journal_frame)
        
        # Preprints subtab
        preprints_frame = ttk.Frame(paper_notebook)
        paper_notebook.add(preprints_frame, text="Preprints Only")
        self.create_preprints_subtab(preprints_frame)
        
        # Generate Embeddings subtab
        embeddings_frame = ttk.Frame(paper_notebook)
        paper_notebook.add(embeddings_frame, text="Generate Embeddings")
        self.create_embeddings_subtab(embeddings_frame)
        
    def create_full_scraper_subtab(self, parent):
        """Create full scraper subtab."""
        # Title
        title_label = ttk.Label(parent, text="Full Scraper (Preprints + Journal Articles)", 
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Description
        desc_text = """This option runs a complete scraping process including:
• PubMed (Journal Articles) - Custom keywords
• xrvix (Preprints) - Biorxiv, Medrxiv (no keywords needed)
• UBR5 API (Semantic Scholar) - Same keywords as PubMed"""
        
        desc_label = ttk.Label(parent, text=desc_text, justify=tk.LEFT)
        desc_label.pack(pady=(0, 20))
        
        # Keywords configuration
        keywords_frame = ttk.LabelFrame(parent, text="Search Keywords Configuration", padding=10)
        keywords_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Load saved keywords
        self.full_scraper_keywords_var = tk.StringVar()
        self.load_saved_keywords_unified()
        
        ttk.Label(keywords_frame, text="Keywords (used for both PubMed and Semantic Scholar):").pack(anchor=tk.W)
        keywords_entry = ttk.Entry(keywords_frame, textvariable=self.full_scraper_keywords_var, width=60)
        keywords_entry.pack(fill=tk.X, pady=(0, 10))
        
        # PubMed max results
        max_results_frame = ttk.Frame(keywords_frame)
        max_results_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(max_results_frame, text="Max PubMed Results:").pack(side=tk.LEFT)
        self.max_results_var = tk.StringVar(value="5000")
        max_results_entry = ttk.Entry(max_results_frame, textvariable=self.max_results_var, width=10)
        max_results_entry.pack(side=tk.LEFT, padx=(10, 0))
        
        # Buttons
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=(0, 20))
        
        run_button = ttk.Button(button_frame, text="🚀 Run Full Scraper", 
                               command=self.run_full_scraper_gui, style='Action.TButton')
        run_button.pack(side=tk.LEFT, padx=(0, 10))
        
        save_keywords_button = ttk.Button(button_frame, text="💾 Save Keywords", 
                                        command=self.save_keywords_unified_gui)
        save_keywords_button.pack(side=tk.LEFT)
        
        # Progress area
        progress_frame = ttk.LabelFrame(parent, text="Real-time Progress", padding=10)
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Progress display for each scraper
        self.full_scraper_progress = ttk.Label(progress_frame, text="Ready to start", 
                                             font=('Courier', 9), foreground='blue')
        self.full_scraper_progress.pack(anchor=tk.W)
        
        # Output area
        output_frame = ttk.LabelFrame(parent, text="Output", padding=10)
        output_frame.pack(fill=tk.BOTH, expand=True)
        
        self.full_scraper_output = scrolledtext.ScrolledText(output_frame, height=15)
        self.full_scraper_output.pack(fill=tk.BOTH, expand=True)
        
    def create_journal_articles_subtab(self, parent):
        """Create journal articles only subtab with individual source options."""
        # Title
        title_label = ttk.Label(parent, text="Journal Articles Only", 
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Description
        desc_text = """This option scrapes journal articles from PubMed and Semantic Scholar.
Choose to scrape both sources together or individual sources separately."""
        
        desc_label = ttk.Label(parent, text=desc_text, justify=tk.LEFT)
        desc_label.pack(pady=(0, 20))
        
        # Source selection
        source_frame = ttk.LabelFrame(parent, text="Source Selection", padding=10)
        source_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.journal_source_var = tk.StringVar(value="both")
        
        ttk.Radiobutton(source_frame, text="📚 Both Sources (PubMed + Semantic Scholar)", 
                       variable=self.journal_source_var, value="both").pack(anchor=tk.W, pady=2)
        
        ttk.Radiobutton(source_frame, text="🔬 PubMed Only (Journal Articles)", 
                       variable=self.journal_source_var, value="pubmed").pack(anchor=tk.W, pady=2)
        
        ttk.Radiobutton(source_frame, text="🧠 Semantic Scholar Only (UBR5 API)", 
                       variable=self.journal_source_var, value="semantic").pack(anchor=tk.W, pady=2)
        
        # Keywords configuration
        keywords_frame = ttk.LabelFrame(parent, text="Search Keywords Configuration", padding=10)
        keywords_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(keywords_frame, text="Keywords (used for PubMed and Semantic Scholar):").pack(anchor=tk.W)
        self.journal_keywords_var = tk.StringVar()
        journal_entry = ttk.Entry(keywords_frame, textvariable=self.journal_keywords_var, width=60)
        journal_entry.pack(fill=tk.X, pady=(0, 10))
        
        # PubMed max results (only relevant for PubMed or both)
        self.max_results_frame = ttk.Frame(keywords_frame)
        self.max_results_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(self.max_results_frame, text="Max PubMed Results:").pack(side=tk.LEFT)
        self.journal_max_results_var = tk.StringVar(value="5000")
        journal_max_results_entry = ttk.Entry(self.max_results_frame, textvariable=self.journal_max_results_var, width=10)
        journal_max_results_entry.pack(side=tk.LEFT, padx=(10, 0))
        
        # Bind source selection to update UI
        self.journal_source_var.trace('w', self.update_journal_ui)
        
        # Buttons
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=(0, 20))
        
        run_button = ttk.Button(button_frame, text="🚀 Run Selected Scraper", 
                               command=self.run_journal_articles_gui, style='Action.TButton')
        run_button.pack(side=tk.LEFT)
        
        # Progress area
        progress_frame = ttk.LabelFrame(parent, text="Real-time Progress", padding=10)
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Progress display for each scraper
        self.journal_progress = ttk.Label(progress_frame, text="Ready to start", 
                                        font=('Courier', 9), foreground='blue')
        self.journal_progress.pack(anchor=tk.W)
        
        # Output area
        output_frame = ttk.LabelFrame(parent, text="Output", padding=10)
        output_frame.pack(fill=tk.BOTH, expand=True)
        
        self.journal_output = scrolledtext.ScrolledText(output_frame, height=15)
        self.journal_output.pack(fill=tk.BOTH, expand=True)
        
    def create_preprints_subtab(self, parent):
        """Create preprints only subtab with individual source options."""
        # Title
        title_label = ttk.Label(parent, text="Preprints Only", 
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Description
        desc_text = """This option processes preprint data from Biorxiv and Medrxiv.
Choose to process both sources together or individual sources separately.
Note: Custom keywords are not applicable for this option."""
        
        desc_label = ttk.Label(parent, text=desc_text, justify=tk.LEFT)
        desc_label.pack(pady=(0, 20))
        
        # Source selection
        source_frame = ttk.LabelFrame(parent, text="Source Selection", padding=10)
        source_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.preprints_source_var = tk.StringVar(value="both")
        
        ttk.Radiobutton(source_frame, text="📄 Both Sources (BioRxiv + MedRxiv)", 
                       variable=self.preprints_source_var, value="both").pack(anchor=tk.W, pady=2)
        
        ttk.Radiobutton(source_frame, text="🧬 BioRxiv Only (Biology Preprints)", 
                       variable=self.preprints_source_var, value="biorxiv").pack(anchor=tk.W, pady=2)
        
        ttk.Radiobutton(source_frame, text="🏥 MedRxiv Only (Medical Preprints)", 
                       variable=self.preprints_source_var, value="medrxiv").pack(anchor=tk.W, pady=2)
        
        # Status check
        status_frame = ttk.LabelFrame(parent, text="Preprint Data Status", padding=10)
        status_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.preprints_status_label = ttk.Label(status_frame, text="Checking status...")
        self.preprints_status_label.pack(anchor=tk.W)
        
        # Buttons
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=(0, 20))
        
        check_status_button = ttk.Button(button_frame, text="🔍 Check Status", 
                                       command=self.check_preprints_status)
        check_status_button.pack(side=tk.LEFT, padx=(0, 10))
        
        run_button = ttk.Button(button_frame, text="🚀 Run Selected Scraper", 
                               command=self.run_preprints_gui, style='Action.TButton')
        run_button.pack(side=tk.LEFT)
        
        # Progress area
        progress_frame = ttk.LabelFrame(parent, text="Real-time Progress", padding=10)
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Progress display for each scraper
        self.preprints_progress = ttk.Label(progress_frame, text="Ready to start", 
                                          font=('Courier', 9), foreground='blue')
        self.preprints_progress.pack(anchor=tk.W)
        
        # Output area
        output_frame = ttk.LabelFrame(parent, text="Output", padding=10)
        output_frame.pack(fill=tk.BOTH, expand=True)
        
        self.preprints_output = scrolledtext.ScrolledText(output_frame, height=15)
        self.preprints_output.pack(fill=tk.BOTH, expand=True)
        
        # Check status on load
        self.root.after(100, self.check_preprints_status)
    
    def update_journal_ui(self, *args):
        """Update journal articles UI based on source selection."""
        source = self.journal_source_var.get()
        
        # Show/hide PubMed max results based on selection
        if source in ["both", "pubmed"]:
            self.max_results_frame.pack(fill=tk.X, pady=(0, 10))
        else:
            self.max_results_frame.pack_forget()
    
    def create_embeddings_subtab(self, parent):
        """Create generate embeddings subtab."""
        # Title
        title_label = ttk.Label(parent, text="Generate Embeddings", 
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Description
        desc_text = """This option generates embeddings for processed data.
Embeddings are typically generated during the scraping process.
Use this option to regenerate embeddings if needed."""
        
        desc_label = ttk.Label(parent, text=desc_text, justify=tk.LEFT)
        desc_label.pack(pady=(0, 20))
        
        # Status check
        status_frame = ttk.LabelFrame(parent, text="Embeddings Status", padding=10)
        status_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.embeddings_status_label = ttk.Label(status_frame, text="Checking status...")
        self.embeddings_status_label.pack(anchor=tk.W)
        
        # Selective generation options
        options_frame = ttk.LabelFrame(parent, text="Selective Embedding Generation", padding=10)
        options_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Radio buttons for selection
        self.embedding_source_var = tk.StringVar(value="all")
        
        ttk.Radiobutton(options_frame, text="🔄 Generate All Embeddings (PubMed + Preprints)", 
                       variable=self.embedding_source_var, value="all").pack(anchor=tk.W, pady=2)
        
        ttk.Radiobutton(options_frame, text="📚 Generate PubMed Embeddings Only (Journal Articles)", 
                       variable=self.embedding_source_var, value="pubmed").pack(anchor=tk.W, pady=2)
        
        ttk.Radiobutton(options_frame, text="📄 Generate Preprint Embeddings Only (BioRxiv + MedRxiv)", 
                       variable=self.embedding_source_var, value="preprints").pack(anchor=tk.W, pady=2)
        
        # Buttons
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=(0, 20))
        
        check_status_button = ttk.Button(button_frame, text="🔍 Check Status", 
                                       command=self.check_embeddings_status)
        check_status_button.pack(side=tk.LEFT, padx=(0, 10))
        
        generate_button = ttk.Button(button_frame, text="🔄 Generate Embeddings", 
                                   command=self.generate_embeddings_gui, style='Action.TButton')
        generate_button.pack(side=tk.LEFT)
        
        # Progress area
        progress_frame = ttk.LabelFrame(parent, text="Embedding Generation Progress", padding=10)
        progress_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Progress heading
        self.embedding_progress_label = ttk.Label(progress_frame, text="Ready to start", 
                                                font=('Arial', 10, 'bold'))
        self.embedding_progress_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Progress bar
        self.embedding_progress_bar = ttk.Progressbar(progress_frame, mode='determinate')
        self.embedding_progress_bar.pack(fill=tk.X, pady=(0, 5))
        
        # File progress info
        self.embedding_file_info = ttk.Label(progress_frame, text="", 
                                           font=('Courier', 9), foreground='blue')
        self.embedding_file_info.pack(anchor=tk.W)
        
        # Output area
        output_frame = ttk.LabelFrame(parent, text="Output", padding=10)
        output_frame.pack(fill=tk.BOTH, expand=True)
        
        self.embeddings_output = scrolledtext.ScrolledText(output_frame, height=15)
        self.embeddings_output.pack(fill=tk.BOTH, expand=True)
        
        # Check status on load
        self.root.after(100, self.check_embeddings_status)
        
    def create_vector_db_tab(self):
        """Create Vector Database Management tab with subtabs."""
        # Create scrollable frame
        canvas = tk.Canvas(self.vector_tab)
        scrollbar = ttk.Scrollbar(self.vector_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Create notebook for subtabs
        vector_notebook = ttk.Notebook(scrollable_frame)
        vector_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Bind mousewheel to canvas
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Load Embeddings subtab
        load_frame = ttk.Frame(vector_notebook)
        vector_notebook.add(load_frame, text="Load Embeddings")
        self.create_load_embeddings_subtab(load_frame)
        
        # Show ChromaDB Data subtab
        show_frame = ttk.Frame(vector_notebook)
        vector_notebook.add(show_frame, text="Show ChromaDB Data")
        self.create_show_chromadb_subtab(show_frame)
        
        # Clear ChromaDB Data subtab
        clear_frame = ttk.Frame(vector_notebook)
        vector_notebook.add(clear_frame, text="Clear ChromaDB Data")
        self.create_clear_chromadb_subtab(clear_frame)
        
    def create_load_embeddings_subtab(self, parent):
        """Create load embeddings subtab."""
        # Title
        title_label = ttk.Label(parent, text="Load Embeddings into ChromaDB", 
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Description
        desc_text = """This option loads processed embeddings into the ChromaDB vector database.
The system will load embeddings from:
• PubMed embeddings
• xrvix embeddings (biorxiv, medrxiv)
• UBR5 API embeddings"""
        
        desc_label = ttk.Label(parent, text=desc_text, justify=tk.LEFT)
        desc_label.pack(pady=(0, 20))
        
        # Status check
        status_frame = ttk.LabelFrame(parent, text="ChromaDB Status", padding=10)
        status_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.load_status_label = ttk.Label(status_frame, text="Checking status...")
        self.load_status_label.pack(anchor=tk.W)
        
        # Selective loading options
        load_options_frame = ttk.LabelFrame(parent, text="Selective Loading Options", padding=10)
        load_options_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Radio buttons for loading selection
        self.load_source_var = tk.StringVar(value="all")
        
        ttk.Radiobutton(load_options_frame, text="📥 Load All Embeddings (PubMed + Preprints + Semantic Scholar)", 
                       variable=self.load_source_var, value="all").pack(anchor=tk.W, pady=2)
        
        ttk.Radiobutton(load_options_frame, text="📚 Load PubMed Embeddings Only (Journal Articles)", 
                       variable=self.load_source_var, value="pubmed").pack(anchor=tk.W, pady=2)
        
        ttk.Radiobutton(load_options_frame, text="📄 Load Preprint Embeddings Only (BioRxiv + MedRxiv)", 
                       variable=self.load_source_var, value="preprints").pack(anchor=tk.W, pady=2)
        
        ttk.Radiobutton(load_options_frame, text="🎓 Load Semantic Scholar Embeddings Only", 
                       variable=self.load_source_var, value="semantic").pack(anchor=tk.W, pady=2)
        
        # Buttons
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=(0, 20))
        
        check_status_button = ttk.Button(button_frame, text="🔍 Check Status", 
                                       command=self.check_chromadb_status)
        check_status_button.pack(side=tk.LEFT, padx=(0, 10))
        
        load_button = ttk.Button(button_frame, text="📥 Load Embeddings", 
                               command=self.load_embeddings_gui, style='Action.TButton')
        load_button.pack(side=tk.LEFT)
        
        # Output area
        output_frame = ttk.LabelFrame(parent, text="Output", padding=10)
        output_frame.pack(fill=tk.BOTH, expand=True)
        
        self.load_output = scrolledtext.ScrolledText(output_frame, height=15)
        self.load_output.pack(fill=tk.BOTH, expand=True)
        
        # Check status on load
        self.root.after(100, self.check_chromadb_status)
        
    def create_show_chromadb_subtab(self, parent):
        """Create show ChromaDB data subtab."""
        # Title
        title_label = ttk.Label(parent, text="Show Current ChromaDB Data", 
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Description
        desc_text = """This option displays information about the current ChromaDB collections
and their contents, including document counts and source breakdown."""
        
        desc_label = ttk.Label(parent, text=desc_text, justify=tk.LEFT)
        desc_label.pack(pady=(0, 20))
        
        # Buttons
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=(0, 20))
        
        show_button = ttk.Button(button_frame, text="📊 Show ChromaDB Data", 
                                command=self.show_chromadb_data_gui, style='Action.TButton')
        show_button.pack(side=tk.LEFT)
        
        refresh_button = ttk.Button(button_frame, text="🔄 Refresh", 
                                   command=self.refresh_chromadb_data)
        refresh_button.pack(side=tk.LEFT, padx=(10, 0))
        
        # Output area
        output_frame = ttk.LabelFrame(parent, text="ChromaDB Information", padding=10)
        output_frame.pack(fill=tk.BOTH, expand=True)
        
        self.show_output = scrolledtext.ScrolledText(output_frame, height=15)
        self.show_output.pack(fill=tk.BOTH, expand=True)
        
    def create_clear_chromadb_subtab(self, parent):
        """Create clear ChromaDB data subtab."""
        # Title
        title_label = ttk.Label(parent, text="Clear ChromaDB Data", 
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Description
        desc_text = """⚠️ WARNING: This option will permanently delete all data from ChromaDB.
This action cannot be undone. Make sure you have backups if needed."""
        
        desc_label = ttk.Label(parent, text=desc_text, justify=tk.LEFT, foreground='red')
        desc_label.pack(pady=(0, 20))
        
        # Confirmation frame
        confirm_frame = ttk.LabelFrame(parent, text="Confirmation Required", padding=10)
        confirm_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.confirm_var = tk.BooleanVar()
        confirm_check = ttk.Checkbutton(confirm_frame, 
                                       text="I understand this will delete all ChromaDB data",
                                       variable=self.confirm_var)
        confirm_check.pack(anchor=tk.W)
        
        # Buttons
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=(0, 20))
        
        clear_button = ttk.Button(button_frame, text="🗑️ Clear ChromaDB Data", 
                                 command=self.clear_chromadb_data_gui, 
                                 style='Danger.TButton')
        clear_button.pack(side=tk.LEFT)
        
        # Output area
        output_frame = ttk.LabelFrame(parent, text="Output", padding=10)
        output_frame.pack(fill=tk.BOTH, expand=True)
        
        self.clear_output = scrolledtext.ScrolledText(output_frame, height=15)
        self.clear_output.pack(fill=tk.BOTH, expand=True)
        
    def create_settings_tab(self):
        """Create Settings & Config tab with subtabs."""
        # Create scrollable frame
        canvas = tk.Canvas(self.settings_tab)
        scrollbar = ttk.Scrollbar(self.settings_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Create notebook for subtabs
        settings_notebook = ttk.Notebook(scrollable_frame)
        settings_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Bind mousewheel to canvas
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Data Status subtab
        status_frame = ttk.Frame(settings_notebook)
        settings_notebook.add(status_frame, text="Data Status")
        self.create_data_status_subtab(status_frame)
        
        # Configurations subtab
        config_frame = ttk.Frame(settings_notebook)
        settings_notebook.add(config_frame, text="Configurations")
        self.create_configurations_subtab(config_frame)
        
    def create_data_status_subtab(self, parent):
        """Create data status subtab."""
        # Title
        title_label = ttk.Label(parent, text="Data Status Overview", 
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Description
        desc_text = """This option shows the status of all available data sources
and provides recommendations for next steps."""
        
        desc_label = ttk.Label(parent, text=desc_text, justify=tk.LEFT)
        desc_label.pack(pady=(0, 20))
        
        # Buttons
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=(0, 20))
        
        show_status_button = ttk.Button(button_frame, text="📊 Show Data Status", 
                                       command=self.show_data_status_gui, style='Action.TButton')
        show_status_button.pack(side=tk.LEFT)
        
        refresh_button = ttk.Button(button_frame, text="🔄 Refresh", 
                                   command=self.refresh_data_status)
        refresh_button.pack(side=tk.LEFT, padx=(10, 0))
        
        # Output area
        output_frame = ttk.LabelFrame(parent, text="Data Status Information", padding=10)
        output_frame.pack(fill=tk.BOTH, expand=True)
        
        self.data_status_output = scrolledtext.ScrolledText(output_frame, height=15)
        self.data_status_output.pack(fill=tk.BOTH, expand=True)
        
    def create_configurations_subtab(self, parent):
        """Create comprehensive configuration editor with subtabs."""
        # Create notebook for configuration subtabs
        config_notebook = ttk.Notebook(parent)
        config_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Processing Configuration Tab
        processing_frame = ttk.Frame(config_notebook)
        config_notebook.add(processing_frame, text="🔄 Processing")
        self.create_processing_config_tab(processing_frame)
        
        # Rate Limiting Configuration Tab
        rate_frame = ttk.Frame(config_notebook)
        config_notebook.add(rate_frame, text="⏱️ Rate Limiting")
        self.create_rate_limiting_config_tab(rate_frame)
        
        # Database Configuration Tab
        db_frame = ttk.Frame(config_notebook)
        config_notebook.add(db_frame, text="🗄️ Database")
        self.create_database_config_tab(db_frame)
        
        # API Configuration Tab
        api_frame = ttk.Frame(config_notebook)
        config_notebook.add(api_frame, text="🔌 API")
        self.create_api_config_tab(api_frame)
        
        # Search Configuration Tab
        search_frame = ttk.Frame(config_notebook)
        config_notebook.add(search_frame, text="🔍 Search")
        self.create_search_config_tab(search_frame)
        
        # Save/Load buttons at the bottom
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        save_config_button = ttk.Button(button_frame, text="💾 Save All Configurations", 
                                      command=self.save_all_configurations, style='Action.TButton')
        save_config_button.pack(side=tk.LEFT, padx=(0, 10))
        
        load_config_button = ttk.Button(button_frame, text="📂 Load Configurations", 
                                      command=self.load_all_configurations, style='Action.TButton')
        load_config_button.pack(side=tk.LEFT, padx=(0, 10))
        
        reset_config_button = ttk.Button(button_frame, text="🔄 Reset to Defaults", 
                                      command=self.reset_configurations, style='Action.TButton')
        reset_config_button.pack(side=tk.LEFT)
    
    def create_processing_config_tab(self, parent):
        """Create processing configuration tab."""
        # Create scrollable frame
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Processing Settings
        processing_frame = ttk.LabelFrame(scrollable_frame, text="Parallel Processing", padding=10)
        processing_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # MAX_WORKERS
        ttk.Label(processing_frame, text="Max Workers:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.max_workers_var = tk.StringVar(value="8")
        max_workers_entry = ttk.Entry(processing_frame, textvariable=self.max_workers_var, width=10)
        max_workers_entry.grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        ttk.Label(processing_frame, text="(Number of parallel workers for processing)").grid(row=0, column=2, sticky=tk.W, padx=(10, 0), pady=2)
        
        # BATCH_SIZE
        ttk.Label(processing_frame, text="Batch Size:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.batch_size_var = tk.StringVar(value="500")
        batch_size_entry = ttk.Entry(processing_frame, textvariable=self.batch_size_var, width=10)
        batch_size_entry.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        ttk.Label(processing_frame, text="(Papers processed per batch)").grid(row=1, column=2, sticky=tk.W, padx=(10, 0), pady=2)
        
        # REQUEST_TIMEOUT
        ttk.Label(processing_frame, text="Request Timeout:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.request_timeout_var = tk.StringVar(value="30")
        request_timeout_entry = ttk.Entry(processing_frame, textvariable=self.request_timeout_var, width=10)
        request_timeout_entry.grid(row=2, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        ttk.Label(processing_frame, text="(Seconds)").grid(row=2, column=2, sticky=tk.W, padx=(10, 0), pady=2)
        
        # SAVE_INTERVAL
        ttk.Label(processing_frame, text="Save Interval:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.save_interval_var = tk.StringVar(value="1000")
        save_interval_entry = ttk.Entry(processing_frame, textvariable=self.save_interval_var, width=10)
        save_interval_entry.grid(row=3, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        ttk.Label(processing_frame, text="(Save progress every N papers)").grid(row=3, column=2, sticky=tk.W, padx=(10, 0), pady=2)
        
        # Text Processing Settings
        text_frame = ttk.LabelFrame(scrollable_frame, text="Text Processing", padding=10)
        text_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # MIN_CHUNK_LENGTH
        ttk.Label(text_frame, text="Min Chunk Length:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.min_chunk_length_var = tk.StringVar(value="50")
        min_chunk_entry = ttk.Entry(text_frame, textvariable=self.min_chunk_length_var, width=10)
        min_chunk_entry.grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        ttk.Label(text_frame, text="(Minimum characters per text chunk)").grid(row=0, column=2, sticky=tk.W, padx=(10, 0), pady=2)
        
        # MAX_CHUNK_LENGTH
        ttk.Label(text_frame, text="Max Chunk Length:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.max_chunk_length_var = tk.StringVar(value="8000")
        max_chunk_entry = ttk.Entry(text_frame, textvariable=self.max_chunk_length_var, width=10)
        max_chunk_entry.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        ttk.Label(text_frame, text="(Maximum characters per text chunk)").grid(row=1, column=2, sticky=tk.W, padx=(10, 0), pady=2)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_rate_limiting_config_tab(self, parent):
        """Create rate limiting configuration tab."""
        # Create scrollable frame
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # General Rate Limiting
        general_frame = ttk.LabelFrame(scrollable_frame, text="General Rate Limiting", padding=10)
        general_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # RATE_LIMIT_DELAY
        ttk.Label(general_frame, text="Rate Limit Delay:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.rate_limit_delay_var = tk.StringVar(value="0.02")
        rate_limit_delay_entry = ttk.Entry(general_frame, textvariable=self.rate_limit_delay_var, width=10)
        rate_limit_delay_entry.grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        ttk.Label(general_frame, text="(Seconds between requests)").grid(row=0, column=2, sticky=tk.W, padx=(10, 0), pady=2)
        
        # Citation Rate Limiting
        citation_frame = ttk.LabelFrame(scrollable_frame, text="Citation Processing", padding=10)
        citation_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # CITATION_RATE_LIMIT
        ttk.Label(citation_frame, text="Citation Rate Limit:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.citation_rate_limit_var = tk.StringVar(value="0.5")
        citation_rate_limit_entry = ttk.Entry(citation_frame, textvariable=self.citation_rate_limit_var, width=10)
        citation_rate_limit_entry.grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        ttk.Label(citation_frame, text="(Seconds between citation requests)").grid(row=0, column=2, sticky=tk.W, padx=(10, 0), pady=2)
        
        # CITATION_TIMEOUT
        ttk.Label(citation_frame, text="Citation Timeout:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.citation_timeout_var = tk.StringVar(value="5")
        citation_timeout_entry = ttk.Entry(citation_frame, textvariable=self.citation_timeout_var, width=10)
        citation_timeout_entry.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        ttk.Label(citation_frame, text="(Seconds)").grid(row=1, column=2, sticky=tk.W, padx=(10, 0), pady=2)
        
        # API Rate Limits
        api_frame = ttk.LabelFrame(scrollable_frame, text="API Rate Limits", padding=10)
        api_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # EMBEDDING_MAX_REQUESTS_PER_MINUTE
        ttk.Label(api_frame, text="Embedding Requests/Min:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.embedding_max_requests_var = tk.StringVar(value="1500")
        embedding_max_requests_entry = ttk.Entry(api_frame, textvariable=self.embedding_max_requests_var, width=10)
        embedding_max_requests_entry.grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        ttk.Label(api_frame, text="(Google Embedding API limit)").grid(row=0, column=2, sticky=tk.W, padx=(10, 0), pady=2)
        
        # GEMINI_MAX_REQUESTS_PER_MINUTE
        ttk.Label(api_frame, text="Gemini Requests/Min:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.gemini_max_requests_var = tk.StringVar(value="1000")
        gemini_max_requests_entry = ttk.Entry(api_frame, textvariable=self.gemini_max_requests_var, width=10)
        gemini_max_requests_entry.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        ttk.Label(api_frame, text="(Gemini API limit)").grid(row=1, column=2, sticky=tk.W, padx=(10, 0), pady=2)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_database_config_tab(self, parent):
        """Create database configuration tab."""
        # Create scrollable frame
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # ChromaDB Settings
        chromadb_frame = ttk.LabelFrame(scrollable_frame, text="ChromaDB Configuration", padding=10)
        chromadb_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # DB_BATCH_SIZE
        ttk.Label(chromadb_frame, text="DB Batch Size:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.db_batch_size_var = tk.StringVar(value="5000")
        db_batch_size_entry = ttk.Entry(chromadb_frame, textvariable=self.db_batch_size_var, width=10)
        db_batch_size_entry.grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        ttk.Label(chromadb_frame, text="(Embeddings per batch - max 5461)").grid(row=0, column=2, sticky=tk.W, padx=(10, 0), pady=2)
        
        # Collection Name
        ttk.Label(chromadb_frame, text="Collection Name:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.collection_name_var = tk.StringVar(value="ubr5_papers")
        collection_name_entry = ttk.Entry(chromadb_frame, textvariable=self.collection_name_var, width=20)
        collection_name_entry.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        ttk.Label(chromadb_frame, text="(ChromaDB collection name)").grid(row=1, column=2, sticky=tk.W, padx=(10, 0), pady=2)
        
        # Persist Directory
        ttk.Label(chromadb_frame, text="Persist Directory:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.persist_directory_var = tk.StringVar(value="./chroma_db")
        persist_directory_entry = ttk.Entry(chromadb_frame, textvariable=self.persist_directory_var, width=30)
        persist_directory_entry.grid(row=2, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        ttk.Label(chromadb_frame, text="(Database storage location)").grid(row=2, column=2, sticky=tk.W, padx=(10, 0), pady=2)
        
        # Storage Settings
        storage_frame = ttk.LabelFrame(scrollable_frame, text="Storage Configuration", padding=10)
        storage_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Embeddings Directory
        ttk.Label(storage_frame, text="Embeddings Directory:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.embeddings_dir_var = tk.StringVar(value="xrvix_embeddings")
        embeddings_dir_entry = ttk.Entry(storage_frame, textvariable=self.embeddings_dir_var, width=30)
        embeddings_dir_entry.grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        ttk.Label(storage_frame, text="(Local embeddings storage)").grid(row=0, column=2, sticky=tk.W, padx=(10, 0), pady=2)
        
        # File Format
        ttk.Label(storage_frame, text="File Format:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.file_format_var = tk.StringVar(value="json")
        file_format_combo = ttk.Combobox(storage_frame, textvariable=self.file_format_var, width=10, state="readonly")
        file_format_combo['values'] = ('json', 'jsonl', 'csv')
        file_format_combo.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        ttk.Label(storage_frame, text="(Data file format)").grid(row=1, column=2, sticky=tk.W, padx=(10, 0), pady=2)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_api_config_tab(self, parent):
        """Create API configuration tab."""
        # Create scrollable frame
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Semantic Scholar API
        semantic_frame = ttk.LabelFrame(scrollable_frame, text="Semantic Scholar API", padding=10)
        semantic_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Base URL
        ttk.Label(semantic_frame, text="Base URL:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.semantic_base_url_var = tk.StringVar(value="https://api.semanticscholar.org/graph/v1")
        semantic_base_url_entry = ttk.Entry(semantic_frame, textvariable=self.semantic_base_url_var, width=40)
        semantic_base_url_entry.grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        
        # Batch Size
        ttk.Label(semantic_frame, text="Batch Size:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.semantic_batch_size_var = tk.StringVar(value="100")
        semantic_batch_size_entry = ttk.Entry(semantic_frame, textvariable=self.semantic_batch_size_var, width=10)
        semantic_batch_size_entry.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        ttk.Label(semantic_frame, text="(Papers per batch)").grid(row=1, column=2, sticky=tk.W, padx=(10, 0), pady=2)
        
        # Max Papers Per Query
        ttk.Label(semantic_frame, text="Max Papers/Query:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.semantic_max_papers_var = tk.StringVar(value="50")
        semantic_max_papers_entry = ttk.Entry(semantic_frame, textvariable=self.semantic_max_papers_var, width=10)
        semantic_max_papers_entry.grid(row=2, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        ttk.Label(semantic_frame, text="(Maximum results per search)").grid(row=2, column=2, sticky=tk.W, padx=(10, 0), pady=2)
        
        # API Key
        ttk.Label(semantic_frame, text="API Key:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.semantic_api_key_var = tk.StringVar()
        semantic_api_key_entry = ttk.Entry(semantic_frame, textvariable=self.semantic_api_key_var, width=40, show="*")
        semantic_api_key_entry.grid(row=3, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        ttk.Label(semantic_frame, text="(Optional - higher rate limits)").grid(row=3, column=2, sticky=tk.W, padx=(10, 0), pady=2)
        
        # Load API key from config button
        load_api_key_button = ttk.Button(semantic_frame, text="📂 Load from keys.json", 
                                       command=self.load_semantic_api_key)
        load_api_key_button.grid(row=4, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        
        # Embedding API
        embedding_frame = ttk.LabelFrame(scrollable_frame, text="Google Embedding API", padding=10)
        embedding_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Model
        ttk.Label(embedding_frame, text="Model:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.embedding_model_var = tk.StringVar(value="text-embedding-004")
        embedding_model_entry = ttk.Entry(embedding_frame, textvariable=self.embedding_model_var, width=20)
        embedding_model_entry.grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        ttk.Label(embedding_frame, text="(Google embedding model)").grid(row=0, column=2, sticky=tk.W, padx=(10, 0), pady=2)
        
        # Max Text Length
        ttk.Label(embedding_frame, text="Max Text Length:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.embedding_max_length_var = tk.StringVar(value="8000")
        embedding_max_length_entry = ttk.Entry(embedding_frame, textvariable=self.embedding_max_length_var, width=10)
        embedding_max_length_entry.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        ttk.Label(embedding_frame, text="(Maximum characters for embedding)").grid(row=1, column=2, sticky=tk.W, padx=(10, 0), pady=2)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_search_config_tab(self, parent):
        """Create search configuration tab."""
        # Create scrollable frame
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Search Keywords
        keywords_frame = ttk.LabelFrame(scrollable_frame, text="Search Keywords", padding=10)
        keywords_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # PubMed Keywords
        ttk.Label(keywords_frame, text="PubMed Keywords:").grid(row=0, column=0, sticky=tk.NW, pady=2)
        self.pubmed_keywords_var = tk.StringVar(value="UBR5,ubr-5,ubr5,tumor immunology,protein degradation")
        pubmed_keywords_text = tk.Text(keywords_frame, height=3, width=50)
        pubmed_keywords_text.grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        pubmed_keywords_text.insert(tk.END, self.pubmed_keywords_var.get())
        self.pubmed_keywords_text_widget = pubmed_keywords_text
        
        # Semantic Scholar Keywords
        ttk.Label(keywords_frame, text="Semantic Scholar Keywords:").grid(row=1, column=0, sticky=tk.NW, pady=2)
        self.semantic_keywords_var = tk.StringVar(value="UBR5,ubr-5,ubr5,tumor immunology,protein degradation")
        semantic_keywords_text = tk.Text(keywords_frame, height=3, width=50)
        semantic_keywords_text.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        semantic_keywords_text.insert(tk.END, self.semantic_keywords_var.get())
        self.semantic_keywords_text_widget = semantic_keywords_text
        
        # Search Settings
        search_frame = ttk.LabelFrame(scrollable_frame, text="Search Settings", padding=10)
        search_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Max Results
        ttk.Label(search_frame, text="Max Results:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.max_results_var = tk.StringVar(value="5000")
        max_results_entry = ttk.Entry(search_frame, textvariable=self.max_results_var, width=10)
        max_results_entry.grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        ttk.Label(search_frame, text="(Maximum papers to collect)").grid(row=0, column=2, sticky=tk.W, padx=(10, 0), pady=2)
        
        # Date Range
        ttk.Label(search_frame, text="Date From:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.date_from_var = tk.StringVar(value="1900")
        date_from_entry = ttk.Entry(search_frame, textvariable=self.date_from_var, width=10)
        date_from_entry.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        ttk.Label(search_frame, text="(YYYY or YYYY-MM-DD)").grid(row=1, column=2, sticky=tk.W, padx=(10, 0), pady=2)
        
        # Deduplication
        ttk.Label(search_frame, text="Enable Deduplication:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.deduplication_var = tk.BooleanVar(value=True)
        deduplication_check = ttk.Checkbutton(search_frame, variable=self.deduplication_var)
        deduplication_check.grid(row=2, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        ttk.Label(search_frame, text="(Remove duplicate papers)").grid(row=2, column=2, sticky=tk.W, padx=(10, 0), pady=2)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def save_all_configurations(self):
        """Save all configuration changes to files."""
        try:
            # Save processing_config.py
            self.save_processing_config()
            
            # Save ubr5_scraper_config.py
            self.save_ubr5_scraper_config()
            
            # Save search_keywords_config.json
            self.save_search_keywords_config()
            
            # Save API keys
            self.save_semantic_api_key()
            
            messagebox.showinfo("Success", "All configurations saved successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configurations: {e}")
    
    def save_semantic_api_key(self):
        """Save Semantic Scholar API key to keys.json."""
        try:
            # Load existing keys
            try:
                with open("config/keys.json", 'r') as f:
                    keys_data = json.load(f)
            except FileNotFoundError:
                keys_data = {}
            
            # Update with new API key
            api_key = self.semantic_api_key_var.get().strip()
            if api_key:
                keys_data["SEMANTIC_SCHOLAR_API_KEY"] = api_key
            else:
                # Remove key if empty
                keys_data.pop("SEMANTIC_SCHOLAR_API_KEY", None)
            
            # Save back to file
            with open("config/keys.json", 'w') as f:
                json.dump(keys_data, f, indent=2)
                
        except Exception as e:
            raise Exception(f"Failed to save API key: {e}")
    
    def save_processing_config(self):
        """Save processing configuration to processing_config.py."""
        config_content = f'''"""
Configuration file for optimizing embedding processing performance.
Optimized for 1 request per second (sequential processing) for maximum reliability.
Adjust these settings based on your system capabilities and API limits.
"""

# --- PARALLEL PROCESSING CONFIGURATION ---
# Optimized for sequential processing (1 req/sec) for maximum reliability
MAX_WORKERS = {self.max_workers_var.get()}  # Sequential processing (1 worker)
BATCH_SIZE = {self.batch_size_var.get()}  # Smaller batches for reliability
RATE_LIMIT_DELAY = {self.rate_limit_delay_var.get()}  # 1 second between requests
REQUEST_TIMEOUT = {self.request_timeout_var.get()}  # Request timeout in seconds
MIN_CHUNK_LENGTH = {self.min_chunk_length_var.get()}  # Minimum chunk length for text splitting
MAX_CHUNK_LENGTH = {self.max_chunk_length_var.get()}  # Maximum chunk length for text splitting
SAVE_INTERVAL = {self.save_interval_var.get()}  # Save progress every N papers

# --- CITATION PROCESSING OPTIMIZATION ---
CITATION_MAX_WORKERS = 1  # Single worker for immediate processing (no parallel needed)
CITATION_TIMEOUT = {self.citation_timeout_var.get()}  # Timeout per citation request (seconds)
CITATION_BATCH_SIZE = 1  # Process citations immediately (no batching needed)
CITATION_RATE_LIMIT = {self.citation_rate_limit_var.get()}  # 1 second between citation requests

# --- VECTOR DATABASE CONFIG ---
DB_BATCH_SIZE = {self.db_batch_size_var.get()}  # Number of embeddings to add to ChromaDB in one operation

# --- TEXT PROCESSING CONFIG ---
MIN_CHUNK_LENGTH = {self.min_chunk_length_var.get()}  # Minimum character length for a chunk
MAX_CHUNK_LENGTH = {self.max_chunk_length_var.get()}  # Maximum character length for a chunk

# --- RATE LIMITING ---
# Optimized for 1 request per second (60 requests per minute)
EMBEDDING_MAX_REQUESTS_PER_MINUTE = {self.embedding_max_requests_var.get()}  # 60 requests per minute (1 per second)
GEMINI_MAX_REQUESTS_PER_MINUTE = {self.gemini_max_requests_var.get()}     # 60 requests per minute (1 per second)
MAX_BATCH_ENQUEUED_TOKENS = 3_000_000     # Max batch enqueued tokens
RATE_LIMIT_WINDOW = 60  # Rate limiting window in seconds

# Use embedding limit for processing (since that's what we use most)
MAX_REQUESTS_PER_MINUTE = EMBEDDING_MAX_REQUESTS_PER_MINUTE

# --- MEMORY OPTIMIZATION ---
SAVE_INTERVAL = {self.save_interval_var.get()}  # Save metadata every N papers

# --- SOURCE CONFIG ---
DUMPS = ["biorxiv", "medrxiv"]  # Only process these sources

# --- PERFORMANCE PROFILES ---
PERFORMANCE_PROFILES = {{
    "sequential_optimized": {{
        "max_workers": {self.max_workers_var.get()},
        "request_timeout": {self.request_timeout_var.get()},
        "rate_limit_delay": {self.rate_limit_delay_var.get()},
        "batch_size": {self.batch_size_var.get()},
        "db_batch_size": {self.db_batch_size_var.get()}
    }}
}}

def get_config(profile="sequential_optimized"):
    """Get configuration for the sequential optimized processing profile."""
    return PERFORMANCE_PROFILES["sequential_optimized"]

def print_config_info():
    """Print current configuration information."""
    print("🔧 Current Processing Configuration (Sequential Optimized):")
    print(f"   Processing mode: Sequential ({{MAX_WORKERS}} worker)")
    print(f"   Request timeout: {{REQUEST_TIMEOUT}}s")
    print(f"   Batch size: {{BATCH_SIZE}}")
    print(f"   Rate limit delay: {{RATE_LIMIT_DELAY}}s (1 request per second)")
    print(f"   Sources: {{', '.join(DUMPS)}}")
    print()
    print("📊 Citation Processing:")
    print(f"   Citation workers: {{CITATION_MAX_WORKERS}} (immediate processing)")
    print(f"   Citation timeout: {{CITATION_TIMEOUT}}s")
    print(f"   Citation rate limit: {{CITATION_RATE_LIMIT}}s between requests")
    print()
    print("💡 Performance Notes:")
    print("   - Sequential processing for maximum reliability")
    print("   - 1 request per second to avoid rate limits")
    print("   - Smaller batch sizes for better error recovery")

if __name__ == "__main__":
    print_config_info()
'''
        
        with open("src/core/processing_config.py", "w", encoding="utf-8") as f:
            f.write(config_content)
    
    def save_ubr5_scraper_config(self):
        """Save UBR5 scraper configuration to ubr5_scraper_config.py."""
        config_content = f'''"""
Configuration file for the UBR5 API scraper.
Contains all configurable parameters for paper collection and processing.
"""

# --- UBR5 SEARCH CONFIGURATION ---
UBR5_SEARCH_TERMS = [
    "UBR5", "ubr5", "Ubr5", "ubiquitin protein ligase E3 component n-recognin 5",
    "EDD1", "edd1", "Edd1", "E3 ubiquitin-protein ligase UBR5",
    "ubiquitin ligase UBR5", "UBR5 function", "UBR5 regulation", 
    "UBR5 expression", "UBR5 mutation", "UBR5 activity", "UBR5 mechanism",
    "UBR5 cancer", "UBR5 disease", "UBR5 pathway", "UBR5 oncology",
    "UBR5 tumor", "UBR5 metastasis", "UBR5 interaction", "UBR5 binding",
    "UBR5 complex", "UBR5 protein-protein", "UBR5 substrate",
    "UBR5 ubiquitination", "UBR5 degradation", "UBR5 signaling",
    "UBR5 transcription", "UBR5 DNA repair", "UBR5 tissue",
    "UBR5 cell line", "UBR5 organ", "UBR5 development"
]

# --- API CONFIGURATION ---
SEMANTIC_SCHOLAR_CONFIG = {{
    "base_url": "{self.semantic_base_url_var.get()}",
    "search_endpoint": "/paper/search",
    "paper_endpoint": "/paper",
    "batch_size": {self.semantic_batch_size_var.get()},
    "max_papers_per_query": {self.semantic_max_papers_var.get()},
    "fields": [
        "paperId", "title", "abstract", "venue", "year", "authors",
        "referenceCount", "citationCount", "openAccessPdf", 
        "publicationDate", "publicationTypes", "fieldsOfStudy",
        "publicationVenue", "externalIds", "url", "isOpenAccess"
    ]
}}

SCHOLARLY_CONFIG = {{
    "max_papers_per_query": 30,
    "search_keywords": ["ubr5", "UBR5", "ubr-5", "UBR-5"]
}}

# --- RATE LIMITING ---
# Optimized for 1 request per second for maximum reliability
RATE_LIMITING = {{
    "semantic_scholar_delay": 1.0,  # 1 second between Semantic Scholar requests
    "scholarly_delay": 1.0,         # 1 second between scholarly requests
    "keyword_delay": 1.0,           # 1 second between search keywords
    "embedding_delay": 1.0,         # 1 second between embedding requests
    "max_retries": 3,
    "timeout": 30,
    "rate_limit_wait": 60
}}

# --- EMBEDDING CONFIGURATION ---
EMBEDDING_CONFIG = {{
    "model": "{self.embedding_model_var.get()}",
    "api_endpoint": "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedText",
    "max_text_length": {self.embedding_max_length_var.get()},
    "text_components": [
        "title", "abstract", "authors", "journal", "year", "fields_of_study"
    ]
}}

# --- PAPER PROCESSING ---
PAPER_PROCESSING = {{
    "min_title_length": 10,
    "min_abstract_length": 50,
    "title_similarity_threshold": 0.8,
    "max_abstract_length": 5000,
    "required_fields": ["title"],
    "optional_fields": ["abstract", "authors", "journal", "year", "doi"]
}}

# --- STORAGE CONFIGURATION ---
STORAGE_CONFIG = {{
    "embeddings_dir": "{self.embeddings_dir_var.get()}",
    "source_name": "ubr5_api",
    "file_format": "{self.file_format_var.get()}",
    "metadata_file": "metadata.json",
    "max_filename_length": 100,
    "save_individual_files": True,
    "save_metadata": True
}}

# --- CHROMADB INTEGRATION ---
CHROMADB_CONFIG = {{
    "collection_name": "{self.collection_name_var.get()}",
    "persist_directory": "{self.persist_directory_var.get()}",
    "metadata_fields": [
        "title", "doi", "authors", "journal", "year", "citation_count",
        "source", "is_preprint", "publication_date", "fields_of_study",
        "publication_types", "abstract"
    ],
    "max_abstract_length": 1000,
    "id_prefix": "ubr5_api"
}}

# --- SEARCH CONFIGURATION ---
SEARCH_CONFIG = {{
    "comprehensive": {{
        "max_papers": {self.max_results_var.get()},
        "use_semantic_scholar": True,
        "use_scholarly": True,
        "search_all_keywords": True,
        "deduplication": {str(self.deduplication_var.get()).lower()}
    }},
    "focused": {{
        "max_papers": 500,
        "use_semantic_scholar": True,
        "use_scholarly": False,
        "search_all_keywords": False,
        "deduplication": True
    }},
    "quick": {{
        "max_papers": 200,
        "use_semantic_scholar": True,
        "use_scholarly": False,
        "search_all_keywords": False,
        "deduplication": False
    }}
}}

# --- LOGGING CONFIGURATION ---
LOGGING_CONFIG = {{
    "level": "INFO",
    "format": "%(asctime)s - %(levelname)s - %(message)s",
    "file": "ubr5_api_scraping.log",
    "console": True,
    "file_handler": True
}}

# --- PERFORMANCE PROFILES ---
PERFORMANCE_PROFILES = {{
    "sequential_optimized": {{
        "rate_limit_delay": 1.0,  # 1 second between requests
        "strategy_delay": 1.0,    # 1 second between strategies
        "max_workers": 1,         # Sequential processing
        "batch_size": 100         # Smaller batches for reliability
    }},
    "conservative": {{
        "rate_limit_delay": 2.0,  # 2 seconds between requests
        "strategy_delay": 2.0,
        "max_workers": 1,
        "batch_size": 50
    }},
    "balanced": {{
        "rate_limit_delay": 0.5,  # 0.5 seconds between requests
        "strategy_delay": 1.0,
        "max_workers": 2,
        "batch_size": 100
    }},
    "aggressive": {{
        "rate_limit_delay": 0.1,  # 0.1 seconds between requests (may hit rate limits)
        "strategy_delay": 0.5,
        "max_workers": 4,
        "batch_size": 200
    }}
}}

def get_config(profile: str = "sequential_optimized") -> dict:
    """Get configuration for a specific performance profile."""
    if profile not in PERFORMANCE_PROFILES:
        profile = "sequential_optimized"
    
    config = PERFORMANCE_PROFILES[profile].copy()
    config.update({{
        "search_keywords": SCHOLARLY_CONFIG["search_keywords"],
        "semantic_scholar": SEMANTIC_SCHOLAR_CONFIG,
        "scholarly": SCHOLARLY_CONFIG,
        "rate_limiting": RATE_LIMITING,
        "embedding": EMBEDDING_CONFIG,
        "paper_processing": PAPER_PROCESSING,
        "storage": STORAGE_CONFIG,
        "chromadb": CHROMADB_CONFIG,
        "logging": LOGGING_CONFIG
    }})
    
    return config

def print_config_info(profile: str = "sequential_optimized"):
    """Print current configuration information."""
    config = get_config(profile)
    
    print("🔧 UBR5 Scraper Configuration (Sequential Optimized):")
    print(f"   Performance profile: {{profile}}")
    print(f"   Processing mode: Sequential (1 worker)")
    print(f"   Rate limit delay: {{config['rate_limiting']['semantic_scholar_delay']}}s (1 request per second)")
    print(f"   Keyword delay: {{config['rate_limiting']['keyword_delay']}}s")
    print(f"   Search keywords: {{len(config['search_keywords'])}} keywords")
    print(f"   Embedding model: {{config['embedding']['model']}}")
    print(f"   Storage directory: {{config['storage']['embeddings_dir']}}")
    print(f"   ChromaDB collection: {{config['chromadb']['collection_name']}}")
    print()
    print("💡 Performance Notes:")
    print("   - Sequential processing for maximum reliability")
    print("   - 1 request per second to avoid rate limits")
    print("   - Optimized for stable, long-running operations")

if __name__ == "__main__":
    print_config_info()
'''
        
        with open("ubr5_scraper_config.py", "w", encoding="utf-8") as f:
            f.write(config_content)
    
    def save_search_keywords_config(self):
        """Save search keywords configuration to search_keywords_config.json."""
        config_data = {
            "pubmed_keywords": self.pubmed_keywords_text_widget.get("1.0", tk.END).strip(),
            "semantic_keywords": self.semantic_keywords_text_widget.get("1.0", tk.END).strip(),
            "last_updated": {"timestamp": datetime.now().isoformat()}
        }
        
        with open("config/search_keywords_config.json", "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    def load_semantic_api_key(self):
        """Load Semantic Scholar API key from keys.json."""
        try:
            with open("config/keys.json", 'r') as f:
                keys_data = json.load(f)
            
            api_key = keys_data.get("SEMANTIC_SCHOLAR_API_KEY", "")
            self.semantic_api_key_var.set(api_key)
            
            if api_key:
                messagebox.showinfo("Success", "Semantic Scholar API key loaded successfully!")
            else:
                messagebox.showinfo("Info", "No Semantic Scholar API key found in keys.json")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load API key: {e}")
    
    def load_all_configurations(self):
        """Load all configurations from files."""
        try:
            # Load processing_config.py
            self.load_processing_config()
            
            # Load ubr5_scraper_config.py
            self.load_ubr5_scraper_config()
            
            # Load search_keywords_config.json
            self.load_search_keywords_config()
            
            # Load API keys
            self.load_semantic_api_key()
            
            messagebox.showinfo("Success", "All configurations loaded successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load configurations: {e}")
    
    def load_processing_config(self):
        """Load processing configuration from processing_config.py."""
        try:
            from src.core import processing_config
            
            # Load values from the imported module
            self.max_workers_var.set(str(processing_config.MAX_WORKERS))
            self.batch_size_var.set(str(processing_config.BATCH_SIZE))
            self.rate_limit_delay_var.set(str(processing_config.RATE_LIMIT_DELAY))
            self.request_timeout_var.set(str(processing_config.REQUEST_TIMEOUT))
            self.min_chunk_length_var.set(str(processing_config.MIN_CHUNK_LENGTH))
            self.max_chunk_length_var.set(str(processing_config.MAX_CHUNK_LENGTH))
            self.save_interval_var.set(str(processing_config.SAVE_INTERVAL))
            self.citation_rate_limit_var.set(str(processing_config.CITATION_RATE_LIMIT))
            self.citation_timeout_var.set(str(processing_config.CITATION_TIMEOUT))
            self.db_batch_size_var.set(str(processing_config.DB_BATCH_SIZE))
            self.embedding_max_requests_var.set(str(processing_config.EMBEDDING_MAX_REQUESTS_PER_MINUTE))
            self.gemini_max_requests_var.set(str(processing_config.GEMINI_MAX_REQUESTS_PER_MINUTE))
            
        except Exception as e:
            print(f"Error loading processing config: {e}")
    
    def load_ubr5_scraper_config(self):
        """Load UBR5 scraper configuration from ubr5_scraper_config.py."""
        try:
            from src.scrapers import ubr5_scraper_config
            
            # Load values from the imported module
            self.semantic_base_url_var.set(ubr5_scraper_config.SEMANTIC_SCHOLAR_CONFIG["base_url"])
            self.semantic_batch_size_var.set(str(ubr5_scraper_config.SEMANTIC_SCHOLAR_CONFIG["batch_size"]))
            self.semantic_max_papers_var.set(str(ubr5_scraper_config.SEMANTIC_SCHOLAR_CONFIG["max_papers_per_query"]))
            self.embedding_model_var.set(ubr5_scraper_config.EMBEDDING_CONFIG["model"])
            self.embedding_max_length_var.set(str(ubr5_scraper_config.EMBEDDING_CONFIG["max_text_length"]))
            self.collection_name_var.set(ubr5_scraper_config.CHROMADB_CONFIG["collection_name"])
            self.persist_directory_var.set(ubr5_scraper_config.CHROMADB_CONFIG["persist_directory"])
            self.embeddings_dir_var.set(ubr5_scraper_config.STORAGE_CONFIG["embeddings_dir"])
            self.file_format_var.set(ubr5_scraper_config.STORAGE_CONFIG["file_format"])
            
        except Exception as e:
            print(f"Error loading UBR5 scraper config: {e}")
    
    def load_search_keywords_config(self):
        """Load search keywords configuration from search_keywords_config.json."""
        try:
            with open("config/search_keywords_config.json", "r", encoding="utf-8") as f:
                config_data = json.load(f)
            
            # Load keywords into text widgets
            self.pubmed_keywords_text_widget.delete("1.0", tk.END)
            self.pubmed_keywords_text_widget.insert("1.0", config_data.get("pubmed_keywords", ""))
            
            self.semantic_keywords_text_widget.delete("1.0", tk.END)
            self.semantic_keywords_text_widget.insert("1.0", config_data.get("semantic_keywords", ""))
            
        except Exception as e:
            print(f"Error loading search keywords config: {e}")
    
    def reset_configurations(self):
        """Reset all configurations to default values."""
        if messagebox.askyesno("Confirm Reset", "Are you sure you want to reset all configurations to default values? This cannot be undone."):
            try:
                # Reset to optimized default values (1 req/sec)
                self.max_workers_var.set("1")  # Sequential processing
                self.batch_size_var.set("100")  # Smaller batches for reliability
                self.rate_limit_delay_var.set("1.0")  # 1 second between requests
                self.request_timeout_var.set("30")
                self.min_chunk_length_var.set("50")
                self.max_chunk_length_var.set("8000")
                self.save_interval_var.set("1000")
                self.citation_rate_limit_var.set("1.0")  # 1 second between citation requests
                self.citation_timeout_var.set("5")
                self.db_batch_size_var.set("5000")
                self.embedding_max_requests_var.set("60")  # 60 requests per minute (1 per second)
                self.gemini_max_requests_var.set("60")  # 60 requests per minute (1 per second)
                
                self.semantic_base_url_var.set("https://api.semanticscholar.org/graph/v1")
                self.semantic_batch_size_var.set("100")
                self.semantic_max_papers_var.set("50")
                self.embedding_model_var.set("text-embedding-004")
                self.embedding_max_length_var.set("8000")
                self.collection_name_var.set("ubr5_papers")
                self.persist_directory_var.set("./chroma_db")
                self.embeddings_dir_var.set("xrvix_embeddings")
                self.file_format_var.set("json")
                
                self.max_results_var.set("5000")
                self.date_from_var.set("1900")
                self.deduplication_var.set(True)
                
                self.pubmed_keywords_text_widget.delete("1.0", tk.END)
                self.pubmed_keywords_text_widget.insert("1.0", "UBR5,ubr-5,ubr5,tumor immunology,protein degradation")
                
                self.semantic_keywords_text_widget.delete("1.0", tk.END)
                self.semantic_keywords_text_widget.insert("1.0", "UBR5,ubr-5,ubr5,tumor immunology,protein degradation")
                
                messagebox.showinfo("Success", "All configurations reset to default values!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to reset configurations: {e}")
    
    def init_config_variables(self):
        """Initialize configuration variables with optimized default values."""
        # Processing configuration variables - optimized for 1 req/sec
        self.max_workers_var = tk.StringVar(value="1")  # Sequential processing
        self.batch_size_var = tk.StringVar(value="100")  # Smaller batches for reliability
        self.rate_limit_delay_var = tk.StringVar(value="1.0")  # 1 second between requests
        self.request_timeout_var = tk.StringVar(value="30")
        self.min_chunk_length_var = tk.StringVar(value="50")
        self.max_chunk_length_var = tk.StringVar(value="8000")
        self.save_interval_var = tk.StringVar(value="1000")
        self.citation_rate_limit_var = tk.StringVar(value="1.0")  # 1 second between citation requests
        self.citation_timeout_var = tk.StringVar(value="5")
        self.db_batch_size_var = tk.StringVar(value="5000")
        self.embedding_max_requests_var = tk.StringVar(value="60")  # 60 requests per minute (1 per second)
        self.gemini_max_requests_var = tk.StringVar(value="60")  # 60 requests per minute (1 per second)
        
        # API configuration variables
        self.semantic_base_url_var = tk.StringVar(value="https://api.semanticscholar.org/graph/v1")
        self.semantic_batch_size_var = tk.StringVar(value="100")
        self.semantic_max_papers_var = tk.StringVar(value="50")
        self.embedding_model_var = tk.StringVar(value="text-embedding-004")
        self.embedding_max_length_var = tk.StringVar(value="8000")
        
        # Database configuration variables
        self.collection_name_var = tk.StringVar(value="ubr5_papers")
        self.persist_directory_var = tk.StringVar(value="./chroma_db")
        self.embeddings_dir_var = tk.StringVar(value="xrvix_embeddings")
        self.file_format_var = tk.StringVar(value="json")
        
        # Search configuration variables
        self.max_results_var = tk.StringVar(value="5000")
        self.date_from_var = tk.StringVar(value="1900")
        self.deduplication_var = tk.BooleanVar(value=True)
        self.pubmed_keywords_var = tk.StringVar(value="UBR5,ubr-5,ubr5,tumor immunology,protein degradation")
        self.semantic_keywords_var = tk.StringVar(value="UBR5,ubr-5,ubr5,tumor immunology,protein degradation")
        
        # Initialize text widgets as None (will be set when tabs are created)
        self.pubmed_keywords_text_widget = None
        self.semantic_keywords_text_widget = None
        
    def create_hypothesis_tab(self):
        """Create Hypothesis Generation tab with subtabs."""
        # Create notebook for subtabs
        hypothesis_notebook = ttk.Notebook(self.hypothesis_tab)
        hypothesis_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Generate Hypotheses subtab
        generate_frame = ttk.Frame(hypothesis_notebook)
        hypothesis_notebook.add(generate_frame, text="Generate Hypotheses")
        self.create_generate_hypotheses_subtab(generate_frame)
        
        # Test Run subtab
        test_frame = ttk.Frame(hypothesis_notebook)
        hypothesis_notebook.add(test_frame, text="Test Run")
        self.create_test_run_subtab(test_frame)
        
    def create_generate_hypotheses_subtab(self, parent):
        """Create generate hypotheses subtab with scrollable content."""
        # Create a canvas and scrollbar for the entire tab
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bind mousewheel to canvas
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Title
        title_label = ttk.Label(scrollable_frame, text="Generate Hypotheses", 
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Description
        desc_text = """This option launches the Enhanced RAG System for hypothesis generation.
The system will use the loaded ChromaDB data to generate novel research hypotheses."""
        
        desc_label = ttk.Label(scrollable_frame, text=desc_text, justify=tk.LEFT)
        desc_label.pack(pady=(0, 20))
        
        # Input fields frame
        input_frame = ttk.LabelFrame(scrollable_frame, text="Hypothesis Generation Parameters", padding=10)
        input_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Prompt input
        prompt_label = ttk.Label(input_frame, text="Research Prompt:")
        prompt_label.pack(anchor=tk.W, pady=(0, 5))
        
        self.hypothesis_prompt = tk.Text(
            input_frame, 
            height=4,  # Increased from 3 to 4
            wrap=tk.WORD,
            font=('Arial', 10)  # Better font for input
        )
        self.hypothesis_prompt.pack(fill=tk.X, pady=(0, 10))
        
        # Add placeholder text
        placeholder_text = "Enter your research question or hypothesis prompt here...\nExample: How does UBR5 regulate cancer immunity through protein ubiquitination?"
        self.hypothesis_prompt.insert("1.0", placeholder_text)
        self.hypothesis_prompt.config(fg='gray')
        
        # Bind events to handle placeholder text
        self.hypothesis_prompt.bind("<FocusIn>", self._on_prompt_focus_in)
        self.hypothesis_prompt.bind("<FocusOut>", self._on_prompt_focus_out)
        
        # Lab and Institution inputs in a row
        lab_inst_frame = ttk.Frame(input_frame)
        lab_inst_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Lab input
        lab_label = ttk.Label(lab_inst_frame, text="Lab Name:")
        lab_label.pack(side=tk.LEFT, padx=(0, 10))
        
        self.hypothesis_lab = tk.StringVar(value="Dr. Xiaojing Ma")
        lab_entry = ttk.Entry(lab_inst_frame, textvariable=self.hypothesis_lab, width=25)
        lab_entry.pack(side=tk.LEFT, padx=(0, 20))
        
        # Institution input
        inst_label = ttk.Label(lab_inst_frame, text="Institution:")
        inst_label.pack(side=tk.LEFT, padx=(0, 10))
        
        self.hypothesis_institution = tk.StringVar(value="Weill Cornell Medicine")
        inst_entry = ttk.Entry(lab_inst_frame, textvariable=self.hypothesis_institution, width=25)
        inst_entry.pack(side=tk.LEFT)
        
        # Hypotheses per meta-hypothesis input
        hypotheses_per_meta_frame = ttk.Frame(input_frame)
        hypotheses_per_meta_frame.pack(fill=tk.X, pady=(0, 10))
        
        hypotheses_per_meta_label = ttk.Label(hypotheses_per_meta_frame, text="Hypotheses per Meta-Hypothesis:")
        hypotheses_per_meta_label.pack(side=tk.LEFT, padx=(0, 10))
        
        self.hypotheses_per_meta = tk.StringVar(value="3")
        hypotheses_per_meta_entry = ttk.Entry(hypotheses_per_meta_frame, textvariable=self.hypotheses_per_meta, width=10)
        hypotheses_per_meta_entry.pack(side=tk.LEFT, padx=(0, 10))
        
        # Add help text
        help_text = ttk.Label(hypotheses_per_meta_frame, text="(Number of accepted hypotheses to generate for each meta-hypothesis)", 
                             font=('Arial', 8), foreground='gray')
        help_text.pack(side=tk.LEFT)
        
        # Prerequisites check
        prereq_frame = ttk.LabelFrame(scrollable_frame, text="Prerequisites Check", padding=10)
        prereq_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.prereq_status_label = ttk.Label(prereq_frame, text="Checking prerequisites...")
        self.prereq_status_label.pack(anchor=tk.W)
        
        # Buttons
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill=tk.X, pady=(0, 20))
        
        check_prereq_button = ttk.Button(button_frame, text="🔍 Check Prerequisites", 
                                       command=self.check_hypothesis_prerequisites)
        check_prereq_button.pack(side=tk.LEFT, padx=(0, 10))
        
        generate_button = ttk.Button(button_frame, text="🧠 Generate Hypotheses", 
                                   command=self.generate_hypotheses_gui, style='Action.TButton')
        generate_button.pack(side=tk.LEFT, padx=(0, 10))
        
        clear_output_button = ttk.Button(button_frame, text="🗑️ Clear Output", 
                                       command=self.clear_hypothesis_output)
        clear_output_button.pack(side=tk.LEFT)
        
        # Progress bar area
        progress_frame = ttk.LabelFrame(scrollable_frame, text="Progress", padding=10)
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Progress step label
        self.progress_step_label = ttk.Label(progress_frame, text="Ready to generate hypotheses", 
                                           font=('Arial', 10, 'bold'))
        self.progress_step_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate', length=400)
        self.progress_bar.pack(fill=tk.X, pady=(0, 5))
        
        # Progress percentage label
        self.progress_percent_label = ttk.Label(progress_frame, text="0%", font=('Arial', 9))
        self.progress_percent_label.pack(anchor=tk.W)
        
        # Output area - Much larger now
        output_frame = ttk.LabelFrame(scrollable_frame, text="Output", padding=10)
        output_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Create a much larger output text area with better scrolling
        self.hypothesis_output = scrolledtext.ScrolledText(
            output_frame, 
            height=40,  # Significantly increased from 25 to 40
            width=120,  # Increased width from 100 to 120
            wrap=tk.WORD,  # Word wrapping
            font=('Consolas', 9)  # Slightly smaller font to fit more content
        )
        self.hypothesis_output.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Check prerequisites on load
        self.root.after(100, self.check_hypothesis_prerequisites)
    
    def _on_prompt_focus_in(self, event):
        """Handle focus in event for prompt text area."""
        if self.hypothesis_prompt.get("1.0", tk.END).strip() == "Enter your research question or hypothesis prompt here...\nExample: How does UBR5 regulate cancer immunity through protein ubiquitination?":
            self.hypothesis_prompt.delete("1.0", tk.END)
            self.hypothesis_prompt.config(fg='black')
    
    def _on_prompt_focus_out(self, event):
        """Handle focus out event for prompt text area."""
        if not self.hypothesis_prompt.get("1.0", tk.END).strip():
            placeholder_text = "Enter your research question or hypothesis prompt here...\nExample: How does UBR5 regulate cancer immunity through protein ubiquitination?"
            self.hypothesis_prompt.insert("1.0", placeholder_text)
            self.hypothesis_prompt.config(fg='gray')
    
    def clear_hypothesis_output(self):
        """Clear the hypothesis output area."""
        self.hypothesis_output.delete("1.0", tk.END)
        self.log_message(self.hypothesis_output, "🗑️ Output cleared")
        self.update_progress("Ready to generate hypotheses", 0, stop_animation=True)
    
    def update_progress(self, step_text, percentage=None, start_animation=False, stop_animation=False):
        """Update the progress bar and step text."""
        # Update step text
        self.progress_step_label.config(text=step_text)
        
        # Update percentage if provided
        if percentage is not None:
            self.progress_percent_label.config(text=f"{percentage}%")
        
        # Control progress bar animation
        if start_animation:
            self.progress_bar.config(mode='indeterminate')
            self.progress_bar.start(10)  # Start animation
        elif stop_animation:
            self.progress_bar.stop()
            self.progress_bar.config(mode='determinate', value=0)
        elif percentage is not None:
            self.progress_bar.config(mode='determinate', value=percentage)
        
        # Force GUI update
        self.root.update_idletasks()
    
    def log_progress(self, step_text, percentage=None, start_animation=False, stop_animation=False):
        """Log progress to both terminal and GUI output."""
        # Update progress bar
        self.update_progress(step_text, percentage, start_animation, stop_animation)
        
        # Log to GUI output
        self.log_message(self.hypothesis_output, f"📊 {step_text}")
        
        # Log to terminal
        print(f"📊 {step_text}")
        
        # Force GUI update
        self.root.update_idletasks()
        
    def create_test_run_subtab(self, parent):
        """Create test run subtab."""
        # Title
        title_label = ttk.Label(parent, text="Test Run: Biomedical Research", 
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Description
        desc_text = """This option runs a test query focused on biomedical research.
It's useful for testing the system with a specific research focus."""
        
        desc_label = ttk.Label(parent, text=desc_text, justify=tk.LEFT)
        desc_label.pack(pady=(0, 20))
        
        # Test query configuration
        query_frame = ttk.LabelFrame(parent, text="Test Query Configuration", padding=10)
        query_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(query_frame, text="Test Query:").pack(anchor=tk.W)
        self.test_query_var = tk.StringVar(value="protein function and disease")
        test_query_entry = ttk.Entry(query_frame, textvariable=self.test_query_var, width=60)
        test_query_entry.pack(fill=tk.X, pady=(0, 10))
        
        # Prerequisites check
        prereq_frame = ttk.LabelFrame(parent, text="Prerequisites Check", padding=10)
        prereq_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.test_prereq_status_label = ttk.Label(prereq_frame, text="Checking prerequisites...")
        self.test_prereq_status_label.pack(anchor=tk.W)
        
        # Buttons
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=(0, 20))
        
        check_prereq_button = ttk.Button(button_frame, text="🔍 Check Prerequisites", 
                                       command=self.check_test_prerequisites)
        check_prereq_button.pack(side=tk.LEFT, padx=(0, 10))
        
        test_button = ttk.Button(button_frame, text="🧪 Run Test", 
                               command=self.test_run_gui, style='Action.TButton')
        test_button.pack(side=tk.LEFT)
        
        # Output area
        output_frame = ttk.LabelFrame(parent, text="Output", padding=10)
        output_frame.pack(fill=tk.BOTH, expand=True)
        
        self.test_output = scrolledtext.ScrolledText(output_frame, height=15)
        self.test_output.pack(fill=tk.BOTH, expand=True)
        
        # Check prerequisites on load
        self.root.after(100, self.check_test_prerequisites)
        
    def create_tutorial_tab(self):
        """Create Tutorial tab with data pipeline overview."""
        # Create scrollable frame for tutorial content
        canvas = tk.Canvas(self.tutorial_tab)
        scrollbar = ttk.Scrollbar(self.tutorial_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Tutorial content
        self.create_tutorial_content(scrollable_frame)
        
        # Bind mousewheel to canvas
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
    def create_tutorial_content(self, parent):
        """Create tutorial content."""
        # Title
        title_label = ttk.Label(parent, text="AI Research Processor Tutorial", 
                               font=('Arial', 18, 'bold'))
        title_label.pack(pady=(20, 30))
        
        # Introduction
        intro_frame = ttk.LabelFrame(parent, text="Introduction", padding=15)
        intro_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        intro_text = """Welcome to the AI Research Processor! This system is designed to help researchers 
generate novel scientific hypotheses by analyzing vast amounts of scientific literature for any biomedical research topic.

The system uses AI-powered techniques to:
• Scrape and process scientific papers from multiple sources
• Generate embeddings for semantic search
• Use advanced RAG (Retrieval-Augmented Generation) to create hypotheses
• Provide insights into biomedical research and scientific discovery"""
        
        intro_label = ttk.Label(intro_frame, text=intro_text, justify=tk.LEFT, wraplength=800)
        intro_label.pack(anchor=tk.W)
        
        # Data Pipeline Overview
        pipeline_frame = ttk.LabelFrame(parent, text="Data Pipeline Overview", padding=15)
        pipeline_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        pipeline_text = """The AI Research Processor follows a structured data pipeline with 4 main stages:

1. 📚 DATA COLLECTION - Scrape scientific papers from multiple sources
2. 🔄 DATA PROCESSING - Generate embeddings and process content
3. 🗄️ VECTOR STORAGE - Store processed data in ChromaDB
4. 🧠 HYPOTHESIS GENERATION - Use AI to generate research hypotheses

Each stage builds upon the previous one, creating a comprehensive research analysis system."""
        
        pipeline_label = ttk.Label(pipeline_frame, text=pipeline_text, justify=tk.LEFT, wraplength=800)
        pipeline_label.pack(anchor=tk.W)
        
        # Stage 1: Data Collection
        stage1_frame = ttk.LabelFrame(parent, text="Stage 1: Data Collection 📚", padding=15)
        stage1_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        stage1_text = """The system collects scientific papers from three main sources:

🔬 PubMed (Journal Articles)
• Source: National Library of Medicine's PubMed database
• Content: Peer-reviewed journal articles
• Keywords: Customizable search terms (default: biomedical research terms)
• Processing: Direct API access with configurable result limits

📄 Preprints (Biorxiv, Medrxiv)
• Source: Preprint servers for biology and medicine
• Content: Latest research before peer review
• Processing: Bulk download and processing of dump files
• Advantage: Access to cutting-edge research

🔍 Semantic Scholar API
• Source: AI-powered academic search engine
• Content: Comprehensive academic literature
• Keywords: Customizable search terms
• Processing: Advanced citation analysis and metadata extraction

💡 TIP: Start with the "Full Scraper" option to collect data from all sources at once."""
        
        stage1_label = ttk.Label(stage1_frame, text=stage1_text, justify=tk.LEFT, wraplength=800)
        stage1_label.pack(anchor=tk.W)
        
        # Stage 2: Data Processing
        stage2_frame = ttk.LabelFrame(parent, text="Stage 2: Data Processing 🔄", padding=15)
        stage2_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        stage2_text = """Once papers are collected, the system processes them for analysis:

📝 Text Extraction
• Extract full text content from papers
• Clean and normalize text data
• Handle different document formats (PDF, HTML, etc.)

🔤 Embedding Generation
• Convert text into numerical vectors (embeddings)
• Use advanced language models for semantic understanding
• Enable similarity search and clustering

📊 Metadata Processing
• Extract key information: authors, journals, citations, impact factors
• Process abstracts, keywords, and references
• Organize data by source and topic

🏷️ Categorization
• Group papers by research area
• Identify key themes and concepts
• Prepare data for vector storage

💡 TIP: Embeddings are automatically generated during the scraping process."""
        
        stage2_label = ttk.Label(stage2_frame, text=stage2_text, justify=tk.LEFT, wraplength=800)
        stage2_label.pack(anchor=tk.W)
        
        # Stage 3: Vector Storage
        stage3_frame = ttk.LabelFrame(parent, text="Stage 3: Vector Storage 🗄️", padding=15)
        stage3_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        stage3_text = """Processed data is stored in ChromaDB, a vector database:

💾 ChromaDB Features
• Persistent storage of embeddings and metadata
• Fast similarity search capabilities
• Automatic indexing and optimization
• Support for large-scale data collections

📦 Data Organization
• Collections organized by data source
• Batch processing for efficient loading
• Source tracking and statistics
• Easy data management and updates

🔍 Search Capabilities
• Semantic similarity search
• Keyword-based filtering
• Citation network analysis
• Temporal analysis of research trends

📊 Monitoring
• Real-time statistics on stored documents
• Source breakdown and analysis
• Data quality metrics
• Storage optimization

💡 TIP: Use "Load Embeddings" to populate ChromaDB with your collected data."""
        
        stage3_label = ttk.Label(stage3_frame, text=stage3_text, justify=tk.LEFT, wraplength=800)
        stage3_label.pack(anchor=tk.W)
        
        # Stage 4: Hypothesis Generation
        stage4_frame = ttk.LabelFrame(parent, text="Stage 4: Hypothesis Generation 🧠", padding=15)
        stage4_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        stage4_text = """The final stage uses AI to generate novel research hypotheses:

🤖 RAG System (Retrieval-Augmented Generation)
• Combines retrieval of relevant papers with AI generation
• Uses advanced language models for hypothesis creation
• Context-aware analysis of research gaps

🔍 Knowledge Synthesis
• Identifies patterns across multiple papers
• Connects disparate research findings
• Highlights unexplored research directions

💡 Hypothesis Types
• Mechanistic hypotheses about protein function
• Therapeutic intervention strategies
• Biomarker discovery opportunities
• Novel research methodologies

🎯 Focus Areas
• Protein function and regulation
• Disease biology and immunology
• Cellular pathways and systems
• Therapeutic target identification

💡 TIP: Start with the "Test Run" to see how hypothesis generation works."""
        
        stage4_label = ttk.Label(stage4_frame, text=stage4_text, justify=tk.LEFT, wraplength=800)
        stage4_label.pack(anchor=tk.W)
        
        # Workflow Guide
        workflow_frame = ttk.LabelFrame(parent, text="Recommended Workflow 🚀", padding=15)
        workflow_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        workflow_text = """Follow this step-by-step workflow for best results:

STEP 1: Check System Status
• Go to Settings → Data Status
• Verify what data sources are available
• Check ChromaDB status

STEP 2: Collect Data (if needed)
• Use Paper Scraping → Full Scraper for comprehensive collection
• Or use specific scrapers for targeted data collection
• Monitor progress in the output areas

STEP 3: Load Data into ChromaDB
• Go to Vector Database → Load Embeddings
• Wait for processing to complete
• Verify data was loaded successfully

STEP 4: Generate Hypotheses
• Go to Hypothesis Generation → Generate Hypotheses
• Or start with Test Run for a quick demonstration
• Review generated hypotheses and insights

STEP 5: Analyze Results
• Use Settings → Data Status to review your data
• Check Vector Database → Show ChromaDB Data for statistics
• Export results for further analysis

💡 TIP: The system remembers your data between sessions - ChromaDB uses persistent storage."""
        
        workflow_label = ttk.Label(workflow_frame, text=workflow_text, justify=tk.LEFT, wraplength=800)
        workflow_label.pack(anchor=tk.W)
        
        # Configuration Tips
        config_frame = ttk.LabelFrame(parent, text="Configuration Tips ⚙️", padding=15)
        config_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        config_text = """Optimize your research with these configuration tips:

🔍 Search Keywords
• Customize keywords for your specific research focus
• Use synonyms and related terms (e.g., protein names, disease terms, pathways)
• Save keyword configurations for reuse

📊 PubMed Limits
• Start with 1000-5000 papers for initial exploration
• Use unlimited (-1) for comprehensive analysis (warning: takes hours)
• Consider your computational resources

🗄️ ChromaDB Management
• Clear data when switching research topics
• Monitor storage usage for large datasets
• Use batch processing for efficiency

🧠 Hypothesis Generation
• Start with test queries to understand the system
• Use specific, focused queries for better results
• Review and refine generated hypotheses

💡 TIP: Check Settings → Configurations to see current system settings."""
        
        config_label = ttk.Label(config_frame, text=config_text, justify=tk.LEFT, wraplength=800)
        config_label.pack(anchor=tk.W)
        
        # Troubleshooting
        troubleshooting_frame = ttk.LabelFrame(parent, text="Troubleshooting 🔧", padding=15)
        troubleshooting_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        troubleshooting_text = """Common issues and solutions:

❌ "No ChromaDB collections found"
• Solution: Run "Load Embeddings" first to populate the database

❌ "No preprint dumps found"
• Solution: The system will automatically download dumps when needed
• Ensure you have internet connection

❌ "Keywords not saved"
• Solution: Check file permissions in the project directory
• Try running as administrator if needed

❌ "Import errors"
• Solution: Install missing dependencies: pip install pandas paperscraper chromadb
• Check Python version compatibility

❌ "Long processing times"
• Solution: This is normal for large datasets
• Use smaller PubMed limits for faster initial runs
• Monitor progress in output areas

💡 TIP: Check the output areas in each tab for detailed error messages and progress updates."""
        
        troubleshooting_label = ttk.Label(troubleshooting_frame, text=troubleshooting_text, justify=tk.LEFT, wraplength=800)
        troubleshooting_label.pack(anchor=tk.W)
        
        # Advanced Features
        advanced_frame = ttk.LabelFrame(parent, text="Advanced Features 🚀", padding=15)
        advanced_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        advanced_text = """Explore these advanced features for power users:

🔬 Custom Research Focus
• Modify keywords for specific protein families or disease areas
• Target specific disease types or biological pathways
• Focus on particular research methodologies

📈 Data Analysis
• Export data for external analysis tools
• Use citation networks for research mapping
• Analyze temporal trends in research

🤖 AI Customization
• Adjust RAG parameters for different hypothesis types
• Fine-tune search strategies
• Experiment with different query formulations

🔍 Research Validation
• Cross-reference hypotheses with recent literature
• Validate findings against multiple data sources
• Track hypothesis evolution over time

💡 TIP: The system is designed to be extensible - check the documentation for advanced customization options."""
        
        advanced_label = ttk.Label(advanced_frame, text=advanced_text, justify=tk.LEFT, wraplength=800)
        advanced_label.pack(anchor=tk.W)
        
        # Getting Help
        help_frame = ttk.LabelFrame(parent, text="Getting Help 📞", padding=15)
        help_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        help_text = """Need assistance? Here's how to get help:

📖 Documentation
• Check GUI_README.md for detailed GUI instructions
• Review README.md for system overview
• Consult processing_config.py for configuration options

🔍 Status Monitoring
• Use Settings → Data Status for system overview
• Check Vector Database → Show ChromaDB Data for database status
• Monitor output areas for real-time progress

🐛 Error Reporting
• Check output areas for detailed error messages
• Look for error codes and suggested solutions
• Note the specific operation that failed

💡 Best Practices
• Start with small datasets to learn the system
• Save your keyword configurations
• Regularly check data status
• Use test runs before large-scale operations

💡 TIP: The system provides helpful error messages and suggestions in the output areas."""
        
        help_label = ttk.Label(help_frame, text=help_text, justify=tk.LEFT, wraplength=800)
        help_label.pack(anchor=tk.W)
        
        # Footer
        footer_label = ttk.Label(parent, text="Happy Researching! 🧬", 
                               font=('Arial', 14, 'bold'))
        footer_label.pack(pady=30)
        
    # GUI Methods for Paper Scraping & Processing
    
    def load_saved_keywords(self):
        """Load saved keywords from configuration file."""
        try:
            if os.path.exists("config/search_keywords_config.json"):
                with open("config/search_keywords_config.json", 'r') as f:
                    config = json.load(f)
                self.pubmed_keywords_var.set(config.get("pubmed_keywords", ""))
                self.semantic_keywords_var.set(config.get("semantic_keywords", ""))
        except Exception as e:
            print(f"Could not load keywords config: {e}")
            
    def load_saved_keywords_unified(self):
        """Load saved keywords from configuration file for unified format."""
        try:
            if os.path.exists("config/search_keywords_config.json"):
                with open("config/search_keywords_config.json", 'r') as f:
                    config = json.load(f)
                # Use pubmed_keywords as the unified keywords (they should be the same)
                unified_keywords = config.get("pubmed_keywords", "")
                self.full_scraper_keywords_var.set(unified_keywords)
        except Exception as e:
            print(f"Could not load keywords config: {e}")
            
    def save_keywords_gui(self):
        """Save keywords configuration."""
        try:
            config = {
                "pubmed_keywords": self.pubmed_keywords_var.get(),
                "semantic_keywords": self.semantic_keywords_var.get(),
                "last_updated": json.dumps({"timestamp": __import__("datetime").datetime.now().isoformat()})
            }
            
            with open("config/search_keywords_config.json", 'w') as f:
                json.dump(config, f, indent=2)
            
            messagebox.showinfo("Success", "Keywords saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save keywords: {e}")
            
    def save_keywords_unified_gui(self):
        """Save unified keywords configuration."""
        try:
            unified_keywords = self.full_scraper_keywords_var.get()
            config = {
                "pubmed_keywords": unified_keywords,
                "semantic_keywords": unified_keywords,
                "last_updated": json.dumps({"timestamp": __import__("datetime").datetime.now().isoformat()})
            }
            
            with open("config/search_keywords_config.json", 'w') as f:
                json.dump(config, f, indent=2)
            
            messagebox.showinfo("Success", "Keywords saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save keywords: {e}")
            
    def run_full_scraper_gui(self):
        """Run full scraper in a separate thread."""
        if not self.full_scraper_keywords_var.get().strip():
            messagebox.showerror("Error", "Please enter keywords")
            return
            
        try:
            max_results = int(self.max_results_var.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for max results")
            return
            
        # Run in separate thread
        thread = threading.Thread(target=self._run_full_scraper_thread, 
                                args=(max_results,))
        thread.daemon = True
        thread.start()
        
    def _run_full_scraper_thread(self, max_results):
        """Thread function for full scraper."""
        self.start_progress()
        self.update_status("Running full scraper...")
        
        # Create progress wrapper
        progress_wrapper = ScraperProgressWrapper(
            self, 
            self.full_scraper_progress, 
            self.full_scraper_output
        )
        
        try:
            self.log_message(self.full_scraper_output, "Starting Full Scraper (Preprints + Journal Articles)")
            self.log_message(self.full_scraper_output, "="*60)
            
            success_count = 0
            total_sources = 3
            
            # Step 1: Process PubMed
            self.log_message(self.full_scraper_output, "Step 1/3: Processing PubMed (Journal Articles)...")
            try:
                self.log_message(self.full_scraper_output, f"Using keywords: {self.full_scraper_keywords_var.get()}")
                progress_wrapper.run_pubmed_with_progress(max_results=max_results)
                self.log_message(self.full_scraper_output, "✅ PubMed processing completed successfully!")
                success_count += 1
            except Exception as e:
                self.log_message(self.full_scraper_output, f"❌ PubMed processing failed: {e}")
            
            # Step 2: Process xrvix
            self.log_message(self.full_scraper_output, "Step 2/3: Processing xrvix (Preprints)...")
            try:
                progress_wrapper.run_xrvix_with_progress()
                self.log_message(self.full_scraper_output, "✅ xrvix processing completed successfully!")
                success_count += 1
            except Exception as e:
                self.log_message(self.full_scraper_output, f"❌ xrvix processing failed: {e}")
            
            # Step 3: Process UBR5
            self.log_message(self.full_scraper_output, "Step 3/3: Processing UBR5 (Semantic Scholar)...")
            try:
                self.log_message(self.full_scraper_output, f"Using keywords: {self.full_scraper_keywords_var.get()}")
                progress_wrapper.run_ubr5_with_progress(self.full_scraper_keywords_var.get())
                self.log_message(self.full_scraper_output, "✅ UBR5 processing completed successfully!")
                success_count += 1
            except Exception as e:
                self.log_message(self.full_scraper_output, f"❌ UBR5 processing failed: {e}")
            
            # Summary
            self.log_message(self.full_scraper_output, f"🎉 Full scraper completed!")
            self.log_message(self.full_scraper_output, f"✅ Successfully processed: {success_count}/{total_sources} sources")
            
        except Exception as e:
            self.log_message(self.full_scraper_output, f"❌ Unexpected error: {e}")
        finally:
            self.stop_progress()
            self.update_status("Ready")
            
    def run_journal_articles_gui(self):
        """Run journal articles scraper in a separate thread."""
        if not self.journal_keywords_var.get().strip():
            messagebox.showerror("Error", "Please enter keywords")
            return
            
        # Get source selection
        source = self.journal_source_var.get()
        
        # Validate max results only if PubMed is selected
        max_results = None
        if source in ["both", "pubmed"]:
            try:
                max_results = int(self.journal_max_results_var.get())
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid number for max results")
                return
            
        # Run in separate thread
        thread = threading.Thread(target=self._run_journal_articles_thread, 
                                args=(max_results, source))
        thread.daemon = True
        thread.start()
        
    def _run_journal_articles_thread(self, max_results, source):
        """Thread function for journal articles scraper."""
        self.start_progress()
        self.update_status("Running journal articles scraper...")
        
        # Create progress wrapper
        progress_wrapper = ScraperProgressWrapper(
            self, 
            self.journal_progress, 
            self.journal_output
        )
        
        try:
            source_name = {
                "both": "PubMed + Semantic Scholar",
                "pubmed": "PubMed Only",
                "semantic": "Semantic Scholar Only"
            }.get(source, "Unknown")
            
            self.log_message(self.journal_output, f"Starting Journal Articles Scraper ({source_name})")
            self.log_message(self.journal_output, "="*60)
            
            success_count = 0
            total_sources = 2 if source == "both" else 1
            
            # Step 1: Process PubMed (if selected)
            if source in ["both", "pubmed"]:
                step_num = "1/2" if source == "both" else "1/1"
                self.log_message(self.journal_output, f"Step {step_num}: Processing PubMed...")
                try:
                    self.log_message(self.journal_output, f"Using keywords: {self.journal_keywords_var.get()}")
                    progress_wrapper.run_pubmed_with_progress(max_results=max_results)
                    self.log_message(self.journal_output, "✅ PubMed processing completed successfully!")
                    success_count += 1
                except Exception as e:
                    self.log_message(self.journal_output, f"❌ PubMed processing failed: {e}")
            
            # Step 2: Process Semantic Scholar (if selected)
            if source in ["both", "semantic"]:
                step_num = "2/2" if source == "both" else "1/1"
                self.log_message(self.journal_output, f"Step {step_num}: Processing Semantic Scholar...")
                try:
                    self.log_message(self.journal_output, f"Using keywords: {self.journal_keywords_var.get()}")
                    progress_wrapper.run_ubr5_with_progress(self.journal_keywords_var.get())
                    self.log_message(self.journal_output, "✅ Semantic Scholar processing completed successfully!")
                    success_count += 1
                except Exception as e:
                    self.log_message(self.journal_output, f"❌ Semantic Scholar processing failed: {e}")
            
            # Summary
            self.log_message(self.journal_output, f"🎉 Journal articles scraper completed!")
            self.log_message(self.journal_output, f"✅ Successfully processed: {success_count}/{total_sources} sources")
            
        except Exception as e:
            self.log_message(self.journal_output, f"❌ Unexpected error: {e}")
        finally:
            self.stop_progress()
            self.update_status("Ready")
            
    def check_preprints_status(self):
        """Check preprint data status."""
        try:
            dump_dir = "data/scraped_data/paperscraper_dumps"
            if os.path.exists(dump_dir):
                biorxiv_dumps = [f for f in os.listdir(dump_dir) if f.startswith('biorxiv')]
                medrxiv_dumps = [f for f in os.listdir(dump_dir) if f.startswith('medrxiv')]
                
                if biorxiv_dumps or medrxiv_dumps:
                    status_text = f"✅ Found {len(biorxiv_dumps)} Biorxiv dumps and {len(medrxiv_dumps)} Medrxiv dumps"
                else:
                    status_text = "❌ No preprint dumps found - will download automatically"
            else:
                status_text = "❌ No preprint dumps directory found - will download automatically"
                
            self.preprints_status_label.config(text=status_text)
        except Exception as e:
            self.preprints_status_label.config(text=f"❌ Error checking status: {e}")
            
    def run_preprints_gui(self):
        """Run preprints scraper in a separate thread."""
        # Get source selection
        source = self.preprints_source_var.get()
        
        # Run in separate thread
        thread = threading.Thread(target=self._run_preprints_thread, args=(source,))
        thread.daemon = True
        thread.start()
        
    def _run_preprints_thread(self, source):
        """Thread function for preprints scraper."""
        self.start_progress()
        self.update_status("Running preprints scraper...")
        
        # Create progress wrapper
        progress_wrapper = ScraperProgressWrapper(
            self, 
            self.preprints_progress, 
            self.preprints_output
        )
        
        try:
            source_name = {
                "both": "BioRxiv + MedRxiv",
                "biorxiv": "BioRxiv Only",
                "medrxiv": "MedRxiv Only"
            }.get(source, "Unknown")
            
            self.log_message(self.preprints_output, f"Starting Preprints Scraper ({source_name})")
            self.log_message(self.preprints_output, "="*60)
            
            # Note: The current xrvix processor handles both BioRxiv and MedRxiv together
            # For individual source processing, we would need to modify the xrvix processor
            # For now, we'll run the full processor and log which sources are being processed
            
            if source == "both":
                self.log_message(self.preprints_output, "Processing both BioRxiv and MedRxiv sources...")
                progress_wrapper.run_xrvix_with_progress()
                self.log_message(self.preprints_output, "✅ Both preprint sources processed successfully!")
            elif source == "biorxiv":
                self.log_message(self.preprints_output, "Processing BioRxiv only...")
                self.log_message(self.preprints_output, "⚠️ Note: Current implementation processes both sources together")
                self.log_message(self.preprints_output, "💡 BioRxiv data will be extracted from the combined processing")
                progress_wrapper.run_xrvix_with_progress()
                self.log_message(self.preprints_output, "✅ BioRxiv processing completed!")
            elif source == "medrxiv":
                self.log_message(self.preprints_output, "Processing MedRxiv only...")
                self.log_message(self.preprints_output, "⚠️ Note: Current implementation processes both sources together")
                self.log_message(self.preprints_output, "💡 MedRxiv data will be extracted from the combined processing")
                progress_wrapper.run_xrvix_with_progress()
                self.log_message(self.preprints_output, "✅ MedRxiv processing completed!")
            
        except Exception as e:
            self.log_message(self.preprints_output, f"❌ Preprints processing failed: {e}")
        finally:
            self.stop_progress()
            self.update_status("Ready")
            
    def check_embeddings_status(self):
        """Check embeddings status."""
        try:
            if os.path.exists("data/embeddings/xrvix_embeddings"):
                status_text = "✅ Embeddings directory exists - embeddings are generated during scraping"
            else:
                status_text = "❌ No embeddings found - run scraping options first"
                
            self.embeddings_status_label.config(text=status_text)
        except Exception as e:
            self.embeddings_status_label.config(text=f"❌ Error checking status: {e}")
            
    def generate_embeddings_gui(self):
        """Generate embeddings based on selected source."""
        source = self.embedding_source_var.get()
        
        self.log_message(self.embeddings_output, f"🔄 Starting embedding generation for: {source}")
        
        if source == "all":
            self.log_message(self.embeddings_output, "📚 Generating embeddings for all sources (PubMed + Preprints)")
            self.log_message(self.embeddings_output, "💡 This will process both journal articles and preprints")
            
        elif source == "pubmed":
            self.log_message(self.embeddings_output, "📚 Generating embeddings for PubMed data only")
            self.log_message(self.embeddings_output, "💡 This will process journal articles from PubMed")
            
        elif source == "preprints":
            self.log_message(self.embeddings_output, "📄 Generating embeddings for preprints only")
            self.log_message(self.embeddings_output, "💡 This will process BioRxiv and MedRxiv data")
        
        # Check if we have the required data
        if source in ["all", "pubmed"]:
            pubmed_dir = "data/scraped_data/pubmed"
            if os.path.exists(pubmed_dir):
                pubmed_files = [f for f in os.listdir(pubmed_dir) if f.endswith('.jsonl')]
                if pubmed_files:
                    self.log_message(self.embeddings_output, f"✅ Found {len(pubmed_files)} PubMed data files")
                else:
                    self.log_message(self.embeddings_output, "❌ No PubMed data files found - run PubMed scraping first")
            else:
                self.log_message(self.embeddings_output, "❌ No PubMed data directory found - run PubMed scraping first")
        
        if source in ["all", "preprints"]:
            xrvix_dir = "data/embeddings/xrvix_embeddings"
            if os.path.exists(xrvix_dir):
                self.log_message(self.embeddings_output, "✅ Found xrvix embeddings directory")
            else:
                self.log_message(self.embeddings_output, "❌ No xrvix embeddings found - run preprints scraping first")
        
        # Check API key
        try:
            with open("config/keys.json", 'r') as f:
                keys = json.load(f)
            google_api_key = keys.get("GOOGLE_API_KEY")
            if google_api_key:
                self.log_message(self.embeddings_output, "✅ Google API key found")
            else:
                self.log_message(self.embeddings_output, "❌ No Google API key found - add to config/keys.json")
                return
        except Exception as e:
            self.log_message(self.embeddings_output, f"❌ Error loading API keys: {e}")
            return
        
        # Reset progress bar
        self.embedding_progress_bar['value'] = 0
        self.embedding_progress_label.config(text="Ready to start")
        self.embedding_file_info.config(text="")
        
        # Start the embedding generation process
        self.start_progress()
        self.update_status("Generating embeddings...")
        
        def run_embedding_generation():
            try:
                if source == "pubmed":
                    self.run_pubmed_embedding_generation()
                elif source == "preprints":
                    self.run_preprints_embedding_generation()
                elif source == "all":
                    self.run_all_embedding_generation()
                    
            except Exception as e:
                self.log_message(self.embeddings_output, f"❌ Error during embedding generation: {e}")
                self.embedding_progress_label.config(text="❌ Error occurred")
                self.embedding_file_info.config(text="")
            finally:
                self.stop_progress()
                self.update_status("Ready")
        
        # Run in separate thread
        threading.Thread(target=run_embedding_generation, daemon=True).start()
    
    def run_pubmed_embedding_generation(self):
        """Run PubMed embedding generation with progress tracking."""
        try:
            import sys
            sys.path.append('src/scrapers')
            from pubmed_scraper_json import process_existing_pubmed_data
            
            # Load API key
            with open("config/keys.json", 'r') as f:
                keys = json.load(f)
            google_api_key = keys.get("GOOGLE_API_KEY")
            
            # Find PubMed files
            pubmed_dir = "data/scraped_data/pubmed"
            pubmed_files = [f for f in os.listdir(pubmed_dir) if f.endswith('.jsonl') and f.startswith('pubmed_final_')]
            
            if not pubmed_files:
                self.log_message(self.embeddings_output, "❌ No PubMed final files found")
                return
            
            # Sort files by size (largest first)
            pubmed_files.sort(key=lambda f: os.path.getsize(os.path.join(pubmed_dir, f)), reverse=True)
            
            total_files = len(pubmed_files)
            self.log_message(self.embeddings_output, f"📚 Found {total_files} PubMed files to process")
            
            # Calculate total expected embeddings across all files
            total_expected_embeddings = 0
            file_embedding_counts = {}
            
            self.log_message(self.embeddings_output, "🔍 Calculating total embeddings across all files...")
            self.embedding_progress_label.config(text="🔍 Calculating total embeddings...")
            
            for filename in pubmed_files:
                file_path = os.path.join(pubmed_dir, filename)
                try:
                    # Count embeddings in this file
                    file_embeddings = 0
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                paper = json.loads(line)
                                title = paper.get('title', '')
                                abstract = paper.get('abstract', '')
                                full_text = f"{title}\n\n{abstract}" if abstract else title
                                
                                # Import chunk_paragraphs function
                                sys.path.append('src/scrapers')
                                from pubmed_scraper_json import chunk_paragraphs
                                paragraphs = chunk_paragraphs(full_text)
                                file_embeddings += len([para for para in paragraphs if para and para.strip()])
                    
                    file_embedding_counts[filename] = file_embeddings
                    total_expected_embeddings += file_embeddings
                    
                except Exception as e:
                    self.log_message(self.embeddings_output, f"⚠️  Error calculating embeddings for {filename}: {e}")
                    file_embedding_counts[filename] = 0
            
            self.log_message(self.embeddings_output, f"📊 Total expected embeddings: {total_expected_embeddings}")
            
            # Set up progress bar for embeddings
            self.embedding_progress_bar['maximum'] = total_expected_embeddings
            self.embedding_progress_bar['value'] = 0
            self.embedding_progress_label.config(text="🚀 Generating Embeddings...")
            
            processed_files = 0
            successful_files = 0
            total_processed_embeddings = 0
            
            for i, filename in enumerate(pubmed_files):
                file_path = os.path.join(pubmed_dir, filename)
                file_size = os.path.getsize(file_path)
                expected_embeddings = file_embedding_counts[filename]
                
                # Update progress info
                files_left = total_files - i
                self.embedding_file_info.config(text=f"📄 {filename} ({expected_embeddings:,} embeddings) | Files left: {files_left}")
                self.log_message(self.embeddings_output, f"📚 Processing file {i+1}/{total_files}: {filename} ({expected_embeddings:,} embeddings)")
                
                # Create progress callback for this file
                def create_progress_callback(file_start_embeddings):
                    def progress_callback(current_embeddings, total_embeddings):
                        # Update GUI progress bar
                        self.embedding_progress_bar['value'] = file_start_embeddings + current_embeddings
                        self.embedding_progress_label.config(text=f"🚀 Generating Embeddings: {file_start_embeddings + current_embeddings:,}/{total_expected_embeddings:,}")
                        self.root.update_idletasks()
                    return progress_callback
                
                # Process the file with progress callback
                success = process_existing_pubmed_data(file_path, google_api_key, 
                                                    progress_callback=create_progress_callback(total_processed_embeddings))
                
                if success:
                    successful_files += 1
                    total_processed_embeddings += expected_embeddings
                    self.log_message(self.embeddings_output, f"✅ Successfully processed: {filename}")
                else:
                    self.log_message(self.embeddings_output, f"❌ Failed to process: {filename}")
                
                processed_files += 1
            
            # Final status
            self.embedding_progress_label.config(text=f"✅ Completed: {total_processed_embeddings:,}/{total_expected_embeddings:,} embeddings")
            self.embedding_file_info.config(text="")
            
            if successful_files > 0:
                self.log_message(self.embeddings_output, f"✅ PubMed embedding generation completed! {total_processed_embeddings:,}/{total_expected_embeddings:,} embeddings processed")
            else:
                self.log_message(self.embeddings_output, "❌ PubMed embedding generation failed!")
                
        except Exception as e:
            self.log_message(self.embeddings_output, f"❌ Error in PubMed embedding generation: {e}")
            self.embedding_progress_label.config(text="❌ Error occurred")
            self.embedding_file_info.config(text="")
    
    def run_preprints_embedding_generation(self):
        """Run preprints embedding generation with progress tracking."""
        try:
            self.embedding_progress_label.config(text="📄 Checking Preprint Embeddings...")
            self.embedding_progress_bar['maximum'] = 100
            self.embedding_progress_bar['value'] = 0
            
            self.log_message(self.embeddings_output, "📄 Preprint embeddings are generated during the scraping process")
            self.log_message(self.embeddings_output, "💡 Run the 'Preprints Only' scraper to generate preprint embeddings")
            
            # Check if preprints embeddings exist
            xrvix_dir = "data/embeddings/xrvix_embeddings"
            if os.path.exists(xrvix_dir):
                self.embedding_progress_bar['value'] = 50
                self.embedding_file_info.config(text="🔍 Scanning preprint directories...")
                self.root.update_idletasks()
                
                batch_files = []
                for subdir in ['biorxiv', 'medrxiv']:
                    subdir_path = os.path.join(xrvix_dir, subdir)
                    if os.path.exists(subdir_path):
                        batch_files.extend([f for f in os.listdir(subdir_path) if f.startswith('batch_')])
                
                self.embedding_progress_bar['value'] = 100
                
                if batch_files:
                    self.log_message(self.embeddings_output, f"✅ Found {len(batch_files)} preprint embedding batch files")
                    self.log_message(self.embeddings_output, "💡 Preprint embeddings are already available for loading")
                    self.embedding_progress_label.config(text=f"✅ Found {len(batch_files)} preprint batch files")
                    self.embedding_file_info.config(text="Preprint embeddings ready for loading")
                else:
                    self.log_message(self.embeddings_output, "❌ No preprint embedding batch files found")
                    self.log_message(self.embeddings_output, "💡 Run the 'Preprints Only' scraper first")
                    self.embedding_progress_label.config(text="❌ No preprint embeddings found")
                    self.embedding_file_info.config(text="Run 'Preprints Only' scraper first")
            else:
                self.log_message(self.embeddings_output, "❌ No xrvix embeddings directory found")
                self.log_message(self.embeddings_output, "💡 Run the 'Preprints Only' scraper first")
                self.embedding_progress_label.config(text="❌ No xrvix directory found")
                self.embedding_file_info.config(text="Run 'Preprints Only' scraper first")
                
        except Exception as e:
            self.log_message(self.embeddings_output, f"❌ Error checking preprint embeddings: {e}")
            self.embedding_progress_label.config(text="❌ Error occurred")
            self.embedding_file_info.config(text="")
    
    def run_all_embedding_generation(self):
        """Run embedding generation for all sources with progress tracking."""
        try:
            self.embedding_progress_label.config(text="🔄 Processing All Sources...")
            self.embedding_progress_bar['maximum'] = 100
            self.embedding_progress_bar['value'] = 0
            
            self.log_message(self.embeddings_output, "🔄 Running embedding generation for all sources...")
            
            # First run PubMed embeddings
            self.log_message(self.embeddings_output, "📚 Step 1: Generating PubMed embeddings...")
            self.embedding_file_info.config(text="📚 Processing PubMed files...")
            self.root.update_idletasks()
            
            self.run_pubmed_embedding_generation()
            
            # Update progress
            self.embedding_progress_bar['value'] = 50
            self.root.update_idletasks()
            
            # Then check preprints
            self.log_message(self.embeddings_output, "📄 Step 2: Checking preprint embeddings...")
            self.embedding_file_info.config(text="📄 Checking preprint embeddings...")
            self.root.update_idletasks()
            
            self.run_preprints_embedding_generation()
            
            # Final progress
            self.embedding_progress_bar['value'] = 100
            self.embedding_progress_label.config(text="✅ All-source processing completed!")
            self.embedding_file_info.config(text="All sources processed successfully")
            
            self.log_message(self.embeddings_output, "✅ All-source embedding generation completed!")
            
        except Exception as e:
            self.log_message(self.embeddings_output, f"❌ Error in all-source embedding generation: {e}")
            self.embedding_progress_label.config(text="❌ Error occurred")
            self.embedding_file_info.config(text="")
            
    # GUI Methods for Vector Database Management
    
    def check_chromadb_status(self):
        """Check ChromaDB status."""
        try:
            manager = ChromaDBManager()
            collections = manager.list_collections()
            
            if collections:
                stats = manager.get_collection_stats()
                total_docs = stats.get('total_documents', 0)
                status_text = f"✅ ChromaDB available with {total_docs} documents in {len(collections)} collection(s)"
            else:
                status_text = "⚠️ ChromaDB available but no collections found"
                
            self.load_status_label.config(text=status_text)
        except Exception as e:
            self.load_status_label.config(text=f"❌ Error checking ChromaDB: {e}")
            
    def load_embeddings_gui(self):
        """Load embeddings into ChromaDB in a separate thread."""
        thread = threading.Thread(target=self._load_embeddings_thread)
        thread.daemon = True
        thread.start()
        
    def _load_embeddings_thread(self):
        """Thread function for loading embeddings."""
        self.start_progress()
        self.update_status("Loading embeddings into ChromaDB...")
        
        try:
            source = self.load_source_var.get()
            self.log_message(self.load_output, f"Loading embeddings into ChromaDB for: {source}")
            self.log_message(self.load_output, "="*60)
            
            # Initialize ChromaDB manager
            manager = ChromaDBManager()
            
            # Create collection
            if not manager.create_collection():
                self.log_message(self.load_output, "❌ Failed to create ChromaDB collection")
                return
            
            # Check if collection already has data
            stats = manager.get_collection_stats()
            if stats.get('total_documents', 0) > 0:
                self.log_message(self.load_output, f"📚 ChromaDB already has {stats.get('total_documents', 0)} documents!")
                self.log_message(self.load_output, "💡 ChromaDB uses persistent storage - data is already saved locally.")
                
                # Ask user if they want to reload anyway
                result = messagebox.askyesno("Reload Data?", 
                                           "ChromaDB already has data. Do you want to reload anyway?")
                if not result:
                    self.log_message(self.load_output, "✅ Using existing ChromaDB data.")
                    return
            
            total_loaded = 0
            
            if source in ["all", "pubmed"]:
                # Load PubMed embeddings from batch files
                pubmed_batch_dir = "data/embeddings/pubmed"
                pubmed_single_file = "data/embeddings/xrvix_embeddings/pubmed_embeddings.json"
                
                # First try to load from batch files (newer format)
                if os.path.exists(pubmed_batch_dir):
                    batch_files = [f for f in os.listdir(pubmed_batch_dir) if f.startswith("batch_") and f.endswith(".json")]
                    if batch_files:
                        self.log_message(self.load_output, f"🔄 Loading PubMed embeddings from {len(batch_files)} batch files...")
                        if manager.add_embeddings_from_directory(pubmed_batch_dir, sources=["pubmed"], db_batch_size=DB_BATCH_SIZE):
                            stats = manager.get_collection_stats()
                            total_docs = stats.get('total_documents', 0)
                            self.log_message(self.load_output, f"✅ Loaded PubMed embeddings from batch files (total documents: {total_docs})")
                            total_loaded = total_docs
                        else:
                            self.log_message(self.load_output, "❌ Failed to load PubMed embeddings from batch files")
                
                # Fallback to single file (legacy format)
                elif os.path.exists(pubmed_single_file):
                    self.log_message(self.load_output, "🔄 Loading PubMed embeddings from single file...")
                    pubmed_data = manager.load_embeddings_from_json(pubmed_single_file)
                    if pubmed_data:
                        if manager.add_embeddings_to_collection(pubmed_data, "pubmed"):
                            total_loaded += len(pubmed_data.get('embeddings', []))
                            self.log_message(self.load_output, f"✅ Loaded {len(pubmed_data.get('embeddings', []))} PubMed embeddings")
                
                else:
                    self.log_message(self.load_output, "⚠️ No PubMed embeddings found")
            
            if source in ["all", "preprints"]:
                # Load preprint embeddings (biorxiv, medrxiv)
                xrvix_path = "data/embeddings/xrvix_embeddings"
                if os.path.exists(xrvix_path):
                    self.log_message(self.load_output, "🔄 Loading preprint embeddings (BioRxiv + MedRxiv)...")
                    # Load only preprint sources
                    preprint_sources = ['biorxiv', 'medrxiv']
                    if manager.add_embeddings_from_directory(xrvix_path, sources=preprint_sources, db_batch_size=DB_BATCH_SIZE):
                        stats = manager.get_collection_stats()
                        total_docs = stats.get('total_documents', 0)
                        self.log_message(self.load_output, f"✅ Loaded preprint embeddings (total documents: {total_docs})")
                        total_loaded = total_docs
                else:
                    self.log_message(self.load_output, "⚠️ No preprint embeddings found")
            
            if source in ["all", "semantic"]:
                # Load Semantic Scholar API embeddings
                semantic_path = "data/embeddings/xrvix_embeddings/semantic_scholar"
                if os.path.exists(semantic_path):
                    self.log_message(self.load_output, "🔄 Loading Semantic Scholar API embeddings...")
                    if manager.add_embeddings_from_directory(semantic_path, db_batch_size=DB_BATCH_SIZE):
                        stats = manager.get_collection_stats()
                        total_docs = stats.get('total_documents', 0)
                        self.log_message(self.load_output, f"✅ Loaded Semantic Scholar API embeddings (total documents: {total_docs})")
                        total_loaded = total_docs
                else:
                    self.log_message(self.load_output, "⚠️ No Semantic Scholar embeddings found")
            
            if source == "all":
                # Load all embeddings from xrvix directory (auto-detects all sources)
                xrvix_path = "data/embeddings/xrvix_embeddings"
                if os.path.exists(xrvix_path):
                    self.log_message(self.load_output, "🔄 Loading all embeddings from xrvix directory (auto-detecting sources)...")
                    if manager.add_embeddings_from_directory(xrvix_path, db_batch_size=DB_BATCH_SIZE):
                        stats = manager.get_collection_stats()
                        total_docs = stats.get('total_documents', 0)
                        self.log_message(self.load_output, f"✅ Loaded all embeddings (total documents: {total_docs})")
                        total_loaded = total_docs
            
            if total_loaded == 0:
                self.log_message(self.load_output, "⚠️ No data was loaded. Make sure you have processed some data first.")
                return
            
            # Display final statistics
            stats = manager.get_collection_stats()
            self.log_message(self.load_output, f"\n📊 Vector Database Statistics:")
            self.log_message(self.load_output, f"   Total documents: {stats.get('total_documents', 0)}")
            self.log_message(self.load_output, f"   Collection name: {stats.get('collection_name', 'N/A')}")
            
            self.log_message(self.load_output, f"\n✅ Successfully loaded {total_loaded} embeddings into vector database!")
            
        except Exception as e:
            self.log_message(self.load_output, f"❌ Error loading data into vector database: {e}")
        finally:
            self.stop_progress()
            self.update_status("Ready")
            
    def show_chromadb_data_gui(self):
        """Show ChromaDB data."""
        thread = threading.Thread(target=self._show_chromadb_data_thread)
        thread.daemon = True
        thread.start()
        
    def _show_chromadb_data_thread(self):
        """Thread function for showing ChromaDB data."""
        try:
            self.log_message(self.show_output, "Current ChromaDB Data:")
            self.log_message(self.show_output, "="*60)
            
            manager = ChromaDBManager()
            collections = manager.list_collections()
            
            if not collections:
                self.log_message(self.show_output, "❌ No collections found in ChromaDB")
                self.log_message(self.show_output, "💡 Run Load Embeddings to load data into ChromaDB")
                return
            
            self.log_message(self.show_output, f"✅ ChromaDB is available with {len(collections)} collection(s)")
            
            # Check each collection
            for collection_name in collections:
                self.log_message(self.show_output, f"\n📚 Collection: {collection_name}")
                
                if not manager.switch_collection(collection_name):
                    self.log_message(self.show_output, "   ❌ Failed to access collection")
                    continue
                
                # Get statistics
                stats = manager.get_collection_stats()
                total_docs = stats.get('total_documents', 0)
                self.log_message(self.show_output, f"   📊 Total documents: {total_docs}")
                
                if total_docs == 0:
                    self.log_message(self.show_output, "   ⚠️ Collection is empty")
                else:
                    self.log_message(self.show_output, "   ✅ Collection has data - ready for searching!")
                    
                    # Show source breakdown
                    batch_stats = manager.get_batch_statistics()
                    if batch_stats and batch_stats.get('sources'):
                        self.log_message(self.show_output, "   📦 Source breakdown:")
                        for source, source_stats in batch_stats['sources'].items():
                            self.log_message(self.show_output, f"      {source}: {source_stats['total_documents']} documents")
            
            self.log_message(self.show_output, f"\n💡 ChromaDB uses persistent storage - data is saved locally")
            
        except Exception as e:
            self.log_message(self.show_output, f"❌ Error checking ChromaDB status: {e}")
            
    def refresh_chromadb_data(self):
        """Refresh ChromaDB data display."""
        self.show_output.delete(1.0, tk.END)
        self.show_chromadb_data_gui()
        
    def clear_chromadb_data_gui(self):
        """Clear ChromaDB data."""
        if not self.confirm_var.get():
            messagebox.showerror("Confirmation Required", "Please check the confirmation box first")
            return
            
        result = messagebox.askyesno("Confirm Deletion", 
                                   "Are you sure you want to delete all ChromaDB data?\nThis action cannot be undone!")
        if not result:
            return
            
        thread = threading.Thread(target=self._clear_chromadb_data_thread)
        thread.daemon = True
        thread.start()
        
    def _clear_chromadb_data_thread(self):
        """Thread function for clearing ChromaDB data."""
        self.start_progress()
        self.update_status("Clearing ChromaDB data...")
        
        try:
            self.log_message(self.clear_output, "Clearing ChromaDB Data:")
            self.log_message(self.clear_output, "="*60)
            
            manager = ChromaDBManager()
            
            # Initialize collection first to ensure it's accessible
            if manager.create_collection():
                if manager.clear_collection():
                    self.log_message(self.clear_output, "✅ ChromaDB collection cleared successfully!")
                else:
                    self.log_message(self.clear_output, "❌ Failed to clear ChromaDB collection")
            else:
                self.log_message(self.clear_output, "❌ Failed to initialize ChromaDB collection")
                
        except Exception as e:
            self.log_message(self.clear_output, f"❌ Error clearing ChromaDB: {e}")
        finally:
            self.stop_progress()
            self.update_status("Ready")
            
    # GUI Methods for Settings & Config
    
    def show_data_status_gui(self):
        """Show data status."""
        thread = threading.Thread(target=self._show_data_status_thread)
        thread.daemon = True
        thread.start()
        
    def _show_data_status_thread(self):
        """Thread function for showing data status."""
        try:
            self.log_message(self.data_status_output, "Data Status Overview:")
            self.log_message(self.data_status_output, "=" * 50)
            
            # Check PubMed data
            self.log_message(self.data_status_output, "\n📚 PubMed Data (Journal Articles):")
            pubmed_path = "data/embeddings/xrvix_embeddings/pubmed_embeddings.json"
            if os.path.exists(pubmed_path):
                try:
                    with open(pubmed_path, 'r') as f:
                        pubmed_data = json.load(f)
                        pubmed_count = len(pubmed_data.get('embeddings', []))
                        size = os.path.getsize(pubmed_path) / 1024
                        self.log_message(self.data_status_output, f"   ✅ Available: {pubmed_count} papers ({size:.1f} KB)")
                except Exception as e:
                    self.log_message(self.data_status_output, f"   ⚠️ Available but error reading: {e}")
            else:
                self.log_message(self.data_status_output, "   ❌ Not available")
            
            # Check Biorxiv data
            self.log_message(self.data_status_output, "\n📄 Biorxiv Data (Preprints):")
            biorxiv_path = "data/embeddings/xrvix_embeddings/biorxiv"
            if os.path.exists(biorxiv_path):
                try:
                    batch_files = [f for f in os.listdir(biorxiv_path) if f.startswith("batch_") and f.endswith(".json")]
                    self.log_message(self.data_status_output, f"   ✅ Available: {len(batch_files)} batch files")
                    if batch_files:
                        # Try to get total count from first batch
                        first_batch = os.path.join(biorxiv_path, batch_files[0])
                        with open(first_batch, 'r') as f:
                            batch_data = json.load(f)
                            sample_count = len(batch_data.get('embeddings', []))
                            estimated_total = sample_count * len(batch_files)
                            self.log_message(self.data_status_output, f"   📊 Estimated papers: ~{estimated_total}")
                except Exception as e:
                    self.log_message(self.data_status_output, f"   ⚠️ Available but error reading: {e}")
            else:
                self.log_message(self.data_status_output, "   ❌ Not available")
            
            # Check Medrxiv data
            self.log_message(self.data_status_output, "\n📄 Medrxiv Data (Preprints):")
            medrxiv_path = "data/embeddings/xrvix_embeddings/medrxiv"
            if os.path.exists(medrxiv_path):
                try:
                    batch_files = [f for f in os.listdir(medrxiv_path) if f.startswith("batch_") and f.endswith(".json")]
                    self.log_message(self.data_status_output, f"   ✅ Available: {len(batch_files)} batch files")
                    if batch_files:
                        # Try to get total count from first batch
                        first_batch = os.path.join(medrxiv_path, batch_files[0])
                        with open(first_batch, 'r') as f:
                            batch_data = json.load(f)
                            sample_count = len(batch_data.get('embeddings', []))
                            estimated_total = sample_count * len(batch_files)
                            self.log_message(self.data_status_output, f"   📊 Estimated papers: ~{estimated_total}")
                except Exception as e:
                    self.log_message(self.data_status_output, f"   ⚠️ Available but error reading: {e}")
            else:
                self.log_message(self.data_status_output, "   ❌ Not available")
            
            # Check Semantic Scholar API data
            self.log_message(self.data_status_output, "\n🔬 Semantic Scholar API Data:")
            semantic_path = "data/embeddings/xrvix_embeddings/semantic_scholar"
            if os.path.exists(semantic_path):
                try:
                    paper_files = [f for f in os.listdir(semantic_path) if f.endswith('.json') and f != 'metadata.json']
                    metadata_file = os.path.join(semantic_path, "metadata.json")
                    if os.path.exists(metadata_file):
                        with open(metadata_file, 'r') as f:
                            semantic_data = json.load(f)
                            semantic_count = semantic_data.get('total_papers', 0)
                            self.log_message(self.data_status_output, f"   ✅ Available: {semantic_count} papers")
                    else:
                        self.log_message(self.data_status_output, f"   ✅ Available: {len(paper_files)} individual paper files")
                except Exception as e:
                    self.log_message(self.data_status_output, f"   ⚠️ Available but error reading: {e}")
            else:
                self.log_message(self.data_status_output, "   ❌ Not available")
            
            # Check ChromaDB
            self.log_message(self.data_status_output, "\n🗄️ ChromaDB Vector Database:")
            if os.path.exists("data/vector_db/chroma_db"):
                try:
                    manager = ChromaDBManager()
                    collections = manager.list_collections()
                    if collections:
                        stats = manager.get_collection_stats()
                        total_docs = stats.get('total_documents', 0)
                        self.log_message(self.data_status_output, f"   ✅ Available: {total_docs} documents")
                    else:
                        self.log_message(self.data_status_output, "   ⚠️ Available but empty")
                except Exception as e:
                    self.log_message(self.data_status_output, f"   ⚠️ Available but error accessing: {e}")
            else:
                self.log_message(self.data_status_output, "   ❌ Not available")
            
            self.log_message(self.data_status_output, "\n💡 Recommendations:")
            self.log_message(self.data_status_output, "   • Use Paper Scraping options to collect missing data sources")
            self.log_message(self.data_status_output, "   • Use Load Embeddings to load data into ChromaDB")
            self.log_message(self.data_status_output, "   • Use Generate Hypotheses to generate hypotheses")
            
        except Exception as e:
            self.log_message(self.data_status_output, f"❌ Error checking data status: {e}")
            
    def refresh_data_status(self):
        """Refresh data status display."""
        self.data_status_output.delete(1.0, tk.END)
        self.show_data_status_gui()
        
    def show_configurations_gui(self):
        """Show configurations."""
        try:
            self.log_message(self.config_output, "Current Configurations:")
            self.log_message(self.config_output, "="*60)
            
            # Capture print_config_info output
            import io
            import sys
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            with redirect_stdout(f):
                print_config_info()
            config_info = f.getvalue()
            
            self.log_message(self.config_output, config_info)
            
        except Exception as e:
            self.log_message(self.config_output, f"❌ Error showing configurations: {e}")
            
    # GUI Methods for Hypothesis Generation
    
    def check_hypothesis_prerequisites(self):
        """Check prerequisites for hypothesis generation."""
        try:
            manager = ChromaDBManager()
            collections = manager.list_collections()
            
            if not collections:
                status_text = "❌ No ChromaDB collections found - run Load Embeddings first"
            else:
                # Ensure collection is properly initialized
                if not manager.create_collection():
                    status_text = "❌ Failed to initialize ChromaDB collection"
                else:
                    stats = manager.get_collection_stats()
                    if stats.get('total_documents', 0) == 0:
                        status_text = "❌ ChromaDB is empty - run Load Embeddings first"
                    else:
                        status_text = f"✅ ChromaDB has {stats.get('total_documents', 0)} documents - ready for hypothesis generation"
                    
            self.prereq_status_label.config(text=status_text)
        except Exception as e:
            self.prereq_status_label.config(text=f"❌ Error checking prerequisites: {e}")
            
    def generate_hypotheses_gui(self):
        """Generate hypotheses using the GUI input fields."""
        try:
            # Get input values
            prompt = self.hypothesis_prompt.get("1.0", tk.END).strip()
            lab_name = self.hypothesis_lab.get().strip()
            institution = self.hypothesis_institution.get().strip()
            hypotheses_per_meta_str = self.hypotheses_per_meta.get().strip()
            
            # Check if prompt is placeholder text
            placeholder_text = "Enter your research question or hypothesis prompt here...\nExample: How does UBR5 regulate cancer immunity through protein ubiquitination?"
            if prompt == placeholder_text:
                prompt = ""
            
            # Validate inputs
            if not prompt:
                self.log_message(self.hypothesis_output, "❌ Please enter a research prompt")
                return
            
            if not lab_name:
                self.log_message(self.hypothesis_output, "❌ Please enter a lab name")
                return
                
            if not institution:
                self.log_message(self.hypothesis_output, "❌ Please enter an institution")
                return
            
            # Validate hypotheses per meta-hypothesis
            try:
                hypotheses_per_meta = int(hypotheses_per_meta_str)
                if hypotheses_per_meta < 1 or hypotheses_per_meta > 10:
                    self.log_message(self.hypothesis_output, "❌ Hypotheses per meta-hypothesis must be between 1 and 10")
                    return
            except ValueError:
                self.log_message(self.hypothesis_output, "❌ Please enter a valid number for hypotheses per meta-hypothesis")
                return
            
            # Clear output and start progress
            self.hypothesis_output.delete("1.0", tk.END)
            self.log_progress("Starting hypothesis generation...", 0, start_animation=True)
            
            # Log input parameters
            self.log_message(self.hypothesis_output, "🧠 Generating Hypotheses with Parameters:")
            self.log_message(self.hypothesis_output, f"📝 Prompt: {prompt}")
            self.log_message(self.hypothesis_output, f"🔬 Lab: {lab_name}")
            self.log_message(self.hypothesis_output, f"🏛️ Institution: {institution}")
            self.log_message(self.hypothesis_output, f"📊 Hypotheses per Meta-Hypothesis: {hypotheses_per_meta}")
            self.log_message(self.hypothesis_output, "=" * 60)
            
            # Check ChromaDB prerequisites
            self.log_progress("Checking ChromaDB prerequisites...", 10)
            manager = ChromaDBManager()
            collections = manager.list_collections()
            
            if not collections:
                self.log_message(self.hypothesis_output, "❌ No ChromaDB collections found")
                self.log_message(self.hypothesis_output, "💡 Run Load Embeddings to load data into ChromaDB first")
                self.update_progress("Failed - No ChromaDB collections", stop_animation=True)
                return
            
            # Ensure collection is properly initialized
            self.log_progress("Initializing ChromaDB collection...", 20)
            if not manager.create_collection():
                self.log_message(self.hypothesis_output, "❌ Failed to initialize ChromaDB collection")
                self.update_progress("Failed - ChromaDB initialization error", stop_animation=True)
                return
            
            stats = manager.get_collection_stats()
            if stats.get('total_documents', 0) == 0:
                self.log_message(self.hypothesis_output, "❌ ChromaDB is empty")
                self.log_message(self.hypothesis_output, "💡 Run Load Embeddings to load data into ChromaDB first")
                self.update_progress("Failed - ChromaDB is empty", stop_animation=True)
                return
            
            self.log_message(self.hypothesis_output, f"✅ ChromaDB has {stats.get('total_documents', 0)} documents")
            self.log_progress("Initializing Enhanced RAG System...", 30)
            
            # Import and initialize the enhanced RAG system
            from src.ai.enhanced_rag_with_chromadb import EnhancedRAGQuery
            
            # Initialize the RAG system
            rag_system = EnhancedRAGQuery(use_chromadb=True, load_data_at_startup=True)
            
            self.log_message(self.hypothesis_output, "✅ RAG System initialized successfully")
            self.log_progress("Starting meta-hypothesis generation...", 40)
            
            # Generate hypotheses using the RAG system
            self.generate_hypotheses_with_rag(rag_system, prompt, lab_name, institution, hypotheses_per_meta)
            
        except Exception as e:
            self.log_message(self.hypothesis_output, f"❌ Failed to generate hypotheses: {e}")
            import traceback
            self.log_message(self.hypothesis_output, f"📋 Error details: {traceback.format_exc()}")
            self.update_progress("Failed - Error occurred", stop_animation=True)
    
    def generate_hypotheses_with_rag(self, rag_system, prompt, lab_name, institution, hypotheses_per_meta=3):
        """Generate hypotheses using the RAG system with custom parameters."""
        try:
            # Update lab configuration temporarily
            import json
            lab_config = {
                "lab_name": lab_name,
                "institution": institution,
                "research_focus": "UBR5, cancer immunology, protein ubiquitination, mechanistic and therapeutic hypotheses"
            }
            
            # Save temporary config
            with open("config/temp_lab_config.json", "w") as f:
                json.dump(lab_config, f)
            
            self.log_message(self.hypothesis_output, f"📋 Using lab configuration: {lab_name} at {institution}")
            
            # Generate hypotheses using the comprehensive session
            self.log_progress("Generating meta-hypotheses (Step 1/3)...", 50)
            self.log_message(self.hypothesis_output, "🧠 Generating hypotheses with comprehensive evaluation...")
            
            # Use the comprehensive hypothesis generation method
            self.log_progress("Processing meta-hypotheses (Step 2/3)...", 70)
            
            # Create progress callback for the RAG system
            def progress_callback(step_text, percentage=None):
                self.log_progress(step_text, percentage)
            
            accepted_hypotheses = rag_system.run_comprehensive_hypothesis_session(
                prompt, 
                max_hypotheses=3, 
                hypotheses_per_meta=hypotheses_per_meta,
                progress_callback=progress_callback
            )
            
            if accepted_hypotheses:
                self.log_progress("Formatting results (Step 3/3)...", 90)
                self.log_message(self.hypothesis_output, "✅ Hypotheses generated successfully!")
                self.log_message(self.hypothesis_output, "=" * 60)
                
                for i, hypothesis_record in enumerate(accepted_hypotheses, 1):
                    self.log_message(self.hypothesis_output, f"📋 HYPOTHESIS {i}:")
                    
                    # Extract hypothesis text from the record
                    if isinstance(hypothesis_record, dict):
                        hypothesis_text = hypothesis_record.get('hypothesis', str(hypothesis_record))
                    else:
                        hypothesis_text = str(hypothesis_record)
                    
                    self.log_message(self.hypothesis_output, hypothesis_text)
                    self.log_message(self.hypothesis_output, "=" * 60)
                
                # Complete
                self.log_progress("✅ Hypothesis generation completed successfully!", 100, stop_animation=True)
            else:
                self.log_message(self.hypothesis_output, "❌ No hypotheses were generated")
                self.update_progress("Failed - No hypotheses generated", stop_animation=True)
            
            # Clean up temporary config
            if os.path.exists("config/temp_lab_config.json"):
                os.remove("config/temp_lab_config.json")
                
        except Exception as e:
            self.log_message(self.hypothesis_output, f"❌ Error during hypothesis generation: {e}")
            import traceback
            self.log_message(self.hypothesis_output, f"📋 Error details: {traceback.format_exc()}")
            self.update_progress("Failed - Error during generation", stop_animation=True)
            
    def check_test_prerequisites(self):
        """Check prerequisites for test run."""
        try:
            manager = ChromaDBManager()
            collections = manager.list_collections()
            
            if not collections:
                status_text = "❌ No ChromaDB collections found - run Load Embeddings first"
            else:
                # Ensure collection is properly initialized
                if not manager.create_collection():
                    status_text = "❌ Failed to initialize ChromaDB collection"
                else:
                    stats = manager.get_collection_stats()
                    if stats.get('total_documents', 0) == 0:
                        status_text = "❌ ChromaDB is empty - run Load Embeddings first"
                    else:
                        status_text = f"✅ ChromaDB has {stats.get('total_documents', 0)} documents - ready for test run"
                    
            self.test_prereq_status_label.config(text=status_text)
        except Exception as e:
            self.test_prereq_status_label.config(text=f"❌ Error checking prerequisites: {e}")
            
    def test_run_gui(self):
        """Run test."""
        try:
            manager = ChromaDBManager()
            collections = manager.list_collections()
            
            if not collections:
                self.log_message(self.test_output, "❌ No ChromaDB collections found")
                self.log_message(self.test_output, "💡 Run Load Embeddings to load data into ChromaDB first")
                return
            
            # Ensure collection is properly initialized
            if not manager.create_collection():
                self.log_message(self.test_output, "❌ Failed to initialize ChromaDB collection")
                return
            
            stats = manager.get_collection_stats()
            if stats.get('total_documents', 0) == 0:
                self.log_message(self.test_output, "❌ ChromaDB is empty")
                self.log_message(self.test_output, "💡 Run Load Embeddings to load data into ChromaDB first")
                return
            
            test_query = self.test_query_var.get()
            self.log_message(self.test_output, f"✅ ChromaDB has {stats.get('total_documents', 0)} documents")
            self.log_message(self.test_output, f"🧪 Running test query: '{test_query}'")
            
            # Launch the enhanced RAG system with a test query
            self.log_message(self.test_output, "🚀 Launching Enhanced RAG System for test run...")
            subprocess.run([sys.executable, "enhanced_rag_with_chromadb.py"])
            
        except Exception as e:
            self.log_message(self.test_output, f"❌ Failed to run test: {e}")


def main():
    """Main function to run the GUI application."""
    root = tk.Tk()
    app = AIResearchProcessorGUI(root)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()


if __name__ == "__main__":
    main()
