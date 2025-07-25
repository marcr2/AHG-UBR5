import streamlit as st
import importlib
import time
import threading
from typing import Any
from tqdm.auto import tqdm
import sys
import logging
import traceback
import io
import os
import glob

st.set_page_config(page_title="RAG & Hypothesis Generator", layout="centered")
st.title("🔬 RAG Search & Hypothesis Generator")

# --- Global suppression of ScriptRunContext warning ---
class SuppressScriptRunContextWarning(io.StringIO):
    def write(self, s):
        if "missing ScriptRunContext! This warning can be ignored when running in bare mode." not in s:
            super().write(s)
sys.stderr = SuppressScriptRunContextWarning()

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("embedding_loader")

class ScriptRunContextFilter(logging.Filter):
    def filter(self, record):
        return "missing ScriptRunContext! This warning can be ignored when running in bare mode." not in record.getMessage()

logging.getLogger().addFilter(ScriptRunContextFilter())

def dual_progress_update(val, tqdm_bar=None):
    if tqdm_bar is not None:
        tqdm_bar.n = int(val * 100)
        tqdm_bar.refresh()
        if val >= 1.0:
            tqdm_bar.close()

# Helper for rerun (compatible with all Streamlit versions)
def rerun():
    if hasattr(st, 'rerun'):
        st.rerun()
    else:
        st.experimental_rerun()

# Import EnhancedRAGQuery
try:
    rag_mod = importlib.import_module("enhanced_rag_with_chromadb")
    EnhancedRAGQuery = getattr(rag_mod, "EnhancedRAGQuery")
except Exception as e:
    logger.error(f"Failed to import EnhancedRAGQuery: {e}")
    st.error(f"Failed to import EnhancedRAGQuery: {e}")
    st.stop()

# --- Caching for embeddings/model loading ---
@st.cache_resource(show_spinner=False)
def get_cached_embeddings(_progress_callback=None) -> Any:
    logger.info("Calling EnhancedRAGQuery for embedding loading...")
    rag = EnhancedRAGQuery(use_chromadb=True, load_data_at_startup=False)
    logger.info("Calling load_all_embeddings...")
    embeddings_data = rag.load_all_embeddings(progress_callback=_progress_callback)
    logger.info("Embedding loading complete.")
    return embeddings_data

# --- Background loading state ---
if "embeddings_loaded" not in st.session_state:
    st.session_state["embeddings_loaded"] = False
if "embeddings_loading" not in st.session_state:
    st.session_state["embeddings_loading"] = False
if "embeddings_progress" not in st.session_state:
    st.session_state["embeddings_progress"] = 0.0
if "embeddings_error" not in st.session_state:
    st.session_state["embeddings_error"] = None

# --- Persistent placeholder for loading UI ---
progress_bar_placeholder = st.empty()

# --- Background loading function ---
def load_embeddings_bg():
    try:
        logger.info("Starting embedding loading background thread...")
        # --- Count total number of batches for tqdm ---
        batch_count = 0
        embeddings_dir = "xrvix_embeddings"
        if os.path.exists(embeddings_dir):
            for source_dir in os.listdir(embeddings_dir):
                source_path = os.path.join(embeddings_dir, source_dir)
                if os.path.isdir(source_path):
                    batch_files = glob.glob(os.path.join(source_path, "batch_*.json"))
                    batch_count += len(batch_files)
        if batch_count == 0:
            batch_count = 100  # fallback to 100 if nothing found
        tqdm_bar = tqdm(total=batch_count, desc="Loading embeddings", file=sys.stdout)
        loaded_batches = [0]
        def update_progress(val):
            st.session_state["embeddings_progress"] = val
            # Update tqdm by incrementing by 1 for each batch (if val increases)
            if loaded_batches[0] < int(val * batch_count):
                tqdm_bar.update(int(val * batch_count) - loaded_batches[0])
                loaded_batches[0] = int(val * batch_count)
            if val >= 1.0:
                tqdm_bar.close()
        logger.info("Calling get_cached_embeddings...")
        embeddings_data = get_cached_embeddings(_progress_callback=update_progress)
        logger.info("Returned from get_cached_embeddings.")
        rag = EnhancedRAGQuery(use_chromadb=True, load_data_at_startup=False)
        rag.embeddings_data = embeddings_data
        st.session_state["rag"] = rag
        st.session_state["embeddings_loaded"] = True
        st.session_state["embeddings_loading"] = False
        st.session_state["embeddings_error"] = None
        logger.info("Embeddings loaded and session state updated.")
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Exception in embedding loading thread: {e}\n{tb}")
        print(f"Exception in embedding loading thread: {e}\n{tb}")
        st.session_state["embeddings_error"] = f"{e}\n{tb}"
        st.session_state["embeddings_loading"] = False

# --- Start background loading if needed ---
if ("rag" not in st.session_state or not st.session_state["embeddings_loaded"]) and not st.session_state["embeddings_loading"]:
    st.session_state["embeddings_loading"] = True
    st.session_state["embeddings_progress"] = 0.0
    st.session_state["embeddings_error"] = None
    threading.Thread(target=load_embeddings_bg, daemon=True).start()
    # Show spinner and progress bar immediately, and poll progress from main thread
    with st.spinner("Loading all embeddings in the background. The UI will be responsive, but you must wait for loading to finish before using the tool."):
        while not st.session_state["embeddings_loaded"] and not st.session_state["embeddings_error"]:
            progress_bar_placeholder.progress(st.session_state["embeddings_progress"])
            time.sleep(0.1)
        if st.session_state["embeddings_error"]:
            progress_bar_placeholder.empty()
            st.error(f"Failed to load embeddings: {st.session_state['embeddings_error']}")
    # Do not rerun here; let the UI update naturally

# --- Polling/loading UI ---
if not st.session_state["embeddings_loaded"]:
    if st.session_state["embeddings_error"]:
        progress_bar_placeholder.empty()
        st.error(f"Failed to load embeddings: {st.session_state['embeddings_error']}" )
    else:
        with st.spinner("Loading all embeddings in the background. The UI will be responsive, but you must wait for loading to finish before using the tool."):
            while not st.session_state["embeddings_loaded"] and not st.session_state["embeddings_error"]:
                progress_bar_placeholder.progress(st.session_state["embeddings_progress"])
                time.sleep(0.1)
            if st.session_state["embeddings_error"]:
                progress_bar_placeholder.empty()
                st.error(f"Failed to load embeddings: {st.session_state['embeddings_error']}")
        # Only rerun if progress has changed significantly or loading is done
        if st.session_state["embeddings_progress"] >= 1.0 or st.session_state["embeddings_loaded"]:
            rerun()
    st.stop()
else:
    progress_bar_placeholder.empty()

# --- Main UI (only after embeddings are loaded) ---
rag = st.session_state["rag"]
try:
    sources = list(rag.embeddings_data["sources"].keys())
except Exception as e:
    st.error(f"Error loading sources: {e}")
    sources = []

selected_sources = st.multiselect("Select sources to search:", sources, default=sources) if sources else []

# --- Search and add to package ---
st.write("### Search and Add to Package (Real Data)")
search_query = st.text_input("Enter search query:")
st.caption("Up to 5000 most relevant chunks will be loaded for each search.")

search_progress_bar = st.empty()
search_details = st.empty()
search_results = None
if st.button("Search and Add to Package"):
    try:
        tqdm_bar = tqdm(total=100, desc="Searching", file=sys.stdout)
        if not selected_sources:
            st.warning("Please select at least one source.")
        else:
            search_start = time.time()
            def update_search_progress(val):
                elapsed = time.time() - search_start
                dual_progress_update(val, tqdm_bar=tqdm_bar)
                eta = (elapsed / val) * (1-val) if val > 0 else 0
                chunks_per_s = int((val * 5000) / elapsed) if elapsed > 0 else 0
                search_details.markdown(f"**Searching...** {val*100:.1f}% | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s | Chunks/s: {chunks_per_s}")
            search_progress_bar.progress(0.0)
            search_details.markdown("**Searching...** 0% | Elapsed: 0s | ETA: -- | Chunks/s: --")
            filter_dict = {"source_file": selected_sources[0]} if len(selected_sources) == 1 else {"source_file": {"$in": selected_sources}}
            search_results = rag.search_hybrid(search_query, top_k=5000, filter_dict=filter_dict, progress_callback=update_search_progress)
            search_progress_bar.empty()
            search_details.empty()
            if search_results:
                rag.add_to_package(search_results)
                st.success(f"Added {len(search_results)} chunks to package.")
            else:
                st.warning("No results found.")
    except Exception as e:
        st.error(f"Search failed: {e}")
        search_progress_bar.empty()
        search_details.empty()

# Show current package contents
try:
    if rag.current_package["chunks"]:
        st.write("#### Current Package Chunks:")
        st.table([
            {"Title": meta.get("title", "No title")[:60], "Source": meta.get("source_file", "?")}
            for meta in rag.current_package["metadata"]
        ])
        if st.button("Clear Package"):
            rag.clear_package()
            rerun()
except Exception as e:
    st.error(f"Error displaying package: {e}")

# --- Hypothesis Generation ---
st.write("### Generate and Critique Hypotheses (Real Pipeline)")
num_hypotheses = st.number_input("How many hypotheses to generate?", min_value=1, max_value=10, value=3, step=1)

if st.button(f"Generate {num_hypotheses} Hypotheses (Real)"):
    try:
        with st.spinner("Generating and critiquing hypotheses using Gemini and real data..."):
            results = rag.generate_hypotheses_from_package(n=num_hypotheses)
        if results:
            st.success(f"Generated {len(results)} hypotheses.")
            for i, res in enumerate(results, 1):
                st.markdown(f"**{i}. {res['hypothesis']}**")
                st.markdown(f"- Score: {res['score']:.1f}")
                st.markdown(f"- Novelty: {res['critique'].get('novelty','N/A')}")
                st.markdown(f"- Accuracy: {res['critique'].get('accuracy','N/A')}")
                st.markdown(f"- Verdict: {res['critique'].get('verdict','N/A')}")
                st.markdown(f"- Critique: {res['critique'].get('critique','N/A')}")
                st.markdown("---")
        else:
            st.warning("No hypotheses generated. Check package size or API limits.")
    except Exception as e:
        st.error(f"Hypothesis generation failed: {e}") 