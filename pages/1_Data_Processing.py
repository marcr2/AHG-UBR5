import streamlit as st
import importlib
from tqdm.auto import tqdm
import sys

st.set_page_config(page_title="Data Scraping and Processing", layout="centered")
st.title("📄 Data Scraping and Processing")

# Dynamically import DUMPS from processing_config
processing_config = importlib.import_module("processing_config")
DUMPS = getattr(processing_config, "DUMPS", ["biorxiv", "medrxiv"])

# User can select sources to process
selected_dumps = st.multiselect("Select sources to process:", DUMPS, default=DUMPS)

# Optionally allow file upload (not implemented: just UI placeholder)
st.file_uploader("Upload a new dump file (not yet implemented)", type=["jsonl", "json"])

progress_placeholder = st.empty()
status_placeholder = st.empty()
stats_placeholder = st.empty()

def dual_progress_update(val, st_progress=None, tqdm_bar=None):
    if st_progress is not None:
        st_progress.progress(val)
    if tqdm_bar is not None:
        tqdm_bar.n = int(val * 100)
        tqdm_bar.refresh()
        if val >= 1.0:
            tqdm_bar.close()

if st.button("Start Processing"):
    import process_xrvix_dumps_json as pxj
    tqdm_bar = tqdm(total=100, desc="Processing Data", file=sys.stdout)
    with st.spinner("Processing data with adaptive parallel system..."):
        pxj.streamlit_process_dumps(selected_dumps, progress_placeholder, status_placeholder, stats_placeholder, tqdm_bar)
    st.success("Data processing complete!") 