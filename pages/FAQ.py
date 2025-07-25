import streamlit as st

st.set_page_config(page_title="FAQ", layout="centered")
st.title("❓ Frequently Asked Questions (FAQ)")

st.header("How to Use This Tool")

st.markdown("""
1. **Data Processing**
   - Go to the Data Processing page to process and embed your data (PubMed, xrvix, etc.).
   - You can select which sources to process and monitor progress with a live progress bar.

2. **RAG Search & Hypothesis Generation**
   - Use the RAG & Hypothesis Generator page to search your embedded data.
   - Select sources (PubMed, xrvix, or both), enter a search query, and specify how many top results to retrieve.
   - Add relevant chunks to your package for hypothesis generation.
   - Generate and critique hypotheses using the real Gemini LLM pipeline.

3. **Package Management**
   - You can clear your package at any time to start a new search or hypothesis session.

4. **Console Debugging**
   - For advanced users, the console shows tqdm progress bars and high-level logs for debugging.

---

**Common Slowdowns & Solutions**

- **Large Dataset Loading**
  - Loading all embeddings (especially for xrvix or PubMed) can take several minutes for large datasets.
  - Progress bars in the UI and console show loading status. Please be patient during this step.
  - For very large datasets, consider processing only the sources you need.

- **Searching Large Embedding Collections**
  - Searching across many embeddings can be slow. Use source filters and limit the number of top results (Top K) to speed up queries.

- **Hypothesis Generation API Rate Limits**
  - The Gemini LLM API has rate limits. If you hit these, you may see slowdowns or errors. Try again later or reduce the number of hypotheses generated at once.

- **Browser Freezing or Crashing**
  - If you select too many sources or try to load/process very large datasets, your browser may become unresponsive. Try reducing the scope of your query or processing in smaller batches.

- **Console/Terminal Debugging**
  - The console shows high-level process starts and tqdm progress bars for batch loading and searching. Use these for troubleshooting if the UI seems stuck.

---

**If you encounter persistent issues:**
- Check the console for errors or warnings.
- Try restarting the app and processing fewer sources at a time.
- Contact the developer for further support.
""") 