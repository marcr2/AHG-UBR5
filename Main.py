import streamlit as st

st.set_page_config(page_title="Hypothesis Generation Dashboard", layout="centered")
st.title("🏠 Hypothesis Generation Dashboard")

st.markdown("""
Welcome to the Hypothesis Generation Dashboard!

- Use the sidebar to navigate between pages:
    - **Data Processing**: Scrape and process new data sources.
    - **RAG & Hypothesis Generator**: Search, package, and generate hypotheses.
    - **Discussion Chat**: View the interactive dialogue between the generator and critic.
    - **FAQ**: Frequently asked questions.

---
""")

# Optionally show high-level status if available
progress = st.session_state.get('progress')
goal = st.session_state.get('goal')
if progress is not None and goal is not None:
    st.progress(progress / goal, text=f"{progress} / {goal} hypotheses generated")

latest_avg = st.session_state.get('average_score')
if latest_avg is not None:
    st.metric("Average Score of Latest Batch", latest_avg) 