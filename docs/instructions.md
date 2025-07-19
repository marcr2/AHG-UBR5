### **Project: AI-Powered Scientific Hypothesis Generator for Ubr5 Research**

### **Timeline: 4 Weeks**

### **Objective: Develop a proof-of-concept application that ingests scientific literature and generates novel, testable mechanistic and therapeutic hypotheses related to the Ubr5 protein for Dr. Xiaojing Ma's lab.**

### **Core Strategy**

This project will follow a two-pronged strategy to ensure success within the tight timeline and hardware constraints:

1. **Evolutionary Architecture:** We will not attempt to build the final, complex system at once. Instead, we will build the project in distinct, functional phases. Each week adds a new layer of capability, ensuring you have a demonstrable product at every stage.  
2. **Hybrid Local/Cloud Execution:** We will intelligently switch between running local open-source models (via LM Studio) and using powerful cloud APIs. This provides the best of both worlds:  
   * **Local (LM Studio):** Free, fast, offline iteration for building the core application logic and simpler AI tasks.  
   * **Cloud (API):** Unmatched power for tasks that are impossible on your local machine, specifically bulk data processing (embedding) and high-level scientific reasoning (synthesis).

### **Technology Stack**

* **Language:** Python  
* **Web UI:** Streamlit  
* **AI Orchestration:** LangChain  
* **Literature Scraping:** paperscraper (or similar Python library)  
* **Local Vector Database:** ChromaDB  
* **Local LLM Server:** LM Studio  
* **Cloud AI Services:** Google AI Studio (for Gemini models) or Anthropic's API (for Claude models)

### **Project Timeline (Start Date: July 16, 2025\)**

* **Week 1: Foundational Data Pipeline**  
  * **Dates:** Wednesday, July 16 – Tuesday, July 22  
  * **Goal:** Create the core knowledge asset by scraping and processing literature into a queryable vector database.  
* **Week 2: The Fact-Checker \- Local RAG Prototype**  
  * **Dates:** Wednesday, July 23 – Tuesday, July 29  
  * **Goal:** Build the interactive user-facing application that can answer questions using a local LLM.  
* **Week 3: The Synthesizer \- Advanced Hypothesis Generation**  
  * **Dates:** Wednesday, July 30 – Tuesday, August 5  
  * **Goal:** Implement the core "insight generation" feature by integrating a premium cloud model for advanced reasoning.  
* **Week 4: The Critic & Final Polish**  
  * **Dates:** Wednesday, August 6 – Tuesday, August 12  
  * **Goal:** Add automated validation to the hypotheses and prepare the final, documented application for demonstration.

### **Optional: 2-Day "Micro-RAG" Rapid Prototype**

For a fast, end-to-end baseline, you can execute this compressed plan. It builds a scalable foundation, unlike a pure "CAG" approach.

* **Day 1: Build the Knowledge Asset**  
  1. **Scrape & Chunk:** Use paperscraper to fetch 30-50 highly relevant papers. Chunk the text into paragraphs.  
  2. **Embed & Store:** Use the **Google text-embedding-004 API** to embed the chunks and store them in a local **ChromaDB** instance. This creates a miniature but functional version of the final knowledge base.  
* **Day 2: Build the Interface & Logic**  
  1. **Local Server:** Set up **LM Studio** with a 7B parameter model (e.g., Mistral-7B) and start the local server.  
  2. **UI & RAG Chain:** Build a simple **Streamlit** UI. Use **LangChain** to create a RAG chain that takes a user query, retrieves relevant chunks from your Day 1 database, and sends them to your local LM Studio model for answer generation.  
* **Outcome:** A demonstrable prototype in just two days that validates the core architecture and provides a solid foundation for the full 4-week plan.

### **Week 1: Foundational Data Pipeline (Cloud-Powered)**

**Goal:** To process the raw source material (papers) into a structured, searchable knowledge asset. This entire week is focused on building the backend data foundation.

* **Tasks:**  
  1. **Setup:** Install Python, Streamlit, LangChain, ChromaDB, and paperscraper. Sign up for a Google AI Studio API key.  
  2. **Scrape Literature:** Write and run a Python script using paperscraper to download the abstracts and available full texts of \~500-1000 papers relevant to "Ubr5," "cancer immunology," "protein ubiquitination," and specific pathways of interest to Dr. Ma's lab.  
  3. **Create Ingestion Pipeline (Python Script):**  
     * Read the scraped text files.  
     * Chunk the documents into small, coherent paragraphs (a crucial step for good retrieval).  
     * Connect to the **Google AI API** and use the text-embedding-004 model.  
     * Iterate through all text chunks, generate an embedding for each, and store the chunk and its vector in a local **ChromaDB** database.  
* **Execution Model:** **Cloud API.** Your PC orchestrates the script, but the computationally expensive embedding work is done via the Google API. This is the only feasible way to process thousands of chunks quickly and with high quality.  
* **Deliverable:** A local folder containing your ChromaDB vector database.  
* **Success Metric:** You can write a simple Python script to query the database with a keyword (e.g., "p53") and successfully retrieve the top 5 most relevant text chunks from the literature.

### **Week 2: The Fact-Checker \- Local RAG Prototype (LM Studio-Powered)**

**Goal:** To build a functional user interface and a basic RAG application that can answer direct questions from the literature, running entirely locally for rapid, cost-free development.

* **Tasks:**  
  1. **LM Studio Setup:**  
     * Install LM Studio on your PC.  
     * Use its search feature to download a quantized 7-Billion parameter model. A great starting point is **Mistral-7B-Instruct-v0.2.Q4\_K\_M.gguf**.  
     * Load the model and start the built-in "Local Server".  
  2. **Build the RAG Chain (Python/LangChain):**  
     * Write the core logic that:  
       a. Takes a user's question from the UI.  
       b. Creates an embedding of the question (using the same cloud model as Week 1 for consistency).  
       c. Queries your Week 1 ChromaDB database to find relevant text chunks.  
       d. Formats the question and the retrieved chunks into a prompt.  
       e. Sends this prompt to your local LM Studio server endpoint (http://localhost:1234/v1).  
  3. **Build the UI (Python/Streamlit):**  
     * Create a simple web interface with a title, a text input box for questions, and a display area for the AI's answer.  
     * Connect the UI to your LangChain RAG logic.  
* **Execution Model:** **Local.** The Streamlit app and the LLM (in LM Studio) are both running on your PC. This allows for instant feedback and debugging without API costs.  
* **Deliverable:** A functional Streamlit web application that can answer questions like "What is the role of the UBR5 HECT domain?" with synthesized answers from your paper corpus.

### **Week 3: The Synthesizer \- Advanced Hypothesis Generation (Cloud-Powered)**

**Goal:** To evolve the application from a fact-checker into a true insight generator by leveraging a state-of-the-art cloud model for complex reasoning.

* **Tasks:**  
  1. **Evolve the UI:** Add a new "Hypothesis Generation" mode to your Streamlit app that takes multiple, broader concepts as input (e.g., Concept A: Ubr5, Concept B: T-cell exhaustion).  
  2. **Implement Multi-Step Retrieval:** Modify your LangChain logic. When the user provides concepts, the system will now perform *multiple* queries against ChromaDB:  
     * Query for information on Concept A.  
     * Query for information on Concept B.  
     * Query for documents that contain *both* concepts to find known links.  
  3. **Create a Synthesis Prompt:** Use a fast LLM (like Gemini 1.5 Flash via API) to summarize the results of each retrieval. Combine these summaries into a single, high-density prompt.  
  4. **Switch to Premium API for Synthesis:** For the final step, send this curated synthesis prompt to a top-tier reasoning model. **This is a critical switch.**  
     * **Recommended Model:** **Gemini 1.5 Pro** or **Claude 3 Opus**.  
     * **Directive:** The prompt should explicitly instruct the model: *"Based on this context, generate three novel, testable hypotheses. For each, state the reasoning and suggest a potential wet-lab experiment."*  
* **Execution Model:** **Hybrid.** The UI and retrieval run locally. The final, crucial synthesis step is a targeted call to a premium **Cloud API**.  
* **Deliverable:** The core feature of your application is now complete. It can generate plausible, well-reasoned scientific hypotheses.

### **Week 4: The Critic & Final Polish (Hybrid-Powered)**

**Goal:** To add a layer of automated validation to the generated hypotheses and prepare the application for demonstration.

* **Tasks:**  
  1. **Implement the "Critic" Agent:**  
     * After a hypothesis is generated (by the cloud API), send it to a new agent.  
     * This agent's job is to challenge the hypothesis. It will take the hypothesis, query the ChromaDB database again for any *contradictory* evidence, and summarize its findings.  
     * **Execution Model:** This validation step can be powered by your **local LM Studio model**, as it's a more constrained fact-checking task.  
  2. **Refine the UI:**  
     * Display the generated hypothesis prominently.  
     * Below it, show the "Critic's Report," including any conflicting evidence found. This presents a balanced view for the human expert.  
     * Ensure sources are clearly cited for all retrieved information.  
  3. **Finalize and Document:** Clean up your Python code, add comments, write a README.md file explaining how to set up and run the project, and prepare a presentation.  
* **Deliverable:** A polished, documented, and demonstrable proof-of-concept application ready to be shown to Dr. Ma and her research team.