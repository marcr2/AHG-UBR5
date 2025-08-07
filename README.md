# Project Title: Ubr5-HypeGen: An AI-Powered Hypothesis Generation Engine for Ubr5 Protein Research

## Project Overview

This project aims to accelerate scientific discovery by developing a specialized AI application capable of generating novel research hypotheses for the **Ubr5 protein**. The core of this project is a two-part system:

1.  **A robust academic literature scraping pipeline** that gathers a comprehensive corpus of research papers focused on the Ubr5 protein.
2.  **A proof-of-concept (PoC) hypothesis generation model** that leverages a carefully selected Large Language Model (LLM) to analyze the scraped literature and propose new, testable scientific hypotheses.

The final application will serve as a powerful tool for researchers, suggesting unexplored connections, potential functions, and novel experimental directions related to Ubr5, an E3 ubiquitin ligase implicated in various cellular processes and diseases. 

---

## Key Features

* **Automated Literature Acquisition:** Utilizes Python libraries (e.g., `paperscraper`) to systematically search and download relevant academic articles from various scientific databases.
* **Targeted Data Corpus:** Creates a specialized, up-to-date knowledge base exclusively focused on the Ubr5 protein.
* **Persistent Vector Database:** ChromaDB stores embeddings locally, eliminating the need to reload data every time you start the system.
* **LLM-Powered Analysis:** Integrates a state-of-the-art LLM to "read" and synthesize the collected scientific literature.
* **Hypothesis Generation:** The primary output is the generation of scientifically plausible hypotheses, complete with underlying citations and reasoning from the source texts.
* **Modular Design:** The scraper and the AI application are developed as distinct modules, allowing for independent testing, validation, and future expansion.

---

## Methodology

### Part 1: Scientific Data Acquisition 

The initial phase focuses on building a scalable and efficient web scraper. The primary candidate for this task is the `paperscraper` Python package, chosen for its ability to interface with scientific publishers and databases. The scraper will be configured to:
- Search for publications containing the keyword "Ubr5" and related terms.
- Extract full-text content, metadata (authors, publication date, journal), and citations.
- Preprocess and structure the scraped data into a clean, usable format (e.g., JSON or a structured database) ready for LLM ingestion.

### Part 2: AI-Driven Hypothesis Generation 

This phase involves designing and building the proof-of-concept AI application. The central challenge is selecting and integrating an appropriate LLM. The development process will follow these steps:

1.  **LLM Selection Analysis:** A critical evaluation of various LLMs will be conducted. This analysis will weigh the trade-offs between:
    * **Proprietary Models (e.g., GPT-4, Claude 3):** High performance, large context windows, but with API costs and data privacy considerations.
    * **Open-Source Models (e.g., Llama 3, Mixtral):** Greater control, fine-tuning capabilities, and local deployment options, but may require more setup and computational resources.
    * **Domain-Specific Models:** Exploration of models pre-trained on scientific or biomedical text.

2.  **Integration Workflow:** A Python-based application will be developed to:
    * Load the preprocessed Ubr5 literature corpus.
    * Implement a Retrieval-Augmented Generation (RAG) architecture. This allows the LLM to access the specific knowledge from our scraped articles when generating hypotheses, ensuring they are grounded in existing research.
    * Develop a prompting strategy to guide the LLM in generating insightful and scientifically valid hypotheses.

3.  **Proof of Concept (PoC):** The PoC will demonstrate the system's ability to take the Ubr5 corpus as input and output a set of coherent and novel hypotheses.

---

## Technologies

* **Language:** Python 3.x
* **Web Scraping/Data Acquisition:** `paperscraper`, `Beautiful Soup`, `requests`, `Scrapy` (as alternatives)
* **Data Handling:** `Pandas`, `JSON`
* **Vector Database:** `ChromaDB` (persistent local storage)
* **AI/LLM Integration:** `LangChain`, `Hugging Face Transformers`, `OpenAI API`, `Anthropic API`, `Google Gemini API`
* **Environment Management:** `Docker` (optional, for reproducibility)

---

## Persistent ChromaDB Storage

The system uses ChromaDB as a persistent vector database that stores embeddings locally in the `./chroma_db/` directory. This provides several key benefits:

* **No Data Reloading:** Once embeddings are loaded into ChromaDB, they persist between sessions
* **Fast Startup:** The Enhanced RAG System can start immediately without reloading large embedding files
* **Efficient Memory Usage:** Data is stored on disk and accessed as needed, rather than loaded entirely into memory
* **Scalability:** Can handle large datasets without memory constraints

### Usage Workflow

1. **First Time Setup:** Run `python master_processor.py` and choose option 4 to load data into ChromaDB
2. **Subsequent Sessions:** Start the Enhanced RAG System directly - it will use the persisted ChromaDB data
3. **Status Check:** Use option 8 in the master processor to check ChromaDB status and data availability

### Data Location

* ChromaDB data is stored in: `./chroma_db/`
* Embedding files are stored in: `./pubmed_embeddings.json` and `./xrvix_embeddings/`
* The system automatically detects if ChromaDB has data and skips reloading

## Enhanced Metadata Collection

The system now includes comprehensive metadata extraction capabilities that significantly enhance the research analysis experience:

### 🎯 Impact Factor Integration
* **Automatic impact factor extraction** from journal metadata
* **Comprehensive journal database** with 60+ high-impact journals
* **Smart estimation** for unknown journals using fuzzy matching
* **Preprint server handling** (bioRxiv, medRxiv, arXiv)

### 📋 Enhanced Metadata Fields
The system now captures these additional fields when available:
* `keywords` - Research keywords and subject tags
* `affiliations` - Author institutional affiliations  
* `funding` - Grant and funding source information
* `license` - Publication license details
* `categories` - Subject classifications
* `language` - Publication language
* `version` - Version info for preprints

### 🔧 Migration and Testing Tools
* **Migration script** for upgrading existing data: `enhanced_metadata_migration.py`
* **Comprehensive testing suite**: `test_enhanced_metadata.py`
* **Validation tools** for data quality assurance
* **Backup and rollback** capabilities for safe upgrades

### 📖 Documentation
See the [Enhanced Metadata Guide](docs/ENHANCED_METADATA_GUIDE.md) for detailed usage instructions, migration procedures, and troubleshooting tips.

## Project Goals & Future Directions

The primary goal is to create a functional PoC that validates this approach to AI-driven scientific discovery. **Recent enhancements** include comprehensive metadata collection and impact factor integration for richer research analysis.

### Completed Enhancements
* ✅ **Enhanced metadata collection** from paperscraper sources
* ✅ **Impact factor integration** with comprehensive journal database
* ✅ **Pipeline integration** ensuring complete metadata preservation
* ✅ **Migration tools** for upgrading existing datasets
* ✅ **Testing suite** for validation and quality assurance

### Future Directions
* Expanding the scraper to include a wider range of databases and pre-print servers
* Fine-tuning an open-source LLM on the Ubr5 corpus for enhanced domain-specific performance
* Developing a user interface (UI) for interacting with the hypothesis generation engine
* Evaluating the generated hypotheses with domain experts
* **Advanced metadata analytics** using the enhanced metadata fields
* **Citation network analysis** using comprehensive metadata
* **Research trend analysis** powered by impact factors and keywords