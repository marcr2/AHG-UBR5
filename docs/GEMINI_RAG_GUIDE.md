# Gemini RAG System - Complete Guide

## 🎉 Overview

Your project now has a comprehensive RAG (Retrieval-Augmented Generation) system that:

1. **Automatically loads embeddings into ChromaDB** after generation
2. **Uses Gemini 2.5 Flash LLM** for high-quality responses
3. **Retrieves relevant context** from your scientific literature
4. **Provides interactive querying** with UBR-5 focused prompts

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install google-generativeai
```

### 2. Set Up API Keys
Make sure your `keys.json` contains:
```json
{
  "GOOGLE_API_KEY": "your_google_embedding_api_key",
  "GEMINI_API_KEY": "your_gemini_api_key"
}
```

### 3. Generate Embeddings (with automatic ChromaDB loading)
```bash
# Process PubMed papers (automatically loads to ChromaDB)
python pubmed_scraper_json.py

# Process xrvix papers (automatically loads to ChromaDB)
python process_xrvix_dumps_json.py
```

### 4. Run the RAG System
```bash
python gemini_rag_system.py
```

---

## 🔍 Using the RAG System

### Interactive Commands

When you run `gemini_rag_system.py`, you can use these commands:

- **Ask questions**: `What is UBR-5?`
- **Filter by source**: `filter pubmed` or `filter xrvix`
- **Clear filter**: `clear filter`
- **View statistics**: `stats`
- **Exit**: `quit`

### Example Session

```
❓ Your question (or command): What is UBR-5?

🔍 Retrieving context for: 'What is UBR-5?'
📚 Retrieved 5 relevant chunks
🧠 Generating response with Gemini...

================================================================================
🤖 GEMINI RESPONSE:
================================================================================
UBR-5 (ubiquitin-protein ligase E3 component n-recognin 5) is an E3 ubiquitin 
ligase that plays crucial roles in protein degradation and cellular regulation...

================================================================================

📚 Show retrieved context? (y/n): n
```

---

## 🏗️ System Architecture

### Components

1. **Embedding Generation**: 
   - `pubmed_scraper_json.py` - PubMed papers
   - `process_xrvix_dumps_json.py` - BioRxiv/MedRxiv papers

2. **Vector Database**: 
   - `ChromaDBManager` - Persistent vector storage
   - Automatic loading after embedding generation

3. **RAG System**: 
   - `GeminiRAGSystem` - Main RAG orchestrator
   - Context retrieval + Gemini LLM generation

4. **LLM Integration**: 
   - `google-generativeai` - Official Gemini SDK
   - Tailored prompts for UBR-5 research

### Data Flow

```
Papers → Embeddings → ChromaDB → RAG Query → Gemini Response
```

---

## 🔧 Advanced Usage

### Custom Queries

The system automatically:
- Retrieves the most relevant chunks from your literature
- Builds context-aware prompts for Gemini
- Generates comprehensive, scientific responses

### Filtering

Filter by data source:
```
filter pubmed    # Only PubMed papers
filter xrvix     # Only BioRxiv/MedRxiv papers
clear filter     # Use all sources
```

### Context Management

- The system stores the last retrieved context
- You can view retrieved chunks after each query
- Context is automatically optimized for Gemini's input limits

---

## 📊 System Statistics

Use the `stats` command to see:
- Total chunks and embeddings
- Data sources and their sizes
- ChromaDB collection status
- Gemini LLM availability

---

## 🎯 UBR-5 Focus

The system is specifically tailored for Dr. Xiaojing Ma's lab research:

- **Specialized prompts** for UBR-5 and cancer immunology
- **Scientific accuracy** with literature citations
- **Research-focused responses** for hypothesis generation

---

## 🛠️ Troubleshooting

### Common Issues

1. **"No embeddings data available"**
   - Run embedding generation scripts first
   - Check that `xrvix_embeddings/pubmed_embeddings.json` or `xrvix_embeddings.json` exist

2. **"Gemini client not available"**
   - Check your `GEMINI_API_KEY` in `keys.json`
   - Ensure `google-generativeai` is installed

3. **"ChromaDB initialization failed"**
   - Check disk space and permissions
   - The system will fall back to traditional search

### Performance Tips

- **First run**: May take time to load embeddings into ChromaDB
- **Subsequent runs**: Fast startup with persistent ChromaDB
- **Large datasets**: Consider filtering by source for faster queries

---

## 🔄 Integration with Existing Tools

Your existing tools still work:
- `enhanced_rag_with_chromadb.py` - Original RAG system
- `master_processor.py` - Batch processing
- `hypothesis_tools.py` - Hypothesis generation/critique

The new `gemini_rag_system.py` provides the most advanced experience with Gemini LLM integration.

---

## 🎉 Success!

You now have a complete, production-ready RAG system that:
- ✅ Automatically manages your knowledge base
- ✅ Provides high-quality, context-aware responses
- ✅ Focuses on UBR-5 and Dr. Ma's research
- ✅ Scales with your data
- ✅ Integrates cutting-edge LLM technology

**Happy researching! 🧬🔬** 