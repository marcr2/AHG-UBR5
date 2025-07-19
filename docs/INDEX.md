# AHG-UBR5 Documentation Index

Welcome to the AHG-UBR5 project documentation. This directory contains all guides, tutorials, and technical documentation.

## 📚 Documentation Guide

### Getting Started
- **[README.md](../README.md)** - Main project overview and quick start guide

### Core System Guides
- **[GEMINI_RAG_GUIDE.md](GEMINI_RAG_GUIDE.md)** - Complete guide to the Enhanced RAG System with Gemini
- **[CHROMADB_USAGE_GUIDE.md](CHROMADB_USAGE_GUIDE.md)** - ChromaDB integration and usage guide
- **[PERFORMANCE_OPTIMIZATION_GUIDE.md](PERFORMANCE_OPTIMIZATION_GUIDE.md)** - Performance tuning and optimization

### Troubleshooting & Support
- **[API_RATE_LIMIT_GUIDE.md](API_RATE_LIMIT_GUIDE.md)** - Handling Gemini API rate limits and quotas
- **[RATE_LIMIT_UPDATE.md](RATE_LIMIT_UPDATE.md)** - Latest rate limit configuration updates
- **[PARALLEL_PROCESSING_FIX.md](PARALLEL_PROCESSING_FIX.md)** - Solutions for parallel processing issues

### Development & Technical
- **[instructions.md](instructions.md)** - Development instructions and project setup
- **[tests/README.md](../tests/README.md)** - Testing guide and test runner documentation

## 🚀 Quick Navigation

### For New Users
1. Start with [README.md](../README.md)
2. Read [GEMINI_RAG_GUIDE.md](GEMINI_RAG_GUIDE.md) for system overview
3. Check [API_RATE_LIMIT_GUIDE.md](API_RATE_LIMIT_GUIDE.md) if you encounter API issues

### For Developers
1. Review [instructions.md](instructions.md) for setup
2. Check [PERFORMANCE_OPTIMIZATION_GUIDE.md](PERFORMANCE_OPTIMIZATION_GUIDE.md) for optimization
3. Use [tests/README.md](../tests/README.md) for testing

### For Troubleshooting
1. Check [API_RATE_LIMIT_GUIDE.md](API_RATE_LIMIT_GUIDE.md) for API issues
2. Review [PARALLEL_PROCESSING_FIX.md](PARALLEL_PROCESSING_FIX.md) for processing problems
3. Consult [CHROMADB_USAGE_GUIDE.md](CHROMADB_USAGE_GUIDE.md) for database issues

## 📁 Project Structure

```
AHG-UBR5/
├── README.md                    # Main project overview
├── docs/                        # Documentation directory
│   ├── INDEX.md                 # This file
│   ├── GEMINI_RAG_GUIDE.md      # RAG system guide
│   ├── CHROMADB_USAGE_GUIDE.md  # ChromaDB guide
│   ├── API_RATE_LIMIT_GUIDE.md  # API troubleshooting
│   ├── PERFORMANCE_OPTIMIZATION_GUIDE.md
│   ├── PARALLEL_PROCESSING_FIX.md
│   └── instructions.md          # Development setup
├── tests/                       # Test files
│   ├── README.md               # Test documentation
│   ├── run_tests.py            # Test runner
│   └── test_*.py               # Individual test files
├── *.py                        # Main Python modules
├── keys.json                   # API keys (gitignored)
├── AHG/                        # Virtual environment
├── chroma_db/                  # ChromaDB data (gitignored)
└── xrvix_embeddings/           # Embeddings data (gitignored)
```

## 🔧 Maintenance

This documentation is maintained alongside the codebase. When making changes:

1. Update relevant documentation files
2. Keep this index current
3. Ensure all links work correctly
4. Test documentation examples

## 📝 Contributing

When adding new documentation:

1. Create the new `.md` file in the `docs/` directory
2. Update this `INDEX.md` file
3. Ensure proper formatting and links
4. Test any code examples

---

*Last updated: July 2025* 