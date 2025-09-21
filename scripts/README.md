# Scripts Directory

This directory contains command-line tools and utilities for the AI Research Processor.

## Available Scripts

### `analyze_citations.py`

Command-line tool for analyzing hypothesis citation mappings.

**Usage:**
```bash
# From root directory
python scripts/analyze_citations.py

# From scripts directory
python analyze_citations.py

# With specific export directory
python scripts/analyze_citations.py hypothesis_export/Hypothesis_Export_09202025-2043
```

**Features:**
- Automatically finds the latest hypothesis export
- Creates summary tables of citations across hypotheses
- Exports citation data to CSV
- Shows citation statistics and examples
- Works with the enhanced citation system

**Requirements:**
- Hypothesis export directory with citation mappings
- Python packages: pandas, json, os, glob

## Adding New Scripts

When adding new command-line tools:

1. Place the script in this `scripts/` directory
2. Update this README with usage information
3. Consider creating a wrapper script in the root directory if needed
4. Update the main README.md with references to new tools
