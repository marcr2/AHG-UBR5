# Tests Directory

This directory contains all test files for the AHG-UBR5 project.

## Test Files

- `test_chromadb.py` - Tests for ChromaDB functionality
- `test_embedding.py` - Tests for embedding generation
- `test_loading_performance.py` - Performance tests for data loading
- `test_xrvix_fix.py` - Tests for xrvix data processing fixes
- `test_xrvix_fix_simple.py` - Simplified xrvix tests

## Running Tests

### Run All Tests
```bash
# From project root
python tests/run_tests.py

# Or from tests directory
cd tests
python run_tests.py
```

### Run Specific Test
```bash
# Run a specific test file
python tests/run_tests.py test_chromadb

# Or run directly
python tests/test_chromadb.py
```

### Using Virtual Environment
```bash
# Use the project's virtual environment
AHG\Scripts\python.exe tests/run_tests.py
```

## Test Structure

Each test file should:
1. Import the modules it's testing
2. Define test functions or classes
3. Include a `main()` function for standalone execution
4. Handle errors gracefully

## Adding New Tests

1. Create a new file with `test_` prefix
2. Import the modules you want to test
3. Add test functions
4. Include a `main()` function
5. Update this README if needed

## Example Test Structure

```python
#!/usr/bin/env python3
"""
Test module for [module_name].
"""

def test_function():
    """Test a specific function."""
    # Test implementation
    pass

def main():
    """Run all tests in this module."""
    # Run tests
    pass

if __name__ == "__main__":
    main()
``` 