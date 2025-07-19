# Vector Database Loading Performance Optimization Guide

## 🚀 Performance Improvements Made

Your vector database loading was running at **2 embeddings/second**, which is extremely slow. The optimizations should improve this to **200-1000+ embeddings/second**.

## 🔧 What Was Optimized

### 1. **Bulk Operations Instead of Individual Inserts**
- **Before**: Each batch file (100 embeddings) was added individually to ChromaDB
- **After**: All embeddings for a source are collected and added in large bulk operations
- **Impact**: 10-50x faster loading

### 2. **Configurable Database Batch Sizes**
- **Before**: Fixed small batches
- **After**: Configurable `db_batch_size` parameter (default: 10,000)
- **Impact**: Better memory usage and faster database operations

### 3. **Reduced Database Calls**
- **Before**: One database call per batch file (~500+ calls for 50k embeddings)
- **After**: One database call per source or per 10k embeddings (~5 calls for 50k embeddings)
- **Impact**: Dramatically reduced database overhead

## 📊 Expected Performance

| Configuration | Expected Rate | Use Case |
|---------------|---------------|----------|
| Conservative | 50-100 docs/sec | Limited memory, stable |
| Balanced | 200-500 docs/sec | Good balance |
| Aggressive | 500-1000+ docs/sec | Fast processing, more memory |

## 🛠️ How to Use the Optimizations

### 1. **Test Current Performance**
```bash
python test_loading_performance.py
```

### 2. **Run Full Benchmark**
```bash
python test_loading_performance.py full
```

### 3. **Use Optimized Loading**
The master processor now automatically uses optimized settings:
```python
# In master_processor.py - now uses DB_BATCH_SIZE from config
manager.add_embeddings_from_directory("xrvix_embeddings", db_batch_size=DB_BATCH_SIZE)
```

### 4. **Adjust Batch Sizes**
Edit `processing_config.py`:
```python
DB_BATCH_SIZE = 10000  # Increase for faster loading, decrease if memory issues
```

## 🎯 Performance Profile

- **Profile Name**: parallel_fixed
- **Max Workers**: 5
- **Request Timeout**: 60
- **Rate Limit Delay**: 0.04 (25 requests/sec)
- **Batch Size**: 100
- **DB Batch Size**: 5000

Only biorxiv and medrxiv are supported as sources. The system is hardcoded to never exceed 25 requests/sec.

This is the only available configuration. All processing uses this fixed parallel profile for optimal throughput under the current API rate limit.

## 🔍 Troubleshooting

### If Loading is Still Slow (< 50 docs/sec)

1. **Check System Resources**
   - Ensure you have enough RAM (8GB+ recommended)
   - Use SSD storage if possible
   - Close other memory-intensive applications

2. **Reduce Batch Size**
   ```python
   # In processing_config.py
   DB_BATCH_SIZE = 5000  # Try smaller value
   ```

3. **Check Disk Space**
   - Ensure you have enough free space for ChromaDB

### If You Get Memory Errors

1. **Reduce Batch Size**
   ```python
   DB_BATCH_SIZE = 2000  # Much smaller
   ```

2. **Process Fewer Sources**
   ```python
   # In master_processor.py, specify only one source
   manager.add_embeddings_from_directory("xrvix_embeddings", sources=["biorxiv"])
   ```

### If You Get Database Errors

1. **Delete and Recreate Collection**
   ```python
   manager.delete_collection()
   manager.create_collection()
   ```

2. **Check ChromaDB Version**
   ```bash
   pip show chromadb
   ```

## 📈 Monitoring Performance

### During Loading
The optimized code now shows:
- Progress indicators every 10 batches
- Bulk insertion progress
- Final statistics

### After Loading
Use the test script to measure actual performance:
```bash
python test_loading_performance.py
```

## 🎉 Expected Results

With these optimizations, you should see:
- **10-50x faster loading** (from 2 to 200+ docs/sec)
- **Better memory efficiency**
- **More stable performance**
- **Progress indicators** during loading

## 🔄 Next Steps

1. **Test the optimizations** with your data
2. **Adjust batch sizes** based on your system capabilities
3. **Monitor performance** and adjust as needed
4. **Consider hardware upgrades** if still too slow (SSD, more RAM)

## 📞 Getting Help

If you're still experiencing slow performance:
1. Run the benchmark script and share results
2. Check system resources (RAM, disk space, CPU)
3. Try different batch sizes
4. Consider processing smaller subsets first 