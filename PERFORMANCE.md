# Performance Tracking Guide

## Overview

The system now includes comprehensive performance tracking that logs the execution time of each major processing stage. This helps identify bottlenecks and optimize the pipeline.

## How It Works

### Automatic Timing

Every analysis run automatically tracks and logs:
1. **Data Preparation** - Converting input to internal format
2. **Embedding Generation** - Creating sentence embeddings with sentence-transformers
3. **Clustering** - HDBSCAN clustering of embeddings
4. **Sentiment Analysis** - Analyzing sentiment per cluster
5. **Insight Generation** - Generating titles and insights with GPT-4o

### Performance Logs

At the end of each analysis, you'll see a performance summary in the logs:

```
============================================================
[PERF] Performance Summary: Standalone Analysis
============================================================
[PERF]   1. Data Preparation.......................   0.003s (  0.1%)
[PERF]   2. Embedding Generation...................  12.456s ( 45.3%)
[PERF]   3. Clustering.............................   3.234s ( 11.8%)
[PERF]   4. Sentiment Analysis.....................   2.567s (  9.3%)
[PERF]   5. Insight Generation.....................   9.234s ( 33.5%)
------------------------------------------------------------
[PERF]   TOTAL.....................................  27.494s (100.0%)
============================================================
```

## Understanding the Output

### Stage Breakdown

| Stage | What It Does | Expected % | Bottleneck If |
|-------|--------------|------------|---------------|
| **Data Preparation** | Convert input format | <1% | Usually not a concern |
| **Embedding Generation** | Create 384-dim vectors | 30-50% | Large datasets (>500 sentences) |
| **Clustering** | HDBSCAN algorithm | 10-20% | Dense datasets, many clusters |
| **Sentiment Analysis** | Classify sentiment per cluster | 5-15% | Many clusters (>20) |
| **Insight Generation** | GPT-4o API calls | 30-50% | Many clusters, API latency |

### Example Scenarios

#### Scenario 1: Embedding-Heavy (Normal)
```
Embedding Generation: 45%
Clustering: 12%
Sentiment: 8%
Insight Generation: 35%
```
**Analysis:** Normal distribution. Embedding is CPU-bound and expected to be significant.

#### Scenario 2: Insight-Heavy (Many Clusters)
```
Embedding Generation: 25%
Clustering: 10%
Sentiment: 5%
Insight Generation: 60%
```
**Analysis:** Many clusters → many GPT-4o API calls. Consider reducing `max_clusters`.

#### Scenario 3: Sentiment-Heavy (Rare)
```
Embedding Generation: 30%
Clustering: 10%
Sentiment: 40%
Insight Generation: 20%
```
**Analysis:** Many sentences per cluster. Sentiment model running on large batches.

## Using Performance Data

### During Development

Run your test with timing enabled:
```bash
python test_standalone.py 2>&1 | tee performance_log.txt
```

Then analyze the performance summary at the end.

### Comparing Configurations

Test different configurations and compare:

```bash
# Configuration 1: Default
python test_standalone.py > perf_default.log 2>&1

# Configuration 2: Smaller batch size
# (modify test script to use batch_size=16)
python test_standalone.py > perf_batch16.log 2>&1

# Compare
grep "\[PERF\]" perf_default.log
grep "\[PERF\]" perf_batch16.log
```

### Identifying Bottlenecks

1. **If Embedding Generation >50%:**
   - Consider caching embeddings for repeated runs
   - Use GPU if available (sentence-transformers supports CUDA)
   - Reduce dataset size for testing

2. **If Clustering >25%:**
   - May indicate very large dataset
   - Consider adjusting `min_cluster_size` parameter
   - Check if HDBSCAN parameters are optimal

3. **If Sentiment Analysis >20%:**
   - Too many clusters or large clusters
   - Consider batching optimization
   - May need to adjust cluster parameters

4. **If Insight Generation >60%:**
   - Too many clusters (reduce `max_clusters`)
   - OpenAI API latency (network issues)
   - Consider parallel API calls (future optimization)

## Performance Optimization Strategies

### Quick Wins

1. **Reduce max_clusters** (if seeing >10 clusters)
   ```python
   analyzer = TextAnalyzer(
       openai_api_key=key,
       max_clusters=5  # Instead of 20
   )
   ```

2. **Adjust min_cluster_size** (to create larger, fewer clusters)
   ```python
   analyzer = TextAnalyzer(
       openai_api_key=key,
       min_cluster_size=5  # Instead of 3
   )
   ```

3. **Use smaller test datasets** during development
   ```python
   baseline_subset = raw_data['baseline'][:30]  # Instead of all
   ```

### Advanced Optimizations

1. **Parallel Insight Generation**
   - Currently sequential (one GPT-4o call at a time)
   - Could parallelize to reduce total time
   - Trade-off: More concurrent API requests

2. **Embedding Caching**
   - Cache embeddings for repeated sentences
   - Useful if analyzing same data multiple times
   - Need cache invalidation strategy

3. **Batch Size Tuning**
   - Currently batch_size=32 for embeddings
   - Larger batches = faster but more memory
   - GPU vs CPU considerations

4. **Model Optimization**
   - Quantize sentiment model (reduce size/time)
   - Use faster embedding model (trade-off: accuracy)
   - Cache loaded models in Lambda (warm starts)

## Lambda Performance Considerations

### Cold Start vs Warm Start

**Cold Start (first request):**
- Model loading: +5-10s
- Total: ~35-40s for 50 sentences

**Warm Start (subsequent requests):**
- No model loading
- Total: ~25-30s for 50 sentences

### Optimization for Lambda

1. **Increase Memory:**
   ```yaml
   MemorySize: 3008  # More memory = more CPU
   ```

2. **Provisioned Concurrency:**
   - Keep instances warm
   - Eliminate cold starts
   - Additional cost

3. **Model Pre-loading:**
   - Load models during Lambda init
   - Currently implemented in service constructors

## Monitoring in Production

### CloudWatch Metrics

Track custom metrics:
- Total execution time
- Time per stage
- Number of clusters generated
- Sentences processed

### Alarms

Set up alarms for:
- Execution time > 10s (or your SLA)
- High percentage in one stage (>70%)
- Timeout errors

### Example CloudWatch Query

```
fields @timestamp, @message
| filter @message like /\[PERF\]/
| filter @message like /TOTAL/
| parse @message "*TOTAL*s*" as prefix, total_time, suffix
| stats avg(total_time), max(total_time), min(total_time) by bin(5m)
```

## Performance Targets

Based on PRD requirements:

| Dataset Size | Target Time | Acceptable Time |
|--------------|-------------|-----------------|
| 10-50 sentences | <3s | <5s |
| 50-100 sentences | <5s | <8s |
| 100-500 sentences | <10s | <15s |
| 500-1000 sentences | <20s | <30s |
| 1000+ sentences | <60s | <90s |

## Example Usage

### Basic Performance Check

```python
# Run test
python test_standalone.py

# Look for performance summary in output
# Should see:
# [PERF] Performance Summary: Standalone Analysis
# [PERF]   1. Data Preparation...
# [PERF]   2. Embedding Generation...
# etc.
```

### Detailed Performance Analysis

```bash
# Run and save logs
python test_standalone.py 2>&1 | tee perf_test.log

# Extract performance data
grep "\[PERF\]" perf_test.log

# Get just the summary
grep -A 10 "Performance Summary" perf_test.log
```

### Compare Before/After Optimization

```bash
# Before
python test_standalone.py 2>&1 | grep "\[PERF\].*TOTAL" > before.txt

# Apply optimization
# ... make changes ...

# After
python test_standalone.py 2>&1 | grep "\[PERF\].*TOTAL" > after.txt

# Compare
diff before.txt after.txt
```

## Troubleshooting

### Performance Logs Not Showing

Check logging level:
```python
logging.basicConfig(level=logging.INFO)  # Or DEBUG
```

### Times Seem Wrong

- Ensure no other processes are using CPU/GPU
- Check if models are being downloaded (first run)
- Verify network connectivity (for OpenAI API)

### Very Slow Execution

1. Check OpenAI API latency
2. Verify GPU is being used (if available)
3. Check dataset size vs expectations
4. Look at percentage breakdown to identify bottleneck

---

**Key Takeaway:** Use performance logs to identify bottlenecks, compare configurations, and optimize the processing pipeline systematically.
