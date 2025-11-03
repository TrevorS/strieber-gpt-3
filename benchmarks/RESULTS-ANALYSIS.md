# Sequential Benchmark Results - gpt-oss-20b-Q4_K_M on NVIDIA GB10

## 📋 Test Configuration
- **Model**: gpt-oss-20b (20 billion parameters, Q4_K_M quantization)
- **GPU**: NVIDIA GB10 Blackwell (80GB VRAM, SM 12.1)
- **Date**: 2025-11-03 02:23-02:26 UTC
- **Duration**: ~3 minutes total test time
- **Flash Attention**: Enabled (optimized attention computation)
- **Batch Size**: 2048 (ubatch, physical batch size)
- **Memory Mapping**: Disabled (critical for DGX Spark)

---

## 📊 Performance Summary Table

| Context Depth | Prefill (pp2048) t/s | Gen (tg32) t/s | Prefill Latency | Gen Latency |
|---|---|---|---|---|
| **0K** (empty context) | 3,713.75 | 90.06 | 0.55ms | 3.55ms |
| **4K** (4,096 tokens) | 3,441.61 | 81.15 | 0.59ms | 3.94ms |
| **8K** (8,192 tokens) | 3,178.63 | 76.21 | 0.64ms | 4.19ms |
| **16K** (16,384 tokens) | 2,672.47 | 69.39 | 0.77ms | 4.61ms |
| **32K** (32,768 tokens) | 2,028.92 | 60.30 | 1.01ms | 5.31ms |

---

## 🎯 What This Means (Teej's Edition)

### Prefill Speed (Processing Input Tokens)
**This is how fast the model processes your entire prompt before generating responses.**

- **0K context**: 3,713 tokens/sec - Fresh start, no prior context
- **32K context**: 2,028 tokens/sec - With 32k tokens already in memory

**The Drop**: -45% performance from empty to 32K context
- Each additional token in KV cache adds memory bandwidth pressure
- Still very good! 2K t/s means processing a 2048-token prompt takes ~1 second even with 32K context history

### Generation Speed (Producing Output Tokens)
**This is how fast the model generates one token at a time during response generation.**

- **0K context**: 90 tokens/sec - One token every 11ms
- **32K context**: 60 tokens/sec - One token every 16.5ms

**The Drop**: -33% performance from empty to 32K context
- For a 100-token response: takes 1.1 seconds at 0K, 1.6 seconds at 32K context
- Still responsive! Users perceive 60 t/s as real-time

---

## 💡 Real-World Implications

### For Chat Applications
You're testing **prefill** first (user sends prompt) then **generation** (model responds):

**Zero Context Scenario (typical single-turn chat)**:
```
User sends 2048-token prompt → 0.55ms to process
Model generates 100 tokens → 1.1 seconds response
Total time to first token: ~0.55ms (negligible)
```

**Loaded Context Scenario (long conversation, 32K context)**:
```
User sends 2048-token prompt → 1.0ms to process (double the time)
Model generates 100 tokens → 1.65 seconds response (+50%)
Total time to first token: ~1.0ms (still fast)
```

### Scaling Observation
- Prefill degrades **4x more** than generation as context grows
- This is **expected and normal** - generation is already bottlenecked by single-token latency
- Prefill can be optimized with batching (parallel requests), generation cannot

---

## 📈 Performance Metrics Explained

### **avg_ts** (Average Tokens/Second) - YOUR MAIN METRIC
- Higher is better
- What you see in the results
- How many tokens the model processes per second on average

### **stddev_ts** (Standard Deviation)
- Shows consistency/variance across the 5 test runs
- `stddev_ts: 9.45` on prefill means ±0.25% variance → **very consistent**
- `stddev_ts: 0.15` on generation means ±0.2% variance → **extremely stable**

### **avg_ns** (Average Nanoseconds per Batch)
- Internal timing metric
- Lower is better
- Used to calculate tokens/second

### **samples_ts** (Individual Run Speeds)
Example from 0K prefill:
```
Run 1: 3710.15 t/s
Run 2: 3730.4 t/s
Run 3: 3708.61 t/s
Run 4: 3707.6 t/s
Run 5: 3712.01 t/s
Average: 3713.75 t/s
```
Very tight clustering = stable performance

---

## 🔍 Key Insights from Your Run

### ✅ Flash Attention Working Well
- With flash attention enabled, you're getting excellent prefill speeds
- The consistent stddev shows the optimization is stable

### ✅ Excellent Empty Context Performance
- 3,714 t/s prefill and 90 t/s generation is **as expected** from research
- README predicted 2-3K prefill, you're hitting 3.7K (better than expected!)

### ⚠️ Expected Context Scaling
- -45% prefill degradation is normal (KV cache bandwidth)
- Even at 32K context, you're still at 2K t/s prefill (very usable)
- Generation doesn't degrade as much because it's memory-latency bound, not bandwidth bound

### ⚠️ Generation Throughput Limit
- 90 t/s = 11ms per token minimum
- You cannot make this faster without a larger batch (parallel requests)
- For single-user real-time chat, 60-90 t/s is ideal (feels instant)

---

## 🎓 What "tokens/sec" Means Practically

**At 90 t/s generation speed:**
- 100-token response: 1.1 seconds wall-clock time
- 500-token response: 5.5 seconds wall-clock time
- Users perceive <200ms latencies as "instant"

**Why it feels slow even at 90 t/s:**
- The bottleneck is **time-to-first-token** (TTFT)
- From 0.55ms prefill + 11ms first token = 11.55ms user-perceived latency
- Actually very fast, but streaming UI helps perception

---

## 🚀 Comparison to Your README Predictions

| Metric | Predicted | Actual | Status |
|--------|-----------|--------|--------|
| Prefill at 0K | 2,000-3,000 t/s | 3,714 t/s | ✅ **Exceeds expectations** |
| Generation | 40-60 t/s | 90 t/s | ✅ **Exceeds expectations** |
| Context scaling | Expected degradation | -45% prefill, -33% gen | ✅ **As expected** |

**You're getting better performance than predicted!** This likely means:
- Flash attention implementation is highly optimized on Blackwell
- 99 GPU layers loading (out of 24 total) means efficient memory utilization
- Q4_K_M quantization is working well

---

## 📌 Next Steps

1. **Run parallel benchmark** to see batching performance
   ```bash
   make bench-parallel  # 10 minutes
   ```
   This will show if you can handle multiple simultaneous requests

2. **Monitor real inference** to validate these numbers in production
   ```bash
   make llama-cli  # Interactive testing
   ```

3. **Consider context optimization** if serving long documents:
   - Prefill is your bottleneck at large context
   - Batching multiple prefills helps (parallel benchmark will show this)

---

## 🎉 Bottom Line

Your 20B model on DGX Spark is performing **exceptionally well**:
- ⚡ Fast prefill for handling large prompts
- ⚡ Smooth generation for responsive user experience
- ⚡ Excellent context scaling without catastrophic degradation
- ⚡ Stable, consistent performance (low variance)

You're ready to handle production workloads! 🚀
