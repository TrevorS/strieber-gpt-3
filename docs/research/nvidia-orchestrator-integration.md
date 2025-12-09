# NVIDIA Orchestrator-8B: Integration Analysis for Local LLM Stack

**Date:** 2025-12-08
**Status:** Research Analysis

## Executive Summary

NVIDIA's Orchestrator-8B represents a paradigm shift from monolithic LLM deployments to **composite AI systems** where a small, specialized routing model coordinates larger models and tools. This document analyzes how a similar approach could transform the strieber-gpt-3 local stack—potentially fronting the 120B GPT-OSS model with a lightweight orchestrator to reduce costs, improve latency, and maintain accuracy.

---

## Part 1: NVIDIA Orchestrator-8B Overview

### What Is It?

Orchestrator-8B is an **8-billion parameter decoder-only transformer** (fine-tuned from Qwen3-8B) that acts as a "routing brain" for agentic AI systems. Instead of throwing every query at a massive frontier model, the orchestrator:

1. **Analyzes the incoming task** complexity and requirements
2. **Considers user preferences** (speed vs. cost vs. accuracy)
3. **Routes to the appropriate resource**—which could be:
   - A basic tool (web search, code interpreter)
   - A specialized LLM (math solver, coding model)
   - A large generalist model (GPT-5, Claude Opus, Llama-Nemotron-Ultra-253B)
4. **Iterates in a multi-turn loop** until task completion

### Key Technical Innovation: ToolOrchestra Training

The model is trained via **Group Relative Policy Optimization (GRPO)** with a novel multi-objective reward function that simultaneously optimizes for:

| Objective | Description |
|-----------|-------------|
| **Accuracy** | Task outcome correctness |
| **Efficiency** | Cost and latency minimization |
| **User Preference Adherence** | Respect for speed/cost/accuracy trade-offs |

This is fundamentally different from prompt-based routing (e.g., "use GPT-4 for complex tasks"), which cannot achieve nuanced trade-off optimization.

### Benchmark Results

| Benchmark | Orchestrator-8B | GPT-5 | Improvement |
|-----------|-----------------|-------|-------------|
| **Humanity's Last Exam** | 37.1% | 35.1% | +2% accuracy |
| **Cost (HLE)** | $0.092 | $0.302 | **70% cheaper** |
| **Latency (HLE)** | 8.2 min | 19.8 min | **2.5x faster** |
| **FRAMES** | Superior | Baseline | ~30% cost |
| **τ²-Bench** | Superior | Baseline | Significant gains |

### Critical Finding: Eliminating Self-Enhancement Bias

When GPT-5 acts as its own router, it overwhelmingly prefers to call GPT-5-mini. Orchestrator-8B, being trained separately, makes **balanced tool calls without bias toward any particular model**.

---

## Part 1.5: Why Not Just Use Qwen3-8B for Routing?

This is a critical question: **If Qwen3-8B has the best tool-calling accuracy (F1: 0.933) among 8B models, why did NVIDIA spend the effort to train a dedicated Orchestrator-8B?**

The answer reveals a fundamental difference between **tool calling** and **orchestration**.

### Tool Calling ≠ Orchestration

| Capability | Tool Calling | Orchestration |
|------------|--------------|---------------|
| **Task** | Execute a specific function correctly | Decide *which* resource to use |
| **Optimization** | Accuracy only | Accuracy + Cost + Latency + User Preference |
| **Decision Space** | "Call weather API with these args" | "Should I call weather API, use small model, or escalate to large model?" |
| **Training Signal** | Did the function call succeed? | Did the system solve the problem efficiently? |

Qwen3-8B excels at the first row. But orchestration requires optimizing across **multiple competing objectives simultaneously**.

### The Self-Enhancement Bias Problem

When you prompt an LLM to "choose the best tool for this task," you get systematic biases:

```
┌─────────────────────────────────────────────────────────────────────┐
│              SELF-ENHANCEMENT BIAS IN ROUTING                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  When GPT-5 is the router:                                         │
│  ├─► Overwhelmingly prefers GPT-5-mini                             │
│  ├─► "Family loyalty" - trusts its own model family                │
│  └─► Rarely delegates to specialized models even when optimal      │
│                                                                     │
│  When Qwen3-8B is the router:                                      │
│  ├─► Defaults to GPT-5 for everything                              │
│  ├─► Lacks confidence in its own abilities                         │
│  └─► Over-delegates, wasting expensive model calls                 │
│                                                                     │
│  When Orchestrator-8B (RL-trained) is the router:                  │
│  ├─► Balanced distribution across all tools                        │
│  ├─► Routes simple tasks to cheap models                           │
│  ├─► Reserves expensive models for genuinely complex tasks         │
│  └─► Respects user cost/speed preferences                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Why Prompting Cannot Solve This

**Prompt-based routing example:**
```
You are a router. Choose the best tool for the user's query:
- small_model: for simple queries
- large_model: for complex reasoning
- math_model: for calculations

User query: "What is 2+2?"
```

**Problems with this approach:**

1. **No cost awareness**: The model doesn't know (or care) that `large_model` costs 10x more
2. **No latency feedback**: The model doesn't learn that `small_model` responds in 2s vs 15s
3. **No outcome optimization**: The model isn't penalized if the routed model fails
4. **No preference tuning**: Can't learn "when user says 'fast', really prefer cheap models"
5. **Instruction following ≠ optimization**: Following instructions doesn't mean making optimal trade-offs

### The RL Training Difference

NVIDIA's key insight: **Make the orchestrator learn from experience with explicit reward signals.**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    THREE-PART REWARD FUNCTION                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. OUTCOME REWARD (Binary)                                        │
│     └─► Did the system solve the problem? +1 or 0                  │
│                                                                     │
│  2. EFFICIENCY REWARD (Continuous)                                 │
│     └─► Cost penalty: -α × dollars_spent                           │
│     └─► Latency penalty: -β × seconds_elapsed                      │
│     └─► Encourages cheap/fast solutions when they work             │
│                                                                     │
│  3. PREFERENCE REWARD (Conditional)                                │
│     └─► If user said "fast": extra reward for low latency          │
│     └─► If user said "accurate": extra reward for high accuracy    │
│     └─► If user said "cheap": extra reward for low cost            │
│                                                                     │
│  Total Reward = Outcome + λ₁(Efficiency) + λ₂(Preference)          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

This reward structure **fundamentally changes the model's behavior**:

- If `small_model` can solve "2+2" correctly, the orchestrator learns to use it (high outcome + high efficiency)
- If `small_model` fails on a calculus problem, the orchestrator learns to escalate (outcome > efficiency)
- If user wants "fast", the orchestrator shifts toward lower-latency options

### Why GRPO Instead of Standard RL?

NVIDIA used **Group Relative Policy Optimization (GRPO)** instead of PPO because:

1. **No critic network needed** → 40% less memory/compute
2. **Comparative assessment** → Evaluates choices relative to alternatives, not absolute
3. **More stable training** → Less sensitive to hyperparameters
4. **Works with sparse rewards** → Outcome reward is often binary

### Empirical Evidence: Balanced Tool Distribution

From the paper, Orchestrator-8B distributes calls more evenly:

| Router | % to Cheap Models | % to Expensive Models | % Direct Tools |
|--------|-------------------|----------------------|----------------|
| GPT-5 (prompted) | 15% | 75% | 10% |
| Qwen3-8B (prompted) | 5% | 85% | 10% |
| **Orchestrator-8B** | **45%** | **35%** | **20%** |

The RL-trained orchestrator routes nearly half of queries to cheap models and aggressively uses direct tool calls—achieving 70% cost reduction.

### Bottom Line

**Qwen3-8B is excellent at tool calling** (when told to call a tool, it does so accurately).

**Orchestrator-8B is excellent at deciding whether and which tool to call** (meta-level decision making with cost awareness).

They solve different problems:
- Qwen3-8B: "Execute `get_weather('NYC')` correctly"
- Orchestrator-8B: "Should I call `get_weather`, use `small_model`, or escalate to `large_model`?"

For a local stack, you could:
1. **Use Orchestrator-8B** for routing decisions (purpose-built)
2. **Use Qwen3-8B** as your small general model (best tool calling accuracy)
3. This gives you the best of both worlds

---

## Part 2: Current strieber-gpt-3 Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    CURRENT ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Frontend (Svelte) ───► Responses API (Rust) ───► gpt-oss-120B │
│                              │                                  │
│                              ├───► MCP Tools                    │
│                              │     • weather                    │
│                              │     • web_search                 │
│                              │     • code_interpreter           │
│                              │     • reader (ReaderLM)          │
│                              │     • zimage_turbo               │
│                              │                                  │
│                              └───► Specialized Models           │
│                                    • Qwen3-VL-2B (vision)       │
│                                    • ReaderLM-v2 (HTML→MD)      │
│                                    • EmbeddingGemma-300M        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Current Routing Logic

The existing system has **no intelligent routing**. All queries flow through the same path:

1. **Frontend** sends request to Responses API
2. **Responses API** translates to Chat Completions format
3. **gpt-oss-120B** processes the request
4. If tool calls are present, execute them via MCP
5. Loop until no more tool calls (max 10 iterations)

**Problem:** Every query—simple or complex—hits the 120B model, consuming significant GPU resources.

### Resource Allocation

| Service | GPU VRAM | Purpose |
|---------|----------|---------|
| gpt-oss-120b | 16GB | Main inference |
| ReaderLM-v2 | 8GB | HTML→Markdown |
| Qwen3-VL-2B | 4GB | Vision tasks |
| EmbeddingGemma | 2GB | Embeddings |
| **Total** | **30GB** | |

---

## Part 3: Orchestrator Integration Proposal

### Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR-ENHANCED ARCHITECTURE                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Frontend (Svelte)                                                      │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              Responses API (Enhanced)                            │   │
│  │  ┌─────────────────────────────────────────────────────────┐    │   │
│  │  │          ORCHESTRATOR LAYER (8B)                        │    │   │
│  │  │                                                          │    │   │
│  │  │  1. Analyze query complexity                             │    │   │
│  │  │  2. Check user preferences (speed/cost/accuracy)         │    │   │
│  │  │  3. Route to optimal resource:                           │    │   │
│  │  │                                                          │    │   │
│  │  │     ┌──────────┐   ┌──────────┐   ┌──────────────┐      │    │   │
│  │  │     │ Direct   │   │ Small    │   │ Large Model  │      │    │   │
│  │  │     │ Tool     │   │ Model    │   │ (gpt-oss-    │      │    │   │
│  │  │     │ Call     │   │ (7-14B)  │   │ 120B)        │      │    │   │
│  │  │     └──────────┘   └──────────┘   └──────────────┘      │    │   │
│  │  │                                                          │    │   │
│  │  │  4. Multi-turn reasoning loop                            │    │   │
│  │  │  5. Synthesize final response                            │    │   │
│  │  └─────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Model Backends                                                         │
│  ├─ orchestrator-8b (NEW)     - 10GB VRAM (INT8)                       │
│  ├─ gpt-oss-120b              - 16GB VRAM (complex reasoning)          │
│  ├─ small-model-7b (NEW)      - 8GB VRAM (simple tasks)                │
│  ├─ math-specialist (NEW)     - 8GB VRAM (math/code)                   │
│  └─ Qwen3-VL-2B               - 4GB VRAM (vision)                      │
│                                                                         │
│  MCP Tools (unchanged)                                                  │
│  ├─ weather, web_search, code_interpreter, reader, zimage_turbo        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Routing Decision Tree

```
User Query
    │
    ▼
┌───────────────────────┐
│   Orchestrator-8B     │
│   Complexity Analysis │
└───────────────────────┘
    │
    ├─── Simple factual? ──────────► small-model-7b
    │    "What's the capital of France?"
    │
    ├─── Math/code heavy? ─────────► math-specialist
    │    "Solve this differential equation"
    │
    ├─── Tool-only task? ──────────► Direct tool call
    │    "What's the weather in NYC?"     (skip LLM entirely)
    │
    ├─── Multi-modal? ─────────────► Qwen3-VL-2B
    │    "Describe this image"
    │
    └─── Complex reasoning? ───────► gpt-oss-120B
         "Analyze this business strategy and compare..."
```

---

## Part 4: Implementation Strategy

### Phase 1: Minimal Integration (Low Risk)

**Goal:** Add orchestrator as optional routing layer without breaking existing flows.

```rust
// backend/responses-api/src/orchestration/router.rs

pub enum RoutingDecision {
    DirectTool { tool_name: String, args: Value },
    SmallModel { model_id: String },
    LargeModel,  // Default to gpt-oss-120b
    Specialist { model_id: String, specialty: Specialty },
}

pub struct OrchestratorRouter {
    orchestrator_client: LlmClient,
    config: OrchestratorConfig,
}

impl OrchestratorRouter {
    pub async fn route(&self, request: &CreateResponseRequest) -> RoutingDecision {
        // If orchestrator disabled, always use large model
        if !self.config.enabled {
            return RoutingDecision::LargeModel;
        }

        // Call orchestrator model for routing decision
        let analysis = self.analyze_complexity(request).await?;

        match analysis.recommendation {
            "direct_tool" => RoutingDecision::DirectTool { ... },
            "small_model" => RoutingDecision::SmallModel { ... },
            "specialist" => RoutingDecision::Specialist { ... },
            _ => RoutingDecision::LargeModel,
        }
    }
}
```

**Configuration:**

```json
{
  "orchestrator": {
    "enabled": true,
    "model_url": "http://orchestrator-8b:8000",
    "user_preference": "balanced",  // "fast" | "cheap" | "accurate" | "balanced"
    "fallback_to_large": true
  },
  "routing_thresholds": {
    "complexity_for_large_model": 0.7,
    "tool_confidence_threshold": 0.9
  }
}
```

### Phase 2: Multi-Model Backend

**Goal:** Add smaller model options to reduce GPU load.

```yaml
# compose.yml additions

  orchestrator-8b:
    image: ghcr.io/ggerganov/llama.cpp:server-cuda
    command: >
      --model /models/orchestrator-8b-q8.gguf
      --ctx-size 8192
      --n-gpu-layers 99
      --port 8000
    ports:
      - "9060:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
              device_ids: ['0']

  small-model-7b:
    image: ghcr.io/ggerganov/llama.cpp:server-cuda
    command: >
      --model /models/qwen2.5-7b-instruct-q8.gguf
      --ctx-size 32768
      --n-gpu-layers 99
      --port 8000
    ports:
      - "9070:8000"
```

### Phase 3: RL Training Pipeline (Advanced)

**Goal:** Train custom orchestrator on actual usage patterns.

This would involve:

1. **Data Collection:** Log all queries with complexity annotations
2. **Reward Design:**
   - Accuracy: Did the routed model produce correct output?
   - Efficiency: GPU-seconds consumed
   - User satisfaction: Explicit feedback or implicit signals
3. **Training:** GRPO optimization similar to NVIDIA's approach

---

## Part 5: Anthropic Application Perspective

### How Anthropic Could Use This

If Anthropic were to adopt an orchestrator pattern for their API, it could work as follows:

```
┌─────────────────────────────────────────────────────────────────┐
│                 ANTHROPIC ORCHESTRATED API                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  User Request                                                   │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────┐                                   │
│  │   Claude Orchestrator   │  (Small, fast, trained to route)  │
│  │   (Haiku-class)         │                                   │
│  └─────────────────────────┘                                   │
│       │                                                         │
│       ├──► Claude Haiku (simple queries, chat, summaries)      │
│       │                                                         │
│       ├──► Claude Sonnet (balanced tasks, most queries)        │
│       │                                                         │
│       ├──► Claude Opus (complex reasoning, research, math)     │
│       │                                                         │
│       └──► External Tools (web, code, retrieval)               │
│                                                                 │
│  User Controls:                                                 │
│  - preference: "fast" | "balanced" | "thorough"                │
│  - max_cost: "$0.10"                                           │
│  - latency_target: "2s"                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Benefits for API Providers

1. **Cost Reduction:** Route 70%+ of queries to cheaper models
2. **Latency Improvement:** Simple queries get instant responses
3. **Transparency:** Users understand why they're charged what they're charged
4. **User Control:** Let users specify their preferences explicitly

### Benefits for Local Stacks (like strieber-gpt-3)

1. **GPU Efficiency:** Don't waste 120B model on "What time is it?"
2. **Parallel Processing:** Small model can handle queue while large model processes complex tasks
3. **Graceful Degradation:** If large model is busy, route to smaller model
4. **Specialized Routing:** Math to math model, code to code model

---

## Part 6: Concrete Integration Points in strieber-gpt-3

### 1. Modify Execution Pipeline

**File:** `backend/responses-api/src/execution/executor.rs`

Add orchestrator decision before calling LLM:

```rust
pub async fn execute(
    &self,
    request: CreateResponseRequest,
    mcp_client: &McpClient,
    model_config: &ModelConfig,
) -> Result<Response, ExecutionError> {
    // NEW: Orchestrator routing
    let routing_decision = self.orchestrator
        .route(&request)
        .await
        .unwrap_or(RoutingDecision::LargeModel);

    // Select model based on routing
    let target_model = match routing_decision {
        RoutingDecision::SmallModel { model_id } =>
            self.get_model_config(&model_id)?,
        RoutingDecision::LargeModel =>
            model_config,
        RoutingDecision::DirectTool { tool_name, args } => {
            // Skip LLM entirely, execute tool directly
            return self.execute_tool_only(tool_name, args, mcp_client).await;
        }
        // ... other cases
    };

    // Continue with existing logic using target_model
    // ...
}
```

### 2. Add Model Configuration

**File:** `backend/responses-api/src/config.rs`

```rust
#[derive(Debug, Clone, Deserialize)]
pub struct OrchestratorConfig {
    pub enabled: bool,
    pub model_url: String,
    pub user_preference: UserPreference,
    pub complexity_threshold: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub enum UserPreference {
    Fast,
    Cheap,
    Accurate,
    Balanced,
}
```

### 3. Frontend Preference Controls

**File:** `frontend/src/lib/components/settings/SettingsPanel.svelte`

Add user preference selector:

```svelte
<div class="setting-group">
  <label>AI Routing Preference</label>
  <select bind:value={$settings.routingPreference}>
    <option value="fast">Fast (smaller models)</option>
    <option value="balanced">Balanced (auto-select)</option>
    <option value="thorough">Thorough (larger models)</option>
  </select>
  <p class="hint">Controls how queries are routed between models</p>
</div>
```

### 4. Metrics & Observability

Track routing decisions for analysis:

```rust
#[derive(Debug, Serialize)]
pub struct RoutingMetrics {
    pub query_id: String,
    pub routing_decision: String,
    pub model_used: String,
    pub latency_ms: u64,
    pub tokens_used: u32,
    pub estimated_cost: f32,
}
```

---

## Part 7: Challenges & Considerations

### Technical Challenges

| Challenge | Mitigation |
|-----------|------------|
| **Added latency** from orchestrator call | Cache routing decisions for similar queries |
| **GPU memory** for additional model | Use INT8 quantization (10GB for 8B model) |
| **Training data** for custom orchestrator | Start with NVIDIA's pre-trained, fine-tune later |
| **Accuracy of routing** | Conservative fallback to large model |

### When NOT to Use Orchestrator

- **Single-model deployments:** Overhead not worth it
- **Homogeneous query types:** If all queries are similar complexity
- **Latency-critical applications:** Extra hop adds ~100-200ms
- **Small scale:** Benefits emerge at scale

### Resource Requirements

| Component | VRAM | Purpose |
|-----------|------|---------|
| Orchestrator-8B (INT8) | 10GB | Routing decisions |
| Small model (7B) | 8GB | Simple queries |
| Large model (120B) | 16GB | Complex reasoning |
| **Total** | **34GB** | (vs 30GB current) |

Note: Can time-share GPU between orchestrator and small model.

---

## Part 8: Recommended Next Steps

### Immediate (This Week)

1. **Download Orchestrator-8B** from HuggingFace
   - Convert to GGUF format for llama.cpp
   - Test inference locally

2. **Benchmark routing accuracy**
   - Sample 100 historical queries
   - Manually label complexity
   - Compare orchestrator decisions vs. labels

### Short-term (This Month)

3. **Implement minimal routing layer**
   - Add `orchestrator` module to responses-api
   - Feature-flag for gradual rollout
   - Log all routing decisions

4. **Add second model backend**
   - Deploy Qwen2.5-7B for simple queries
   - Configure model selection in executor

### Medium-term (This Quarter)

5. **Collect training data**
   - Log query complexity, model used, outcome quality
   - Build dataset for fine-tuning

6. **Train custom orchestrator**
   - Use ToolOrchestra training code
   - Optimize for local model lineup

---

## References

- [NVIDIA Orchestrator-8B Model](https://huggingface.co/nvidia/Orchestrator-8B)
- [ToolOrchestra Paper (arXiv)](https://arxiv.org/abs/2511.21689)
- [ToolOrchestra GitHub](https://github.com/NVlabs/ToolOrchestra/)
- [NVIDIA Technical Blog: Train Small Orchestration Agents](https://developer.nvidia.com/blog/train-small-orchestration-agents-to-solve-big-problems/)
- [NVIDIA AI Blueprint for LLM Routing](https://developer.nvidia.com/blog/deploying-the-nvidia-ai-blueprint-for-cost-efficient-llm-routing/)
- [ToolOrchestra Research Page](https://research.nvidia.com/labs/lpr/ToolOrchestra/)

---

## Appendix A: NVIDIA Orchestrator-8B Prompt Format

Based on HuggingFace model card, the orchestrator expects:

```
<|im_start|>system
You are a helpful assistant that coordinates tools and models to solve user tasks.
Available tools: {tool_definitions}
Available models: {model_definitions}
User preference: {preference}
<|im_end|>
<|im_start|>user
{user_query}
<|im_end|>
<|im_start|>assistant
```

The model outputs structured routing decisions in its response.

## Appendix B: Estimated Cost Savings

Assuming query distribution:
- 60% simple (routable to 7B model)
- 25% medium (require 120B)
- 15% tool-only (no LLM needed)

| Scenario | GPU-seconds/query | Queries/hour | GPU Utilization |
|----------|-------------------|--------------|-----------------|
| **Current (all 120B)** | 5.0s | 720 | 100% |
| **With Orchestrator** | 1.8s avg | 2000 | 36% |

**Projected improvement: 2.8x throughput increase**

---

## Part 9: Detailed Setup Guide

### Step 1: Download and Convert Orchestrator-8B

The GGUF-quantized version is available from bartowski on HuggingFace:

```bash
# Install huggingface CLI
pip install -U "huggingface_hub[cli]"

# Download Q4_K_M quantization (5.03GB, good quality/size balance)
huggingface-cli download bartowski/nvidia_Orchestrator-8B-GGUF \
  --include "nvidia_Orchestrator-8B-Q4_K_M.gguf" \
  --local-dir ~/models/llama-cpp/

# Or for higher quality (6.73GB):
huggingface-cli download bartowski/nvidia_Orchestrator-8B-GGUF \
  --include "nvidia_Orchestrator-8B-Q6_K.gguf" \
  --local-dir ~/models/llama-cpp/
```

**Available Quantizations:**

| Quant | Size | Quality | VRAM Needed |
|-------|------|---------|-------------|
| Q6_K | 6.73GB | Very High | ~8GB |
| Q5_K_M | 5.85GB | High | ~7GB |
| **Q4_K_M** | **5.03GB** | **Good (Recommended)** | **~6GB** |
| Q3_K_M | 4.12GB | Medium | ~5GB |
| Q2_K | 3.28GB | Lower | ~4GB |

### Step 2: Add Orchestrator Service to Docker Compose

```yaml
# compose.yml - Add these services

  # Orchestrator Model (routing decisions)
  llama-server-orchestrator:
    image: ghcr.io/ggerganov/llama.cpp:server-cuda
    command: >
      --model /models/nvidia_Orchestrator-8B-Q4_K_M.gguf
      --ctx-size 8192
      --n-gpu-layers 99
      --flash-attn
      --port 8000
      --host 0.0.0.0
    volumes:
      - ~/models/llama-cpp:/models:ro
    ports:
      - "9060:8000"
    networks:
      - strieber-net
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
              device_ids: ['0']
    restart: unless-stopped

  # Small General Model (simple queries)
  llama-server-small:
    image: ghcr.io/ggerganov/llama.cpp:server-cuda
    command: >
      --model /models/qwen3-8b-instruct-q4_k_m.gguf
      --ctx-size 32768
      --n-gpu-layers 99
      --flash-attn
      --port 8000
      --host 0.0.0.0
    volumes:
      - ~/models/llama-cpp:/models:ro
    ports:
      - "9070:8000"
    networks:
      - strieber-net
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
              device_ids: ['0']
    restart: unless-stopped

  # Math Specialist (optional)
  llama-server-math:
    image: ghcr.io/ggerganov/llama.cpp:server-cuda
    command: >
      --model /models/deepseek-r1-distill-qwen-7b-q4_k_m.gguf
      --ctx-size 32768
      --n-gpu-layers 99
      --flash-attn
      --port 8000
      --host 0.0.0.0
    volumes:
      - ~/models/llama-cpp:/models:ro
    ports:
      - "9080:8000"
    networks:
      - strieber-net
    profiles:
      - specialists
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
              device_ids: ['0']

  # Coding Specialist (optional)
  llama-server-coder:
    image: ghcr.io/ggerganov/llama.cpp:server-cuda
    command: >
      --model /models/qwen2.5-coder-7b-instruct-q4_k_m.gguf
      --ctx-size 32768
      --n-gpu-layers 99
      --flash-attn
      --port 8000
      --host 0.0.0.0
    volumes:
      - ~/models/llama-cpp:/models:ro
    ports:
      - "9090:8000"
    networks:
      - strieber-net
    profiles:
      - specialists
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
              device_ids: ['0']
```

### Step 3: Update Model Configuration

```json
{
  "models": [
    {
      "id": "orchestrator",
      "url": "http://llama-server-orchestrator:8000",
      "owned_by": "nvidia",
      "role": "orchestrator"
    },
    {
      "id": "gpt-oss-120b",
      "url": "http://llama-server:8000",
      "owned_by": "local",
      "role": "large",
      "reasoning": {"effort": "high"}
    },
    {
      "id": "qwen3-8b",
      "url": "http://llama-server-small:8000",
      "owned_by": "local",
      "role": "small"
    },
    {
      "id": "deepseek-r1-math",
      "url": "http://llama-server-math:8000",
      "owned_by": "local",
      "role": "specialist",
      "specialty": "math"
    },
    {
      "id": "qwen-coder-7b",
      "url": "http://llama-server-coder:8000",
      "owned_by": "local",
      "role": "specialist",
      "specialty": "coding"
    }
  ]
}
```

---

## Part 10: Tool Call Shapes

### NVIDIA Orchestrator Tool Call Format

The Orchestrator-8B uses a specific tool/model definition schema:

```json
{
  "tools": [
    {
      "name": "enhance_reasoning",
      "description": "Analyze problems through code execution and intermediate results",
      "models": [
        {
          "id": "reasoner-1",
          "input_cost": 1.25,
          "output_cost": 10.0,
          "latency_seconds": 31,
          "description": "Strong understanding and reasoning capabilities"
        },
        {
          "id": "reasoner-2",
          "input_cost": 0.25,
          "output_cost": 2.0,
          "latency_seconds": 25,
          "description": "May hallucinate on difficult tasks"
        }
      ]
    },
    {
      "name": "answer",
      "description": "Provide final response to user query",
      "models": [
        {
          "id": "answer-1",
          "input_cost": 1.25,
          "output_cost": 10.0,
          "latency_seconds": 96,
          "description": "Strong functional calling abilities"
        },
        {
          "id": "answer-math-1",
          "input_cost": 0.9,
          "output_cost": 0.9,
          "latency_seconds": 13,
          "description": "Middle-school math capability"
        }
      ]
    },
    {
      "name": "search",
      "description": "Search for relevant information",
      "models": [
        {
          "id": "search-1",
          "input_cost": 1.25,
          "output_cost": 10.0,
          "latency_seconds": 22
        }
      ]
    }
  ]
}
```

**Orchestrator Output Format:**

```json
{
  "name": "answer",
  "arguments": {
    "model": "answer-math-1",
    "problem": "What is the integral of x^2?",
    "context_str": "Previous reasoning steps..."
  }
}
```

### Current strieber-gpt-3 Tool Call Format

The codebase uses OpenAI-compatible function calling:

**Tool Definition (sent to LLM):**

```json
{
  "type": "function",
  "name": "weather__get_forecast",
  "description": "Get weather forecast for a location",
  "parameters": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "City name or coordinates"
      },
      "days": {
        "type": "integer",
        "description": "Number of forecast days (1-7)"
      }
    },
    "required": ["location"]
  }
}
```

**LLM Tool Call Output:**

```json
{
  "id": "call_abc123",
  "type": "function",
  "function": {
    "name": "weather__get_forecast",
    "arguments": "{\"location\": \"San Francisco\", \"days\": 3}"
  }
}
```

**MCP Tool Execution:**

```json
{
  "method": "tools/call",
  "params": {
    "name": "weather__get_forecast",
    "arguments": {
      "location": "San Francisco",
      "days": 3
    }
  }
}
```

### Proposed Unified Routing Format

For the orchestrator integration, define a routing-specific tool schema:

```json
{
  "name": "route_request",
  "description": "Route the user request to the optimal resource",
  "parameters": {
    "type": "object",
    "properties": {
      "action": {
        "type": "string",
        "enum": ["use_model", "call_tool", "multi_step"],
        "description": "What type of action to take"
      },
      "target": {
        "type": "string",
        "description": "Model ID or tool name"
      },
      "reasoning": {
        "type": "string",
        "description": "Brief explanation of routing decision"
      },
      "confidence": {
        "type": "number",
        "description": "Confidence in routing decision (0-1)"
      },
      "tool_args": {
        "type": "object",
        "description": "If action=call_tool, the arguments to pass"
      }
    },
    "required": ["action", "target", "reasoning"]
  }
}
```

**Example Orchestrator Responses:**

```json
// Simple query → Small model
{
  "action": "use_model",
  "target": "qwen3-8b",
  "reasoning": "Factual question requiring simple lookup",
  "confidence": 0.95
}

// Weather query → Direct tool call
{
  "action": "call_tool",
  "target": "weather__get_forecast",
  "reasoning": "Direct tool request, no LLM needed",
  "confidence": 0.99,
  "tool_args": {"location": "NYC", "days": 1}
}

// Complex analysis → Large model
{
  "action": "use_model",
  "target": "gpt-oss-120b",
  "reasoning": "Multi-step reasoning required for business analysis",
  "confidence": 0.85
}

// Math problem → Specialist
{
  "action": "use_model",
  "target": "deepseek-r1-math",
  "reasoning": "Calculus problem benefits from math-specialized model",
  "confidence": 0.92
}
```

---

## Part 11: Model Selection Guide

### What We Want From Each Model Role

#### Orchestrator Model (8B)

**Primary Requirements:**
- **Fast inference** (<500ms for routing decision)
- **Accurate task classification** (understand query complexity)
- **Tool awareness** (know what tools can do without executing them)
- **No hallucination on routing** (when uncertain, route to large model)
- **User preference adherence** (respect fast/cheap/accurate trade-offs)

**NOT Required:**
- Deep domain knowledge (that's what routed models provide)
- Long context (queries are usually short)
- Creative writing ability
- Multi-turn conversation (stateless routing)

**Best Options:**

| Model | Why | VRAM (Q4) |
|-------|-----|-----------|
| **NVIDIA Orchestrator-8B** | Purpose-built, RL-trained for routing | 5-6GB |
| Qwen3-8B | Best tool-calling accuracy (F1: 0.933) | 5-6GB |
| Llama-3.1-8B | Stable, good instruction following | 5-6GB |

#### Small General Model (7-8B)

**Primary Requirements:**
- **Fast inference** (handle high throughput)
- **Good instruction following**
- **Factual accuracy** for simple queries
- **Conversational ability** for chat
- **Tool calling support** (may need to call tools)

**Best For:**
- "What is the capital of France?"
- "Summarize this short text"
- "Translate this sentence"
- Simple Q&A, greetings, chitchat

**Best Options:**

| Model | Strengths | VRAM (Q4) |
|-------|-----------|-----------|
| **Qwen3-8B** | Best tool calling, thinking mode | 5-6GB |
| Llama-3.1-8B | Stable, fast, multilingual | 5-6GB |
| Mistral-7B | Very fast, efficient | 4-5GB |

#### Large General Model (70B-120B+)

**Primary Requirements:**
- **Deep reasoning** for complex problems
- **Long context** for document analysis
- **High accuracy** on difficult tasks
- **Nuanced understanding** of complex requests

**Best For:**
- Complex multi-step reasoning
- Document analysis and synthesis
- Creative writing with constraints
- Tasks where errors are costly

**Your Current Model:**
- gpt-oss-120b (already deployed, 16GB VRAM)

**Alternatives:**

| Model | Strengths | VRAM |
|-------|-----------|------|
| Llama-3.1-70B | Strong reasoning, long context | 40GB+ |
| Qwen3-72B | Excellent all-around | 42GB+ |
| Llama-Nemotron-Ultra-253B | Top benchmark scores | Enterprise |

#### Math Specialist (7-14B)

**Primary Requirements:**
- **Step-by-step reasoning** (chain-of-thought)
- **Mathematical accuracy** (correct calculations)
- **Formula handling** (LaTeX, symbolic math)
- **Problem decomposition** (break down complex problems)

**Best For:**
- Calculus, algebra, statistics
- Physics word problems
- Financial calculations
- Logical proofs

**Best Options:**

| Model | Strengths | VRAM (Q4) |
|-------|-----------|-----------|
| **DeepSeek-R1-Distill-Qwen-7B** | Best math reasoning, MIT license | 5GB |
| DeepSeek-R1-Distill-Llama-8B | Strong math, Llama-based | 5GB |
| Qwen3-Math-7B | Good math with faster inference | 5GB |

**Benchmark (MATH-500):**
- DeepSeek-R1: 97.3%
- GPT-5: ~95%
- Qwen3-8B: ~85%

#### Coding Specialist (7-14B)

**Primary Requirements:**
- **Multi-language support** (Python, JS, Rust, etc.)
- **Code completion accuracy**
- **Bug detection and fixing**
- **Code explanation ability**

**Best For:**
- Code generation and completion
- Debugging and code review
- Refactoring suggestions
- Documentation generation

**Best Options:**

| Model | Strengths | Languages | VRAM (Q4) |
|-------|-----------|-----------|-----------|
| **Qwen2.5-Coder-7B** | HumanEval: 88.4%, 92 languages | All major | 5GB |
| DeepSeek-Coder-V2-Lite | 300+ languages, great completion | All | 5GB |
| Codestral-22B | Strong on benchmarks | Major | 13GB |

**Benchmark (HumanEval):**
- Qwen2.5-Coder-7B: 88.4%
- GPT-4: 87.1%
- DeepSeek-Coder-V2-Lite: 81.1%

---

## Part 12: Recommended Model Stack

### Minimal Setup (Single GPU, 24GB)

Best for: RTX 4090, A5000

```
┌─────────────────────────────────────────────┐
│         MINIMAL ORCHESTRATED STACK          │
├─────────────────────────────────────────────┤
│                                             │
│  Orchestrator: Qwen3-8B (Q4)         ~5GB  │
│  Small Model:  (shared with orchestrator)   │
│  Large Model:  gpt-oss-120b (Q4)    ~16GB  │
│                                             │
│  Strategy: Time-share orchestrator/small    │
│  Total VRAM: ~21GB                          │
└─────────────────────────────────────────────┘
```

**How it works:**
- Orchestrator and small model share same Qwen3-8B instance
- Simple queries: Orchestrator decides → same model responds
- Complex queries: Route to gpt-oss-120b

### Standard Setup (Single GPU, 48GB)

Best for: RTX A6000, 2x RTX 3090

```
┌─────────────────────────────────────────────┐
│         STANDARD ORCHESTRATED STACK         │
├─────────────────────────────────────────────┤
│                                             │
│  Orchestrator: Orchestrator-8B (Q4)   ~5GB │
│  Small Model:  Qwen3-8B (Q4)          ~5GB │
│  Large Model:  gpt-oss-120b (Q4)     ~16GB │
│  Math:         DeepSeek-R1-7B (Q4)    ~5GB │
│  Coder:        Qwen-Coder-7B (Q4)     ~5GB │
│                                             │
│  Total VRAM: ~36GB                          │
└─────────────────────────────────────────────┘
```

### Production Setup (Multi-GPU)

Best for: Multiple GPUs or cloud deployment

```
┌─────────────────────────────────────────────┐
│        PRODUCTION ORCHESTRATED STACK        │
├─────────────────────────────────────────────┤
│                                             │
│  GPU 0 (Routing + Small):                   │
│    Orchestrator: Orchestrator-8B      ~6GB │
│    Small:        Qwen3-8B             ~6GB │
│    Math:         DeepSeek-R1-7B       ~5GB │
│    Coder:        Qwen-Coder-7B        ~5GB │
│                                  Total: 22GB│
│                                             │
│  GPU 1 (Large Model):                       │
│    Large:        gpt-oss-120b        ~16GB │
│                                             │
│  GPU 2 (Specialists - Optional):            │
│    Vision:       Qwen3-VL-8B          ~6GB │
│    ReaderLM:     ReaderLM-v2          ~8GB │
│                                             │
└─────────────────────────────────────────────┘
```

---

## Part 13: Orchestrator System Prompt

### Full System Prompt for Routing

```
<|im_start|>system
You are a request router for an AI system. Your job is to analyze incoming queries and route them to the most appropriate resource.

## Available Resources

### Models
1. **small** (qwen3-8b): Fast, good for simple factual queries, chat, summaries
   - Latency: ~2s, Cost: Low
   - Best for: Greetings, simple Q&A, short summaries, translations

2. **large** (gpt-oss-120b): Powerful, best for complex reasoning
   - Latency: ~15s, Cost: High
   - Best for: Complex analysis, multi-step reasoning, nuanced writing

3. **math** (deepseek-r1): Specialized for mathematical reasoning
   - Latency: ~8s, Cost: Medium
   - Best for: Calculus, algebra, proofs, physics problems

4. **coder** (qwen-coder): Specialized for code tasks
   - Latency: ~5s, Cost: Medium
   - Best for: Code generation, debugging, code explanation

### Tools (can call directly without model)
1. **weather__get_forecast**: Get weather for a location
2. **web_search__search**: Search the web
3. **code_interpreter__execute_python**: Run Python code
4. **reader__fetch_url**: Fetch and parse web content
5. **zimage__generate**: Generate images from text

## User Preference
{preference}  // "fast" | "balanced" | "accurate"

## Your Task
Analyze the user query and respond with a JSON routing decision:

```json
{
  "action": "use_model" | "call_tool" | "multi_step",
  "target": "<model_id or tool_name>",
  "reasoning": "<brief explanation>",
  "confidence": <0.0-1.0>,
  "tool_args": {} // only if action=call_tool
}
```

## Routing Guidelines
- When uncertain, route to "large" model
- Direct tool calls for explicit tool requests (weather, search)
- Use "math" for anything with equations, calculations, proofs
- Use "coder" for code generation, debugging, explanation
- Use "small" for chitchat, simple facts, translations
- Use "large" for analysis, complex questions, creative writing
- Respect user preference: "fast" favors small/tools, "accurate" favors large/specialists
<|im_end|>
```

### Example Routing Conversations

**Query: "What's 2+2?"**
```json
{"action": "use_model", "target": "small", "reasoning": "Trivial arithmetic", "confidence": 0.99}
```

**Query: "What's the weather in Tokyo?"**
```json
{"action": "call_tool", "target": "weather__get_forecast", "reasoning": "Direct weather request", "confidence": 0.98, "tool_args": {"location": "Tokyo"}}
```

**Query: "Prove that the square root of 2 is irrational"**
```json
{"action": "use_model", "target": "math", "reasoning": "Mathematical proof requiring step-by-step reasoning", "confidence": 0.95}
```

**Query: "Write a Python function to sort a list using quicksort"**
```json
{"action": "use_model", "target": "coder", "reasoning": "Code generation task", "confidence": 0.97}
```

**Query: "Analyze the geopolitical implications of AI advancement on US-China relations"**
```json
{"action": "use_model", "target": "large", "reasoning": "Complex multi-faceted analysis requiring deep reasoning", "confidence": 0.92}
```

---

## Part 14: Concrete Integration with Your Models

This section provides the exact configuration to integrate NVIDIA Orchestrator-8B with your specific models: **gpt-oss-120b** and **Qwen3-VL-8B**.

### Which Qwen3-VL-8B? Instruct vs Thinking

**Recommendation: Use Qwen3-VL-8B-Instruct** for this orchestrated stack.

| Factor | Instruct | Thinking | Winner for Routing |
|--------|----------|----------|-------------------|
| **Response time** | 15-25% faster | Slower (reasoning overhead) | **Instruct** |
| **Throughput** | 1.5-2x more req/GPU | Lower | **Instruct** |
| **Token budget** | 16K max | 40K max | **Instruct** (cheaper) |
| **Simple tasks accuracy** | Equivalent | Equivalent | **Instruct** |
| **Complex reasoning** | Good | 10-18% better | Thinking |
| **Vision accuracy (MMMU)** | ~70% | ~72% | Marginal |

**Why Instruct wins for your use case:**

1. **The orchestrator handles routing** — complex reasoning goes to gpt-oss-120b anyway
2. **Qwen3-VL handles vision + simple tasks** — Instruct is equivalent on these
3. **Latency matters for routing** — faster small model = better UX
4. **Cost efficiency** — Instruct uses fewer tokens per response

**When to use Thinking mode instead:**
- If you're NOT using the orchestrator (single model doing everything)
- If most queries are complex reasoning without vision
- If you need auditable reasoning chains for compliance

### Exact tools.json Configuration

This is the NVIDIA ToolOrchestra format adapted for your stack:

```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "answer",
        "description": "Generate a response to the user query using an appropriate model",
        "parameters": {
          "type": "object",
          "properties": {
            "model": {
              "type": "string",
              "description": "Select the model based on task complexity and user preference:\n\n| Model | Cost/1M tokens | Latency | Best For |\n|-------|----------------|---------|----------|\n| small | $0 (local) | ~2s | Simple Q&A, chat, summaries |\n| large | $0 (local) | ~12s | Complex reasoning, analysis |\n| vision | $0 (local) | ~3s | Image understanding, OCR |",
              "enum": ["small", "large", "vision"]
            },
            "query": {
              "type": "string",
              "description": "The user's query to answer"
            }
          },
          "required": ["model", "query"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "use_tool",
        "description": "Call a tool directly without involving an LLM",
        "parameters": {
          "type": "object",
          "properties": {
            "tool": {
              "type": "string",
              "description": "Available tools:\n- weather: Get weather forecast\n- web_search: Search the internet\n- code_interpreter: Execute Python code\n- reader: Fetch and parse web pages\n- zimage: Generate images from text",
              "enum": ["weather", "web_search", "code_interpreter", "reader", "zimage"]
            },
            "args": {
              "type": "object",
              "description": "Arguments to pass to the tool"
            }
          },
          "required": ["tool", "args"]
        }
      }
    }
  ]
}
```

### Model Mapping (What the Orchestrator Sees → Actual Backend)

```json
{
  "model_mapping": {
    "small": {
      "backend_url": "http://llama-server-qwen-vl:8000",
      "model_id": "qwen3-vl-8b-instruct",
      "context_size": 32768,
      "supports_vision": true,
      "avg_latency_ms": 2000,
      "description": "Fast responses, vision capable, good for simple tasks"
    },
    "large": {
      "backend_url": "http://llama-server:8000",
      "model_id": "gpt-oss-120b",
      "context_size": 131072,
      "supports_vision": false,
      "avg_latency_ms": 12000,
      "description": "Deep reasoning, complex analysis, long context"
    },
    "vision": {
      "backend_url": "http://llama-server-qwen-vl:8000",
      "model_id": "qwen3-vl-8b-instruct",
      "context_size": 32768,
      "supports_vision": true,
      "avg_latency_ms": 3000,
      "description": "Image understanding, OCR, visual Q&A"
    }
  }
}
```

### Concrete System Prompt for Your Stack

```
<|im_start|>system
You are a request router. Analyze the user's query and decide how to handle it.

## Your Models

### small (Qwen3-VL-8B-Instruct)
- Latency: ~2 seconds
- Strengths: Fast, vision-capable, multilingual, tool calling
- Use for: Greetings, simple Q&A, translations, image descriptions, OCR
- Limitations: Struggles with complex multi-step reasoning

### large (gpt-oss-120b)
- Latency: ~12 seconds
- Strengths: Deep reasoning, long context (128K), nuanced writing
- Use for: Complex analysis, research, creative writing, difficult problems
- Limitations: No vision, slower, higher resource usage

### vision (Qwen3-VL-8B-Instruct)
- Same as "small" but explicitly for image/video tasks
- Use when: User provides an image or asks about visual content

## Your Tools (can skip LLM entirely)

| Tool | When to Use |
|------|-------------|
| weather | "What's the weather in X?" |
| web_search | "Search for X", "Find information about X" |
| code_interpreter | "Run this code", "Calculate X" |
| reader | "Summarize this URL", "What does X website say?" |
| zimage | "Generate an image of X", "Create a picture of X" |

## User Preference: {preference}
- "fast": Strongly prefer small model and direct tools
- "balanced": Use judgment based on complexity
- "thorough": Prefer large model for non-trivial queries

## Output Format

Respond with exactly one JSON object:

```json
{
  "action": "answer" | "use_tool",
  "model": "small" | "large" | "vision",  // if action=answer
  "tool": "weather" | "web_search" | ...,  // if action=use_tool
  "tool_args": {},                          // if action=use_tool
  "reasoning": "Brief explanation"
}
```

## Decision Rules

1. If query includes an image → action=answer, model=vision
2. If query is explicitly about weather/search/code → action=use_tool
3. If query is simple factual/chat → action=answer, model=small
4. If query requires analysis/reasoning/creativity → action=answer, model=large
5. When uncertain → default to model=large (better to over-deliver)
6. Respect user preference (fast→small, thorough→large)
<|im_end|>
```

### Concrete Request/Response Flow

**Step 1: User sends query**
```json
{
  "model": "orchestrator",
  "input": "What's in this image and should I be concerned?",
  "attachments": [{"type": "image", "url": "data:image/png;base64,..."}],
  "tools": [{"type": "weather"}, {"type": "web_search"}],
  "metadata": {"preference": "balanced"}
}
```

**Step 2: Orchestrator analyzes and routes**

Orchestrator receives the query with image attachment:
```
<|im_start|>user
Query: "What's in this image and should I be concerned?"
Attachments: [1 image]
<|im_end|>
```

Orchestrator responds:
```json
{
  "action": "answer",
  "model": "vision",
  "reasoning": "Image analysis required, using vision-capable model"
}
```

**Step 3: Responses API routes to Qwen3-VL**

```
POST http://llama-server-qwen-vl:8000/v1/chat/completions
{
  "model": "qwen3-vl-8b-instruct",
  "messages": [
    {"role": "user", "content": [
      {"type": "text", "text": "What's in this image and should I be concerned?"},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    ]}
  ]
}
```

**Step 4: Qwen3-VL responds**
```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "This image shows a mole on skin with irregular borders. The asymmetry and color variation suggest you should consult a dermatologist for evaluation..."
    }
  }]
}
```

### Another Example: Complex Reasoning

**User query:** "Compare the economic policies of Keynesianism vs Austrian economics and explain which would be better for addressing the current inflation situation"

**Orchestrator decision:**
```json
{
  "action": "answer",
  "model": "large",
  "reasoning": "Complex comparative analysis requiring deep economic understanding"
}
```

→ Routes to gpt-oss-120b for thorough response.

### Another Example: Direct Tool Call

**User query:** "What's the weather in San Francisco?"

**Orchestrator decision:**
```json
{
  "action": "use_tool",
  "tool": "weather",
  "tool_args": {"location": "San Francisco", "days": 1},
  "reasoning": "Direct weather request, no LLM needed"
}
```

→ Skips LLM entirely, calls MCP weather tool directly.

### Docker Compose for This Exact Setup

```yaml
services:
  # ORCHESTRATOR (NVIDIA Orchestrator-8B)
  llama-server-orchestrator:
    image: ghcr.io/ggerganov/llama.cpp:server-cuda
    command: >
      --model /models/nvidia_Orchestrator-8B-Q4_K_M.gguf
      --ctx-size 4096
      --n-gpu-layers 99
      --flash-attn
      --port 8000
      --host 0.0.0.0
    volumes:
      - ${MODELS_PATH:-~/models/llama-cpp}:/models:ro
    ports:
      - "9060:8000"
    networks:
      - strieber-net
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
              device_ids: ['0']

  # SMALL + VISION MODEL (Qwen3-VL-8B-Instruct)
  llama-server-qwen-vl:
    image: ghcr.io/ggerganov/llama.cpp:server-cuda
    command: >
      --model /models/qwen3-vl-8b-instruct-q4_k_m.gguf
      --ctx-size 32768
      --n-gpu-layers 99
      --flash-attn
      --port 8000
      --host 0.0.0.0
      --mmproj /models/qwen3-vl-8b-mmproj.gguf
    volumes:
      - ${MODELS_PATH:-~/models/llama-cpp}:/models:ro
    ports:
      - "9020:8000"
    networks:
      - strieber-net
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
              device_ids: ['0']

  # LARGE MODEL (gpt-oss-120b) - Your existing config
  llama-server:
    image: ghcr.io/ggerganov/llama.cpp:server-cuda
    command: >
      --model /models/gpt-oss-120b-q4_k_m.gguf
      --ctx-size 131072
      --n-gpu-layers 99
      --flash-attn
      --port 8000
      --host 0.0.0.0
    volumes:
      - ${MODELS_PATH:-~/models/llama-cpp}:/models:ro
    ports:
      - "9010:8000"
    networks:
      - strieber-net
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
              device_ids: ['0']
```

### VRAM Budget for This Setup

| Model | Quantization | VRAM |
|-------|--------------|------|
| Orchestrator-8B | Q4_K_M | ~5GB |
| Qwen3-VL-8B-Instruct | Q4_K_M | ~6GB |
| gpt-oss-120b | Q4_K_M | ~16GB |
| **Total** | | **~27GB** |

**Fits on:** Single RTX 4090 (24GB) if you time-share orchestrator with Qwen3-VL, or comfortably on 2x RTX 3090 (48GB total).

### Rust Integration Point

In `backend/responses-api/src/execution/executor.rs`, the orchestrator integration would look like:

```rust
impl Executor {
    pub async fn execute_with_orchestrator(
        &self,
        req: &CreateResponseRequest,
        previous_messages: Vec<ChatMessage>,
    ) -> Result<Response, ExecutionError> {
        // Step 1: Ask orchestrator how to route this request
        let routing = self.get_routing_decision(req).await?;

        match routing.action.as_str() {
            "use_tool" => {
                // Direct tool call - skip LLM entirely
                let tool_name = routing.tool.unwrap();
                let tool_args = routing.tool_args.unwrap_or_default();
                self.execute_tool_directly(&tool_name, tool_args).await
            }
            "answer" => {
                // Route to appropriate model
                let model_id = match routing.model.as_deref() {
                    Some("small") | Some("vision") => "qwen3-vl-8b-instruct",
                    Some("large") => "gpt-oss-120b",
                    _ => "gpt-oss-120b",  // fallback to large
                };

                // Modify request to use selected model
                let mut routed_req = req.clone();
                routed_req.model = model_id.to_string();

                // Execute with the routed model
                self.execute(&routed_req, previous_messages).await
            }
            _ => {
                // Unknown action, fallback to large model
                self.execute(req, previous_messages).await
            }
        }
    }

    async fn get_routing_decision(
        &self,
        req: &CreateResponseRequest,
    ) -> Result<RoutingDecision, ExecutionError> {
        let orchestrator_config = self.get_model("orchestrator")
            .ok_or(ExecutionError::ModelNotFound("orchestrator".into()))?;

        // Build orchestrator prompt
        let system_prompt = self.build_orchestrator_system_prompt(req);
        let user_content = self.format_query_for_routing(req);

        let chat_req = ChatCompletionRequest {
            model: "orchestrator".to_string(),
            messages: vec![
                ChatMessage::system(system_prompt),
                ChatMessage::user(user_content),
            ],
            temperature: Some(0.1),  // Low temp for consistent routing
            max_tokens: Some(256),   // Routing decisions are short
            ..Default::default()
        };

        let response = self.call_llm(&chat_req).await?;
        let content = response.choices[0].message.content.as_ref()
            .ok_or(ExecutionError::Llm("Empty orchestrator response".into()))?;

        // Parse JSON routing decision
        serde_json::from_str(content)
            .map_err(|e| ExecutionError::Llm(format!("Invalid routing JSON: {}", e)))
    }
}

#[derive(Debug, Deserialize)]
struct RoutingDecision {
    action: String,           // "answer" or "use_tool"
    model: Option<String>,    // "small", "large", "vision"
    tool: Option<String>,     // tool name if action=use_tool
    tool_args: Option<Value>, // tool arguments
    reasoning: String,        // explanation
}
```

### Expected Behavior

| Query Type | Orchestrator Decision | Actual Backend |
|------------|----------------------|----------------|
| "Hello!" | small | Qwen3-VL-8B |
| "What's in this image?" | vision | Qwen3-VL-8B |
| "Explain quantum entanglement" | large | gpt-oss-120b |
| "What's the weather in NYC?" | use_tool:weather | MCP (no LLM) |
| "Search for latest AI news" | use_tool:web_search | MCP (no LLM) |
| "Write me a novel opening" | large | gpt-oss-120b |
| "Translate 'hello' to Spanish" | small | Qwen3-VL-8B |
| "Analyze this contract" + image | vision | Qwen3-VL-8B |

### Cost/Latency Savings Estimate

Assuming 1000 queries with this distribution:
- 40% simple (routed to small): 400 × 2s = 800s
- 30% complex (routed to large): 300 × 12s = 3600s
- 15% vision (routed to vision): 150 × 3s = 450s
- 15% tools (direct call): 150 × 1s = 150s

**With orchestrator:** 5000s total + 1000 × 0.3s routing = 5300s
**Without orchestrator (all large):** 1000 × 12s = 12000s

**Savings: 56% reduction in total inference time**
