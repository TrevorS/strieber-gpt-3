# NVIDIA Orchestrator-8B: Integration Analysis for Local LLM Stack

**Date:** 2025-12-08
**Status:** Research Analysis

## Executive Summary

NVIDIA's Orchestrator-8B represents a paradigm shift from monolithic LLM deployments to **composite AI systems** where a small, specialized routing model coordinates larger models and tools. This document analyzes how a similar approach could transform the strieber-gpt-3 local stack—potentially fronting the 120B GPT-OSS model with a lightweight orchestrator to reduce costs, improve latency, and maintain accuracy.

> **Important Clarification:** The orchestrator IS an LLM (8B parameters). When we say "skip the large model" or "direct tool call," we mean:
> - The orchestrator LLM (8B, ~300ms) analyzes the query and decides the routing
> - For simple tool calls, we avoid calling the large LLM (120B, ~12s)
> - Total latency: ~0.8s (orchestrator + tool) vs ~25s (full round-trip through 120B)
>
> The orchestrator doesn't "skip all LLMs"—it *is* the LLM that decides whether a larger, slower model is needed.

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
    │    "What's the weather in NYC?"     (skip large model, ~12s saved)
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
            // Skip large model, execute tool directly (orchestrator already called)
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
- 15% tool-only (orchestrator only, skip large model)

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
  "reasoning": "Direct tool request, skip large model",
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
        "description": "Call a tool directly (skip large model, orchestrator handles routing)",
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

## Your Tools (can skip the large model, orchestrator decides)

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
  "reasoning": "Direct weather request, skip large model"
}
```

→ Skips the 120B model (~12s saved), orchestrator (8B, ~300ms) calls MCP weather tool directly.

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
                // Direct tool call - skip large model (orchestrator already called)
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
| "What's the weather in NYC?" | use_tool:weather | MCP (skip 120B) |
| "Search for latest AI news" | use_tool:web_search | MCP (skip 120B) |
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

---

## Part 15: Complete Rust Implementation

This section provides the full implementation for adding orchestrator routing to the existing `responses-api` codebase.

### File Structure

```
backend/responses-api/src/
├── orchestration/           # NEW MODULE
│   ├── mod.rs              # Module exports
│   ├── router.rs           # Orchestrator routing logic
│   ├── config.rs           # Orchestrator configuration
│   └── prompt.rs           # System prompt builder
├── execution/
│   ├── executor.rs         # MODIFIED: Add orchestration entry point
│   └── ...
├── config/
│   └── mod.rs              # MODIFIED: Add orchestrator config
└── ...
```

### Step 1: Create Orchestration Module

**File: `src/orchestration/mod.rs`**

```rust
//! Orchestration module for intelligent request routing.
//!
//! Uses a small LLM (NVIDIA Orchestrator-8B) to route requests to the
//! optimal backend model or direct tool call.

mod config;
mod prompt;
mod router;

pub use config::{OrchestratorConfig, ModelRole, UserPreference};
pub use router::{OrchestratorRouter, RoutingDecision, RoutingAction};
```

**File: `src/orchestration/config.rs`**

```rust
use serde::{Deserialize, Serialize};

/// Configuration for orchestrator routing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestratorConfig {
    /// Whether orchestration is enabled
    #[serde(default)]
    pub enabled: bool,

    /// Model ID for the orchestrator (must match a model in MODELS_CONFIG)
    #[serde(default = "default_orchestrator_model")]
    pub model_id: String,

    /// Default user preference
    #[serde(default)]
    pub default_preference: UserPreference,

    /// Model role mappings (orchestrator's "small" -> actual "qwen3-vl-8b")
    #[serde(default)]
    pub role_mapping: RoleMapping,

    /// Fallback model if orchestrator fails
    #[serde(default = "default_fallback_model")]
    pub fallback_model: String,

    /// Max tokens for orchestrator response
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,

    /// Temperature for orchestrator (low = consistent routing)
    #[serde(default = "default_temperature")]
    pub temperature: f32,
}

fn default_orchestrator_model() -> String { "orchestrator".to_string() }
fn default_fallback_model() -> String { "gpt-oss-120b".to_string() }
fn default_max_tokens() -> u32 { 256 }
fn default_temperature() -> f32 { 0.1 }

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model_id: default_orchestrator_model(),
            default_preference: UserPreference::Balanced,
            role_mapping: RoleMapping::default(),
            fallback_model: default_fallback_model(),
            max_tokens: default_max_tokens(),
            temperature: default_temperature(),
        }
    }
}

/// User preference for routing trade-offs.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum UserPreference {
    /// Prefer fast responses (use small model more often)
    Fast,
    /// Balance between speed and quality
    #[default]
    Balanced,
    /// Prefer quality (use large model more often)
    Thorough,
}

/// Maps orchestrator role names to actual model IDs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleMapping {
    /// Model for "small" role
    pub small: String,
    /// Model for "large" role
    pub large: String,
    /// Model for "vision" role (if different from small)
    pub vision: Option<String>,
}

impl Default for RoleMapping {
    fn default() -> Self {
        Self {
            small: "qwen3-vl-8b-instruct".to_string(),
            large: "gpt-oss-120b".to_string(),
            vision: None, // Uses small if not specified
        }
    }
}

/// Role a model plays in the orchestrated system.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelRole {
    /// The orchestrator model itself
    Orchestrator,
    /// Small/fast model for simple queries
    Small,
    /// Large model for complex reasoning
    Large,
    /// Vision-capable model
    Vision,
    /// Specialist model (math, code, etc.)
    Specialist,
}
```

**File: `src/orchestration/prompt.rs`**

```rust
use super::config::UserPreference;
use crate::models::{Input, InputItem, ContentPart, CreateResponseRequest};

/// Build the system prompt for the orchestrator.
pub fn build_system_prompt(
    preference: UserPreference,
    available_tools: &[String],
) -> String {
    let preference_str = match preference {
        UserPreference::Fast => "fast",
        UserPreference::Balanced => "balanced",
        UserPreference::Thorough => "thorough",
    };

    let tools_list = if available_tools.is_empty() {
        "No tools available".to_string()
    } else {
        available_tools.iter()
            .map(|t| format!("- {}", t))
            .collect::<Vec<_>>()
            .join("\n")
    };

    format!(r#"You are a request router. Analyze the user's query and decide how to handle it.

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
- Limitations: No vision, slower

### vision (Qwen3-VL-8B-Instruct)
- Same as "small" but explicitly for image/video tasks
- Use when: User provides an image or asks about visual content

## Your Tools (can skip the large model, orchestrator decides)
{tools_list}

## User Preference: {preference_str}
- "fast": Strongly prefer small model and direct tools
- "balanced": Use judgment based on complexity
- "thorough": Prefer large model for non-trivial queries

## Output Format

Respond with exactly one JSON object (no markdown, no explanation):

{{"action": "answer", "model": "small|large|vision", "reasoning": "..."}}
OR
{{"action": "use_tool", "tool": "tool_name", "tool_args": {{}}, "reasoning": "..."}}

## Decision Rules

1. If query includes an image → action=answer, model=vision
2. If query explicitly requests weather/search/etc → action=use_tool
3. If query is simple factual/chat → action=answer, model=small
4. If query requires analysis/reasoning/creativity → action=answer, model=large
5. When uncertain → default to model=large (better to over-deliver)
6. Respect user preference
"#, tools_list = tools_list, preference_str = preference_str)
}

/// Format the user's request for routing.
pub fn format_user_query(req: &CreateResponseRequest) -> String {
    let mut parts = Vec::new();

    // Extract text content
    let text = match &req.input {
        Input::Text(s) => s.clone(),
        Input::Items(items) => {
            items.iter()
                .filter_map(|item| match item {
                    InputItem::Message(msg) => Some(format!("{:?}: {:?}", msg.role, msg.content)),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n")
        }
        Input::Empty => String::new(),
    };

    if !text.is_empty() {
        parts.push(format!("Query: {}", text));
    }

    // Check for images
    let has_image = match &req.input {
        Input::Items(items) => items.iter().any(|item| {
            matches!(item, InputItem::Message(msg) => {
                match &msg.content {
                    crate::models::MessageContent::Parts(parts) => {
                        parts.iter().any(|p| matches!(p, ContentPart::InputImage { .. }))
                    }
                    _ => false,
                }
            })
        }),
        _ => false,
    };

    if has_image {
        parts.push("Attachments: [image]".to_string());
    }

    parts.join("\n")
}
```

**File: `src/orchestration/router.rs`**

```rust
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::ModelConfig;
use crate::execution::ExecutionError;
use crate::models::{ChatCompletionRequest, ChatMessage, ChatRole, ChatContent, CreateResponseRequest};

use super::config::{OrchestratorConfig, UserPreference};
use super::prompt::{build_system_prompt, format_user_query};

/// The routing decision made by the orchestrator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingDecision {
    pub action: RoutingAction,
    pub reasoning: String,
}

/// The action to take based on routing.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum RoutingAction {
    /// Route to a model
    Answer {
        model: String,
    },
    /// Call a tool directly
    UseTool {
        tool: String,
        tool_args: Value,
    },
}

/// Raw JSON response from orchestrator (for parsing).
#[derive(Debug, Deserialize)]
struct RawRoutingResponse {
    action: String,
    model: Option<String>,
    tool: Option<String>,
    tool_args: Option<Value>,
    reasoning: String,
}

/// The orchestrator router.
pub struct OrchestratorRouter {
    config: OrchestratorConfig,
}

impl OrchestratorRouter {
    pub fn new(config: OrchestratorConfig) -> Self {
        Self { config }
    }

    /// Check if orchestration is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get the preference from request metadata or use default.
    pub fn get_preference(&self, req: &CreateResponseRequest) -> UserPreference {
        req.metadata
            .as_ref()
            .and_then(|m| m.get("preference"))
            .and_then(|v| v.as_str())
            .and_then(|s| match s {
                "fast" => Some(UserPreference::Fast),
                "balanced" => Some(UserPreference::Balanced),
                "thorough" => Some(UserPreference::Thorough),
                _ => None,
            })
            .unwrap_or(self.config.default_preference)
    }

    /// Build the chat completion request for the orchestrator.
    pub fn build_routing_request(
        &self,
        req: &CreateResponseRequest,
        available_tools: Vec<String>,
    ) -> ChatCompletionRequest {
        let preference = self.get_preference(req);
        let system_prompt = build_system_prompt(preference, &available_tools);
        let user_query = format_user_query(req);

        ChatCompletionRequest {
            model: self.config.model_id.clone(),
            messages: vec![
                ChatMessage {
                    role: ChatRole::System,
                    content: Some(ChatContent::Text(system_prompt)),
                    reasoning_content: None,
                    tool_calls: None,
                    tool_call_id: None,
                },
                ChatMessage {
                    role: ChatRole::User,
                    content: Some(ChatContent::Text(user_query)),
                    reasoning_content: None,
                    tool_calls: None,
                    tool_call_id: None,
                },
            ],
            tools: None,
            tool_choice: None,
            max_tokens: Some(self.config.max_tokens),
            temperature: Some(self.config.temperature),
            top_p: None,
            stream: false,
        }
    }

    /// Parse the orchestrator's response into a routing decision.
    pub fn parse_response(&self, content: &str) -> Result<RoutingDecision, ExecutionError> {
        // Try to extract JSON from the response (handle markdown code blocks)
        let json_str = if content.contains("```") {
            content
                .split("```")
                .nth(1)
                .map(|s| s.trim_start_matches("json").trim())
                .unwrap_or(content)
        } else {
            content.trim()
        };

        let raw: RawRoutingResponse = serde_json::from_str(json_str)
            .map_err(|e| ExecutionError::Llm(format!("Invalid routing JSON: {} in '{}'", e, json_str)))?;

        let action = match raw.action.as_str() {
            "answer" => {
                let model = raw.model.unwrap_or_else(|| "large".to_string());
                RoutingAction::Answer { model }
            }
            "use_tool" => {
                let tool = raw.tool
                    .ok_or_else(|| ExecutionError::Llm("use_tool action requires tool field".into()))?;
                let tool_args = raw.tool_args.unwrap_or(Value::Object(Default::default()));
                RoutingAction::UseTool { tool, tool_args }
            }
            other => {
                return Err(ExecutionError::Llm(format!("Unknown routing action: {}", other)));
            }
        };

        Ok(RoutingDecision {
            action,
            reasoning: raw.reasoning,
        })
    }

    /// Map an orchestrator role to an actual model ID.
    pub fn resolve_model(&self, role: &str) -> String {
        match role {
            "small" => self.config.role_mapping.small.clone(),
            "large" => self.config.role_mapping.large.clone(),
            "vision" => self.config.role_mapping.vision
                .clone()
                .unwrap_or_else(|| self.config.role_mapping.small.clone()),
            _ => self.config.fallback_model.clone(),
        }
    }

    /// Get the fallback model ID.
    pub fn fallback_model(&self) -> &str {
        &self.config.fallback_model
    }
}
```

### Step 2: Modify Executor

**File: `src/execution/executor.rs`** (additions)

```rust
use crate::orchestration::{OrchestratorRouter, RoutingAction};

impl Executor {
    /// Execute a request with orchestrator routing.
    ///
    /// If orchestration is enabled, this:
    /// 1. Asks the orchestrator which model/tool to use
    /// 2. Routes to the appropriate backend
    /// 3. Falls back to large model on error
    pub async fn execute_orchestrated(
        &self,
        req: &CreateResponseRequest,
        previous_messages: Vec<ChatMessage>,
        orchestrator: &OrchestratorRouter,
    ) -> Result<Response, ExecutionError> {
        // If orchestrator is disabled, use normal execution
        if !orchestrator.is_enabled() {
            return self.execute(req, previous_messages).await;
        }

        // Get available tool names for the orchestrator prompt
        let available_tools: Vec<String> = self.mcp
            .list_all_tools()
            .await
            .iter()
            .map(|t| t.name.to_string())
            .collect();

        // Ask orchestrator for routing decision
        let routing_req = orchestrator.build_routing_request(req, available_tools);

        tracing::info!(
            model = %routing_req.model,
            "Calling orchestrator for routing decision"
        );

        let routing_result = match self.call_llm(&routing_req).await {
            Ok(resp) => {
                let content = resp.choices.get(0)
                    .and_then(|c| c.message.content.as_ref())
                    .map(|c| match c {
                        ChatContent::Text(t) => t.as_str(),
                        _ => "",
                    })
                    .unwrap_or("");

                orchestrator.parse_response(content)
            }
            Err(e) => {
                tracing::warn!("Orchestrator call failed: {}, falling back", e);
                Err(e)
            }
        };

        // Handle routing decision
        match routing_result {
            Ok(decision) => {
                tracing::info!(
                    action = ?decision.action,
                    reasoning = %decision.reasoning,
                    "Orchestrator routing decision"
                );

                match decision.action {
                    RoutingAction::Answer { model } => {
                        // Route to the selected model
                        let actual_model = orchestrator.resolve_model(&model);
                        tracing::info!(
                            role = %model,
                            actual_model = %actual_model,
                            "Routing to model"
                        );

                        let mut routed_req = req.clone();
                        routed_req.model = actual_model;
                        self.execute(&routed_req, previous_messages).await
                    }
                    RoutingAction::UseTool { tool, tool_args } => {
                        // Direct tool call - skip large model (orchestrator already called)
                        tracing::info!(
                            tool = %tool,
                            "Direct tool call, skipping large model"
                        );
                        self.execute_direct_tool(&tool, tool_args, req).await
                    }
                }
            }
            Err(e) => {
                // Fallback to large model
                tracing::warn!(
                    error = %e,
                    fallback = %orchestrator.fallback_model(),
                    "Routing failed, using fallback"
                );
                let mut fallback_req = req.clone();
                fallback_req.model = orchestrator.fallback_model().to_string();
                self.execute(&fallback_req, previous_messages).await
            }
        }
    }

    /// Execute a tool call directly (orchestrator routes, skips large model).
    async fn execute_direct_tool(
        &self,
        tool_name: &str,
        args: Value,
        original_req: &CreateResponseRequest,
    ) -> Result<Response, ExecutionError> {
        tracing::info!(tool = %tool_name, "Executing direct tool call");

        let result = self.mcp.call_tool(tool_name, args).await?;

        // Build a response with the tool result
        let mut text_parts = Vec::new();
        for content in &result.content {
            if let rmcp::model::RawContent::Text(tc) = &content.raw {
                text_parts.push(tc.text.as_str());
            }
        }
        let text = text_parts.join("\n");

        // Create a synthetic response
        Ok(crate::translation::build_tool_only_response(
            original_req,
            tool_name,
            text,
        ))
    }
}
```

### Step 3: Update Config Module

**File: `src/config/mod.rs`** (additions)

```rust
use crate::orchestration::OrchestratorConfig;

/// Main configuration for the Responses API service.
#[derive(Debug, Clone)]
pub struct Config {
    // ... existing fields ...

    /// Orchestrator configuration
    pub orchestrator: OrchestratorConfig,
}

impl Config {
    pub fn from_env() -> Self {
        let mut config = Self::default();

        // ... existing env parsing ...

        // Parse orchestrator configuration
        if let Ok(json) = env::var("ORCHESTRATOR_CONFIG") {
            match serde_json::from_str::<OrchestratorConfig>(&json) {
                Ok(orch_config) => config.orchestrator = orch_config,
                Err(e) => tracing::error!("Failed to parse ORCHESTRATOR_CONFIG: {}", e),
            }
        }

        config
    }
}
```

### Step 4: Environment Configuration

**Docker Compose environment variables:**

```yaml
services:
  responses-api:
    environment:
      MODELS_CONFIG: |
        {
          "models": [
            {
              "id": "orchestrator",
              "url": "http://llama-server-orchestrator:8000",
              "owned_by": "nvidia"
            },
            {
              "id": "qwen3-vl-8b-instruct",
              "url": "http://llama-server-qwen-vl:8000",
              "owned_by": "local",
              "supports_vision": true
            },
            {
              "id": "gpt-oss-120b",
              "url": "http://llama-server:8000",
              "owned_by": "local",
              "reasoning": {"effort": "high"}
            }
          ]
        }

      ORCHESTRATOR_CONFIG: |
        {
          "enabled": true,
          "model_id": "orchestrator",
          "default_preference": "balanced",
          "role_mapping": {
            "small": "qwen3-vl-8b-instruct",
            "large": "gpt-oss-120b",
            "vision": "qwen3-vl-8b-instruct"
          },
          "fallback_model": "gpt-oss-120b",
          "max_tokens": 256,
          "temperature": 0.1
        }

      # MCP tools config (unchanged)
      MCP_CONFIG: |
        {
          "servers": [
            {"name": "weather", "url": "http://mcp-weather:8000/mcp", "builtin_type": "weather"},
            {"name": "web_search", "url": "http://mcp-web-search:8000/mcp", "builtin_type": "web_search"},
            {"name": "code_interpreter", "url": "http://mcp-code-interpreter:8000/mcp", "builtin_type": "code_interpreter"}
          ]
        }
```

### Step 5: Download Models

```bash
# 1. NVIDIA Orchestrator-8B (Q4_K_M = 5GB)
huggingface-cli download bartowski/nvidia_Orchestrator-8B-GGUF \
  --include "nvidia_Orchestrator-8B-Q4_K_M.gguf" \
  --local-dir ~/models/llama-cpp/

# 2. Qwen3-VL-8B-Instruct (need both model and mmproj)
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct-GGUF \
  --include "qwen3-vl-8b-instruct-q4_k_m.gguf" \
  --include "qwen3-vl-8b-mmproj.gguf" \
  --local-dir ~/models/llama-cpp/

# 3. gpt-oss-120b (your existing model, already present)
```

### Step 6: Run the Stack

```bash
# Start all services
docker compose up -d llama-server-orchestrator llama-server-qwen-vl llama-server responses-api

# Check logs
docker compose logs -f responses-api
```

### Complete Request Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATED REQUEST FLOW                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. User Request                                                            │
│     POST /v1/responses                                                      │
│     {"model": "auto", "input": "What's 2+2?"}                              │
│                                                                             │
│  2. Handler (handlers.rs)                                                   │
│     └─► execute_orchestrated(req, prev_msgs, orchestrator)                 │
│                                                                             │
│  3. Orchestrator Router (router.rs)                                         │
│     ├─► build_routing_request(req, available_tools)                        │
│     ├─► call_llm(orchestrator_req)  ──────────────────┐                    │
│     │                                                  │                    │
│     │   ┌──────────────────────────────────────────────▼──────────────┐   │
│     │   │  llama-server-orchestrator:8000                             │   │
│     │   │  NVIDIA Orchestrator-8B                                      │   │
│     │   │  Input: "Query: What's 2+2?"                                │   │
│     │   │  Output: {"action":"answer","model":"small","reasoning":..} │   │
│     │   └──────────────────────────────────────────────┬──────────────┘   │
│     │                                                  │                    │
│     └─► parse_response(content)  ◄─────────────────────┘                   │
│         └─► RoutingDecision { action: Answer { model: "small" }, ... }     │
│                                                                             │
│  4. Route to Model                                                          │
│     └─► resolve_model("small") → "qwen3-vl-8b-instruct"                   │
│     └─► execute(routed_req, prev_msgs)                                     │
│                                                                             │
│     ┌──────────────────────────────────────────────────────────────────┐   │
│     │  llama-server-qwen-vl:8000                                       │   │
│     │  Qwen3-VL-8B-Instruct                                            │   │
│     │  Input: "What's 2+2?"                                            │   │
│     │  Output: "2 + 2 = 4"                                             │   │
│     └──────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  5. Return Response                                                         │
│     {"id": "resp_...", "output": [{"type": "message", "content": "4"}]}   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Alternative Flow: Direct Tool Call

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      DIRECT TOOL CALL FLOW                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. User: "What's the weather in NYC?"                                      │
│                                                                             │
│  2. Orchestrator decides:                                                   │
│     {"action": "use_tool", "tool": "weather__get_forecast",                │
│      "tool_args": {"location": "NYC"}}                                     │
│                                                                             │
│  3. Execute Direct Tool (SKIP LARGE MODEL!)                                 │
│     └─► mcp.call_tool("weather__get_forecast", {"location": "NYC"})        │
│                                                                             │
│     ┌──────────────────────────────────────────────────────────────────┐   │
│     │  mcp-weather:8000                                                │   │
│     │  Returns: {"temp": 72, "conditions": "sunny", ...}               │   │
│     └──────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  4. Return Response (built from tool result)                                │
│     {"output": [{"type": "message", "content": "NYC: 72°F, sunny"}]}       │
│                                                                             │
│  ⚡ Total LLM calls: 1 (orchestrator only)                                  │
│  ⚡ Latency: ~0.3s (orchestrator) + ~0.5s (tool) = ~0.8s                   │
│     vs. ~12s if routed to gpt-oss-120b                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Testing the Integration

```bash
# Test routing to small model
curl -X POST http://localhost:9150/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "input": "Hello! How are you?",
    "metadata": {"preference": "fast"}
  }'

# Test routing to large model
curl -X POST http://localhost:9150/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "input": "Explain the philosophical implications of Gödels incompleteness theorems",
    "metadata": {"preference": "thorough"}
  }'

# Test direct tool call
curl -X POST http://localhost:9150/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "input": "What is the weather in San Francisco?",
    "tools": [{"type": "weather"}]
  }'
```

### Monitoring & Debugging

Add these log queries to monitor routing:

```bash
# Watch routing decisions
docker compose logs -f responses-api | grep "Orchestrator routing"

# Watch model selection
docker compose logs -f responses-api | grep "Routing to model"

# Watch direct tool calls
docker compose logs -f responses-api | grep "Direct tool call"
```

### Summary: What Changes

| File | Change |
|------|--------|
| `src/orchestration/mod.rs` | NEW - Module exports |
| `src/orchestration/config.rs` | NEW - Config types |
| `src/orchestration/prompt.rs` | NEW - Prompt builder |
| `src/orchestration/router.rs` | NEW - Routing logic |
| `src/execution/executor.rs` | ADD - `execute_orchestrated()` method |
| `src/config/mod.rs` | ADD - `orchestrator` field, env parsing |
| `src/server/handlers.rs` | MODIFY - Call `execute_orchestrated` |
| `src/lib.rs` | ADD - `mod orchestration;` |
| `compose.yml` | ADD - orchestrator service |
| Environment | ADD - `ORCHESTRATOR_CONFIG` |

---

## Part 16: Complete Docker Service Setup

This section provides production-ready Docker Compose service definitions that follow the existing `compose.yml` patterns exactly.

### Port Assignment Strategy

The existing codebase uses a structured port scheme:

| Port Range | Purpose | Examples |
|------------|---------|----------|
| 9010-9019 | Main LLM inference | 9010: gpt-oss-120b |
| 9020-9029 | Vision/multimodal models | 9020: Qwen3-VL-2B |
| 9030-9039 | Specialized models | 9030: ReaderLM |
| 9050-9059 | Embeddings | 9050: EmbeddingGemma |
| **9060-9069** | **Orchestrator** (NEW) | **9060: Orchestrator-8B** |
| **9070-9079** | **Small general models** (NEW) | **9070: Qwen3-8B** |
| 9100-9199 | MCP tool servers | 9100-9141: Various MCP |
| 9150 | Responses API | 9150: responses-api |
| 9200-9299 | Web UIs | 9200: Open WebUI |
| 9300-9399 | Frontend | 9300: Chat UI |

### Service Definition: NVIDIA Orchestrator-8B

Add this to `compose.yml`:

```yaml
  # ==========================================================================
  # llama-server-orchestrator: NVIDIA Orchestrator-8B for intelligent routing
  # ABOUTME: Purpose-trained routing model that decides which backend to use
  # Fine-tuned from Qwen3-8B via GRPO for multi-objective optimization
  # ==========================================================================
  llama-server-orchestrator:
    build:
      context: .
      dockerfile: Dockerfile.llamacpp
      args:
        CUDA_VERSION: "13.0.1"
        UBUNTU_VERSION: "22.04"
        CUDA_DOCKER_ARCH: "121"  # Blackwell GB10

    image: strieber-llama-server:latest  # Reuse same image
    container_name: strieber-llama-server-orchestrator

    # Restart policy
    restart: unless-stopped

    # GPU configuration (shares GPU 0 with other services)
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

    # Shared memory for CUDA operations
    shm_size: 4g  # Small model, minimal overhead

    # IPC mode for better GPU communication
    ipc: host

    # Port mapping: 9060 = orchestrator routing layer
    ports:
      - "9060:8000"

    # Volume mounts
    volumes:
      # Models directory: NVIDIA Orchestrator GGUF
      - /home/trevor/models/llama-cpp/orchestrator:/models

    # Environment variables
    environment:
      - LLAMA_ARG_HOST=0.0.0.0
      - LLAMA_ARG_PORT=8000
      - LLAMA_LOG_VERBOSITY=0
      - LLAMA_LOG_TIMESTAMPS=true
      - LLAMA_LOG_PREFIX=true

    # Server command optimized for Orchestrator-8B routing decisions
    # Low temperature for consistent routing, limited context (queries are short)
    command:
      - "-m"
      - "/models/nvidia_Orchestrator-8B-Q4_K_M.gguf"
      - "--port"
      - "8000"
      - "--host"
      - "0.0.0.0"
      - "-c"
      - "8192"                              # Context: 8K tokens (sufficient for routing)
      - "--n-gpu-layers"
      - "999"                               # Auto-detect layers for GPU offload
      - "--jinja"                           # Enable chat template support
      - "-b"
      - "512"                               # Batch size: standard
      - "-ub"
      - "512"                               # Ubatch size: physical batch
      - "--flash-attn"
      - "on"                                # Flash attention: efficiency
      - "--cont-batching"                   # Continuous batching
      - "--parallel"
      - "4"                                 # Higher parallelism for routing (fast decisions)
      - "--no-mmap"                         # Disable mmap
      - "-t"
      - "0.1"                               # Temperature: low for consistent routing

    # Health check
    healthcheck:
      test: ["CMD", "curl", "-sf", "http://localhost:8000/health"]
      interval: 30s
      timeout: 5s
      start_period: 30s
      retries: 3

    # Network configuration
    networks:
      - strieber-net
```

### Service Definition: Qwen3-VL-8B-Instruct (Upgraded)

Replace the existing `llama-server-qwen-vl` with the 8B version:

```yaml
  # ==========================================================================
  # llama-server-qwen-vl: Qwen3-VL-8B-Instruct vision-language model
  # ABOUTME: Upgraded from 2B to 8B for better quality on simple tasks
  # Serves dual purpose: vision tasks AND small general queries via orchestrator
  # ==========================================================================
  llama-server-qwen-vl:
    build:
      context: .
      dockerfile: Dockerfile.llamacpp
      args:
        CUDA_VERSION: "13.0.1"
        UBUNTU_VERSION: "22.04"
        CUDA_DOCKER_ARCH: "121"  # Blackwell GB10

    image: strieber-llama-server:latest  # Reuse same image
    container_name: strieber-llama-server-qwen-vl

    # Restart policy
    restart: unless-stopped

    # GPU configuration (shares GPU 0 with other llama-server instances)
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

    # Shared memory for CUDA operations
    shm_size: 8g  # Larger than 2B version

    # IPC mode for better GPU communication
    ipc: host

    # Port mapping: 9020 = vision/multimodal + small general tasks
    ports:
      - "9020:8000"

    # Volume mounts
    volumes:
      # Models directory: Qwen-VL 8B model and mmproj
      - /home/trevor/models/llama-cpp/qwen-vl:/models

    # Environment variables
    environment:
      - LLAMA_ARG_HOST=0.0.0.0
      - LLAMA_ARG_PORT=8000
      - LLAMA_LOG_VERBOSITY=0
      - LLAMA_LOG_TIMESTAMPS=true
      - LLAMA_LOG_PREFIX=true

    # Server command optimized for Qwen3-VL-8B-Instruct
    # Handles both vision tasks and general queries routed by orchestrator
    command:
      - "-m"
      - "/models/Qwen3-VL-8B-Instruct-Q4_K_M.gguf"
      - "--mmproj"
      - "/models/mmproj-Qwen3-VL-8B-F16.gguf"   # Multimodal projector for vision
      - "--port"
      - "8000"
      - "--host"
      - "0.0.0.0"
      - "-c"
      - "32768"                               # Context: 32k tokens
      - "--n-gpu-layers"
      - "999"                                 # Auto-detect layers for GPU offload
      - "--jinja"                             # Enable chat template support
      - "-b"
      - "512"                                 # Batch size
      - "-ub"
      - "512"                                 # Ubatch size
      - "--flash-attn"
      - "on"                                  # Flash attention: efficiency
      - "--cont-batching"                     # Continuous batching
      - "--parallel"
      - "2"                                   # Concurrent requests
      - "--no-mmap"                           # Disable mmap

    # Health check
    healthcheck:
      test: ["CMD", "curl", "-sf", "http://localhost:8000/health"]
      interval: 30s
      timeout: 5s
      start_period: 60s                       # Longer startup for 8B model
      retries: 3

    # Network configuration
    networks:
      - strieber-net
```

### Model Download Commands

```bash
# Create model directories
mkdir -p ~/models/llama-cpp/orchestrator
mkdir -p ~/models/llama-cpp/qwen-vl

# 1. Download NVIDIA Orchestrator-8B (Q4_K_M = 5GB)
huggingface-cli download bartowski/nvidia_Orchestrator-8B-GGUF \
  --include "nvidia_Orchestrator-8B-Q4_K_M.gguf" \
  --local-dir ~/models/llama-cpp/orchestrator

# 2. Download Qwen3-VL-8B-Instruct (model + mmproj)
#    Note: File names may vary - check HuggingFace for exact names
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct-GGUF \
  --include "*.gguf" \
  --local-dir ~/models/llama-cpp/qwen-vl

# Alternative: If Qwen official GGUF not available, use bartowski's conversion
huggingface-cli download bartowski/Qwen3-VL-8B-Instruct-GGUF \
  --include "*Q4_K_M*.gguf" \
  --include "*mmproj*.gguf" \
  --local-dir ~/models/llama-cpp/qwen-vl
```

### Updated responses-api Service

Update the `responses-api` service to include orchestrator configuration:

```yaml
  # ==========================================================================
  # responses-api: OpenAI Responses API adapter with Orchestrator routing
  # ABOUTME: Now includes intelligent routing via NVIDIA Orchestrator-8B
  # ==========================================================================
  responses-api:
    build:
      context: ./backend/responses-api
      dockerfile: Dockerfile
    image: strieber-responses-api:latest
    container_name: strieber-responses-api
    restart: unless-stopped
    ports:
      - "9150:8000"
    environment:
      - PORT=8000
      - HOST=0.0.0.0
      # Models: Include orchestrator + small + large
      - MODELS_CONFIG={"models":[{"id":"orchestrator","url":"http://llama-server-orchestrator:8000","owned_by":"nvidia"},{"id":"qwen3-vl-8b-instruct","url":"http://llama-server-qwen-vl:8000","supports_vision":true,"supported_tools":[]},{"id":"gpt-oss-120b","url":"http://llama-server:8000","reasoning":{"effort":"high"}}]}
      # Orchestrator configuration
      - ORCHESTRATOR_CONFIG={"enabled":true,"model_id":"orchestrator","default_preference":"balanced","role_mapping":{"small":"qwen3-vl-8b-instruct","large":"gpt-oss-120b","vision":"qwen3-vl-8b-instruct"},"fallback_model":"gpt-oss-120b","max_tokens":256,"temperature":0.1}
      # MCP servers (unchanged)
      - MCP_CONFIG={"servers":[{"name":"weather","url":"http://mcp-weather:8000/mcp","builtin_type":"weather"},{"name":"web_search","url":"http://mcp-web-search:8000/mcp","builtin_type":"web_search"},{"name":"code_interpreter","url":"http://mcp-code-interpreter:8000/mcp","builtin_type":"code_interpreter"},{"name":"reader","url":"http://mcp-reader:8000/mcp","builtin_type":"reader"},{"name":"zimage","url":"http://mcp-comfy-zimage:8000/mcp","builtin_type":"zimage_turbo"}]}
      - MAX_TOOL_ITERATIONS=10
      - TIMEOUT_SECS=300
      - RESPONSE_TTL_SECS=3600
      - RUST_LOG=responses_api=info,tower_http=info
    volumes:
      - responses-data:/data
    depends_on:
      llama-server:
        condition: service_healthy
      llama-server-orchestrator:
        condition: service_healthy
      llama-server-qwen-vl:
        condition: service_healthy
      mcp-weather:
        condition: service_started
      mcp-web-search:
        condition: service_started
      mcp-code-interpreter:
        condition: service_started
      mcp-reader:
        condition: service_started
      mcp-comfy-zimage:
        condition: service_started
    healthcheck:
      test: ["CMD", "curl", "-sf", "http://localhost:8000/health"]
      interval: 30s
      timeout: 5s
      start_period: 10s
      retries: 3
    networks:
      - strieber-net
```

### VRAM Budget Analysis

| Service | Model | VRAM (Q4_K_M) | Purpose |
|---------|-------|---------------|---------|
| llama-server | gpt-oss-120b | ~16GB | Complex reasoning |
| llama-server-orchestrator | Orchestrator-8B | ~6GB | Routing decisions |
| llama-server-qwen-vl | Qwen3-VL-8B | ~8GB | Simple tasks + vision |
| llama-server-readerlm | ReaderLM-v2 | ~2GB | HTML→Markdown |
| embeddinggemma | EmbeddingGemma | ~1GB | Embeddings |
| **Total** | | **~33GB** | |

**Note:** DGX Spark has 128GB unified memory, so this fits comfortably with room for:
- ComfyUI image generation
- KV cache overhead
- Multiple concurrent requests

### Startup Order

The services should start in this order (handled by `depends_on`):

```
1. llama-server (gpt-oss-120b)           ← Base, always needed
2. llama-server-orchestrator             ← Routing layer
3. llama-server-qwen-vl                  ← Small model for routed queries
4. llama-server-readerlm                 ← Specialized (reader tool)
5. mcp-* servers                         ← Tools
6. responses-api                         ← API layer (depends on all above)
7. chat-ui                               ← Frontend (depends on responses-api)
```

### Quick Start Commands

```bash
# Build custom llama.cpp image (if not already built)
docker compose build llama-server

# Start the full orchestrated stack
docker compose up -d \
  llama-server \
  llama-server-orchestrator \
  llama-server-qwen-vl \
  mcp-weather \
  mcp-web-search \
  mcp-code-interpreter \
  responses-api \
  chat-ui

# Watch startup progress
docker compose logs -f llama-server-orchestrator llama-server-qwen-vl

# Test orchestrator is working
curl -s http://localhost:9060/health | jq .

# Test Qwen-VL is working
curl -s http://localhost:9020/health | jq .

# Test full stack via responses-api
curl -X POST http://localhost:9150/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "input": "Hello, what can you help me with?"}' | jq .
```

### Debugging Tips

```bash
# Check GPU allocation across services
nvidia-smi

# View orchestrator routing logs
docker compose logs responses-api | grep -E "(Orchestrator|Routing|route)"

# Check model load status
curl -s http://localhost:9060/v1/models | jq .
curl -s http://localhost:9020/v1/models | jq .

# Test orchestrator directly (bypass responses-api)
curl -X POST http://localhost:9060/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia_Orchestrator-8B-Q4_K_M",
    "messages": [
      {"role": "system", "content": "You are a router. Return JSON: {\"action\":\"answer\",\"model\":\"small\"}"},
      {"role": "user", "content": "What is 2+2?"}
    ],
    "temperature": 0.1,
    "max_tokens": 100
  }' | jq .
```

### Expected Latency Breakdown

| Route | Orchestrator | Backend | Tool | Total |
|-------|--------------|---------|------|-------|
| Simple → small | 300ms | 2000ms | - | **2.3s** |
| Complex → large | 300ms | 12000ms | - | **12.3s** |
| Tool → weather | 300ms | - | 500ms | **0.8s** |
| Vision → small | 300ms | 3000ms | - | **3.3s** |

**Comparison without orchestrator:**
- All queries → large: 12000ms
- Savings on simple queries: 12.3s → 2.3s = **81% faster**
- Savings on tool queries: 12.3s → 0.8s = **93% faster**

---

## Part 17: Orchestrator-8B Serving Best Practices

This section documents the specific requirements and best practices for serving NVIDIA Orchestrator-8B with llama.cpp, based on research from official sources.

### Model Architecture Details

| Property | Value | Notes |
|----------|-------|-------|
| **Base Model** | Qwen3-8B | Uses Qwen chat template |
| **Parameters** | 8B | Decoder-only transformer |
| **Chat Template** | `<\|im_start\|>...<\|im_end\|>` | Qwen/ChatML format |
| **Tool Calling** | Hermes format | Single tool-call per turn only |
| **Training** | GRPO (RL) | Multi-objective reward optimization |

### GGUF Quantization Options

From [bartowski/nvidia_Orchestrator-8B-GGUF](https://huggingface.co/bartowski/nvidia_Orchestrator-8B-GGUF):

| Quantization | Size | Quality | Recommended For |
|--------------|------|---------|-----------------|
| **Q4_K_M** | 5.03GB | Good | **Production (recommended)** |
| Q5_K_M | 5.85GB | High | Quality-sensitive deployments |
| Q6_K | 6.73GB | Very High | Maximum accuracy |
| Q8_0 | 8.71GB | Excellent | Testing/validation |
| IQ4_NL | ~4.5GB | Good | ARM/AVX with online repacking |

**Recommendation:** Use **Q4_K_M** for production. It provides the best balance of quality, speed, and VRAM usage (~6GB).

### Critical Constraint: Single Tool-Call Per Turn

The Orchestrator-8B model **only supports single tool-calls at once**. This is enforced in the official Jinja template:

```
"This model only supports single tool-calls at once!"
```

**Implications:**
- The orchestrator will output ONE routing decision per query
- Multi-step orchestration requires multiple turns
- Our simple use case (one routing decision per request) is perfectly aligned

### Chat Template Format

The model uses ChatML/Qwen format:

```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_query}<|im_end|>
<|im_start|>assistant
```

**llama.cpp Configuration:** Use `--jinja` flag to enable chat template support.

### Tool Call Output Format

When the orchestrator decides to call a tool/model, it outputs JSON in Hermes format:

```json
{"name": "tool_name", "parameters": {"key": "value"}}
```

For our routing use case, we'll structure it as:

```json
{
  "action": "answer",
  "model": "small",
  "reasoning": "Simple greeting, no complex reasoning needed"
}
```

Or for direct tool calls:

```json
{
  "action": "use_tool",
  "tool": "weather",
  "tool_args": {"location": "NYC"},
  "reasoning": "Direct weather request"
}
```

### Recommended Inference Parameters

Based on the official evaluation code and best practices:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Temperature** | 0.1 | Low for consistent routing decisions |
| **Max Tokens** | 256 | Routing decisions are short |
| **Top-P** | 0.9 | Standard for focused output |
| **Context Size** | 8192 | Sufficient for query + system prompt |
| **Repeat Penalty** | 1.0 | No penalty needed for short outputs |

### llama.cpp Server Flags

Optimal configuration for llama-server:

```bash
llama-server \
  -m /models/nvidia_Orchestrator-8B-Q4_K_M.gguf \
  --port 8000 \
  --host 0.0.0.0 \
  -c 8192 \                    # Context size (8K sufficient for routing)
  --n-gpu-layers 999 \         # Full GPU offload
  --jinja \                    # Enable chat template support
  -b 512 \                     # Batch size
  -ub 512 \                    # Ubatch size
  --flash-attn on \            # Flash attention for efficiency
  --cont-batching \            # Continuous batching for throughput
  --parallel 4 \               # Multiple concurrent routing requests
  --no-mmap \                  # Better for unified memory
  -t 0.1                       # Low temperature for consistent routing
```

### vLLM Alternative (Reference)

If using vLLM instead of llama.cpp:

```bash
vllm serve nvidia/Orchestrator-8B \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --max-model-len 8192 \
  --tensor-parallel-size 1
```

### System Prompt for Routing

Based on the official evaluation code structure:

```
<|im_start|>system
You are an intelligent request router. Analyze the user's query and decide the best way to handle it.

## Available Resources

### Models
- small: Fast 8B model for simple queries, greetings, translations (~2s latency)
- large: Powerful 120B model for complex reasoning, analysis (~12s latency)
- vision: 8B multimodal model for image understanding (~3s latency)

### Tools (skip large model, call directly)
- weather: Get weather forecasts for any location
- web_search: Search the internet for current information
- code_interpreter: Execute Python code
- reader: Fetch and parse web pages
- zimage: Generate images from text descriptions

## User Preference: {preference}
- "fast": Prefer small model and direct tools
- "balanced": Use judgment based on complexity
- "thorough": Prefer large model for non-trivial queries

## Output Format
Respond with ONLY a JSON object:

For model routing:
{"action": "answer", "model": "small|large|vision", "reasoning": "brief explanation"}

For direct tool use:
{"action": "use_tool", "tool": "tool_name", "tool_args": {...}, "reasoning": "brief explanation"}

## Decision Guidelines
1. Image in query → model: vision
2. Explicit tool request (weather, search, generate image) → action: use_tool
3. Simple greetings, facts, translations → model: small
4. Complex analysis, multi-step reasoning, creative writing → model: large
5. When uncertain → default to large (better to over-deliver)
<|im_end|>
```

### Health Check Endpoint

llama.cpp server exposes `/health` endpoint:

```bash
curl http://localhost:9060/health
# Returns: {"status": "ok"}
```

### Performance Expectations

On DGX Spark (GB10 Blackwell) with Q4_K_M:

| Metric | Expected Value |
|--------|----------------|
| Time to First Token | ~50ms |
| Tokens/second | ~100-150 tok/s |
| Routing Decision Time | 200-400ms |
| VRAM Usage | ~6GB |
| Concurrent Requests | 4-8 |

### Monitoring Routing Quality

Track these metrics to ensure routing is working correctly:

```bash
# Check routing distribution over time
docker compose logs responses-api | grep "Routing to" | \
  awk '{print $NF}' | sort | uniq -c

# Expected healthy distribution:
#   45%  small
#   35%  large
#   15%  tools
#   5%   vision
```

### Sources

- [NVIDIA Orchestrator-8B Model Card](https://huggingface.co/nvidia/Orchestrator-8B)
- [bartowski GGUF Quantizations](https://huggingface.co/bartowski/nvidia_Orchestrator-8B-GGUF)
- [ToolOrchestra GitHub Repository](https://github.com/NVlabs/ToolOrchestra)
- [ToolOrchestra Research Page](https://research.nvidia.com/labs/lpr/ToolOrchestra/)
- [NVIDIA Technical Blog](https://developer.nvidia.com/blog/train-small-orchestration-agents-to-solve-big-problems/)

---

## Part 18: Orchestration Layer Design

This section describes the detailed design for integrating the orchestration layer into the existing `responses-api` codebase.

### Current Architecture Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CURRENT REQUEST FLOW                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. POST /v1/responses                                                      │
│     │                                                                       │
│     ▼                                                                       │
│  2. handlers.rs: create_response()                                          │
│     ├─► resolve_previous_response_chain()                                   │
│     │                                                                       │
│     ▼                                                                       │
│  3. executor.execute(&req, previous_messages)                               │
│     │                                                                       │
│     ├─► validate model exists                                               │
│     ├─► expand_tools() - convert built-in to function definitions           │
│     │                                                                       │
│     ▼                                                                       │
│  4. Tool Loop:                                                              │
│     ├─► call_llm() - POST to llama-server                                   │
│     ├─► if tool_calls: execute via MCP                                      │
│     └─► repeat until no more tool calls                                     │
│                                                                             │
│  5. Return Response                                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Files:**
- `src/server/handlers.rs:create_response()` - Entry point
- `src/execution/executor.rs:Executor::execute()` - Main execution loop
- `src/config/mod.rs:Config` - Configuration from environment
- `src/models/request.rs:CreateResponseRequest` - Request structure

### Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATED REQUEST FLOW                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. POST /v1/responses                                                      │
│     {"model": "auto", "input": "What's the weather in NYC?", ...}          │
│     │                                                                       │
│     ▼                                                                       │
│  2. handlers.rs: create_response()                                          │
│     ├─► resolve_previous_response_chain()                                   │
│     │                                                                       │
│     ▼                                                                       │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │ 3. NEW: Orchestration Check                                           │ │
│  │    if orchestrator.is_enabled() && req.model == "auto":               │ │
│  │        routing = orchestrator.route(&req).await                       │ │
│  │                                                                        │ │
│  │        ┌─────────────────────────────────────────────────────────────┐│ │
│  │        │ Call Orchestrator-8B (llama-server-orchestrator:9060)       ││ │
│  │        │ Input: System prompt + user query                           ││ │
│  │        │ Output: {"action":"use_tool","tool":"weather",...}          ││ │
│  │        └─────────────────────────────────────────────────────────────┘│ │
│  │                                                                        │ │
│  │    match routing.action:                                              │ │
│  │        Answer { model } → modify req.model, call execute()            │ │
│  │        UseTool { tool, args } → call MCP directly, build response     │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│     │                                                                       │
│     ▼                                                                       │
│  4. executor.execute() OR direct_tool_response()                            │
│     │                                                                       │
│     ▼                                                                       │
│  5. Return Response                                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Module Structure

```
backend/responses-api/src/
├── orchestration/           # NEW MODULE
│   ├── mod.rs              # Module exports
│   ├── config.rs           # OrchestratorConfig, RoleMapping
│   ├── router.rs           # OrchestratorRouter - main routing logic
│   ├── prompt.rs           # System prompt builder
│   └── types.rs            # RoutingDecision, RoutingAction
├── config/
│   └── mod.rs              # MODIFY: Add orchestrator config parsing
├── server/
│   └── handlers.rs         # MODIFY: Add orchestration check
├── execution/
│   └── executor.rs         # MODIFY: Add execute_direct_tool()
└── lib.rs                  # MODIFY: Add mod orchestration
```

### Type Definitions

**File: `src/orchestration/types.rs`**

```rust
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Action the orchestrator decided to take.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum RoutingAction {
    /// Route to a model for answering
    Answer {
        /// Model role: "small", "large", or "vision"
        model: String,
    },
    /// Call a tool directly (skip large model)
    UseTool {
        /// Tool name (e.g., "weather", "web_search")
        tool: String,
        /// Tool arguments
        #[serde(default)]
        tool_args: Value,
    },
}

/// Complete routing decision from the orchestrator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingDecision {
    /// The action to take
    #[serde(flatten)]
    pub action: RoutingAction,
    /// Explanation of why this routing was chosen
    pub reasoning: String,
}
```

**File: `src/orchestration/config.rs`**

```rust
use serde::{Deserialize, Serialize};

/// Mapping from role names to actual model IDs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleMapping {
    /// Model for simple queries (e.g., "qwen3-vl-8b-instruct")
    pub small: String,
    /// Model for complex queries (e.g., "gpt-oss-120b")
    pub large: String,
    /// Model for vision queries (e.g., "qwen3-vl-8b-instruct")
    #[serde(default)]
    pub vision: Option<String>,
}

impl Default for RoleMapping {
    fn default() -> Self {
        Self {
            small: "qwen3-vl-8b-instruct".to_string(),
            large: "gpt-oss-120b".to_string(),
            vision: Some("qwen3-vl-8b-instruct".to_string()),
        }
    }
}

/// Configuration for the orchestrator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestratorConfig {
    /// Whether orchestration is enabled
    #[serde(default)]
    pub enabled: bool,

    /// Model ID of the orchestrator (must exist in MODELS_CONFIG)
    #[serde(default = "default_model_id")]
    pub model_id: String,

    /// Default user preference when not specified in request
    #[serde(default = "default_preference")]
    pub default_preference: String,

    /// Role to model ID mapping
    #[serde(default)]
    pub role_mapping: RoleMapping,

    /// Fallback model when orchestration fails
    #[serde(default = "default_fallback")]
    pub fallback_model: String,

    /// Max tokens for orchestrator response
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,

    /// Temperature for orchestrator (low for consistency)
    #[serde(default = "default_temperature")]
    pub temperature: f32,
}

fn default_model_id() -> String { "orchestrator".to_string() }
fn default_preference() -> String { "balanced".to_string() }
fn default_fallback() -> String { "gpt-oss-120b".to_string() }
fn default_max_tokens() -> u32 { 256 }
fn default_temperature() -> f32 { 0.1 }

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model_id: default_model_id(),
            default_preference: default_preference(),
            role_mapping: RoleMapping::default(),
            fallback_model: default_fallback(),
            max_tokens: default_max_tokens(),
            temperature: default_temperature(),
        }
    }
}
```

### Router Implementation

**File: `src/orchestration/router.rs`**

```rust
use reqwest::Client;
use serde_json::Value;

use crate::config::ModelConfig;
use crate::mcp::McpClient;
use crate::models::{ChatCompletionRequest, ChatCompletionResponse, ChatMessage, CreateResponseRequest};

use super::config::OrchestratorConfig;
use super::prompt::build_system_prompt;
use super::types::{RoutingAction, RoutingDecision};

/// Error during orchestration routing.
#[derive(Debug, thiserror::Error)]
pub enum RoutingError {
    #[error("Orchestrator model not found: {0}")]
    ModelNotFound(String),
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("Failed to parse routing decision: {0}")]
    ParseError(String),
    #[error("Orchestrator returned error: {0}")]
    OrchestratorError(String),
}

/// The orchestrator router that decides how to handle requests.
pub struct OrchestratorRouter {
    config: OrchestratorConfig,
    http: Client,
}

impl OrchestratorRouter {
    /// Create a new orchestrator router.
    pub fn new(config: OrchestratorConfig) -> Self {
        let http = Client::builder()
            .timeout(std::time::Duration::from_secs(30)) // Fast timeout for routing
            .build()
            .expect("failed to create HTTP client");

        Self { config, http }
    }

    /// Check if orchestration is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get the fallback model ID.
    pub fn fallback_model(&self) -> &str {
        &self.config.fallback_model
    }

    /// Route a request to the appropriate backend.
    pub async fn route(
        &self,
        req: &CreateResponseRequest,
        model_config: &ModelConfig,
        available_tools: Vec<String>,
    ) -> Result<RoutingDecision, RoutingError> {
        // Extract user preference from metadata
        let preference = req.metadata
            .as_ref()
            .and_then(|m| m.get("preference"))
            .and_then(|v| v.as_str())
            .unwrap_or(&self.config.default_preference);

        // Check if request has images
        let has_images = self.request_has_images(req);

        // Build the orchestrator request
        let system_prompt = build_system_prompt(
            &self.config.role_mapping,
            &available_tools,
            preference,
        );

        let user_content = self.format_user_query(req, has_images);

        let chat_req = ChatCompletionRequest {
            model: self.config.model_id.clone(),
            messages: vec![
                ChatMessage::system(system_prompt),
                ChatMessage::user(user_content),
            ],
            temperature: Some(self.config.temperature),
            max_tokens: Some(self.config.max_tokens),
            ..Default::default()
        };

        // Call the orchestrator
        let url = format!("{}/v1/chat/completions", model_config.url);
        let response = self.http.post(&url)
            .json(&chat_req)
            .send()
            .await?;

        if !response.status().is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(RoutingError::OrchestratorError(body));
        }

        let chat_resp: ChatCompletionResponse = response.json().await?;

        // Parse the routing decision from the response
        self.parse_routing_decision(&chat_resp)
    }

    /// Resolve a model role to an actual model ID.
    pub fn resolve_model(&self, role: &str) -> String {
        match role {
            "small" => self.config.role_mapping.small.clone(),
            "large" => self.config.role_mapping.large.clone(),
            "vision" => self.config.role_mapping.vision
                .clone()
                .unwrap_or_else(|| self.config.role_mapping.small.clone()),
            _ => self.config.fallback_model.clone(),
        }
    }

    /// Check if the request contains images.
    fn request_has_images(&self, req: &CreateResponseRequest) -> bool {
        // Check input for image content parts
        match &req.input {
            crate::models::Input::Items(items) => {
                items.iter().any(|item| {
                    if let crate::models::InputItem::Message(msg) = item {
                        if let crate::models::MessageContent::Parts(parts) = &msg.content {
                            return parts.iter().any(|p| {
                                matches!(p, crate::models::ContentPart::InputImage { .. })
                            });
                        }
                    }
                    false
                })
            }
            _ => false,
        }
    }

    /// Format the user query for the orchestrator.
    fn format_user_query(&self, req: &CreateResponseRequest, has_images: bool) -> String {
        let query_text = match &req.input {
            crate::models::Input::Text(t) => t.clone(),
            crate::models::Input::Items(items) => {
                // Extract text from message items
                items.iter()
                    .filter_map(|item| {
                        if let crate::models::InputItem::Message(msg) = item {
                            match &msg.content {
                                crate::models::MessageContent::Text(t) => Some(t.clone()),
                                crate::models::MessageContent::Parts(parts) => {
                                    let texts: Vec<_> = parts.iter()
                                        .filter_map(|p| {
                                            if let crate::models::ContentPart::InputText { text } = p {
                                                Some(text.clone())
                                            } else {
                                                None
                                            }
                                        })
                                        .collect();
                                    Some(texts.join(" "))
                                }
                            }
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            }
            crate::models::Input::Empty => String::new(),
        };

        if has_images {
            format!("[Query includes {} image(s)]\n{}", 1, query_text)
        } else {
            query_text
        }
    }

    /// Parse the routing decision from orchestrator response.
    fn parse_routing_decision(
        &self,
        resp: &ChatCompletionResponse,
    ) -> Result<RoutingDecision, RoutingError> {
        let content = resp.choices.first()
            .and_then(|c| c.message.content.as_ref())
            .ok_or_else(|| RoutingError::ParseError("Empty response".to_string()))?;

        // Try to extract JSON from the response
        // The orchestrator might return markdown-wrapped JSON
        let json_str = if content.contains("```json") {
            content
                .split("```json")
                .nth(1)
                .and_then(|s| s.split("```").next())
                .unwrap_or(content)
                .trim()
        } else if content.contains("```") {
            content
                .split("```")
                .nth(1)
                .unwrap_or(content)
                .trim()
        } else {
            content.trim()
        };

        serde_json::from_str(json_str)
            .map_err(|e| RoutingError::ParseError(format!("{}: {}", e, json_str)))
    }
}
```

### System Prompt Builder

**File: `src/orchestration/prompt.rs`**

```rust
use super::config::RoleMapping;

/// Build the system prompt for the orchestrator.
pub fn build_system_prompt(
    roles: &RoleMapping,
    available_tools: &[String],
    preference: &str,
) -> String {
    let tools_section = if available_tools.is_empty() {
        "No tools available.".to_string()
    } else {
        let tool_list = available_tools.iter()
            .map(|t| format!("- {}", t))
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "### Tools (skip large model, call directly)\n{}",
            tool_list
        )
    };

    format!(r#"You are an intelligent request router. Analyze the user's query and decide the best way to handle it.

## Available Resources

### Models
- small ({small}): Fast 8B model for simple queries, greetings, translations (~2s latency)
- large ({large}): Powerful 120B model for complex reasoning, analysis (~12s latency)
- vision ({vision}): 8B multimodal model for image understanding (~3s latency)

{tools_section}

## User Preference: {preference}
- "fast": Strongly prefer small model and direct tools
- "balanced": Use judgment based on complexity
- "thorough": Prefer large model for non-trivial queries

## Output Format
Respond with ONLY a JSON object (no markdown, no explanation):

For model routing:
{{"action": "answer", "model": "small|large|vision", "reasoning": "brief explanation"}}

For direct tool use:
{{"action": "use_tool", "tool": "tool_name", "tool_args": {{}}, "reasoning": "brief explanation"}}

## Decision Guidelines
1. Image in query → action: answer, model: vision
2. Explicit tool request (weather, search, image generation) → action: use_tool
3. Simple greetings, facts, translations → action: answer, model: small
4. Complex analysis, multi-step reasoning, creative writing → action: answer, model: large
5. When uncertain → default to model: large (better to over-deliver)
6. Respect user preference (fast→small, thorough→large)"#,
        small = roles.small,
        large = roles.large,
        vision = roles.vision.as_ref().unwrap_or(&roles.small),
        tools_section = tools_section,
        preference = preference,
    )
}
```

### Integration: Config Changes

**File: `src/config/mod.rs`** (additions)

```rust
use crate::orchestration::OrchestratorConfig;

// Add to Config struct:
pub struct Config {
    // ... existing fields ...

    /// Orchestrator configuration
    pub orchestrator: OrchestratorConfig,
}

// Add to Config::from_env():
impl Config {
    pub fn from_env() -> Self {
        let mut config = Self::default();

        // ... existing parsing ...

        // Parse orchestrator configuration
        if let Ok(json) = env::var("ORCHESTRATOR_CONFIG") {
            match serde_json::from_str::<OrchestratorConfig>(&json) {
                Ok(orch_config) => config.orchestrator = orch_config,
                Err(e) => tracing::error!("Failed to parse ORCHESTRATOR_CONFIG: {}", e),
            }
        }

        config
    }
}
```

### Integration: Handler Changes

**File: `src/server/handlers.rs`** (modifications)

```rust
use crate::orchestration::{OrchestratorRouter, RoutingAction};

// Add to AppState:
pub struct AppState {
    pub executor: Executor,
    pub store: InMemoryStore,
    pub config: Config,
    pub mcp: McpClient,
    pub containers: ContainerStore,
    pub orchestrator: OrchestratorRouter,  // NEW
}

// Modify create_response():
pub async fn create_response(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateResponseRequest>,
) -> Result<impl IntoResponse, ApiError> {
    // ... existing chain resolution ...

    // NEW: Check if orchestration should be applied
    let effective_req = if state.orchestrator.is_enabled() && req.model == "auto" {
        orchestrate_request(&state, &req, &previous_messages).await?
    } else {
        req.clone()
    };

    // Continue with existing flow using effective_req
    let response = state
        .executor
        .execute(&effective_req, previous_messages)
        .await
        .map_err(execution_error)?;

    // ... rest of handler ...
}

/// Apply orchestration to route the request.
async fn orchestrate_request(
    state: &AppState,
    req: &CreateResponseRequest,
    _previous_messages: &[ChatMessage],
) -> Result<CreateResponseRequest, ApiError> {
    // Get orchestrator model config
    let orch_model = state.config.get_model(&state.config.orchestrator.model_id)
        .ok_or_else(|| orchestration_error("Orchestrator model not configured"))?;

    // Get available tool names
    let available_tools = state.mcp.get_tool_names().await;

    // Get routing decision
    let routing = state.orchestrator
        .route(req, orch_model, available_tools)
        .await
        .map_err(|e| orchestration_error(&e.to_string()))?;

    tracing::info!(
        action = ?routing.action,
        reasoning = %routing.reasoning,
        "Orchestrator routing decision"
    );

    match routing.action {
        RoutingAction::Answer { model } => {
            // Resolve role to actual model ID
            let actual_model = state.orchestrator.resolve_model(&model);
            let mut routed_req = req.clone();
            routed_req.model = actual_model;
            Ok(routed_req)
        }
        RoutingAction::UseTool { tool, tool_args } => {
            // For direct tool calls, we'll handle this specially
            // TODO: Implement direct tool execution path
            // For now, fall back to large model with the tool
            let mut routed_req = req.clone();
            routed_req.model = state.orchestrator.fallback_model().to_string();
            Ok(routed_req)
        }
    }
}

fn orchestration_error(msg: &str) -> ApiError {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(json!({
            "error": {
                "type": "orchestration_error",
                "message": msg
            }
        })),
    )
}
```

### Environment Configuration

```yaml
# compose.yml - responses-api environment
environment:
  MODELS_CONFIG: |
    {
      "models": [
        {"id": "orchestrator", "url": "http://llama-server-orchestrator:8000", "owned_by": "nvidia"},
        {"id": "qwen3-vl-8b-instruct", "url": "http://llama-server-qwen-vl:8000", "supports_vision": true},
        {"id": "gpt-oss-120b", "url": "http://llama-server:8000", "reasoning": {"effort": "high"}}
      ]
    }
  ORCHESTRATOR_CONFIG: |
    {
      "enabled": true,
      "model_id": "orchestrator",
      "default_preference": "balanced",
      "role_mapping": {
        "small": "qwen3-vl-8b-instruct",
        "large": "gpt-oss-120b",
        "vision": "qwen3-vl-8b-instruct"
      },
      "fallback_model": "gpt-oss-120b",
      "max_tokens": 256,
      "temperature": 0.1
    }
```

### Request Flow Examples

**Example 1: Simple greeting → routed to small model**

```
User: {"model": "auto", "input": "Hello!"}
                    │
                    ▼
         Orchestrator-8B (~300ms)
         {"action": "answer", "model": "small", "reasoning": "Simple greeting"}
                    │
                    ▼
         Resolve: "small" → "qwen3-vl-8b-instruct"
                    │
                    ▼
         Execute with model="qwen3-vl-8b-instruct" (~2s)
                    │
                    ▼
         Response: "Hello! How can I help you today?"

Total: ~2.3s (vs ~12.3s without orchestration)
```

**Example 2: Complex analysis → routed to large model**

```
User: {"model": "auto", "input": "Explain the philosophical implications of Gödel's theorems"}
                    │
                    ▼
         Orchestrator-8B (~300ms)
         {"action": "answer", "model": "large", "reasoning": "Complex philosophical analysis"}
                    │
                    ▼
         Resolve: "large" → "gpt-oss-120b"
                    │
                    ▼
         Execute with model="gpt-oss-120b" (~12s)
                    │
                    ▼
         Response: [detailed philosophical explanation]

Total: ~12.3s (orchestrator adds minimal overhead for complex queries)
```

**Example 3: Weather request → direct tool call**

```
User: {"model": "auto", "input": "What's the weather in NYC?", "tools": [{"type": "weather"}]}
                    │
                    ▼
         Orchestrator-8B (~300ms)
         {"action": "use_tool", "tool": "weather", "tool_args": {"location": "NYC"}}
                    │
                    ▼
         Direct MCP call: weather.get_forecast({"location": "NYC"}) (~500ms)
                    │
                    ▼
         Build synthetic response from tool result
                    │
                    ▼
         Response: "NYC: 72°F, sunny, humidity 45%"

Total: ~0.8s (vs ~12.5s with full LLM round-trip)
```

### Implementation Priority

1. **Phase 1 - Basic Routing (MVP)**
   - Implement `OrchestratorConfig` and environment parsing
   - Implement `OrchestratorRouter` with model routing only
   - Modify handlers to use orchestrator when `model == "auto"`
   - Direct tool calls fall back to large model (not optimized yet)

2. **Phase 2 - Direct Tool Execution**
   - Add `execute_direct_tool()` to Executor
   - Build synthetic responses from tool results
   - Skip LLM for pure tool calls

3. **Phase 3 - Metrics & Tuning**
   - Add routing metrics (decision distribution, latency)
   - Log routing decisions for analysis
   - Tune system prompt based on real usage patterns

### Testing Strategy

```bash
# Test 1: Verify orchestrator routes simple queries to small model
curl -X POST http://localhost:9150/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "input": "Hello!"}'
# Expected: Fast response (~2s), routed to qwen3-vl-8b-instruct

# Test 2: Verify complex queries go to large model
curl -X POST http://localhost:9150/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "input": "Write a detailed analysis of quantum computing applications in cryptography"}'
# Expected: Thorough response (~12s), routed to gpt-oss-120b

# Test 3: Verify preference override
curl -X POST http://localhost:9150/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "input": "Explain photosynthesis", "metadata": {"preference": "fast"}}'
# Expected: Concise response (~2s), routed to small model despite moderate complexity

# Test 4: Verify fallback when orchestrator disabled
curl -X POST http://localhost:9150/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-oss-120b", "input": "Hello!"}'
# Expected: Normal flow, no orchestration
```

---

## Part 19: System Integration Architecture

This section documents how the orchestrator fits into the existing strieber-gpt-3 system architecture, following established patterns.

### Current System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        STRIEBER-GPT-3 SYSTEM ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────┐                                                            │
│  │   chat-ui       │  Frontend (Svelte 5)                                       │
│  │   :9300         │  - ModelSelector fetches /v1/models                        │
│  └────────┬────────┘  - Sends requests with user-selected model                 │
│           │                                                                     │
│           │ POST /v1/responses {model: "gpt-oss-120b", input: "..."}            │
│           ▼                                                                     │
│  ┌─────────────────┐                                                            │
│  │ responses-api   │  Backend (Rust + Axum)                                     │
│  │   :9150         │  - Config::from_env() loads MODELS_CONFIG, MCP_CONFIG      │
│  │                 │  - McpClient connects to MCP servers                       │
│  │                 │  - Executor handles requests, tool loops                   │
│  └────────┬────────┘                                                            │
│           │                                                                     │
│           │ POST /v1/chat/completions                                           │
│           ▼                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐                                       │
│  │ llama-server    │  │ llama-server    │  LLM Backends (llama.cpp)             │
│  │ (gpt-oss-120b)  │  │ (qwen3-vl-2b)   │  - Each model has own service         │
│  │   :9010         │  │   :9020         │  - Registered in MODELS_CONFIG        │
│  └─────────────────┘  └─────────────────┘                                       │
│                                                                                 │
│           │ MCP tool calls                                                      │
│           ▼                                                                     │
│  ┌───────────┬───────────┬───────────┬───────────┐                              │
│  │mcp-weather│mcp-search │mcp-code   │mcp-reader │  MCP Tool Servers            │
│  │  :9100    │  :9110    │  :9120    │  :9130    │  - Each registered with      │
│  └───────────┴───────────┴───────────┴───────────┘    builtin_type in MCP_CONFIG│
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Configuration Pattern Analysis

**Environment Variables (compose.yml):**
```yaml
environment:
  # Models: Parsed by Config::from_env() → config.models
  - MODELS_CONFIG={"models":[
      {"id":"gpt-oss-120b","url":"http://llama-server:8000",...},
      {"id":"qwen3-vl-2b","url":"http://llama-server-qwen-vl:8000",...}
    ]}

  # MCP: Parsed by Config::from_env() → config.mcp_servers
  - MCP_CONFIG={"servers":[
      {"name":"weather","url":"http://mcp-weather:8000/mcp","builtin_type":"weather"},
      ...
    ]}
```

**Rust Config Loading (src/config/mod.rs):**
```rust
pub fn from_env() -> Self {
    if let Ok(json) = env::var("MODELS_CONFIG") {
        config.models = serde_json::from_str(&json)?;
    }
    if let Ok(json) = env::var("MCP_CONFIG") {
        config.mcp_servers = serde_json::from_str(&json)?;
    }
}
```

**Frontend Model Discovery:**
```typescript
// ModelSelector.svelte fetches /v1/models
const response = await fetch(`${getApiBaseUrl()}/models`);
// Returns: {data: [{id: "gpt-oss-120b", ...}, {id: "qwen3-vl-2b", ...}]}
```

### Orchestrator Integration: The Three-Layer Approach

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     ORCHESTRATED SYSTEM ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────┐                                                            │
│  │   chat-ui       │  - Shows "auto" in ModelSelector (intelligent routing)     │
│  │   :9300         │  - User can select specific model to bypass orchestrator   │
│  └────────┬────────┘                                                            │
│           │                                                                     │
│           │ POST /v1/responses {model: "auto", input: "Hello!"}                 │
│           ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                        responses-api :9150                              │    │
│  │  ┌──────────────────────────────────────────────────────────────────┐   │    │
│  │  │ OrchestratorRouter (NEW)                                         │   │    │
│  │  │ - Intercepts model="auto" requests                               │   │    │
│  │  │ - Calls orchestrator model for routing decision                  │   │    │
│  │  │ - Resolves "small"→"qwen3-vl-8b-instruct"                       │   │    │
│  │  └──────────────────────────────────────────────────────────────────┘   │    │
│  └────────┬───────────────────────────────┬────────────────────────────────┘    │
│           │                               │                                     │
│           │ Routing (~300ms)              │ Request with resolved model         │
│           ▼                               ▼                                     │
│  ┌─────────────────┐             ┌─────────────────┐  ┌─────────────────┐       │
│  │ llama-server    │             │ llama-server    │  │ llama-server    │       │
│  │ orchestrator    │◄────────────│ qwen-vl-8b      │  │ gpt-oss-120b    │       │
│  │ :9060           │  decides    │ :9020 (small)   │  │ :9010 (large)   │       │
│  └─────────────────┘             └─────────────────┘  └─────────────────┘       │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Configuration Design

**1. MODELS_CONFIG - Add "auto" virtual model:**

```json
{
  "models": [
    {
      "id": "auto",
      "url": "",
      "owned_by": "system",
      "description": "Intelligent routing - automatically selects the best model",
      "is_virtual": true
    },
    {
      "id": "orchestrator",
      "url": "http://llama-server-orchestrator:8000",
      "owned_by": "nvidia",
      "hidden": true
    },
    {
      "id": "qwen3-vl-8b-instruct",
      "url": "http://llama-server-qwen-vl:8000",
      "supports_vision": true
    },
    {
      "id": "gpt-oss-120b",
      "url": "http://llama-server:8000",
      "reasoning": {"effort": "high"}
    }
  ]
}
```

**2. ORCHESTRATOR_CONFIG - New environment variable (follows MCP_CONFIG pattern):**

```json
{
  "enabled": true,
  "model_id": "orchestrator",
  "auto_model_id": "auto",
  "default_preference": "balanced",
  "role_mapping": {
    "small": "qwen3-vl-8b-instruct",
    "large": "gpt-oss-120b",
    "vision": "qwen3-vl-8b-instruct"
  },
  "fallback_model": "gpt-oss-120b",
  "max_tokens": 256,
  "temperature": 0.1,
  "timeout_secs": 30
}
```

### Module Structure (Following MCP Pattern)

```
src/orchestration/           # Similar to src/mcp/
├── mod.rs                   # pub use statements
├── config.rs                # OrchestratorConfig (like McpServerConfig)
├── router.rs                # OrchestratorRouter (like McpClient)
├── prompt.rs                # System prompt builder
└── types.rs                 # RoutingAction, RoutingDecision

vs

src/mcp/
├── mod.rs                   # pub use statements
└── client.rs                # McpClient, McpServerConfig
```

### Integration Points

**1. src/config/mod.rs - Add orchestrator config:**

```rust
use crate::orchestration::OrchestratorConfig;

pub struct Config {
    pub models: Vec<ModelConfig>,
    pub mcp_servers: Vec<McpServerConfig>,
    pub orchestrator: OrchestratorConfig,  // NEW
    // ...
}

impl Config {
    pub fn from_env() -> Self {
        // ... existing code ...

        // NEW: Parse ORCHESTRATOR_CONFIG
        if let Ok(json) = env::var("ORCHESTRATOR_CONFIG") {
            match serde_json::from_str::<OrchestratorConfig>(&json) {
                Ok(orch) => config.orchestrator = orch,
                Err(e) => tracing::warn!("Failed to parse ORCHESTRATOR_CONFIG: {}", e),
            }
        }

        config
    }
}
```

**2. src/main.rs - Create and wire OrchestratorRouter:**

```rust
use responses_api::orchestration::OrchestratorRouter;

async fn main() {
    let config = Config::from_env();

    // Existing
    let mcp_client = McpClient::new(config.mcp_servers.clone());
    mcp_client.connect_all().await?;

    // NEW
    let orchestrator = OrchestratorRouter::new(config.orchestrator.clone());

    let state = Arc::new(AppState {
        executor,
        store: InMemoryStore::new(),
        config: config.clone(),
        mcp: mcp_client,
        containers,
        orchestrator,  // NEW
    });
}
```

**3. src/server/handlers.rs - Add orchestration to request flow:**

```rust
pub struct AppState {
    pub executor: Executor,
    pub store: InMemoryStore,
    pub config: Config,
    pub mcp: McpClient,
    pub containers: ContainerStore,
    pub orchestrator: OrchestratorRouter,  // NEW
}

pub async fn create_response(...) -> Result<impl IntoResponse, ApiError> {
    // ... chain resolution ...

    // NEW: Apply orchestration if model is "auto"
    let effective_req = apply_orchestration(&state, req.clone()).await?;

    // Existing: Execute with (potentially modified) request
    let response = state.executor.execute(&effective_req, previous_messages).await?;

    // ...
}

async fn apply_orchestration(
    state: &AppState,
    mut req: CreateResponseRequest,
) -> Result<CreateResponseRequest, ApiError> {
    if !state.orchestrator.is_enabled() || req.model != "auto" {
        return Ok(req);
    }

    let orch_model = state.config.get_model(&state.config.orchestrator.model_id)?;
    let tools = state.mcp.available_tools().await;

    match state.orchestrator.route(&req, orch_model, tools).await {
        Ok(decision) => {
            let actual_model = state.orchestrator.resolve_model(&decision.model());
            req.model = actual_model;
            Ok(req)
        }
        Err(e) => {
            tracing::warn!(error = %e, "Orchestration failed, using fallback");
            req.model = state.orchestrator.fallback_model().to_string();
            Ok(req)
        }
    }
}
```

### Docker Compose Integration

```yaml
services:
  llama-server-orchestrator:
    # ... (from Part 16)

  responses-api:
    environment:
      - MODELS_CONFIG={"models":[{"id":"auto","url":"","owned_by":"system","is_virtual":true},{"id":"orchestrator","url":"http://llama-server-orchestrator:8000","owned_by":"nvidia","hidden":true},{"id":"qwen3-vl-8b-instruct","url":"http://llama-server-qwen-vl:8000","supports_vision":true},{"id":"gpt-oss-120b","url":"http://llama-server:8000","reasoning":{"effort":"high"}}]}
      - MCP_CONFIG={"servers":[...]}
      - ORCHESTRATOR_CONFIG={"enabled":true,"model_id":"orchestrator","auto_model_id":"auto","default_preference":"balanced","role_mapping":{"small":"qwen3-vl-8b-instruct","large":"gpt-oss-120b","vision":"qwen3-vl-8b-instruct"},"fallback_model":"gpt-oss-120b","max_tokens":256,"temperature":0.1}
    depends_on:
      llama-server-orchestrator:  # NEW
        condition: service_healthy
```

### Graceful Degradation

The system never fails due to orchestrator issues:

```rust
match state.orchestrator.route(&req, ...).await {
    Ok(decision) => { /* use routing */ }
    Err(e) => {
        tracing::warn!("Orchestration failed: {}, using fallback", e);
        req.model = state.orchestrator.fallback_model().to_string();
        Ok(req)  // Continue with fallback - never fail the request
    }
}
```

### Summary: Files to Change

| File | Change | Lines |
|------|--------|-------|
| `src/lib.rs` | Add `pub mod orchestration;` | +1 |
| `src/orchestration/mod.rs` | NEW module exports | ~10 |
| `src/orchestration/config.rs` | NEW config types | ~80 |
| `src/orchestration/types.rs` | NEW routing types | ~40 |
| `src/orchestration/router.rs` | NEW router logic | ~150 |
| `src/orchestration/prompt.rs` | NEW prompt builder | ~60 |
| `src/config/mod.rs` | Add orchestrator field + parsing | ~15 |
| `src/server/handlers.rs` | Add AppState field + apply_orchestration | ~50 |
| `src/main.rs` | Create router + add to state | ~10 |
| `compose.yml` | Update env vars, depends_on | ~10 |
| **Total** | | **~425** |

---

## Part 20: Tool Calling Architecture for Phase 2

### The Question: Who Calls Tools?

In the current strieber-gpt-3 system, **any model** can do tool calling. The executor loop in `executor.rs` handles tool calls for any model via the MCP client. But for Phase 2 orchestration, we need to decide the architecture.

### Option A: Orchestrator-Centric Tool Calling (Recommended)

The orchestrator and small model gather **all context** via tools. The large model receives a complete context package and generates the final response.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    OPTION A: ORCHESTRATOR-CENTRIC                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  User Query                                                                 │
│       │                                                                     │
│       ▼                                                                     │
│  ┌──────────────┐                                                           │
│  │ Orchestrator │ ──► CallTool(weather) ──► MCP ──► Result                 │
│  │    (8B)      │ ◄───────────────────────────────────────                 │
│  └──────────────┘                                                           │
│       │                                                                     │
│       ▼                                                                     │
│  ┌──────────────┐                                                           │
│  │ Small Model  │ ──► CallTool(search) ──► MCP ──► Result                  │
│  │   (8B VL)    │ ◄───────────────────────────────────────                 │
│  └──────────────┘                                                           │
│       │                                                                     │
│       │  All context gathered: weather data, search results                 │
│       ▼                                                                     │
│  ┌──────────────┐                                                           │
│  │ Large Model  │ ──► Generates final response (NO tool calls)             │
│  │   (120B)     │     Context already complete                             │
│  └──────────────┘                                                           │
│       │                                                                     │
│       ▼                                                                     │
│  Final Response                                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Pros:**
- Large model gets complete context in one shot (no back-and-forth)
- Don't pay 12s latency multiple times for tool iterations
- Cleaner separation: fast models gather, large model synthesizes

### Option B: Layered Tool Calling

Orchestrator routes, then the destination model does its own tool calling.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    OPTION B: LAYERED TOOL CALLING                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  User Query                                                                 │
│       │                                                                     │
│       ▼                                                                     │
│  ┌──────────────┐                                                           │
│  │ Orchestrator │ ──► Routes to Large Model                                 │
│  │    (8B)      │                                                           │
│  └──────────────┘                                                           │
│       │                                                                     │
│       ▼                                                                     │
│  ┌──────────────┐                                                           │
│  │ Large Model  │ ──► CallTool(weather) ──► MCP ──► Result ──► 12s         │
│  │   (120B)     │ ──► CallTool(search) ──► MCP ──► Result ──► 12s          │
│  └──────────────┘ ──► Generate response ────────────────────► 12s          │
│       │                                                                     │
│       │  Total: 36s+ (3 large model calls)                                 │
│       ▼                                                                     │
│  Final Response                                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Pros:**
- Large model can do complex multi-step reasoning with tools
- Better for tasks that need the large model's intelligence for tool selection

**Cons:**
- Much higher latency (each tool call iteration costs 12s)
- Expensive - pays full cost for each iteration

### Recommendation: Hybrid with Fallback

For Phase 2, **Option A is the default**, but with a fallback to Option B when needed:

```rust
/// How a model handles tools when it's the final responder
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ModelToolCapability {
    /// Model receives pre-gathered context, generates response only (default)
    #[default]
    ResponseOnly,
    /// Model can also make tool calls if trajectory is incomplete
    ToolCallingFallback,
    /// Model always does its own tool calling (bypasses orchestrator gathering)
    ToolCallingPrimary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleConfig {
    pub model_id: String,
    /// How this model handles tools
    #[serde(default)]
    pub tool_capability: ModelToolCapability,
    /// Max context tokens to include in handoff
    #[serde(default = "default_max_context")]
    pub max_context_tokens: usize,
}

fn default_max_context() -> usize { 8000 }
```

### Phase 2 Trajectory Building with Tool Capability

```rust
pub async fn execute_orchestrated(
    &self,
    req: &CreateResponseRequest,
    previous_messages: Vec<ChatMessage>,
) -> Result<Response, ExecutionError> {
    let mut trajectory = Trajectory::new(&req.input);
    let max_steps = self.config.max_orchestration_steps; // default: 5

    for step in 0..max_steps {
        let decision = self.orchestrator.decide(
            &req,
            &trajectory,
            &self.mcp.available_tools().await
        ).await?;

        match decision {
            OrchestratorDecision::CallTool { tool, args } => {
                // Orchestrator gathers context via tools
                let result = self.mcp.call_tool_text(&tool, args).await?;
                trajectory.add_tool_result(&tool, &result);
            }

            OrchestratorDecision::CallSmall { query, capability } => {
                // Small model processes with gathered context
                // Small model can also call tools (fast - 300ms per iteration)
                let small_config = self.config.get_role("small")?;
                let response = match small_config.tool_capability {
                    ModelToolCapability::ResponseOnly => {
                        // Just generate response from context
                        self.call_model_response_only(&small_config.model_id, &query, &trajectory).await?
                    }
                    _ => {
                        // Allow small model to also call tools
                        self.call_model_with_tools(&small_config.model_id, &query, &trajectory).await?
                    }
                };
                trajectory.add_model_response("small", &query, &response);
            }

            OrchestratorDecision::Respond { answer } => {
                // Orchestrator or small model can answer directly
                return Ok(self.build_response_from_trajectory(&trajectory, &answer, req));
            }

            OrchestratorDecision::Escalate { reason, task } => {
                // Hand off to large model with complete trajectory
                let large_config = self.config.get_role("large")?;
                let context_prompt = trajectory.build_handoff_prompt(&reason, &task);

                match large_config.tool_capability {
                    ModelToolCapability::ResponseOnly => {
                        // Large model just synthesizes (recommended)
                        return self.call_large_response_only(&context_prompt).await;
                    }
                    ModelToolCapability::ToolCallingFallback => {
                        // Large model can call tools if context incomplete
                        let response = self.call_large_with_optional_tools(&context_prompt).await?;
                        if response.needs_more_tools {
                            // Let large model do one more tool call round
                            return self.call_large_with_tools(&context_prompt).await;
                        }
                        return Ok(response);
                    }
                    ModelToolCapability::ToolCallingPrimary => {
                        // Large model does all its own tool calling (bypass orchestrator)
                        return self.call_large_with_tools(&context_prompt).await;
                    }
                }
            }
        }
    }

    // Max steps reached - escalate with what we have
    let context_prompt = trajectory.build_emergency_handoff();
    self.call_large_response_only(&context_prompt).await
}
```

### Why Orchestrator-Centric Works Better

The key insight is **latency asymmetry**:

| Model | Per-Call Latency | Tool Iteration Cost |
|-------|------------------|---------------------|
| Orchestrator (8B) | ~300ms | ~800ms total |
| Small Model (8B VL) | ~400ms | ~900ms total |
| Large Model (120B) | ~12s | ~15s total |

If a task needs 3 tool calls:
- **Option A**: 3 × 800ms = 2.4s (orchestrator gathers) + 12s (large synthesizes) = **14.4s**
- **Option B**: 3 × 15s = **45s** (large model does all tool calls)

### Configuration for Phase 2

```yaml
ORCHESTRATOR_CONFIG: |
  {
    "enabled": true,
    "model_id": "orchestrator",
    "auto_model_id": "auto",
    "max_orchestration_steps": 5,
    "default_preference": "balanced",
    "role_mapping": {
      "small": {
        "model_id": "qwen3-vl-8b-instruct",
        "tool_capability": "tool_calling_fallback",
        "max_context_tokens": 4000
      },
      "large": {
        "model_id": "gpt-oss-120b",
        "tool_capability": "response_only",
        "max_context_tokens": 16000
      },
      "vision": {
        "model_id": "qwen3-vl-8b-instruct",
        "tool_capability": "response_only",
        "max_context_tokens": 4000
      }
    },
    "fallback_model": "gpt-oss-120b"
  }
```

### Summary: Tool Calling Architecture

| Component | Tool Calling? | When |
|-----------|---------------|------|
| **Orchestrator (8B)** | ✅ Yes | Always - gathers context, decides routing |
| **Small Model (8B VL)** | ✅ Optional | Can extend trajectory if needed |
| **Large Model (120B)** | ⚠️ Fallback only | Only if trajectory incomplete |

**The Goal**: Build complete trajectories with fast models (orchestrator + small) before handing off to the slow model (large). The large model should ideally just synthesize the gathered context into a final response.

This aligns with your insight: *"get more complete trajectories to hand off using the faster model before calling the slower"*
