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
