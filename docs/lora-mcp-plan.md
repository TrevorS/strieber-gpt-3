# Z-Image Turbo LoRA Training MCP Tooling

Comprehensive implementation plan for adding LoRA training and inference capabilities to the strieber-gpt-3 stack using ai-toolkit.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Frontend Changes](#frontend-changes)
4. [Responses API Changes](#responses-api-changes)
5. [MCP Server Implementation](#mcp-server-implementation)
6. [Image Upload Flow](#image-upload-flow)
7. [Docker Infrastructure](#docker-infrastructure)
8. [Training Pipeline](#training-pipeline)
9. [Inference Pipeline](#inference-pipeline)
10. [Testing Strategy](#testing-strategy)
11. [Implementation Sequence](#implementation-sequence)

---

## Overview

### Goals

Add full LoRA training and inference pipeline:
- **Dataset Management** - Create datasets, upload training images, validate readiness
- **Training Control** - Start/stop/monitor training jobs via MCP tools
- **Inference Enhancement** - Use trained LoRAs in `zimage_turbo` generation

### Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Training framework | ai-toolkit | Native z-image turbo support, active development |
| Storage location | Host path `/home/trevor/lora-training/` | Easy inspection, manual backup |
| Training execution | Docker container spawned by MCP server | Isolated GPU workload, resource limits |
| Progress reporting | Job-based polling (NOT streaming) | Training takes 15-60+ minutes; ctx.report_progress() is server-side only |
| Sample image storage | ContainerStore with file citations | Matches existing code_interpreter image pattern |

---

## Critical Architecture Insights

This section documents how the existing stack handles images and long-running operations. The LoRA training implementation must follow these patterns exactly.

### 1. Image Storage Pattern (responses-api)

**Key Files:**
- `backend/responses-api/src/containers/store.rs` - Container/file storage
- `backend/responses-api/src/execution/streaming.rs:1142-1221` - Image extraction from tool results
- `backend/responses-api/src/containers/handlers.rs` - File serving endpoint

**How it works:**

```
Tool returns ImageContent(data=base64, mimeType="image/png")
         │
         ▼
responses-api: extract_content_from_result()
         │
         ├─► Decode base64 to bytes
         ├─► Determine extension from MIME type
         ├─► Generate filename: "output_N.ext"
         ├─► Store in ContainerStore: containers.add_file()
         └─► Create GeneratedFile { file_id: "cfile_xxx", container_id: "cntr_xxx" }
                    │
                    ▼
            Build Annotation::ContainerFileCitation
                    │
                    ▼
            Attach to final message output
                    │
                    ▼
Frontend: extractFileCitations() → builds URL:
    /v1/containers/{container_id}/files/{file_id}/content
```

**Container ID format:** `cntr_` + UUID
**File ID format:** `cfile_` + UUID
**Container TTL:** 20 minutes (DEFAULT_CONTAINER_TTL_SECS = 1200)

**Implication for LoRA training:**
- Sample images from training should be returned as `ImageContent` blocks
- responses-api will automatically store them in containers
- Frontend will display them via file citations
- No custom image handling needed in responses-api

### 2. Progress Reporting Reality

**Critical Finding:** `ctx.report_progress()` in MCP tools is **server-side logging only**. It does NOT stream back to the frontend.

**Evidence from streaming.rs:**
```rust
// Tool execution is awaited completely - no progress streaming
let result_text = match mcp.call_tool(&tc.name, arguments.clone()).await {
    Ok(tool_result) => {
        // AFTER execution completes, process the result
        let (text, generated_files) = extract_content_from_result(...)
        text
    }
    Err(e) => format!("Error: {}", e)
};
```

**Current tool execution times:**
- code_interpreter: ~5-30 seconds
- comfy_zimage: ~10-60 seconds
- web_search: ~3-10 seconds
- reader: ~2-10 seconds

**LoRA training: 15-60+ minutes** - Cannot use synchronous tool execution!

**Solution: Job-based polling pattern:**
1. `lora_start_training()` → Returns immediately with job_id
2. Training runs in background Docker container
3. `lora_training_status(job_id)` → Returns progress, loss, sample images
4. User/LLM polls status periodically
5. Final status includes all checkpoints

### 3. Frontend Output Item Flow

**Key Files:**
- `frontend/src/lib/api/responses.ts` - SSE event handling
- `frontend/src/lib/stores/conversations.svelte.ts` - State management
- `frontend/src/lib/components/chat/tools/OutputItemRenderer.svelte` - Rendering

**SSE Event Lifecycle:**
```
Backend sends:
  response.output_item.added    → Item appears with status="in_progress"
  response.function_call_arguments.delta → Arguments stream in
  response.output_item.done     → Item updated with status="completed"
  response.completed            → Final items with annotations

Frontend:
  onOutputItem(item, 'added')   → setOutputItem() adds to rawOutput[]
  onFunctionCallArgumentsDelta  → updateFunctionCallArguments() appends
  onOutputItem(item, 'done')    → setOutputItem() updates existing item
  response.completed            → Process message items with file citations
```

**Status transitions:** `in_progress` → `completed` | `failed`

**Implication:** LoRA tools will appear as standard function_call items. No special output item type needed initially - we can use existing FunctionCallDisplay.

### 4. How Images Flow from MCP to Frontend

**Complete trace:**

```
1. MCP Tool (Python)
   └─► Returns List[TextContent | ImageContent]
       ImageContent(type="image", data="base64...", mimeType="image/png")

2. responses-api (Rust) - streaming.rs:1142-1221
   └─► extract_content_from_result()
       ├─► Detects RawContent::Image
       ├─► Decodes base64 to bytes
       ├─► Stores via containers.add_file()
       └─► Returns GeneratedFile { file_id, filename, container_id }

3. responses-api (Rust) - streaming.rs:940-948
   └─► Builds Annotation::ContainerFileCitation
       └─► Attached to message's OutputContent::OutputText

4. SSE Stream to Frontend
   └─► response.completed event includes:
       output: [{ type: "message", content: [{ type: "output_text", annotations: [...] }] }]

5. Frontend - responses.ts (event processing)
   └─► onOutputItem(messageItem, 'done')
       └─► conversationStore.setOutputItem()

6. Frontend - AssistantMessage.svelte
   └─► extractFileCitations(rawOutput)
       └─► Finds container_file_citation annotations
       └─► Builds URL: /v1/containers/{cid}/files/{fid}/content

7. Frontend - Render
   └─► <img src={file.url} /> displays the image
```

**No custom handling needed** - just return `ImageContent` from MCP tools and the existing pipeline handles everything.

### 5. Error Handling Pattern

**Key File:** `backend/tools/mcp_servers/common/error_handling.py`

```python
from common.error_handling import create_error_result, ERROR_INVALID_INPUT

return create_error_result(
    error_message="Dataset has fewer than 5 images",
    error_code=ERROR_INVALID_INPUT,
    error_type="validation_error",
    additional_metadata={"image_count": 3, "minimum_required": 5}
)
```

**Critical:** Set `isError=True` on `CallToolResult` so responses-api knows the tool failed.

### 6. Design Implications Summary

| Aspect | Pattern to Follow |
|--------|-------------------|
| Sample images | Return `ImageContent` blocks - auto-stored in containers |
| Training progress | Job-based polling, NOT synchronous execution |
| Frontend display | Use existing `FunctionCallDisplay` initially |
| File citations | Automatic via `container_file_citation` annotations |
| Error handling | Use `create_error_result()` with `isError=True` |
| Long operations | Background Docker container + status polling |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ ChatInput.svelte                                                        ││
│  │ - File upload (images for training datasets)                            ││
│  │ - Text input with tool triggers                                         ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ ToolToggles.svelte                                                      ││
│  │ - [x] Image Generation (zimage_turbo - enhanced with LoRA)              ││
│  │ - [x] LoRA Training (NEW)                                               ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ tools/OutputItemRenderer.svelte                                         ││
│  │ - LoraTrainingDisplay.svelte (NEW - progress, metrics, samples)         ││
│  │ - FunctionCallDisplay.svelte (existing - for zimage_turbo + LoRA)       ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ HTTP (SSE streaming)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RESPONSES API (Rust)                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ execution/streaming.rs                                                  ││
│  │ - Tool execution loop (no changes needed)                               ││
│  │ - Image placeholder injection for non-vision models                     ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ mcp/client.rs                                                           ││
│  │ - Tool routing (auto-discovers new lora_trainer tools)                  ││
│  │ - No code changes needed (config-driven)                                ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ MCP_CONFIG environment variable                                         ││
│  │ + {"name":"lora_trainer","url":"http://mcp-lora-trainer:8000/mcp",     ││
│  │     "builtin_type":"lora_trainer"}                                      ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ MCP Protocol (HTTP)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            MCP SERVERS (Python)                              │
│                                                                              │
│  ┌───────────────────────────────┐  ┌───────────────────────────────────┐  │
│  │ mcp-lora-trainer :9145 (NEW)  │  │ mcp-comfy-zimage :9141 (ENHANCED) │  │
│  │                               │  │                                   │  │
│  │ Dataset Tools:                │  │ zimage_turbo (enhanced):          │  │
│  │ - lora_create_dataset         │  │ + lora_name: Optional[str]        │  │
│  │ - lora_upload_images          │  │ + lora_strength: float = 1.0      │  │
│  │ - lora_validate_dataset       │  │                                   │  │
│  │ - lora_list_datasets          │  │ zimage_controlnet (unchanged)     │  │
│  │                               │  │                                   │  │
│  │ Training Tools:               │  │ lora_list_available (NEW):        │  │
│  │ - lora_start_training         │  │ - List .safetensors in loras/     │  │
│  │ - lora_training_status        │  │                                   │  │
│  │ - lora_stop_training          │  └───────────────────────────────────┘  │
│  │ - lora_list_jobs              │                                         │
│  │                               │                                         │
│  │ Checkpoint Tools:             │                                         │
│  │ - lora_list_checkpoints       │                                         │
│  │ - lora_promote_checkpoint     │                                         │
│  └───────────────────────────────┘                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Docker API
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AI-TOOLKIT TRAINER (GPU)                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ Container: strieber-ai-toolkit                                          ││
│  │ - ai-toolkit Python environment                                         ││
│  │ - Training adapters (v1, v2)                                            ││
│  │ - GPU access via NVIDIA runtime                                         ││
│  │ - Spawned on-demand by lora_start_training                              ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Volume mounts
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            STORAGE (Host Paths)                              │
│                                                                              │
│  /home/trevor/lora-training/          /home/trevor/models/comfyui/loras/    │
│  ├── datasets/                        ├── my_character.safetensors          │
│  │   └── my_character/                ├── pixel_art_style.safetensors       │
│  │       ├── metadata.json            └── ...                               │
│  │       └── images/                                                        │
│  ├── configs/                                                               │
│  ├── outputs/                                                               │
│  │   └── job_abc123/                                                        │
│  │       ├── checkpoints/                                                   │
│  │       └── samples/                                                       │
│  └── jobs.json                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Frontend Changes

### 1. Tool Configuration

**File**: `frontend/src/lib/components/settings/ToolToggles.svelte`

Add new tool toggle:

```typescript
const AVAILABLE_TOOLS = [
  // ... existing tools ...
  { id: 'lora_trainer', name: 'LoRA Training', description: 'Train custom image models', icon: Zap },
];
```

### 2. API Integration

**File**: `frontend/src/lib/api/responses.ts`

Add `lora_trainer` to tool list in `sendMessageStreaming()` (line ~180):

```typescript
const enabledTools = [];
if (tools.web_search) enabledTools.push({ type: 'web_search' });
if (tools.code_interpreter) enabledTools.push({ type: 'code_interpreter' });
// ... existing tools ...
if (tools.lora_trainer) enabledTools.push({ type: 'lora_trainer' });  // NEW
```

### 3. Output Item Type Definition

**File**: `frontend/src/lib/stores/types.ts`

Add type guard for LoRA training output items:

```typescript
export interface LoraTrainingCallOutput {
  type: 'lora_training_call';
  id: string;
  status: 'in_progress' | 'completed' | 'failed';
  call_id: string;
  name: string;  // Tool name (e.g., 'lora_start_training')
  arguments: string;  // JSON string of arguments
  output?: string;  // Tool result

  // Training-specific fields (populated during training)
  job_id?: string;
  dataset_name?: string;
  current_step?: number;
  total_steps?: number;
  latest_loss?: number;
  checkpoints?: string[];
  sample_images?: string[];  // Base64 sample images
}

export function isLoraTrainingItem(item: OutputItem): item is LoraTrainingCallOutput {
  return item.type === 'lora_training_call';
}
```

### 4. Training Display Component

**File**: `frontend/src/lib/components/chat/tools/LoraTrainingDisplay.svelte` (NEW)

```svelte
<script lang="ts">
  import { Zap } from 'lucide-svelte';
  import ToolCallWrapper from './ToolCallWrapper.svelte';
  import * as Progress from '$lib/components/ui/progress';
  import type { LoraTrainingCallOutput } from '$lib/stores/types';

  interface Props {
    item: LoraTrainingCallOutput;
  }

  let { item }: Props = $props();

  const progressPercent = $derived(
    item.total_steps ? Math.round((item.current_step || 0) / item.total_steps * 100) : 0
  );
</script>

<ToolCallWrapper
  title={item.name === 'lora_start_training' ? 'Training LoRA' : item.name}
  status={item.status}
  icon={Zap}
  defaultOpen={true}
>
  <div class="space-y-3">
    <!-- Training Progress -->
    {#if item.job_id}
      <div class="text-sm">
        <span class="text-muted-foreground">Job:</span> {item.job_id}
      </div>

      {#if item.current_step !== undefined}
        <div class="space-y-1">
          <div class="flex justify-between text-sm">
            <span>Step {item.current_step} / {item.total_steps}</span>
            <span>{progressPercent}%</span>
          </div>
          <Progress.Root value={progressPercent} class="h-2" />
        </div>
      {/if}

      {#if item.latest_loss !== undefined}
        <div class="text-sm">
          <span class="text-muted-foreground">Loss:</span> {item.latest_loss.toFixed(4)}
        </div>
      {/if}
    {/if}

    <!-- Sample Images -->
    {#if item.sample_images?.length}
      <div class="grid grid-cols-2 gap-2">
        {#each item.sample_images as img}
          <img src="data:image/png;base64,{img}" alt="Training sample" class="rounded" />
        {/each}
      </div>
    {/if}

    <!-- Arguments (collapsed) -->
    <details class="text-xs">
      <summary class="cursor-pointer text-muted-foreground">Arguments</summary>
      <pre class="mt-1 p-2 bg-muted rounded overflow-x-auto">{item.arguments}</pre>
    </details>

    <!-- Output -->
    {#if item.output}
      <pre class="text-xs p-2 bg-muted rounded overflow-x-auto whitespace-pre-wrap">{item.output}</pre>
    {/if}
  </div>
</ToolCallWrapper>
```

### 5. Renderer Integration

**File**: `frontend/src/lib/components/chat/tools/OutputItemRenderer.svelte`

Add condition for LoRA training items:

```svelte
<script lang="ts">
  // ... existing imports ...
  import LoraTrainingDisplay from './LoraTrainingDisplay.svelte';
  import { isLoraTrainingItem } from '$lib/stores/types';
</script>

{#if isReasoningItem(item)}
  <ReasoningDisplay {item} />
{:else if isLoraTrainingItem(item)}
  <LoraTrainingDisplay {item} />
{:else if isWebSearchItem(item)}
  <!-- ... existing conditions ... -->
{/if}
```

### 6. Dataset Image Upload Support

**File**: `frontend/src/lib/utils/files.ts`

The existing image upload flow already supports the formats needed. Training images are uploaded as base64 via the normal chat attachment flow. The MCP tool `lora_upload_images` receives them as base64 strings.

No changes needed - existing `createAttachment()` handles image conversion.

---

## Responses API Changes

### Overview

The responses-api requires **minimal changes** because:
1. Tool routing is config-driven via `MCP_CONFIG`
2. Tool discovery happens automatically via `list_tools` RPC
3. Image handling already works (extract → placeholder → pass to tool)

### 1. MCP Configuration Update

**File**: `compose.yml` (environment section for responses-api)

Add lora_trainer server to MCP_CONFIG:

```yaml
- MCP_CONFIG={"servers":[
    {"name":"weather","url":"http://mcp-weather:8000/mcp","builtin_type":"weather"},
    {"name":"web_search","url":"http://mcp-web-search:8000/mcp","builtin_type":"web_search"},
    {"name":"code_interpreter","url":"http://mcp-code-interpreter:8000/mcp","builtin_type":"code_interpreter"},
    {"name":"reader","url":"http://mcp-reader:8000/mcp","builtin_type":"reader"},
    {"name":"zimage","url":"http://mcp-comfy-zimage:8000/mcp","builtin_type":"zimage_turbo"},
    {"name":"lora_trainer","url":"http://mcp-lora-trainer:8000/mcp","builtin_type":"lora_trainer"}
  ]}
```

### 2. No Code Changes Required

The following components work automatically:

| Component | Why No Changes Needed |
|-----------|----------------------|
| `mcp/client.rs` | Auto-discovers tools via `list_tools` RPC |
| `execution/executor.rs` | Generic tool execution loop handles any MCP tool |
| `execution/streaming.rs` | Progress events already supported |
| `translation/request.rs` | Image extraction works for all tools |

### 3. Image Flow for Training

When user attaches training images:

1. **Frontend** converts to base64 data URLs
2. **Request translation** (`request.rs:333-363`) extracts images
3. **Placeholder injection** (`streaming.rs:143-159`) for non-vision models creates:
   ```
   [Attached image: image_0. Use image_data: "image_0" in tool calls to reference this image.]
   ```
4. **Tool execution** passes `image_0`, `image_1`, etc. to MCP tools
5. **MCP server** (`lora_upload_images`) receives image references and retrieves actual data

**Note**: For bulk image uploads (5-15 training images), we'll need a batched upload approach since attaching 15 images to a chat message is impractical. See [Image Upload Flow](#image-upload-flow) section.

---

## MCP Server Implementation

### New Server: `mcp-lora-trainer`

**Location**: `backend/tools/mcp_servers/lora_trainer/`

```
lora_trainer/
├── __init__.py
├── server.py           # MCP tool definitions
├── dataset_manager.py  # Dataset CRUD operations
├── training_runner.py  # Docker-based training execution
├── job_store.py        # Job state persistence
├── models.py           # Pydantic schemas
└── config/
    └── zimage_turbo_base.yaml
```

### models.py - Data Schemas

```python
"""Pydantic models for LoRA training."""

from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class LoRAType(str, Enum):
    CHARACTER = "character"
    STYLE = "style"
    CONCEPT = "concept"


class TrainingStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class DatasetMetadata(BaseModel):
    """Metadata for a training dataset."""
    name: str
    trigger_token: str
    lora_type: LoRAType
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    image_count: int = 0
    has_captions: bool = False


class TrainingConfig(BaseModel):
    """ai-toolkit training configuration."""
    model: str = "z-image-turbo"
    training_adapter: str = "/weights/z-image-turbo/training_adapter_v2.safetensors"
    dataset: str
    image_size: int = 1024
    steps: int = 3000
    batch_size: int = 1
    lr: float = 0.0001
    lora_rank: int = 8
    checkpoint_every: int = 500
    sample_every: int = 250
    sample_prompts: List[str] = []


class TrainingJob(BaseModel):
    """Training job state."""
    job_id: str
    dataset_name: str
    trigger_token: str
    config: TrainingConfig
    status: TrainingStatus = TrainingStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    current_step: int = 0
    total_steps: int = 3000
    latest_loss: Optional[float] = None
    checkpoints: List[str] = []
    sample_images: List[str] = []  # Paths to sample images
    error_message: Optional[str] = None
    container_id: Optional[str] = None  # Docker container ID
```

### server.py - MCP Tool Definitions

```python
"""MCP server for LoRA training management."""

import base64
import json
import logging
from pathlib import Path
from typing import List, Literal, Optional

from mcp.server.fastmcp import Context, FastMCP
from mcp.types import TextContent, ImageContent

from lora_trainer.models import (
    DatasetMetadata, TrainingConfig, TrainingJob,
    LoRAType, TrainingStatus
)
from lora_trainer.dataset_manager import DatasetManager
from lora_trainer.training_runner import TrainingRunner
from lora_trainer.job_store import JobStore


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("lora_trainer", host="0.0.0.0")

# Initialize managers
BASE_PATH = Path("/data")
dataset_manager = DatasetManager(BASE_PATH / "datasets")
job_store = JobStore(BASE_PATH / "jobs.json")
training_runner = TrainingRunner(
    datasets_path=BASE_PATH / "datasets",
    outputs_path=BASE_PATH / "outputs",
    configs_path=BASE_PATH / "configs",
    job_store=job_store,
)


# ============================================================================
# Dataset Management Tools
# ============================================================================

@mcp.tool()
async def lora_create_dataset(
    name: str,
    trigger_token: str,
    lora_type: Literal["character", "style", "concept"] = "character",
    description: Optional[str] = None,
    ctx: Context = None,
) -> List[TextContent]:
    """Create a new LoRA training dataset.

    TRIGGER TOKEN GUIDELINES:
    - Use unique, non-dictionary words (e.g., "ohwx", "sks", "xyz123")
    - Keep it short (3-6 characters)
    - Avoid common words that appear in training data

    LORA TYPES:
    - character: Person/subject identity (faces, full body)
    - style: Artistic style transfer (painting style, color palette)
    - concept: Object or abstract concept

    Args:
        name: Dataset name (alphanumeric + underscores)
        trigger_token: Unique token to trigger the LoRA (e.g., "ohwx")
        lora_type: Type of LoRA being trained
        description: Optional description

    Returns:
        Confirmation with dataset path.
    """
    try:
        metadata = dataset_manager.create_dataset(
            name=name,
            trigger_token=trigger_token,
            lora_type=LoRAType(lora_type),
            description=description,
        )
        return [TextContent(
            type="text",
            text=f"Created dataset '{name}' with trigger token '{trigger_token}'.\n"
                 f"Type: {lora_type}\n"
                 f"Path: {dataset_manager.get_dataset_path(name)}\n\n"
                 f"Next: Upload training images with lora_upload_images."
        )]
    except ValueError as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


@mcp.tool()
async def lora_upload_images(
    dataset_name: str,
    images: List[str],
    captions: Optional[List[str]] = None,
    ctx: Context = None,
) -> List[TextContent]:
    """Upload training images to a dataset.

    IMAGE REQUIREMENTS:
    - Resolution: 1024x1024 or 1536x1536 optimal
    - Format: PNG or JPEG (base64 encoded)
    - Count: 5-15 images minimum (9 sufficient for identity)
    - Diversity: Vary poses, expressions, lighting

    CAPTIONS (optional):
    - If provided, include trigger token in each caption
    - Example: "ohwx, portrait photo, studio lighting"

    Args:
        dataset_name: Name of existing dataset
        images: List of base64-encoded images
        captions: Optional list of captions (same length as images)

    Returns:
        Upload summary with validation results.
    """
    try:
        if ctx:
            await ctx.report_progress(0, len(images), "Starting upload...")

        results = []
        for i, img_data in enumerate(images):
            # Decode and save image
            img_bytes = base64.b64decode(img_data)
            caption = captions[i] if captions and i < len(captions) else None
            filename = dataset_manager.add_image(dataset_name, img_bytes, caption)
            results.append(filename)

            if ctx:
                await ctx.report_progress(i + 1, len(images), f"Uploaded {filename}")

        metadata = dataset_manager.get_metadata(dataset_name)
        return [TextContent(
            type="text",
            text=f"Uploaded {len(results)} images to '{dataset_name}'.\n"
                 f"Total images: {metadata.image_count}\n"
                 f"Has captions: {metadata.has_captions}\n\n"
                 f"Use lora_validate_dataset to check readiness."
        )]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


@mcp.tool()
async def lora_validate_dataset(
    dataset_name: str,
    ctx: Context = None,
) -> List[TextContent]:
    """Validate dataset readiness for training.

    Checks:
    - Image count (minimum 5)
    - Image resolutions
    - Caption format (if present)
    - Trigger token consistency

    Args:
        dataset_name: Name of dataset to validate

    Returns:
        Validation report with any issues found.
    """
    try:
        report = dataset_manager.validate_dataset(dataset_name)
        return [TextContent(type="text", text=report)]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


@mcp.tool()
async def lora_list_datasets(
    ctx: Context = None,
) -> List[TextContent]:
    """List all available training datasets.

    Returns:
        List of datasets with metadata.
    """
    datasets = dataset_manager.list_datasets()
    if not datasets:
        return [TextContent(type="text", text="No datasets found.")]

    lines = ["Available datasets:\n"]
    for ds in datasets:
        lines.append(f"- {ds.name}")
        lines.append(f"  Trigger: {ds.trigger_token}")
        lines.append(f"  Type: {ds.lora_type.value}")
        lines.append(f"  Images: {ds.image_count}")
        lines.append("")

    return [TextContent(type="text", text="\n".join(lines))]


# ============================================================================
# Training Control Tools
# ============================================================================

@mcp.tool()
async def lora_start_training(
    dataset_name: str,
    steps: int = 3000,
    learning_rate: float = 0.0001,
    lora_rank: int = 8,
    checkpoint_every: int = 500,
    sample_every: int = 250,
    sample_prompts: Optional[List[str]] = None,
    ctx: Context = None,
) -> List[TextContent]:
    """Start LoRA training job.

    RECOMMENDED SETTINGS:
    - steps: 3000 for 5-15 image datasets
    - learning_rate: 0.0001 (decrease to 0.00005 if overfitting)
    - lora_rank: 8 (increase to 16 for more capacity)
    - checkpoint_every: 500 (for recovery and comparison)
    - sample_every: 250 (monitor convergence)

    SAMPLE PROMPTS:
    - Include trigger token in each prompt
    - Use fixed seeds for consistent comparison
    - Example: ["ohwx, portrait photo", "ohwx on a beach, sunset"]

    Args:
        dataset_name: Name of prepared dataset
        steps: Total training steps (default 3000)
        learning_rate: Learning rate (default 0.0001)
        lora_rank: LoRA rank/dimension (default 8)
        checkpoint_every: Save checkpoint interval (default 500)
        sample_every: Generate sample interval (default 250)
        sample_prompts: Prompts for sample generation

    Returns:
        Job ID for status polling.
    """
    try:
        # Validate dataset exists and is ready
        metadata = dataset_manager.get_metadata(dataset_name)
        if metadata.image_count < 5:
            return [TextContent(
                type="text",
                text=f"Error: Dataset has {metadata.image_count} images. Minimum 5 required."
            )]

        # Build config
        config = TrainingConfig(
            dataset=dataset_name,
            steps=steps,
            lr=learning_rate,
            lora_rank=lora_rank,
            checkpoint_every=checkpoint_every,
            sample_every=sample_every,
            sample_prompts=sample_prompts or [
                f"{metadata.trigger_token}, portrait, studio lighting",
                f"{metadata.trigger_token}, outdoor, natural light",
            ],
        )

        # Start training
        job = await training_runner.start_training(
            dataset_name=dataset_name,
            trigger_token=metadata.trigger_token,
            config=config,
        )

        return [TextContent(
            type="text",
            text=f"Started training job: {job.job_id}\n"
                 f"Dataset: {dataset_name}\n"
                 f"Steps: {steps}\n"
                 f"LoRA rank: {lora_rank}\n"
                 f"Learning rate: {learning_rate}\n\n"
                 f"Use lora_training_status('{job.job_id}') to monitor progress."
        )]
    except Exception as e:
        logger.error(f"Training start error: {e}", exc_info=True)
        return [TextContent(type="text", text=f"Error: {str(e)}")]


@mcp.tool()
async def lora_training_status(
    job_id: str,
    ctx: Context = None,
) -> List[TextContent | ImageContent]:
    """Get training job status and progress.

    Args:
        job_id: Job ID from lora_start_training

    Returns:
        Status, progress, loss, and sample images (if available).
    """
    try:
        job = job_store.get_job(job_id)
        if not job:
            return [TextContent(type="text", text=f"Job not found: {job_id}")]

        content: List[TextContent | ImageContent] = []

        # Status text
        status_lines = [
            f"Job: {job.job_id}",
            f"Dataset: {job.dataset_name}",
            f"Status: {job.status.value}",
            f"Progress: {job.current_step}/{job.total_steps} ({job.current_step/job.total_steps*100:.1f}%)",
        ]

        if job.latest_loss is not None:
            status_lines.append(f"Latest loss: {job.latest_loss:.4f}")

        if job.checkpoints:
            status_lines.append(f"Checkpoints: {len(job.checkpoints)}")
            status_lines.append(f"  Latest: {job.checkpoints[-1]}")

        if job.error_message:
            status_lines.append(f"Error: {job.error_message}")

        content.append(TextContent(type="text", text="\n".join(status_lines)))

        # Include sample images if available
        for sample_path in job.sample_images[-2:]:  # Last 2 samples
            try:
                with open(sample_path, "rb") as f:
                    img_base64 = base64.b64encode(f.read()).decode()
                content.append(ImageContent(
                    type="image",
                    data=img_base64,
                    mimeType="image/png",
                ))
            except Exception:
                pass

        return content
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


@mcp.tool()
async def lora_stop_training(
    job_id: str,
    ctx: Context = None,
) -> List[TextContent]:
    """Stop a running training job.

    The latest checkpoint will be preserved.

    Args:
        job_id: Job ID to stop

    Returns:
        Confirmation with available checkpoints.
    """
    try:
        job = await training_runner.stop_training(job_id)
        return [TextContent(
            type="text",
            text=f"Stopped job: {job_id}\n"
                 f"Checkpoints available: {len(job.checkpoints)}\n"
                 f"Use lora_list_checkpoints('{job_id}') to see options."
        )]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


@mcp.tool()
async def lora_list_jobs(
    status: Optional[Literal["pending", "running", "completed", "failed", "stopped"]] = None,
    ctx: Context = None,
) -> List[TextContent]:
    """List all training jobs.

    Args:
        status: Filter by status (optional)

    Returns:
        List of jobs with summary info.
    """
    jobs = job_store.list_jobs(status=TrainingStatus(status) if status else None)
    if not jobs:
        return [TextContent(type="text", text="No jobs found.")]

    lines = ["Training Jobs:\n"]
    for job in jobs:
        lines.append(f"- {job.job_id} ({job.status.value})")
        lines.append(f"  Dataset: {job.dataset_name}")
        lines.append(f"  Progress: {job.current_step}/{job.total_steps}")
        if job.checkpoints:
            lines.append(f"  Checkpoints: {len(job.checkpoints)}")
        lines.append("")

    return [TextContent(type="text", text="\n".join(lines))]


# ============================================================================
# Checkpoint Management Tools
# ============================================================================

@mcp.tool()
async def lora_list_checkpoints(
    job_id: str,
    ctx: Context = None,
) -> List[TextContent]:
    """List all checkpoints for a training job.

    Args:
        job_id: Job ID

    Returns:
        List of checkpoints with step numbers.
    """
    try:
        job = job_store.get_job(job_id)
        if not job:
            return [TextContent(type="text", text=f"Job not found: {job_id}")]

        if not job.checkpoints:
            return [TextContent(type="text", text="No checkpoints available yet.")]

        lines = [f"Checkpoints for job {job_id}:\n"]
        for ckpt in job.checkpoints:
            lines.append(f"- {ckpt}")

        lines.append(f"\nUse lora_promote_checkpoint to copy to loras directory.")
        return [TextContent(type="text", text="\n".join(lines))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


@mcp.tool()
async def lora_promote_checkpoint(
    job_id: str,
    checkpoint_name: str,
    output_name: Optional[str] = None,
    ctx: Context = None,
) -> List[TextContent]:
    """Promote a checkpoint to the active LoRA directory.

    Copies the checkpoint to ComfyUI's loras directory, making it
    available for inference with zimage_turbo.

    Args:
        job_id: Job ID
        checkpoint_name: Checkpoint filename (from lora_list_checkpoints)
        output_name: Output filename (default: dataset_name.safetensors)

    Returns:
        Confirmation with usage instructions.
    """
    try:
        job = job_store.get_job(job_id)
        if not job:
            return [TextContent(type="text", text=f"Job not found: {job_id}")]

        output_filename = training_runner.promote_checkpoint(
            job_id=job_id,
            checkpoint_name=checkpoint_name,
            output_name=output_name or job.dataset_name,
        )

        return [TextContent(
            type="text",
            text=f"Promoted checkpoint to: {output_filename}\n\n"
                 f"Usage with zimage_turbo:\n"
                 f"  lora_name: \"{Path(output_filename).stem}\"\n"
                 f"  lora_strength: 1.0\n"
                 f"  prompt: \"{job.trigger_token}, your description here\""
        )]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


# ============================================================================
# Server Instance
# ============================================================================

class LoraTrainerServer:
    def __init__(self):
        self.mcp = mcp
        self.logger = logger

    def get_mcp(self):
        return self.mcp


server = LoraTrainerServer()


def get_mcp():
    return server.get_mcp()


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
```

### training_runner.py - Docker-Based Training

```python
"""Training job execution via Docker."""

import asyncio
import json
import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import docker
from docker.errors import ContainerError, ImageNotFound, APIError

from lora_trainer.models import TrainingConfig, TrainingJob, TrainingStatus
from lora_trainer.job_store import JobStore


logger = logging.getLogger(__name__)

DOCKER_IMAGE = "strieber-ai-toolkit:latest"
LORAS_OUTPUT_PATH = Path("/output/loras")  # Mounted from host


class TrainingRunner:
    """Manages ai-toolkit training jobs via Docker."""

    def __init__(
        self,
        datasets_path: Path,
        outputs_path: Path,
        configs_path: Path,
        job_store: JobStore,
    ):
        self.datasets_path = datasets_path
        self.outputs_path = outputs_path
        self.configs_path = configs_path
        self.job_store = job_store
        self._docker = docker.from_env()
        self._active_containers: dict[str, str] = {}  # job_id -> container_id

    async def start_training(
        self,
        dataset_name: str,
        trigger_token: str,
        config: TrainingConfig,
    ) -> TrainingJob:
        """Start a training job in Docker container."""
        job_id = str(uuid.uuid4())[:8]

        # Create job
        job = TrainingJob(
            job_id=job_id,
            dataset_name=dataset_name,
            trigger_token=trigger_token,
            config=config,
            total_steps=config.steps,
        )
        self.job_store.save_job(job)

        # Create output directory
        job_output_path = self.outputs_path / job_id
        job_output_path.mkdir(parents=True, exist_ok=True)
        (job_output_path / "checkpoints").mkdir()
        (job_output_path / "samples").mkdir()

        # Generate config YAML
        config_path = self._write_config(job_id, dataset_name, trigger_token, config)

        # Start training in background
        asyncio.create_task(self._run_training(job_id, config_path))

        return job

    def _write_config(
        self,
        job_id: str,
        dataset_name: str,
        trigger_token: str,
        config: TrainingConfig,
    ) -> Path:
        """Generate ai-toolkit config YAML."""
        config_content = {
            "job_id": job_id,
            "model": {
                "name": config.model,
                "training_adapter": config.training_adapter,
                "quantize": False,  # DGX Spark has plenty of VRAM
            },
            "dataset": {
                "path": f"/datasets/{dataset_name}/images",
                "resolution": config.image_size,
                "trigger_word": trigger_token,
            },
            "training": {
                "steps": config.steps,
                "batch_size": config.batch_size,
                "learning_rate": config.lr,
                "lora_rank": config.lora_rank,
                "save_every": config.checkpoint_every,
            },
            "sample": {
                "every": config.sample_every,
                "prompts": [
                    {"text": p, "seed": 42 + i}
                    for i, p in enumerate(config.sample_prompts)
                ],
            },
            "output": {
                "path": f"/outputs/{job_id}",
                "checkpoints_dir": "checkpoints",
                "samples_dir": "samples",
            },
        }

        config_path = self.configs_path / f"{job_id}.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        import yaml
        with open(config_path, "w") as f:
            yaml.dump(config_content, f)

        return config_path

    async def _run_training(self, job_id: str, config_path: Path):
        """Run training in Docker container."""
        job = self.job_store.get_job(job_id)
        if not job:
            return

        job.status = TrainingStatus.RUNNING
        job.started_at = datetime.utcnow()
        self.job_store.save_job(job)

        try:
            # Run container
            container = await asyncio.to_thread(
                self._docker.containers.run,
                DOCKER_IMAGE,
                command=["python", "run.py", f"/configs/{job_id}.yaml"],
                volumes={
                    str(self.datasets_path.parent): {"bind": "/data", "mode": "rw"},
                    str(config_path.parent): {"bind": "/configs", "mode": "ro"},
                },
                device_requests=[
                    docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
                ],
                shm_size="16g",
                ipc_mode="host",
                detach=True,
                remove=False,
            )

            self._active_containers[job_id] = container.id
            job.container_id = container.id
            self.job_store.save_job(job)

            # Monitor progress
            await self._monitor_progress(job_id, container)

            # Wait for completion
            result = await asyncio.to_thread(container.wait)

            if result["StatusCode"] == 0:
                job.status = TrainingStatus.COMPLETED
            else:
                job.status = TrainingStatus.FAILED
                logs = await asyncio.to_thread(container.logs, tail=100)
                job.error_message = logs.decode()[-500:]

        except ContainerError as e:
            job.status = TrainingStatus.FAILED
            job.error_message = str(e)
        except ImageNotFound:
            job.status = TrainingStatus.FAILED
            job.error_message = f"Docker image not found: {DOCKER_IMAGE}"
        except APIError as e:
            job.status = TrainingStatus.FAILED
            job.error_message = f"Docker API error: {e}"
        except Exception as e:
            job.status = TrainingStatus.FAILED
            job.error_message = str(e)
            logger.error(f"Training error: {e}", exc_info=True)
        finally:
            job.completed_at = datetime.utcnow()
            self.job_store.save_job(job)
            self._active_containers.pop(job_id, None)

            # Cleanup container
            try:
                container = self._docker.containers.get(job.container_id)
                await asyncio.to_thread(container.remove)
            except Exception:
                pass

    async def _monitor_progress(self, job_id: str, container):
        """Monitor training progress by parsing logs."""
        job = self.job_store.get_job(job_id)

        # Poll logs for progress
        while True:
            await asyncio.sleep(5)

            # Check if container still running
            try:
                container.reload()
                if container.status != "running":
                    break
            except Exception:
                break

            # Parse logs for progress
            try:
                logs = await asyncio.to_thread(container.logs, tail=50)
                self._parse_progress(job_id, logs.decode())
            except Exception:
                pass

            # Check for new checkpoints/samples
            self._scan_outputs(job_id)

    def _parse_progress(self, job_id: str, logs: str):
        """Parse ai-toolkit output for step/loss updates."""
        job = self.job_store.get_job(job_id)
        if not job:
            return

        # Parse patterns like "Step 500/3000 | Loss: 0.0234"
        import re
        step_pattern = r"Step (\d+)/(\d+)"
        loss_pattern = r"Loss: ([\d.]+)"

        for line in logs.split("\n"):
            step_match = re.search(step_pattern, line)
            if step_match:
                job.current_step = int(step_match.group(1))
                job.total_steps = int(step_match.group(2))

            loss_match = re.search(loss_pattern, line)
            if loss_match:
                job.latest_loss = float(loss_match.group(1))

        self.job_store.save_job(job)

    def _scan_outputs(self, job_id: str):
        """Scan for new checkpoints and sample images."""
        job = self.job_store.get_job(job_id)
        if not job:
            return

        output_path = self.outputs_path / job_id

        # Scan checkpoints
        checkpoints_dir = output_path / "checkpoints"
        if checkpoints_dir.exists():
            checkpoints = sorted(checkpoints_dir.glob("*.safetensors"))
            job.checkpoints = [str(c.name) for c in checkpoints]

        # Scan samples
        samples_dir = output_path / "samples"
        if samples_dir.exists():
            samples = sorted(samples_dir.glob("*.png"))
            job.sample_images = [str(s) for s in samples]

        self.job_store.save_job(job)

    async def stop_training(self, job_id: str) -> TrainingJob:
        """Stop a running training job."""
        job = self.job_store.get_job(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        if job.container_id:
            try:
                container = self._docker.containers.get(job.container_id)
                await asyncio.to_thread(container.stop, timeout=10)
            except Exception as e:
                logger.warning(f"Failed to stop container: {e}")

        job.status = TrainingStatus.STOPPED
        job.completed_at = datetime.utcnow()
        self.job_store.save_job(job)

        return job

    def promote_checkpoint(
        self,
        job_id: str,
        checkpoint_name: str,
        output_name: str,
    ) -> str:
        """Copy checkpoint to loras directory."""
        job = self.job_store.get_job(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        src = self.outputs_path / job_id / "checkpoints" / checkpoint_name
        if not src.exists():
            raise ValueError(f"Checkpoint not found: {checkpoint_name}")

        dst = LORAS_OUTPUT_PATH / f"{output_name}.safetensors"
        shutil.copy2(src, dst)

        return str(dst)
```

### Enhanced comfy_zimage Server

**File**: `backend/tools/mcp_servers/comfy_zimage/server.py`

Changes to `zimage_turbo` function (add parameters):

```python
# Add these parameters to zimage_turbo function signature
@mcp.tool()
async def zimage_turbo(
    prompt: str,
    size: ImageSize = "1024x1024",
    n: int = 1,
    seed: Optional[int] = None,
    steps: int = 8,
    lora_name: Optional[str] = None,      # NEW
    lora_strength: float = 1.0,           # NEW
    ctx: Context = None,
) -> List[TextContent | ImageContent]:
    """Generate images from text descriptions.

    [... existing docstring ...]

    LORA SUPPORT:
    If you have trained a custom LoRA, you can use it:
    - lora_name: LoRA filename (without .safetensors extension)
    - lora_strength: Influence strength (0.0-2.0, default 1.0)
    - Include the trigger token in your prompt

    Example: "ohwx, portrait photo, studio lighting" with lora_name="my_character"

    Use lora_list_available to see available LoRAs.
    """
```

Add workflow selection logic:

```python
    # Select workflow based on LoRA usage
    if lora_name:
        workflow = json.loads(json.dumps(ZIMAGE_LORA_WORKFLOW))
        nodes = ZIMAGE_LORA_NODES

        # Configure LoRA loader node
        workflow[nodes["lora_loader"]]["inputs"]["lora_name"] = f"{lora_name}.safetensors"
        workflow[nodes["lora_loader"]]["inputs"]["strength_model"] = lora_strength
    else:
        workflow = json.loads(json.dumps(ZIMAGE_WORKFLOW))
        nodes = ZIMAGE_NODES

    # Rest of implementation uses `nodes` variable for node IDs
```

Add new tool:

```python
@mcp.tool()
async def lora_list_available(
    ctx: Context = None,
) -> List[TextContent]:
    """List available LoRAs for z-image turbo inference.

    Returns LoRA names that can be used with zimage_turbo's lora_name parameter.
    """
    lora_path = Path("/models/loras")

    if not lora_path.exists():
        return [TextContent(type="text", text="No LoRAs directory found.")]

    loras = []
    for f in sorted(lora_path.glob("*.safetensors")):
        size_mb = f.stat().st_size / (1024 * 1024)
        loras.append(f"- {f.stem} ({size_mb:.1f} MB)")

    if not loras:
        return [TextContent(type="text", text="No LoRAs found in loras directory.")]

    return [TextContent(
        type="text",
        text="Available LoRAs:\n" + "\n".join(loras) + "\n\n"
             "Usage: zimage_turbo(prompt='...', lora_name='name_here', lora_strength=1.0)"
    )]
```

---

## Image Upload Flow

### Challenge

Training requires 5-15 images. Attaching 15 images to a single chat message is impractical and could exceed context limits.

### Solution: Batched Upload via Tool Calls

The LLM orchestrates multiple `lora_upload_images` calls:

```
User: "I want to train a LoRA of my cat. Here are 3 photos to start."
[Attaches 3 images]

LLM: "I'll create a dataset and upload these images."
     1. Calls lora_create_dataset(name="my_cat", trigger_token="mycat", lora_type="character")
     2. Calls lora_upload_images(dataset_name="my_cat", images=[img0, img1, img2])

User: "Here are 6 more photos"
[Attaches 6 images]

LLM: "Adding more images to the dataset."
     Calls lora_upload_images(dataset_name="my_cat", images=[img0, img1, img2, img3, img4, img5])

User: "That's enough, start training"

LLM: "Let me validate and start training."
     1. Calls lora_validate_dataset(dataset_name="my_cat")
     2. Calls lora_start_training(dataset_name="my_cat", ...)
```

### Image Data Flow

```
Frontend                    responses-api                MCP Server
   │                             │                            │
   │ Attach images               │                            │
   │ (converted to base64)       │                            │
   │─────────────────────────────►                            │
   │                             │                            │
   │                             │ extract_attached_images()  │
   │                             │ images → [image_0, ...]    │
   │                             │                            │
   │                             │ For non-vision model:      │
   │                             │ replace with placeholder   │
   │                             │                            │
   │                             │ Tool call with image refs  │
   │                             │────────────────────────────►
   │                             │                            │
   │                             │        lora_upload_images  │
   │                             │        (images=["image_0"])│
   │                             │                            │
   │                             │◄────────────────────────────
   │                             │                            │
   │◄─────────────────────────────                            │
```

### Alternative: Direct File Upload Endpoint

For bulk uploads, consider adding a separate REST endpoint:

```
POST /datasets/{name}/images
Content-Type: multipart/form-data

files: [image1.png, image2.png, ...]
```

This bypasses the chat flow for large batches. Can be implemented later if needed.

---

## Docker Infrastructure

### New Dockerfile: ai-toolkit Trainer

**File**: `backend/tools/ai-toolkit/Dockerfile`

```dockerfile
# AI-Toolkit training container for Z-Image Turbo LoRA training
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Clone ai-toolkit
RUN git clone https://github.com/ostris/ai-toolkit.git . && \
    git submodule update --init --recursive

# Create venv and install dependencies
RUN python3.11 -m venv venv && \
    . venv/bin/activate && \
    pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 && \
    pip install -r requirements.txt && \
    pip install accelerate transformers diffusers huggingface_hub pyyaml

# Download training adapters
RUN mkdir -p /weights/z-image-turbo && \
    wget -O /weights/z-image-turbo/training_adapter_v1.safetensors \
        "https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/resolve/main/training_adapter_v1.safetensors" && \
    wget -O /weights/z-image-turbo/training_adapter_v2.safetensors \
        "https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/resolve/main/training_adapter_v2.safetensors"

ENV PATH="/app/venv/bin:$PATH"

ENTRYPOINT ["python", "run.py"]
```

### Compose Service Additions

**File**: `compose.yml`

```yaml
  # ==========================================================================
  # mcp-lora-trainer: LoRA training management MCP server
  # ==========================================================================
  mcp-lora-trainer:
    build:
      context: ./backend/tools/mcp_servers
      dockerfile: Dockerfile.mcp-server
      args:
        SERVER_MODULE: lora_trainer
    image: strieber-mcp-lora-trainer:latest
    container_name: strieber-mcp-lora-trainer
    restart: unless-stopped
    ports:
      - "9145:8000"
    environment:
      - PORT=8000
      - TRAINING_DATA_PATH=/data
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock  # For spawning training containers
      - /home/trevor/lora-training:/data
      - /home/trevor/models:/models
    networks:
      - strieber-net

  # ==========================================================================
  # ai-toolkit-trainer: GPU container for LoRA training (spawned on-demand)
  # ==========================================================================
  ai-toolkit-trainer:
    build:
      context: ./backend/tools/ai-toolkit
      dockerfile: Dockerfile
    image: strieber-ai-toolkit:latest
    # Note: This service is not started by compose - it's spawned by mcp-lora-trainer
    profiles:
      - training
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    shm_size: 16g
    ipc: host
    volumes:
      - /home/trevor/lora-training:/data
      - /home/trevor/models:/weights:ro
      - /home/trevor/models/comfyui/loras:/output/loras
    networks:
      - strieber-net
```

### Launcher Updates

**File**: `backend/tools/mcp_servers/launcher.py`

Add case for lora_trainer:

```python
elif server_module == "lora_trainer":
    from lora_trainer import server as lora_trainer_server
    mcp_instance = lora_trainer_server.get_mcp()
```

### Dockerfile.mcp-server Updates

**File**: `backend/tools/mcp_servers/Dockerfile.mcp-server`

Add lora_trainer copy:

```dockerfile
# Copy MCP server modules
COPY weather.py ./
COPY web_search.py ./
COPY code_interpreter.py ./
COPY reader ./reader/
COPY comfy_zimage ./comfy_zimage/
COPY lora_trainer ./lora_trainer/  # NEW
```

Add docker SDK dependency:

```dockerfile
RUN pip install docker pyyaml
```

---

## Training Pipeline

### Full Training Flow

```
1. CREATE DATASET
   User: "I want to train a LoRA for my character"
   → lora_create_dataset(name="my_char", trigger_token="mychar", lora_type="character")

2. UPLOAD IMAGES (batch)
   User: [attaches 3 images] "Here are some reference photos"
   → lora_upload_images(dataset_name="my_char", images=[...])
   User: [attaches more] "And some more"
   → lora_upload_images(dataset_name="my_char", images=[...])

3. VALIDATE
   → lora_validate_dataset(dataset_name="my_char")
   Returns: "9 images, all 1024x1024, ready for training"

4. START TRAINING
   → lora_start_training(
       dataset_name="my_char",
       steps=3000,
       lora_rank=8,
       sample_prompts=["mychar, portrait, studio lighting"]
     )
   Returns: job_id="abc123"

5. MONITOR PROGRESS
   (User asks periodically or LLM checks autonomously)
   → lora_training_status(job_id="abc123")
   Returns: status, progress (1500/3000), loss (0.023), sample images

6. COMPLETION
   → lora_training_status(job_id="abc123")
   Returns: status="completed", 6 checkpoints available

7. PROMOTE CHECKPOINT
   → lora_list_checkpoints(job_id="abc123")
   → lora_promote_checkpoint(job_id="abc123", checkpoint_name="step_3000.safetensors")
   Returns: "Promoted to /home/trevor/models/comfyui/loras/my_char.safetensors"

8. INFERENCE
   → zimage_turbo(
       prompt="mychar, portrait in a garden, golden hour",
       lora_name="my_char",
       lora_strength=1.0
     )
   Returns: Generated image using the trained LoRA
```

### ai-toolkit Configuration

Default config template (`zimage_turbo_base.yaml`):

```yaml
model:
  name: z-image-turbo
  training_adapter: /weights/z-image-turbo/training_adapter_v2.safetensors
  quantize: false

dataset:
  path: /datasets/{dataset_name}/images
  resolution: 1024
  trigger_word: "{trigger_token}"
  caption_ext: ".txt"  # Optional captions

training:
  steps: 3000
  batch_size: 1
  learning_rate: 0.0001
  lora_rank: 8
  save_every: 500
  gradient_checkpointing: true
  mixed_precision: bf16

sample:
  every: 250
  prompts:
    - text: "{trigger_token}, portrait, studio lighting"
      seed: 42
    - text: "{trigger_token}, outdoor, natural light"
      seed: 1337

output:
  path: /outputs/{job_id}
  checkpoints_dir: checkpoints
  samples_dir: samples
```

---

## Inference Pipeline

### LoRA-Enabled Workflow

**File**: `backend/tools/mcp_servers/comfy_zimage/workflows/zimage_lora_api.json`

```json
{
  "_comment": "z-image turbo with LoRA support (API format)",
  "_node_mapping": {
    "clip_loader": "1",
    "vae_loader": "2",
    "unet_loader": "3",
    "lora_loader": "4",
    "empty_latent": "5",
    "positive_prompt": "6",
    "negative_zero": "7",
    "model_sampling": "8",
    "sampler": "9",
    "vae_decode": "10",
    "save_image": "11"
  },
  "1": {
    "inputs": { "clip_name": "qwen_3_4b.safetensors", "type": "lumina2", "device": "default" },
    "class_type": "CLIPLoader"
  },
  "2": {
    "inputs": { "vae_name": "ae.safetensors" },
    "class_type": "VAELoader"
  },
  "3": {
    "inputs": { "unet_name": "z_image_turbo_bf16.safetensors", "weight_dtype": "default" },
    "class_type": "UNETLoader"
  },
  "4": {
    "inputs": {
      "lora_name": "PLACEHOLDER.safetensors",
      "strength_model": 1.0,
      "model": ["3", 0]
    },
    "class_type": "LoraLoaderModelOnly"
  },
  "5": {
    "inputs": { "width": 1024, "height": 1024, "batch_size": 1 },
    "class_type": "EmptySD3LatentImage"
  },
  "6": {
    "inputs": { "text": "PLACEHOLDER_PROMPT", "clip": ["1", 0] },
    "class_type": "CLIPTextEncode"
  },
  "7": {
    "inputs": { "conditioning": ["6", 0] },
    "class_type": "ConditioningZeroOut"
  },
  "8": {
    "inputs": { "model": ["4", 0], "shift": 3.0 },
    "class_type": "ModelSamplingAuraFlow"
  },
  "9": {
    "inputs": {
      "seed": 0,
      "steps": 9,
      "cfg": 1.0,
      "sampler_name": "res_multistep",
      "scheduler": "simple",
      "denoise": 1.0,
      "model": ["8", 0],
      "positive": ["6", 0],
      "negative": ["7", 0],
      "latent_image": ["5", 0]
    },
    "class_type": "KSampler"
  },
  "10": {
    "inputs": { "samples": ["9", 0], "vae": ["2", 0] },
    "class_type": "VAEDecode"
  },
  "11": {
    "inputs": { "filename_prefix": "zimage_lora", "images": ["10", 0] },
    "class_type": "SaveImage"
  }
}
```

Key difference from base workflow:
- Node 4 (`LoraLoaderModelOnly`) inserted between UNETLoader and ModelSamplingAuraFlow
- Model connections updated: `UNETLoader → LoraLoader → ModelSamplingAuraFlow`

---

## Testing Strategy

### Unit Tests

**File**: `backend/tools/mcp_servers/tests/test_lora_trainer.py`

```python
"""Tests for LoRA trainer MCP server."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from lora_trainer.models import (
    DatasetMetadata, TrainingConfig, TrainingJob,
    LoRAType, TrainingStatus
)
from lora_trainer.dataset_manager import DatasetManager
from lora_trainer.training_runner import TrainingRunner


class TestModels:
    """Tests for Pydantic models."""

    def test_dataset_metadata_defaults(self):
        meta = DatasetMetadata(
            name="test",
            trigger_token="ohwx",
            lora_type=LoRAType.CHARACTER,
        )
        assert meta.image_count == 0
        assert meta.has_captions is False

    def test_training_config_defaults(self):
        config = TrainingConfig(dataset="test")
        assert config.steps == 3000
        assert config.lora_rank == 8
        assert config.lr == 0.0001


class TestDatasetManager:
    """Tests for dataset management."""

    @pytest.fixture
    def manager(self, tmp_path):
        return DatasetManager(tmp_path)

    def test_create_dataset(self, manager):
        meta = manager.create_dataset(
            name="my_char",
            trigger_token="ohwx",
            lora_type=LoRAType.CHARACTER,
        )
        assert meta.name == "my_char"
        assert manager.dataset_exists("my_char")

    def test_duplicate_dataset_error(self, manager):
        manager.create_dataset("test", "tok", LoRAType.CHARACTER)
        with pytest.raises(ValueError, match="already exists"):
            manager.create_dataset("test", "tok", LoRAType.CHARACTER)

    def test_add_image(self, manager):
        manager.create_dataset("test", "tok", LoRAType.CHARACTER)

        # Create dummy image bytes
        img_bytes = b'\x89PNG\r\n\x1a\n' + b'\x00' * 100

        filename = manager.add_image("test", img_bytes)
        assert filename.endswith(".png")

        meta = manager.get_metadata("test")
        assert meta.image_count == 1


class TestTrainingRunner:
    """Tests for training execution."""

    @pytest.fixture
    def mock_docker(self):
        with patch('docker.from_env') as mock:
            client = MagicMock()
            mock.return_value = client
            yield client

    @pytest.mark.asyncio
    async def test_start_training(self, mock_docker, tmp_path):
        job_store = MagicMock()
        job_store.get_job.return_value = None

        runner = TrainingRunner(
            datasets_path=tmp_path / "datasets",
            outputs_path=tmp_path / "outputs",
            configs_path=tmp_path / "configs",
            job_store=job_store,
        )

        config = TrainingConfig(dataset="test", steps=100)
        job = await runner.start_training("test", "ohwx", config)

        assert job.job_id is not None
        assert job.status == TrainingStatus.PENDING
        job_store.save_job.assert_called()
```

### Integration Tests

**File**: `backend/tools/mcp_servers/tests/test_zimage_lora.py`

```python
"""Integration tests for LoRA-enhanced zimage_turbo."""

import pytest
from unittest.mock import AsyncMock, patch


class TestZimageTurboLora:
    """Tests for LoRA parameter in zimage_turbo."""

    @pytest.mark.asyncio
    async def test_lora_workflow_selection(self):
        """Test that LoRA workflow is used when lora_name provided."""
        with patch('comfy_zimage.server.comfy_client') as mock_client:
            mock_client.queue_prompt = AsyncMock(return_value="prompt-123")
            mock_client.progress = AsyncMock(return_value=iter([100]))
            mock_client.collect_output_files = AsyncMock(return_value=[
                ("test.png", b"PNG...")
            ])

            from comfy_zimage.server import zimage_turbo

            result = await zimage_turbo(
                prompt="ohwx, portrait",
                lora_name="my_character",
                lora_strength=0.8,
            )

            # Verify LoRA node was configured
            call_args = mock_client.queue_prompt.call_args[0][0]
            assert "4" in call_args  # LoRA loader node
            assert "my_character.safetensors" in str(call_args["4"])
```

### E2E Testing Checklist

1. **Dataset Flow**
   - [ ] Create dataset with valid parameters
   - [ ] Create dataset with invalid name (should error)
   - [ ] Upload 5 images
   - [ ] Validate dataset (should pass)
   - [ ] Validate empty dataset (should fail)

2. **Training Flow**
   - [ ] Start training with valid dataset
   - [ ] Check status (returns progress)
   - [ ] Stop training mid-run
   - [ ] Verify checkpoint saved

3. **Inference Flow**
   - [ ] List available LoRAs
   - [ ] Generate with LoRA
   - [ ] Generate without LoRA (baseline)
   - [ ] Compare results

---

## Implementation Sequence

### Phase 1: Infrastructure Setup
1. Create `backend/tools/mcp_servers/lora_trainer/` directory structure
2. Implement `models.py` with Pydantic schemas
3. Implement `dataset_manager.py` for dataset CRUD
4. Implement `job_store.py` for job persistence
5. Update `launcher.py` and `Dockerfile.mcp-server`

### Phase 2: Training Backend
1. Create `backend/tools/ai-toolkit/Dockerfile`
2. Implement `training_runner.py` with Docker integration
3. Implement `server.py` with dataset and training tools
4. Add ai-toolkit-trainer service to `compose.yml`

### Phase 3: Inference Enhancement
1. Create `workflows/zimage_lora_api.json`
2. Modify `comfy_zimage/server.py` - add LoRA parameters
3. Add `lora_list_available` tool
4. Test LoRA loading in ComfyUI

### Phase 4: Frontend Integration
1. Add `lora_trainer` to `ToolToggles.svelte`
2. Add `lora_trainer` to `responses.ts` tool list
3. Create `LoraTrainingDisplay.svelte` component
4. Update `OutputItemRenderer.svelte`
5. Add type definitions to `types.ts`

### Phase 5: Testing & Polish
1. Write unit tests for all components
2. Write integration tests
3. E2E testing of full pipeline
4. Update CLAUDE.md with new tools
5. Create user documentation

---

## Files Summary

### New Files to Create

| File | Purpose |
|------|---------|
| `backend/tools/mcp_servers/lora_trainer/__init__.py` | Module init |
| `backend/tools/mcp_servers/lora_trainer/server.py` | MCP tool definitions |
| `backend/tools/mcp_servers/lora_trainer/models.py` | Pydantic schemas |
| `backend/tools/mcp_servers/lora_trainer/dataset_manager.py` | Dataset operations |
| `backend/tools/mcp_servers/lora_trainer/training_runner.py` | Training execution |
| `backend/tools/mcp_servers/lora_trainer/job_store.py` | Job persistence |
| `backend/tools/mcp_servers/lora_trainer/config/zimage_turbo_base.yaml` | Config template |
| `backend/tools/mcp_servers/comfy_zimage/workflows/zimage_lora_api.json` | LoRA workflow |
| `backend/tools/ai-toolkit/Dockerfile` | Training container |
| `frontend/src/lib/components/chat/tools/LoraTrainingDisplay.svelte` | UI component |
| `backend/tools/mcp_servers/tests/test_lora_trainer.py` | Unit tests |
| `backend/tools/mcp_servers/tests/test_zimage_lora.py` | Integration tests |

### Files to Modify

| File | Changes |
|------|---------|
| `compose.yml` | Add mcp-lora-trainer, ai-toolkit-trainer services; update MCP_CONFIG |
| `backend/tools/mcp_servers/launcher.py` | Add lora_trainer case |
| `backend/tools/mcp_servers/Dockerfile.mcp-server` | Add lora_trainer copy, docker dependency |
| `backend/tools/mcp_servers/comfy_zimage/server.py` | Add lora_name, lora_strength params; add lora_list_available |
| `frontend/src/lib/components/settings/ToolToggles.svelte` | Add lora_trainer toggle |
| `frontend/src/lib/api/responses.ts` | Add lora_trainer to enabled tools |
| `frontend/src/lib/stores/types.ts` | Add LoraTrainingCallOutput type |
| `frontend/src/lib/components/chat/tools/OutputItemRenderer.svelte` | Add LoraTrainingDisplay routing |

---

## References

- [Training a LoRA for Z-Image Turbo with Ostris AI Toolkit](https://huggingface.co/blog/content-and-code/training-a-lora-for-z-image-turbo)
- [GitHub - ostris/ai-toolkit](https://github.com/ostris/ai-toolkit)
- [Z-Image Turbo LoRA Training Full Tutorial](https://github.com/FurkanGozukara/Stable-Diffusion/wiki/Z-Image-Turbo-LoRA-training-with-AI-Toolkit-and-Z-Image-ControlNet-Full-Tutorial-for-Highest-Quality)
- [ComfyUI LoRA Loader Documentation](https://comfyui-wiki.com/en/comfyui-nodes/loaders/lora-loader)
