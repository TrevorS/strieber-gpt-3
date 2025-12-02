# Svelte Chat UI - Project Specification

## Overview

Build a modern chat interface for the OpenAI Responses API using Svelte 5, SvelteKit 2, and shadcn-svelte. This replaces Open WebUI as the primary chat interface for the strieber-gpt-3 project.

The frontend communicates with an existing Rust-based Responses API backend (`responses-api` service) that translates requests to llama.cpp inference backends and executes tools via MCP servers.

## Goals

1. **OpenAI SDK Compatible**: Use the official `openai` npm package with a custom `baseURL` pointing to our backend. This gives us TypeScript types for free and ensures API compatibility.

2. **Streaming-First**: Real-time token-by-token rendering using Server-Sent Events (SSE).

3. **Tool-Aware**: Display tool calls (web search, code interpreter, reasoning) with appropriate UI components.

4. **Conversation Management**: Multi-turn conversations using `previous_response_id` chaining, with localStorage for conversation metadata persistence.

5. **File Upload Support**: Paste, drag-drop, and file picker for images and documents.

---

## Technical Stack

| Component | Technology | Version |
|-----------|------------|---------|
| Framework | SvelteKit | 2.x |
| UI Library | Svelte | 5.x (runes) |
| Components | shadcn-svelte | latest (Svelte 5 compatible) |
| Styling | Tailwind CSS | 4.x |
| API Client | openai | latest |
| Icons | lucide-svelte | latest |
| Markdown | marked | 12.x |
| Syntax Highlighting | highlight.js | 11.x |
| Runtime | Node.js | 22.x (adapter-node) |

### Key Svelte 5 Patterns

Use Svelte 5 runes for all state management:

```typescript
// Reactive state
let count = $state(0);

// Derived values
let doubled = $derived(count * 2);

// Class-based stores for complex state
class ConversationStore {
  conversations = $state<Conversation[]>([]);
  activeId = $state<string | null>(null);

  active = $derived(
    this.conversations.find(c => c.id === this.activeId)
  );
}
```

---

## Project Structure

```
frontend/
├── src/
│   ├── lib/
│   │   ├── api/
│   │   │   ├── client.ts           # OpenAI client wrapper
│   │   │   ├── streaming.ts        # SSE stream event parser
│   │   │   └── types.ts            # Re-exports from openai package
│   │   ├── components/
│   │   │   ├── chat/               # Chat-specific components
│   │   │   ├── tools/              # Tool call display components
│   │   │   ├── sidebar/            # Navigation and conversation list
│   │   │   └── ui/                 # shadcn-svelte components
│   │   ├── stores/
│   │   │   ├── conversations.svelte.ts  # Conversation state
│   │   │   ├── streaming.svelte.ts      # Active stream state
│   │   │   └── settings.svelte.ts       # User preferences
│   │   └── utils/
│   │       ├── markdown.ts         # Markdown rendering with syntax highlighting
│   │       ├── files.ts            # File reading and base64 encoding
│   │       └── storage.ts          # localStorage helpers
│   ├── routes/
│   │   ├── +layout.svelte          # App shell with sidebar
│   │   ├── +page.svelte            # New conversation / home
│   │   ├── c/
│   │   │   └── [id]/
│   │   │       └── +page.svelte    # Conversation by ID
│   │   └── health/
│   │       └── +server.ts          # Health check endpoint
│   ├── app.css                     # Global styles, Tailwind imports
│   └── app.html                    # HTML template
├── static/
├── Dockerfile
├── package.json
├── svelte.config.js
├── tailwind.config.ts
├── vite.config.ts
└── components.json                 # shadcn-svelte config
```

---

## API Integration

### OpenAI Client Configuration

```typescript
import OpenAI from 'openai';

const client = new OpenAI({
  baseURL: import.meta.env.VITE_RESPONSES_API_URL || 'http://localhost:9150/v1',
  apiKey: 'not-needed', // Backend doesn't require auth
  dangerouslyAllowBrowser: true, // For client-side usage
});
```

### Backend Endpoints

The `responses-api` backend provides these endpoints:

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/v1/responses` | Create response (streaming with `stream: true`) |
| `GET` | `/v1/responses/{id}` | Retrieve stored response |
| `DELETE` | `/v1/responses/{id}` | Delete response |
| `GET` | `/v1/models` | List available models |
| `GET` | `/v1/containers/{id}/files/{file_id}/content` | Download code interpreter output files |
| `GET` | `/health` | Health check |

### Request Format

The Responses API accepts requests in this format:

```typescript
interface CreateResponseRequest {
  model: string;                    // e.g., "gpt-oss-120b"
  input: string | InputItem[];      // Simple string or array of items
  instructions?: string;            // System prompt
  tools?: Tool[];                   // Available tools
  tool_choice?: ToolChoice;         // "auto" | "required" | "none"
  previous_response_id?: string;    // Chain to previous response
  temperature?: number;             // 0-2, default 1
  top_p?: number;                   // 0-1, default 1
  max_output_tokens?: number;       // Token limit
  stream?: boolean;                 // Enable SSE streaming
  store?: boolean;                  // Store for later retrieval (default true)
}
```

### Input Item Types

```typescript
type InputItem =
  | { type: "message"; role: "user" | "assistant" | "system"; content: MessageContent }
  | { type: "reasoning"; id?: string; content: ReasoningContent[] }
  | { type: "function_call"; call_id: string; name: string; arguments: string }
  | { type: "function_call_output"; call_id: string; output: string };

type MessageContent = string | ContentPart[];

type ContentPart =
  | { type: "input_text"; text: string }
  | { type: "input_image"; image_url: string | { url: string; detail?: "auto" | "low" | "high" } }
  | { type: "input_file"; filename: string; file_data: string };  // data URL format
```

### Response Format

```typescript
interface Response {
  id: string;                       // e.g., "resp_abc123"
  object: "response";
  created_at: number;               // Unix timestamp
  status: "completed" | "failed" | "in_progress" | "cancelled";
  model: string;
  output: OutputItem[];             // Interleaved reasoning, messages, tool calls
  usage: {
    input_tokens: number;
    output_tokens: number;
    total_tokens: number;
  };
  // ... other fields
}
```

### Output Item Types

The `output` array contains interleaved items of these types:

```typescript
type OutputItem =
  | { type: "reasoning"; id: string; content: ReasoningContent[]; status: OutputStatus }
  | { type: "message"; id: string; role: "assistant"; content: OutputContent[]; status: OutputStatus }
  | { type: "function_call"; id: string; call_id: string; name: string; arguments: string; status: OutputStatus }
  | { type: "web_search_call"; id: string; status: OutputStatus; action?: WebSearchAction }
  | { type: "code_interpreter_call"; id: string; status: OutputStatus; code?: string; outputs?: CodeOutput[] }
  // ... other tool types

type OutputContent =
  | { type: "output_text"; text: string; annotations?: Annotation[] }
  | { type: "refusal"; refusal: string };

type Annotation =
  | { type: "url_citation"; url: string; title?: string; index: number }
  | { type: "file_citation"; file_id: string; filename?: string; index: number }
  | { type: "container_file_citation"; container_id: string; file_id: string; filename: string };
```

---

## Streaming Events

When `stream: true`, the backend returns Server-Sent Events. Each event has a `type` field:

### Lifecycle Events (emitted once)

| Event | When | Action |
|-------|------|--------|
| `response.created` | Response started | Show loading indicator |
| `response.in_progress` | Processing | Update status |
| `response.completed` | Done | Hide loading, enable actions |
| `response.failed` | Error occurred | Show error message |

### Content Events (emitted multiple times)

| Event | When | Action |
|-------|------|--------|
| `response.output_item.added` | New item (message, tool call, reasoning) | Add to output list |
| `response.output_item.done` | Item completed | Mark item as finalized |
| `response.output_text.delta` | Text token generated | Append to current message text |
| `response.output_text.done` | Text complete | Finalize text content |
| `response.content_part.added` | New content part | Add to message content |
| `response.content_part.done` | Content part complete | Finalize content part |

### Tool-Specific Events

| Event | When | Action |
|-------|------|--------|
| `response.function_call_arguments.delta` | Function args streaming | Show tool preparing |
| `response.function_call_arguments.done` | Args complete | Tool ready to execute |
| `response.web_search_call.searching` | Web search in progress | Show search indicator |
| `response.web_search_call.completed` | Search done | Display sources |
| `response.code_interpreter_call.code_delta` | Code being written | Stream code |
| `response.code_interpreter_call.interpreting` | Code executing | Show execution indicator |
| `response.code_interpreter_call.completed` | Execution done | Display output/images |

### SSE Parsing

Events arrive as:
```
event: response.output_text.delta
data: {"type":"response.output_text.delta","delta":"Hello"}

event: response.output_text.delta
data: {"type":"response.output_text.delta","delta":" world"}
```

The stream ends with:
```
data: [DONE]
```

---

## Conversation State Management

### Conversation Model

```typescript
interface Conversation {
  id: string;                       // UUID
  title: string;                    // Auto-generated or user-set
  createdAt: number;                // Unix timestamp
  updatedAt: number;
  lastResponseId: string | null;    // For chaining with previous_response_id
  messages: Message[];              // UI representation of conversation
}

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;                  // Rendered text (for display)
  rawInput?: InputItem[];           // Original input items (for re-sending)
  rawOutput?: OutputItem[];         // Original output items (for context)
  attachments?: Attachment[];       // Files/images attached
  createdAt: number;
}

interface Attachment {
  id: string;
  type: "image" | "file";
  filename: string;
  mimeType: string;
  dataUrl: string;                  // Base64 data URL
  previewUrl?: string;              // For images, same as dataUrl
}
```

### Multi-Turn Conversation Flow

1. **First message**: Send request without `previous_response_id`
2. **Subsequent messages**: Include `previous_response_id` from last response
3. **Store response ID**: Save `response.id` to conversation for next turn
4. **Backend chains context**: Server prepends previous conversation automatically

```typescript
// First turn
const response1 = await client.responses.create({
  model: "gpt-oss-120b",
  input: "Tell me a joke",
  store: true,
});

// Second turn - server has full context
const response2 = await client.responses.create({
  model: "gpt-oss-120b",
  previous_response_id: response1.id,
  input: [{ role: "user", content: "Explain why that's funny" }],
  store: true,
});
```

### localStorage Schema

```typescript
// Key: "strieber-conversations"
interface StoredConversations {
  version: 1;
  conversations: Conversation[];
  activeId: string | null;
}

// Key: "strieber-settings"
interface StoredSettings {
  version: 1;
  theme: "light" | "dark" | "system";
  model: string;
  temperature: number;
  // ... other preferences
}
```

---

## File Upload

### Supported Formats

| Type | Extensions | Max Size | Input Type |
|------|------------|----------|------------|
| Images | png, jpg, jpeg, gif, webp | 20MB | `input_image` |
| Documents | pdf | 20MB | `input_file` |

### Data URL Format

Files are sent as base64 data URLs:

```typescript
// Image
{
  type: "input_image",
  image_url: "data:image/png;base64,iVBORw0KGgo..."
}

// With detail level
{
  type: "input_image",
  image_url: {
    url: "data:image/png;base64,iVBORw0KGgo...",
    detail: "high"  // "auto" | "low" | "high"
  }
}

// PDF or other file
{
  type: "input_file",
  filename: "document.pdf",
  file_data: "data:application/pdf;base64,JVBERi0xLjQK..."
}
```

### Upload Methods

1. **Paste**: Handle `paste` event, extract files from `clipboardData`
2. **Drag-drop**: Handle `dragover`, `drop` events on input area
3. **File picker**: `<input type="file">` triggered by button click

### Processing Flow

```typescript
async function processFile(file: File): Promise<Attachment> {
  const buffer = await file.arrayBuffer();
  const base64 = btoa(String.fromCharCode(...new Uint8Array(buffer)));
  const mimeType = file.type;
  const dataUrl = `data:${mimeType};base64,${base64}`;

  return {
    id: crypto.randomUUID(),
    type: mimeType.startsWith("image/") ? "image" : "file",
    filename: file.name,
    mimeType,
    dataUrl,
    previewUrl: mimeType.startsWith("image/") ? dataUrl : undefined,
  };
}
```

---

## UI Components

### Layout Structure

```
┌─────────────────────────────────────────────────────────────────┐
│ Header (optional - model selector, settings)                    │
├──────────────┬──────────────────────────────────────────────────┤
│              │                                                  │
│   Sidebar    │              Main Chat Area                      │
│              │                                                  │
│ ┌──────────┐ │  ┌────────────────────────────────────────────┐  │
│ │ New Chat │ │  │                                            │  │
│ └──────────┘ │  │           Message List                     │  │
│              │  │         (scrollable)                       │  │
│ Conversation │  │                                            │  │
│    List      │  │  ┌─ User ─────────────────────────────┐   │  │
│              │  │  │ Message content                     │   │  │
│ ┌──────────┐ │  │  └─────────────────────────────────────┘   │  │
│ │ Today    │ │  │                                            │  │
│ │ ├─ Conv1 │ │  │  ┌─ Assistant ─────────────────────────┐   │  │
│ │ └─ Conv2 │ │  │  │ [Reasoning block - collapsible]     │   │  │
│ │ Yesterday│ │  │  │                                     │   │  │
│ │ └─ Conv3 │ │  │  │ Response text with **markdown**     │   │  │
│ └──────────┘ │  │  │                                     │   │  │
│              │  │  │ [Tool calls - web search, code]     │   │  │
│              │  │  └─────────────────────────────────────┘   │  │
│              │  │                                            │  │
│              │  └────────────────────────────────────────────┘  │
│              │                                                  │
│              │  ┌────────────────────────────────────────────┐  │
│              │  │ [+] [attachments] │ Input area...   [Send] │  │
│              │  └────────────────────────────────────────────┘  │
└──────────────┴──────────────────────────────────────────────────┘
```

### Component Responsibilities

**Sidebar**
- New conversation button
- Conversation list grouped by date (Today, Yesterday, Previous 7 days, etc.)
- Active conversation highlight
- Conversation rename/delete actions
- Collapsible on mobile

**Message List**
- Virtual scrolling for long conversations (optional, can defer)
- Auto-scroll to bottom on new messages
- Scroll-to-bottom button when scrolled up
- Loading skeleton during initial load

**Message Component**
- User messages: Simple text with optional attachments
- Assistant messages: Complex with multiple content types
- Copy button for message content
- Regenerate button for last assistant message
- Timestamp on hover

**Reasoning Block**
- Collapsible by default
- Subtle styling (muted colors, smaller text)
- "Thinking..." indicator while streaming
- Shows token count when complete

**Tool Call Displays**
- Web Search: Query, loading state, then expandable sources list with favicons
- Code Interpreter: Syntax-highlighted code block, execution status, output/errors, image outputs
- Generic function call: Name, arguments (JSON), result

**Input Area**
- Auto-resizing textarea
- Attachment preview thumbnails with remove button
- Paste handler for images
- Drag-drop zone (visual indicator on dragover)
- File picker button (+)
- Send button (enabled when input or attachments present)
- Keyboard: Enter to send, Shift+Enter for newline
- Stop button during streaming

**Settings Panel** (dialog or slide-over)
- Model selector dropdown
- Temperature slider (0-2)
- Dark/light/system theme toggle

---

## Markdown Rendering

Use `marked` for parsing and `highlight.js` for code blocks:

```typescript
import { marked } from 'marked';
import hljs from 'highlight.js';

marked.setOptions({
  highlight: (code, lang) => {
    if (lang && hljs.getLanguage(lang)) {
      return hljs.highlight(code, { language: lang }).value;
    }
    return hljs.highlightAuto(code).value;
  },
});

function renderMarkdown(text: string): string {
  return marked.parse(text);
}
```

### Code Block Features

- Language label in top-right corner
- Copy button
- Syntax highlighting for common languages (python, javascript, typescript, rust, bash, json, etc.)

### Annotation Handling

When output contains annotations (citations), render them as:
- Inline superscript numbers `[1]`
- Clickable links to sources
- Citation list at end of message (optional)

---

## shadcn-svelte Components

Install these components via the shadcn-svelte CLI:

```bash
npx shadcn-svelte@latest add button
npx shadcn-svelte@latest add input
npx shadcn-svelte@latest add textarea
npx shadcn-svelte@latest add scroll-area
npx shadcn-svelte@latest add separator
npx shadcn-svelte@latest add avatar
npx shadcn-svelte@latest add badge
npx shadcn-svelte@latest add dialog
npx shadcn-svelte@latest add dropdown-menu
npx shadcn-svelte@latest add tooltip
npx shadcn-svelte@latest add sheet           # Mobile sidebar
npx shadcn-svelte@latest add collapsible     # Reasoning blocks
npx shadcn-svelte@latest add card
npx shadcn-svelte@latest add select          # Model selector
npx shadcn-svelte@latest add slider          # Temperature
npx shadcn-svelte@latest add skeleton        # Loading states
npx shadcn-svelte@latest add toast           # Notifications
```

---

## Docker Deployment

### Dockerfile

```dockerfile
# Build stage
FROM node:22-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build
RUN npm prune --production

# Runtime stage
FROM node:22-alpine
WORKDIR /app
RUN apk add --no-cache curl
COPY --from=builder /app/build build/
COPY --from=builder /app/node_modules node_modules/
COPY package.json .

RUN addgroup -g 1001 -S appgroup && \
    adduser -S appuser -u 1001 -G appgroup
USER appuser

ENV NODE_ENV=production PORT=3000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -sf http://localhost:3000/health || exit 1
EXPOSE 3000
CMD ["node", "build"]
```

### compose.yml Service

```yaml
chat-ui:
  build:
    context: ./frontend
    dockerfile: Dockerfile
  image: strieber-chat-ui:latest
  container_name: strieber-chat-ui
  restart: unless-stopped
  ports:
    - "${CHAT_UI_PORT:-9300}:3000"
  environment:
    - PORT=3000
    - ORIGIN=http://localhost:9300
    - PUBLIC_RESPONSES_API_URL=http://responses-api:8000
  depends_on:
    responses-api:
      condition: service_healthy
  networks:
    - strieber-net
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 3000 | Server port |
| `ORIGIN` | http://localhost:3000 | SvelteKit origin for CSRF |
| `PUBLIC_RESPONSES_API_URL` | http://localhost:9150 | Responses API base URL |

---

## Backend Changes Required

### Implement `previous_response_id` Support

Currently, the `previous_response_id` field is accepted but not used. Implementation needed:

**In the executor, before processing:**

1. If `previous_response_id` is set, load the stored response
2. Get the original request that created it
3. Reconstruct conversation: `[previous_input] + [previous_output] + [new_input]`
4. Continue with normal execution

**Conversion logic:**

| Output Type | → | Input Type |
|-------------|---|------------|
| `message` (assistant) | → | `message` (role: assistant) |
| `reasoning` | → | `reasoning` |
| `function_call` | → | `function_call` |
| `web_search_call` | → | (skip - built-in tool result) |
| `code_interpreter_call` | → | (skip - built-in tool result) |

---

## Development Workflow

### Local Development

```bash
# Terminal 1: Start backend services
cd /path/to/strieber-gpt-3
docker compose up -d responses-api llama-server mcp-weather mcp-web-search

# Terminal 2: Run frontend with hot reload
cd frontend
npm install
npm run dev -- --host 0.0.0.0
```

Frontend will be available at `http://localhost:5173` (Vite default).

### Testing Against Backend

```bash
# Check backend is running
curl http://localhost:9150/health

# List models
curl http://localhost:9150/v1/models

# Test response (non-streaming)
curl -X POST http://localhost:9150/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-oss-120b","input":"Hello"}'
```

---

## Implementation Phases

### Phase 1: Project Setup
- Initialize SvelteKit with TypeScript
- Install and configure Tailwind v4
- Set up shadcn-svelte
- Configure path aliases
- Create health endpoint
- Verify Docker build works

### Phase 2: Core Chat (Non-Streaming)
- OpenAI client wrapper
- Basic layout (sidebar + main)
- Message list rendering
- Input area with send
- End-to-end message flow

### Phase 3: Streaming
- SSE stream parser
- Streaming state store
- Token-by-token rendering
- Loading states
- Error handling

### Phase 4: Conversations
- localStorage persistence
- Conversation CRUD
- Sidebar list with grouping
- `previous_response_id` chaining
- **Backend: Implement previous_response_id**

### Phase 5: Tool Displays
- Reasoning blocks (collapsible)
- Web search results
- Code interpreter output
- Annotations/citations

### Phase 6: File Upload
- Image paste
- Drag-drop
- File picker
- Preview and remove
- Base64 encoding

### Phase 7: Polish
- Model selector
- Theme toggle
- Settings panel
- Keyboard shortcuts
- Mobile responsive
- Error toasts

---

## Success Criteria

1. **Functional**: Complete chat flow with streaming responses
2. **Tool Support**: Proper display of web search, code interpreter, reasoning
3. **File Upload**: Images and PDFs can be attached and sent
4. **Conversations**: Multi-turn with history, persisted locally
5. **Docker**: Builds and runs in compose alongside other services
6. **Responsive**: Works on desktop and mobile viewports
