# Svelte Chat UI - Implementation Tasks

Breaking down `PROJECT.md` into specific, implementable development tasks.

---

## Testing Infrastructure

Unit and component testing with Vitest. Run tests with `npm test`.

**Setup (completed)**:
- Vitest configured with jsdom environment
- `@testing-library/svelte` for component testing
- Test files: `src/**/*.{test,spec}.ts`

**Commands**:
```bash
npm test           # Run tests once
npm run test:watch # Watch mode
npm run test:coverage # With coverage
```

**Test Conventions**:
- Unit tests in `__tests__/` directories next to source
- Name: `<module>.test.ts`
- Use `describe`/`it` blocks with clear descriptions

---

## Linting & Formatting

Code quality with Biome (fast Rust-based linter/formatter).

**Setup (completed)**:
- Biome 2.x configured for TypeScript files
- Svelte files excluded (Biome doesn't understand Svelte template syntax)
- Test files have relaxed `any` rules

**Commands**:
```bash
npm run lint        # Check for lint issues
npm run lint:fix    # Fix auto-fixable lint issues
npm run format      # Format code
npm run format:check # Check formatting without changing
npm run ci          # Combined lint + format check (for CI)
```

**Configuration** (`biome.json`):
- Tabs for indentation
- Single quotes
- No trailing commas
- 100 char line width
- Recommended rules with sensible overrides

**Docker Usage**:
```bash
docker compose run --rm frontend-dev npm run lint
docker compose run --rm frontend-dev npm run format
```

---

## E2E Testing (Playwright)

Visual and interaction testing with Playwright in Docker.

**Setup (completed)**:
- Playwright configured with headless Chromium
- Tests run against production build (`npm run build && npm run preview`)
- Screenshots saved to `test-results/screenshots/`
- Docker service: `playwright-test` (uses Microsoft Playwright image)

**Commands**:
```bash
# Run in Docker (recommended)
docker compose run --rm playwright-test

# Run locally
npm run test:e2e
npm run test:e2e:headed   # With browser UI
npm run test:e2e:debug    # Debug mode
```

**Test Conventions**:
- E2E tests in `e2e/` directory
- Name: `<feature>.spec.ts`
- Screenshots: `await page.screenshot({ path: 'test-results/screenshots/<name>.png' })`

**Claude Workflow**:
After UI changes, Claude can:
1. Run: `docker compose run --rm playwright-test`
2. Read: `frontend/test-results/screenshots/*.png`
3. Verify: Layout looks correct, no visual regressions

---

## Slice 1: Minimal Streaming Chat (MVP)

Get end-to-end streaming chat working with minimal UI.

**Milestone**: Can send messages and see streaming responses with markdown

---

### Task 1.1: Project Initialization

**Description**: Initialize SvelteKit project with TypeScript, Tailwind v4, and shadcn-svelte

**Acceptance Criteria**:
- SvelteKit 2.x project created with TypeScript strict mode
- Tailwind CSS v4 configured and working
- shadcn-svelte initialized with required components
- Path aliases configured (`$lib/*`)
- `pnpm dev` starts without errors
- `pnpm build` produces production build

**Implementation Approach**:
```bash
cd frontend/
npx sv create . --template minimal --types ts --no-add-ons --no-install
pnpm install
pnpm add -D tailwindcss @tailwindcss/vite
npx shadcn-svelte@latest init
```

**Required shadcn-svelte Components** (install as needed):
- button, input, textarea, scroll-area, separator
- avatar, badge, dialog, dropdown-menu, tooltip
- sheet, collapsible, card, select, slider, skeleton, toast

**Configuration Files**:
- `svelte.config.js` - adapter-node, path aliases
- `vite.config.ts` - Tailwind plugin
- `components.json` - shadcn-svelte config
- `tsconfig.json` - Strict TypeScript

**Test Requirements**:
- Dev server starts without errors
- Tailwind classes apply correctly
- shadcn Button component renders

---

### Task 1.2: OpenAI Client Wrapper ✅

**Status**: Complete

**Description**: Create API client using openai npm package with custom baseURL

**Acceptance Criteria**:
- [x] OpenAI client configured with environment variable baseURL
- [x] TypeScript types properly exported
- [x] Client works in browser context (`dangerouslyAllowBrowser: true`)
- [x] Non-streaming request/response works

**Files Created**:
- `src/lib/api/client.ts` - Client wrapper with `createClient()`, `getApiBaseUrl()`
- `src/lib/api/types.ts` - Re-exported OpenAI types (Response, ChatCompletion, errors)
- `src/lib/api/index.ts` - Barrel export

---

### Task 1.3: SSE Stream Parser ✅

**Status**: Complete

**Description**: Implement Server-Sent Events parser for streaming responses

**Acceptance Criteria**:
- [x] Parses SSE format (`event:`, `data:`)
- [x] Handles all event types from spec (response.*, output_text.delta, etc.)
- [x] Detects `[DONE]` terminator
- [x] Provides typed event objects
- [x] Handles connection errors gracefully

**Files Created**:
- `src/lib/api/streaming.ts` - SSE parser with type guards
- `src/lib/api/__tests__/streaming.test.ts` - 25 unit tests

**Test Coverage** (25 tests):
- `parseSSEData`: JSON parsing, [DONE] detection, error handling
- `parseSSEStream`: Single/multiple events, buffering, termination
- Type guards: `isTextDeltaEvent`, `isCompletedEvent`, `isFailedEvent`, `isErrorEvent`
- Edge cases: Partial chunks, comments, unicode, special characters

---

### Task 1.4: Conversation State Store ✅

**Status**: Complete

**Description**: Create Svelte 5 runes-based store for conversation management

**Acceptance Criteria**:
- [x] `Conversation` and `Message` types defined
- [x] CRUD operations for conversations
- [x] Active conversation tracking
- [x] `lastResponseId` tracking for chaining
- [x] Reactive state using `$state` and `$derived`

**Files Created**:
- `src/lib/stores/types.ts` - Conversation/Message interfaces with helper functions
- `src/lib/stores/conversations.svelte.ts` - Svelte 5 runes store class
- `src/lib/stores/index.ts` - Barrel export
- `src/lib/stores/__tests__/conversations.test.ts` - 31 unit tests

**Test Coverage** (31 tests):
- CRUD: create, delete, get, clear, load
- Active tracking: setActive, automatic switching on delete
- Messages: addMessage, updateMessageContent, setMessageStreaming
- Context: updateLastResponseId, updateTitle
- Derived: sorted getter

---

### Task 1.5: Basic Layout Shell ✅

**Status**: Complete

**Description**: Create app layout with sidebar placeholder and main content area

**Acceptance Criteria**:
- [x] Two-column layout (sidebar + main)
- [x] Sidebar shows placeholder for now
- [x] Main area fills remaining space
- [x] Responsive-ready structure (hidden on mobile via `hidden md:flex`)

**Files Modified**:
- `src/routes/+layout.svelte` - Flex container with sidebar + main
- `src/routes/+page.svelte` - Centered placeholder text

---

### Task 1.6: Chat Input Area ✅

**Status**: Complete

**Description**: Auto-resizing textarea with send functionality

**Acceptance Criteria**:
- [x] Auto-resizes up to max height (200px)
- [x] Enter to send, Shift+Enter for newline
- [x] Send button enabled when content present
- [x] Disabled state during streaming
- [x] Placeholder text

**Files Created**:
- `src/lib/components/chat/ChatInput.svelte` - Input component with auto-resize
- `src/lib/components/chat/index.ts` - Barrel export
- `src/lib/components/chat/__tests__/ChatInput.test.ts` - 14 unit tests

**Testing Infrastructure Updated**:
- Added `@testing-library/jest-dom` for DOM matchers
- Updated `vitest.config.ts` to use `svelte()` + `svelteTesting()` plugins
- Added `vitest-setup.ts` for test setup

**Implementation Approach**:
```svelte
<!-- src/lib/components/chat/ChatInput.svelte -->
<script lang="ts">
  import { Button } from '$lib/components/ui/button';
  import { Send } from 'lucide-svelte';

  let { onsubmit, disabled = false } = $props<{
    onsubmit: (text: string) => void;
    disabled?: boolean;
  }>();

  let value = $state('');
  let textarea: HTMLTextAreaElement;

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }

  function submit() {
    const text = value.trim();
    if (text && !disabled) {
      onsubmit(text);
      value = '';
      // Reset height
      if (textarea) textarea.style.height = 'auto';
    }
  }

  function autoResize() {
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
    }
  }
</script>

<div class="border-t p-4">
  <div class="flex gap-2 items-end max-w-3xl mx-auto">
    <textarea
      bind:this={textarea}
      bind:value
      onkeydown={handleKeydown}
      oninput={autoResize}
      {disabled}
      placeholder="Send a message..."
      rows="1"
      class="flex-1 resize-none rounded-lg border p-3 focus:outline-none focus:ring-2 focus:ring-ring"
    ></textarea>
    <Button onclick={submit} disabled={disabled || !value.trim()} size="icon">
      <Send class="h-4 w-4" />
    </Button>
  </div>
</div>
```

**Files**:
- `src/lib/components/chat/ChatInput.svelte`

**Dependencies**:
```bash
pnpm add lucide-svelte
npx shadcn-svelte@latest add button
```

**Test Requirements**:
- Auto-resize works
- Keyboard shortcuts work
- Disabled state works

---

### Task 1.7: Message List Component ✅

**Status**: Complete

**Description**: Scrollable list of messages with auto-scroll

**Acceptance Criteria**:
- [x] Displays user and assistant messages
- [x] Auto-scrolls to bottom on new messages
- [x] Scrollable container

**Files Created**:
- `src/lib/components/chat/MessageList.svelte` - Scrollable list with auto-scroll
- `src/lib/components/chat/UserMessage.svelte` - User message (placeholder)
- `src/lib/components/chat/AssistantMessage.svelte` - Assistant message (placeholder)
- `src/lib/components/chat/__tests__/MessageList.test.ts` - 8 unit tests

**Implementation Approach**:
```svelte
<!-- src/lib/components/chat/MessageList.svelte -->
<script lang="ts">
  import type { Message } from '$lib/stores/types';
  import UserMessage from './UserMessage.svelte';
  import AssistantMessage from './AssistantMessage.svelte';

  let { messages } = $props<{ messages: Message[] }>();

  let container: HTMLDivElement;

  $effect(() => {
    // Auto-scroll when messages change
    messages; // track dependency
    if (container) {
      container.scrollTop = container.scrollHeight;
    }
  });
</script>

<div bind:this={container} class="flex-1 overflow-y-auto p-4">
  <div class="max-w-3xl mx-auto space-y-4">
    {#each messages as message (message.id)}
      {#if message.role === 'user'}
        <UserMessage {message} />
      {:else}
        <AssistantMessage {message} />
      {/if}
    {/each}
  </div>
</div>
```

**Files**:
- `src/lib/components/chat/MessageList.svelte`

**Test Requirements**:
- Messages render
- Auto-scroll triggers on new messages

---

### Task 1.8: User Message Component ✅

**Status**: Complete

**Description**: Render user messages (text only for MVP)

**Acceptance Criteria**:
- [x] Displays text content
- [x] Right-aligned or distinct styling
- [x] Minimal styling for MVP

**Files Created**:
- `src/lib/components/chat/UserMessage.svelte` - Right-aligned user messages
- `src/lib/components/chat/__tests__/UserMessage.test.ts` - 10 unit tests

**Implementation Approach**:
```svelte
<!-- src/lib/components/chat/UserMessage.svelte -->
<script lang="ts">
  import type { Message } from '$lib/stores/types';

  let { message } = $props<{ message: Message }>();
</script>

<div class="flex justify-end">
  <div class="bg-primary text-primary-foreground rounded-lg px-4 py-2 max-w-[80%]">
    <p class="whitespace-pre-wrap">{message.content}</p>
  </div>
</div>
```

**Files**:
- `src/lib/components/chat/UserMessage.svelte`

**Test Requirements**:
- Renders text correctly
- Styling looks reasonable

---

### Task 1.9: Assistant Message Component ✅

**Status**: Complete

**Description**: Render assistant messages with markdown (MVP version)

**Acceptance Criteria**:
- [x] Renders markdown content
- [x] Left-aligned styling
- [x] Handles code blocks with syntax highlighting

**Files Created**:
- `src/lib/components/chat/AssistantMessage.svelte` - Left-aligned with markdown
- `src/lib/components/chat/__tests__/AssistantMessage.test.ts` - 12 unit tests

**Implementation Approach**:
```svelte
<!-- src/lib/components/chat/AssistantMessage.svelte -->
<script lang="ts">
  import type { Message } from '$lib/stores/types';
  import MarkdownContent from './MarkdownContent.svelte';

  let { message } = $props<{ message: Message }>();
</script>

<div class="flex justify-start">
  <div class="bg-muted rounded-lg px-4 py-2 max-w-[80%]">
    <MarkdownContent content={message.content} />
  </div>
</div>
```

**Files**:
- `src/lib/components/chat/AssistantMessage.svelte`

**Test Requirements**:
- Renders markdown correctly
- Code blocks highlighted

---

### Task 1.10: Markdown Renderer ✅

**Status**: Complete

**Description**: Render markdown with syntax-highlighted code blocks

**Acceptance Criteria**:
- [x] Uses marked for parsing
- [x] highlight.js for code blocks
- [ ] Language label on code blocks (future enhancement)
- [ ] Copy button on code blocks (future enhancement)
- [x] Safe HTML rendering

**Files Created**:
- `src/lib/utils/markdown.ts` - Markdown rendering with highlight.js
- `src/lib/components/chat/MarkdownContent.svelte` - Rendered markdown component
- `src/lib/utils/__tests__/markdown.test.ts` - 27 unit tests

**Implementation Approach**:
```typescript
// src/lib/utils/markdown.ts
import { marked } from 'marked';
import hljs from 'highlight.js';

marked.setOptions({
  highlight: (code: string, lang: string) => {
    if (lang && hljs.getLanguage(lang)) {
      return hljs.highlight(code, { language: lang }).value;
    }
    return hljs.highlightAuto(code).value;
  },
});

export function renderMarkdown(text: string): string {
  return marked.parse(text) as string;
}
```

```svelte
<!-- src/lib/components/chat/MarkdownContent.svelte -->
<script lang="ts">
  import { renderMarkdown } from '$lib/utils/markdown';

  let { content } = $props<{ content: string }>();

  let html = $derived(renderMarkdown(content));
</script>

<div class="prose prose-sm dark:prose-invert max-w-none">
  {@html html}
</div>
```

**Files**:
- `src/lib/utils/markdown.ts`
- `src/lib/components/chat/MarkdownContent.svelte`

**Dependencies**:
```bash
pnpm add marked highlight.js
pnpm add -D @tailwindcss/typography
```

**Test Requirements**:
- Markdown renders correctly (headers, lists, bold, etc.)
- Code blocks have syntax highlighting
- No XSS vulnerabilities

---

### Task 1.11: Streaming Message Flow ✅

**Status**: Complete

**Description**: Real-time token-by-token response streaming

**Acceptance Criteria**:
- [x] Send with `stream: true`
- [x] Parse SSE events
- [x] Update message content in real-time
- [x] Handle `response.output_text.delta` events
- [x] Finalize on `response.completed`

**Files Created**:
- `src/lib/api/responses.ts` - sendMessageStreaming + sendMessage functions
- `src/lib/api/__tests__/responses.test.ts` - 11 unit tests

**Implementation Approach**:
```typescript
// src/lib/api/responses.ts
import { client } from './client';
import { parseSSEStream } from './streaming';

export async function sendMessageStreaming(
  input: string,
  previousResponseId: string | null,
  onDelta: (text: string) => void,
  onComplete: (responseId: string) => void,
  onError: (error: Error) => void,
): Promise<void> {
  const response = await fetch(
    `${import.meta.env.VITE_RESPONSES_API_URL || 'http://localhost:9150'}/v1/responses`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'gpt-oss-120b',
        input,
        previous_response_id: previousResponseId,
        stream: true,
        store: true,
      }),
    }
  );

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  let responseId = '';
  let fullText = '';

  for await (const event of parseSSEStream(response)) {
    switch (event.type) {
      case 'response.created':
        responseId = event.response.id;
        break;
      case 'response.output_text.delta':
        fullText += event.delta;
        onDelta(fullText);
        break;
      case 'response.completed':
        onComplete(responseId);
        break;
      case 'response.failed':
        onError(new Error('Response failed'));
        break;
    }
  }
}
```

**Files**:
- `src/lib/api/responses.ts`
- `src/lib/stores/streaming.svelte.ts` - Track streaming state

**Test Requirements**:
- Integration test with backend
- Delta updates work
- Completion callback fires

---

### Task 1.12: Wire Up Home Page ✅

**Status**: Complete

**Description**: Connect all components on the home page

**Acceptance Criteria**:
- [x] Input sends messages
- [x] Messages appear in list
- [x] Streaming responses render token-by-token
- [x] New conversation created on first message

**Files Modified**:
- `src/routes/+page.svelte` - Connected ChatInput, MessageList, and conversationStore with sendMessageStreaming

**Implementation Approach**:
```svelte
<!-- src/routes/+page.svelte -->
<script lang="ts">
  import { conversationStore } from '$lib/stores/conversations.svelte';
  import { sendMessageStreaming } from '$lib/api/responses';
  import MessageList from '$lib/components/chat/MessageList.svelte';
  import ChatInput from '$lib/components/chat/ChatInput.svelte';

  let isStreaming = $state(false);

  async function handleSubmit(text: string) {
    // Create conversation if needed
    let conv = conversationStore.active;
    if (!conv) {
      conv = conversationStore.create();
    }

    // Add user message
    const userMessage = {
      id: crypto.randomUUID(),
      role: 'user' as const,
      content: text,
      createdAt: Date.now(),
    };
    conversationStore.addMessage(conv.id, userMessage);

    // Create placeholder for assistant
    const assistantMessage = {
      id: crypto.randomUUID(),
      role: 'assistant' as const,
      content: '',
      createdAt: Date.now(),
    };
    conversationStore.addMessage(conv.id, assistantMessage);

    isStreaming = true;

    try {
      await sendMessageStreaming(
        text,
        conv.lastResponseId,
        (content) => {
          // Update assistant message content
          assistantMessage.content = content;
        },
        (responseId) => {
          conversationStore.updateLastResponseId(conv!.id, responseId);
          isStreaming = false;
        },
        (error) => {
          console.error('Stream error:', error);
          isStreaming = false;
        }
      );
    } catch (error) {
      console.error('Request error:', error);
      isStreaming = false;
    }
  }

  let messages = $derived(conversationStore.active?.messages ?? []);
</script>

<MessageList {messages} />
<ChatInput onsubmit={handleSubmit} disabled={isStreaming} />
```

**Files**:
- `src/routes/+page.svelte`

**Test Requirements**:
- Full end-to-end flow works
- Messages persist in conversation
- Streaming updates render

---

## Slice 2: Conversations & Persistence

Add multi-conversation support and localStorage.

**Milestone**: Multiple conversations, persisted, with history

---

### Task 2.1: localStorage Persistence ✅

**Status**: Complete

**Description**: Persist conversations to localStorage

**Acceptance Criteria**:
- [x] Conversations saved on every change
- [x] Data loads on app init
- [x] Handles missing/corrupt data gracefully
- [x] Version field for future migrations

**Files Created**:
- `src/lib/utils/storage.ts` - Save/load/clear functions with version field
- `src/lib/utils/__tests__/storage.test.ts` - 17 unit tests

**Implementation Approach**:
```typescript
// src/lib/utils/storage.ts
const CONVERSATIONS_KEY = 'strieber-conversations';
const STORAGE_VERSION = 1;

interface StoredData {
  version: number;
  conversations: Conversation[];
  activeId: string | null;
}

export function loadConversations(): StoredData | null {
  try {
    const raw = localStorage.getItem(CONVERSATIONS_KEY);
    if (!raw) return null;
    const data = JSON.parse(raw) as StoredData;
    if (data.version !== STORAGE_VERSION) {
      // Handle migration in future
      return null;
    }
    return data;
  } catch {
    return null;
  }
}

export function saveConversations(conversations: Conversation[], activeId: string | null): void {
  const data: StoredData = {
    version: STORAGE_VERSION,
    conversations,
    activeId,
  };
  localStorage.setItem(CONVERSATIONS_KEY, JSON.stringify(data));
}
```

Update conversation store to auto-save:
```typescript
// In conversations.svelte.ts
$effect(() => {
  saveConversations(this.conversations, this.activeId);
});
```

**Files**:
- `src/lib/utils/storage.ts`
- Update `src/lib/stores/conversations.svelte.ts`

**Test Requirements**:
- Data persists across refreshes
- Corrupt data doesn't crash app

---

### Task 2.2: Sidebar Conversation List ✅

**Status**: Complete

**Description**: Display conversations grouped by date with actions

**Acceptance Criteria**:
- [x] Conversations listed with titles
- [x] Active conversation highlighted
- [x] Click to switch conversation
- [x] "New Chat" button
- [x] Delete action (rename can come later)
- [x] Date grouping (Today, Yesterday, Previous 7 Days, Older)

**Files Created**:
- `src/lib/components/sidebar/ConversationItem.svelte` - Individual item with hover delete
- `src/lib/components/sidebar/ConversationList.svelte` - Full list with date grouping
- `src/lib/components/sidebar/index.ts` - Barrel export
- `src/lib/components/sidebar/__tests__/ConversationItem.test.ts` - 7 unit tests
- `src/lib/components/sidebar/__tests__/ConversationList.test.ts` - 8 unit tests
- `src/lib/utils/dates.ts` - Date grouping utility
- `src/lib/utils/__tests__/dates.test.ts` - 13 unit tests

**Implementation Approach**:
```svelte
<!-- src/lib/components/sidebar/ConversationList.svelte -->
<script lang="ts">
  import { conversationStore } from '$lib/stores/conversations.svelte';
  import { Button } from '$lib/components/ui/button';
  import { Plus, Trash2 } from 'lucide-svelte';
  import { goto } from '$app/navigation';

  function newChat() {
    const conv = conversationStore.create();
    goto(`/c/${conv.id}`);
  }

  function selectConversation(id: string) {
    conversationStore.activeId = id;
    goto(`/c/${id}`);
  }

  function deleteConversation(id: string, e: Event) {
    e.stopPropagation();
    conversationStore.delete(id);
    if (!conversationStore.activeId) {
      goto('/');
    }
  }
</script>

<div class="p-4 space-y-2">
  <Button onclick={newChat} class="w-full" variant="outline">
    <Plus class="h-4 w-4 mr-2" /> New Chat
  </Button>

  <div class="space-y-1 mt-4">
    {#each conversationStore.conversations as conv (conv.id)}
      <button
        onclick={() => selectConversation(conv.id)}
        class="w-full text-left px-3 py-2 rounded-lg hover:bg-accent flex justify-between items-center group"
        class:bg-accent={conv.id === conversationStore.activeId}
      >
        <span class="truncate">{conv.title}</span>
        <button
          onclick={(e) => deleteConversation(conv.id, e)}
          class="opacity-0 group-hover:opacity-100 p-1 hover:bg-destructive/20 rounded"
        >
          <Trash2 class="h-4 w-4" />
        </button>
      </button>
    {/each}
  </div>
</div>
```

**Files**:
- `src/lib/components/sidebar/ConversationList.svelte`
- Update `src/routes/+layout.svelte`

**Test Requirements**:
- Conversations display
- Selection works
- Delete works

---

### Task 2.3: Conversation Routes ✅

**Status**: Complete

**Description**: SvelteKit routes for home and conversation pages

**Acceptance Criteria**:
- [x] `/` - New conversation / home
- [x] `/c/[id]` - Specific conversation by ID
- [x] Navigation updates URL
- [x] Direct URL access loads correct conversation

**Files Created**:
- `src/routes/c/[id]/+page.svelte` - Conversation view with redirect on not found

**Files Modified**:
- `src/routes/+page.svelte` - Added navigation after creating conversation
- `src/routes/+layout.svelte` - Integrated persistence and sidebar

**Implementation Approach**:
```svelte
<!-- src/routes/c/[id]/+page.svelte -->
<script lang="ts">
  import { page } from '$app/stores';
  import { conversationStore } from '$lib/stores/conversations.svelte';
  import { goto } from '$app/navigation';
  import MessageList from '$lib/components/chat/MessageList.svelte';
  import ChatInput from '$lib/components/chat/ChatInput.svelte';
  // ... same logic as home page but load conversation by ID

  $effect(() => {
    const id = $page.params.id;
    const conv = conversationStore.conversations.find(c => c.id === id);
    if (conv) {
      conversationStore.activeId = id;
    } else {
      // Conversation not found, redirect home
      goto('/');
    }
  });
</script>
```

**Files**:
- `src/routes/c/[id]/+page.svelte`
- Update `src/routes/+page.svelte`

**Test Requirements**:
- Direct URL access works
- Invalid IDs redirect to home

---

### Task 2.4: Multi-Turn Chaining ✅

**Status**: Complete (built into Slice 1)

**Description**: Use `previous_response_id` for context

**Acceptance Criteria**:
- [x] First message has no `previous_response_id`
- [x] Subsequent messages include last response's ID
- [x] Context preserved across turns

**Verification**:
- E2E test confirms context chaining works (user provides name, asks "What is my name?", model correctly responds)
- Screenshot evidence in `test-results/screenshots/chat-context.png`

---

## Slice 3: Tool Displays

Rich display for model capabilities.

**Milestone**: Full tool call visualization

---

### Task 3.1: Reasoning Block Component

**Description**: Collapsible display for model reasoning/thinking

**Acceptance Criteria**:
- Collapsed by default
- "Thinking..." indicator while streaming
- Muted styling
- Uses shadcn collapsible

**Implementation Approach**:
```svelte
<!-- src/lib/components/tools/ReasoningBlock.svelte -->
<script lang="ts">
  import * as Collapsible from '$lib/components/ui/collapsible';
  import { Brain, ChevronDown } from 'lucide-svelte';

  let { content, isStreaming = false } = $props<{
    content: string;
    isStreaming?: boolean;
  }>();

  let open = $state(false);
</script>

<Collapsible.Root bind:open class="border rounded-lg bg-muted/30">
  <Collapsible.Trigger class="flex items-center gap-2 w-full p-3 text-sm text-muted-foreground">
    <Brain class="h-4 w-4" />
    {isStreaming ? 'Thinking...' : 'Reasoning'}
    <ChevronDown class="h-4 w-4 ml-auto transition-transform" class:rotate-180={open} />
  </Collapsible.Trigger>
  <Collapsible.Content class="px-3 pb-3">
    <p class="text-sm text-muted-foreground whitespace-pre-wrap">{content}</p>
  </Collapsible.Content>
</Collapsible.Root>
```

**Files**:
- `src/lib/components/tools/ReasoningBlock.svelte`

**Dependencies**:
```bash
npx shadcn-svelte@latest add collapsible
```

**Test Requirements**:
- Collapse/expand works
- Streaming indicator shows

---

### Task 3.2: Web Search Tool Display

**Description**: Display web search queries and results

**Acceptance Criteria**:
- Shows search query
- Loading state during search
- Source list with links
- Links open in new tab

**Implementation Approach**:
```svelte
<!-- src/lib/components/tools/WebSearchCall.svelte -->
<script lang="ts">
  import { Search, ExternalLink } from 'lucide-svelte';

  let { status, action } = $props<{
    status: string;
    action?: { query?: string; results?: Array<{ url: string; title?: string }> };
  }>();

  let isSearching = $derived(status === 'in_progress');
</script>

<div class="border rounded-lg p-3 space-y-2">
  <div class="flex items-center gap-2 text-sm">
    <Search class="h-4 w-4" />
    {#if isSearching}
      <span class="text-muted-foreground">Searching...</span>
    {:else}
      <span>Searched: "{action?.query}"</span>
    {/if}
  </div>

  {#if action?.results?.length}
    <ul class="text-sm space-y-1 pl-6">
      {#each action.results as result}
        <li>
          <a
            href={result.url}
            target="_blank"
            rel="noopener"
            class="text-blue-600 hover:underline flex items-center gap-1"
          >
            {result.title || result.url}
            <ExternalLink class="h-3 w-3" />
          </a>
        </li>
      {/each}
    </ul>
  {/if}
</div>
```

**Files**:
- `src/lib/components/tools/WebSearchCall.svelte`

**Test Requirements**:
- Loading state displays
- Results render with links

---

### Task 3.3: Code Interpreter Display

**Description**: Display code execution and outputs

**Acceptance Criteria**:
- Syntax-highlighted code block
- Execution status indicator
- stdout/stderr output
- Image outputs rendered

**Implementation Approach**:
```svelte
<!-- src/lib/components/tools/CodeInterpreterCall.svelte -->
<script lang="ts">
  import { Code, Play, CheckCircle, XCircle } from 'lucide-svelte';
  import MarkdownContent from '$lib/components/chat/MarkdownContent.svelte';

  let { status, code, outputs } = $props<{
    status: string;
    code?: string;
    outputs?: Array<{ type: string; content?: string; image_url?: string }>;
  }>();

  let isRunning = $derived(status === 'in_progress');
</script>

<div class="border rounded-lg overflow-hidden">
  <div class="flex items-center gap-2 p-2 bg-muted border-b text-sm">
    <Code class="h-4 w-4" />
    <span>Code Interpreter</span>
    {#if isRunning}
      <Play class="h-4 w-4 animate-pulse" />
    {:else if status === 'completed'}
      <CheckCircle class="h-4 w-4 text-green-600" />
    {:else if status === 'failed'}
      <XCircle class="h-4 w-4 text-red-600" />
    {/if}
  </div>

  {#if code}
    <pre class="p-3 bg-zinc-900 text-zinc-100 text-sm overflow-x-auto"><code>{code}</code></pre>
  {/if}

  {#if outputs?.length}
    <div class="p-3 space-y-2 border-t">
      {#each outputs as output}
        {#if output.type === 'text' && output.content}
          <pre class="text-sm bg-muted p-2 rounded">{output.content}</pre>
        {:else if output.type === 'image' && output.image_url}
          <img src={output.image_url} alt="Output" class="max-w-full rounded" />
        {/if}
      {/each}
    </div>
  {/if}
</div>
```

**Files**:
- `src/lib/components/tools/CodeInterpreterCall.svelte`

**Test Requirements**:
- Code displays with highlighting
- Outputs render correctly

---

### Task 3.4: Annotation/Citation Rendering

**Description**: Render inline citations and source links

**Acceptance Criteria**:
- Inline superscript numbers
- Clickable links to sources
- URL citations link externally

**Implementation Approach**:
Update `MarkdownContent.svelte` to process annotations and inject citation markers. Build a citation list component.

**Files**:
- `src/lib/utils/annotations.ts`
- Update `MarkdownContent.svelte`

**Test Requirements**:
- Citations render inline
- Links work correctly

---

## Slice 4: Settings & Polish

Configuration and UX improvements.

**Milestone**: Configurable, polished experience

---

### Task 4.1: Model Selector

**Description**: Dropdown to select inference model

**Acceptance Criteria**:
- Fetches models from `/v1/models`
- Dropdown in header
- Persists selection
- Shows current model

**Files**:
- `src/lib/components/ModelSelector.svelte`
- `src/lib/stores/settings.svelte.ts`

**Dependencies**:
```bash
npx shadcn-svelte@latest add select
```

---

### Task 4.2: Settings Panel

**Description**: Dialog for user preferences

**Acceptance Criteria**:
- Model selector
- Temperature slider (0-2)
- Theme toggle
- Persists to localStorage

**Files**:
- `src/lib/components/settings/SettingsPanel.svelte`
- `src/lib/stores/settings.svelte.ts`

**Dependencies**:
```bash
npx shadcn-svelte@latest add dialog slider
```

---

### Task 4.3: Theme Support

**Description**: Light/dark/system theme

**Acceptance Criteria**:
- Toggle between modes
- Persists preference
- System respects OS setting

**Files**:
- `src/lib/utils/theme.ts`
- Update `src/app.html`

---

### Task 4.4: Error Handling & Toasts

**Description**: User-friendly error display

**Acceptance Criteria**:
- API errors show toast
- Network errors handled
- Uses shadcn toast

**Files**:
- `src/lib/utils/errors.ts`
- Toast provider setup

**Dependencies**:
```bash
npx shadcn-svelte@latest add toast
```

---

### Task 4.5: Stop Streaming Button

**Description**: Cancel in-progress response

**Acceptance Criteria**:
- Appears during streaming
- Cancels request via AbortController
- Updates UI state

**Files**:
- Update `ChatInput.svelte`
- `src/lib/stores/streaming.svelte.ts`

---

### Task 4.6: Regenerate Response

**Description**: Re-send last user message

**Acceptance Criteria**:
- Button on last assistant message
- Removes last response and re-sends
- Works with streaming

**Files**:
- Update `AssistantMessage.svelte`

---

## Slice 5: File Upload

Image and document support.

**Milestone**: Full file upload support

---

### Task 5.1: File Processing Utilities

**Description**: Base64 encoding and validation

**Acceptance Criteria**:
- Converts files to data URLs
- Validates file types and sizes
- Max 20MB per file

**Files**:
- `src/lib/utils/files.ts`

---

### Task 5.2: Paste Handler

**Description**: Handle image paste

**Acceptance Criteria**:
- Detects images in clipboard
- Adds to attachments
- Shows preview

**Files**:
- Update `ChatInput.svelte`

---

### Task 5.3: Drag and Drop

**Description**: Handle file drag-drop

**Acceptance Criteria**:
- Visual indicator on dragover
- Accepts images and PDFs
- Validates and rejects invalid files

**Files**:
- Update `ChatInput.svelte`

---

### Task 5.4: File Picker Button

**Description**: Button to open file picker

**Acceptance Criteria**:
- Opens native dialog
- Filters to supported types
- Multiple file selection

**Files**:
- Update `ChatInput.svelte`

---

### Task 5.5: Attachment Preview Strip

**Description**: Show pending attachments

**Acceptance Criteria**:
- Image thumbnails
- File icons with names
- Remove button

**Files**:
- `src/lib/components/chat/AttachmentStrip.svelte`

---

## Slice 6: Mobile & Final Polish

Final polish for production.

**Milestone**: Production-ready

---

### Task 6.1: Mobile Responsiveness

**Description**: Mobile-friendly UI

**Acceptance Criteria**:
- Sidebar collapses to hamburger
- Touch-friendly targets
- Works on small screens

**Files**:
- Various component updates

**Dependencies**:
```bash
npx shadcn-svelte@latest add sheet
```

---

### Task 6.2: Keyboard Shortcuts

**Description**: Power user shortcuts

**Acceptance Criteria**:
- Cmd/Ctrl+N - New chat
- Cmd/Ctrl+/ - Toggle sidebar
- Escape - Cancel action

**Files**:
- `src/lib/utils/shortcuts.ts`

---

### Task 6.3: Docker Build

**Description**: Production Dockerfile

**Acceptance Criteria**:
- Multi-stage build
- Health check endpoint
- Non-root user
- <200MB image

**Files**:
- `frontend/Dockerfile`
- `src/routes/health/+server.ts`

---

## Summary

| Slice | Tasks | Milestone |
|-------|-------|-----------|
| 1 | 12 | Streaming chat with markdown |
| 2 | 4 | Multi-conversation with persistence |
| 3 | 4 | Tool call visualization |
| 4 | 6 | Settings, theme, error handling |
| 5 | 5 | File upload |
| 6 | 3 | Mobile, shortcuts, Docker |

**Total: 34 tasks**
