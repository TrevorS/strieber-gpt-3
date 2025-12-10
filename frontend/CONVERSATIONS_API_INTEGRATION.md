# Conversations API Integration Guide

This guide describes how to migrate the frontend from the current client-side conversation management to the new server-side Conversations API.

## Overview

### Current Architecture
- Conversations stored in localStorage via `ConversationStore`
- Multi-turn context via `previous_response_id` chaining
- Client manages all conversation state (messages, metadata, IDs)

### New Architecture
- Conversations stored on the server with unique `conv_*` IDs
- Items (messages, tool calls, reasoning) stored as conversation items
- Response outputs automatically appended to conversation
- Client becomes a thin layer that syncs with server state

## API Reference

### Conversations Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/conversations` | Create a new conversation |
| `GET` | `/v1/conversations/{id}` | Get conversation metadata |
| `POST` | `/v1/conversations/{id}` | Update conversation metadata |
| `DELETE` | `/v1/conversations/{id}` | Delete a conversation |

### Conversation Items Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/v1/conversations/{id}/items` | List items (paginated) |
| `POST` | `/v1/conversations/{id}/items` | Add items (max 20) |
| `GET` | `/v1/conversations/{id}/items/{item_id}` | Get single item |
| `DELETE` | `/v1/conversations/{id}/items/{item_id}` | Delete an item |

### Using Conversations with Responses API

Pass a `conversation` parameter instead of `previous_response_id`:

```typescript
// Before: Using previous_response_id
const response = await fetch('/v1/responses', {
  method: 'POST',
  body: JSON.stringify({
    model: 'gpt-4o',
    input: 'Hello',
    previous_response_id: 'resp_abc123',  // Chain to previous response
    stream: true,
    store: true
  })
});

// After: Using conversation
const response = await fetch('/v1/responses', {
  method: 'POST',
  body: JSON.stringify({
    model: 'gpt-4o',
    input: 'Hello',
    conversation: { id: 'conv_xyz789' },  // Use conversation context
    stream: true,
    store: true
  })
});
```

**Important**: `conversation` and `previous_response_id` are mutually exclusive.

## TypeScript Types

Add these types to `frontend/src/lib/api/types.ts`:

```typescript
// ============================================================================
// Conversations API Types
// ============================================================================

/**
 * Metadata object - up to 16 key-value pairs.
 * Keys max 64 chars, values max 512 chars.
 */
export type ConversationMetadata = Record<string, string>;

/**
 * Server-side conversation object.
 */
export interface Conversation {
  id: string;                           // conv_<uuid>
  object: 'conversation';
  created_at: number;                   // Unix timestamp
  metadata?: ConversationMetadata;
}

/**
 * Conversation item - can be input or output type.
 */
export interface ConversationItem {
  id: string;                           // msg_*, fc_*, rs_*, item_*
  status: 'in_progress' | 'completed' | 'incomplete';
  type: string;                         // 'message', 'function_call', etc.
  [key: string]: unknown;               // Type-specific fields
}

/**
 * Paginated list response.
 */
export interface ListResponse<T> {
  object: 'list';
  data: T[];
  has_more: boolean;
  first_id: string | null;
  last_id: string | null;
}

/**
 * Delete confirmation response.
 */
export interface ConversationDeleted {
  id: string;
  object: 'conversation.deleted';
  deleted: true;
}

// Request types
export interface CreateConversationRequest {
  items?: InputItem[];                  // Initial items (max 20)
  metadata?: ConversationMetadata;
}

export interface UpdateConversationRequest {
  metadata: ConversationMetadata;
}

export interface CreateItemsRequest {
  items: InputItem[];                   // 1-20 items
}

// Pagination query params
export interface PaginationQuery {
  limit?: number;                       // Default 20, max 100
  order?: 'asc' | 'desc';              // Default 'desc'
  after?: string;                       // Cursor for pagination
  before?: string;                      // Cursor for pagination
}
```

## API Client Implementation

Create `frontend/src/lib/api/conversations.ts`:

```typescript
import { getApiClient } from './client';
import type {
  Conversation,
  ConversationItem,
  ConversationDeleted,
  ListResponse,
  CreateConversationRequest,
  UpdateConversationRequest,
  CreateItemsRequest,
  PaginationQuery,
  ConversationMetadata
} from './types';

const BASE_URL = import.meta.env.VITE_RESPONSES_API_URL || 'http://localhost:9150';

/**
 * Create a new conversation.
 */
export async function createConversation(
  request: CreateConversationRequest = {}
): Promise<Conversation> {
  const response = await fetch(`${BASE_URL}/v1/conversations`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request)
  });

  if (!response.ok) {
    throw new Error(`Failed to create conversation: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get a conversation by ID.
 */
export async function getConversation(id: string): Promise<Conversation> {
  const response = await fetch(`${BASE_URL}/v1/conversations/${id}`);

  if (!response.ok) {
    if (response.status === 404) {
      throw new Error(`Conversation ${id} not found`);
    }
    throw new Error(`Failed to get conversation: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Update conversation metadata.
 */
export async function updateConversation(
  id: string,
  metadata: ConversationMetadata
): Promise<Conversation> {
  const response = await fetch(`${BASE_URL}/v1/conversations/${id}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ metadata })
  });

  if (!response.ok) {
    throw new Error(`Failed to update conversation: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Delete a conversation.
 */
export async function deleteConversation(id: string): Promise<ConversationDeleted> {
  const response = await fetch(`${BASE_URL}/v1/conversations/${id}`, {
    method: 'DELETE'
  });

  if (!response.ok) {
    throw new Error(`Failed to delete conversation: ${response.statusText}`);
  }

  return response.json();
}

/**
 * List items in a conversation.
 */
export async function listItems(
  conversationId: string,
  query: PaginationQuery = {}
): Promise<ListResponse<ConversationItem>> {
  const params = new URLSearchParams();
  if (query.limit) params.set('limit', String(query.limit));
  if (query.order) params.set('order', query.order);
  if (query.after) params.set('after', query.after);
  if (query.before) params.set('before', query.before);

  const url = `${BASE_URL}/v1/conversations/${conversationId}/items?${params}`;
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`Failed to list items: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Add items to a conversation.
 */
export async function addItems(
  conversationId: string,
  items: InputItem[]
): Promise<ListResponse<ConversationItem>> {
  const response = await fetch(
    `${BASE_URL}/v1/conversations/${conversationId}/items`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ items })
    }
  );

  if (!response.ok) {
    throw new Error(`Failed to add items: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get a single item.
 */
export async function getItem(
  conversationId: string,
  itemId: string
): Promise<ConversationItem> {
  const response = await fetch(
    `${BASE_URL}/v1/conversations/${conversationId}/items/${itemId}`
  );

  if (!response.ok) {
    throw new Error(`Failed to get item: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Delete an item from a conversation.
 */
export async function deleteItem(
  conversationId: string,
  itemId: string
): Promise<Conversation> {
  const response = await fetch(
    `${BASE_URL}/v1/conversations/${conversationId}/items/${itemId}`,
    { method: 'DELETE' }
  );

  if (!response.ok) {
    throw new Error(`Failed to delete item: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Fetch all items from a conversation (handles pagination).
 */
export async function fetchAllItems(
  conversationId: string
): Promise<ConversationItem[]> {
  const allItems: ConversationItem[] = [];
  let after: string | undefined;

  while (true) {
    const response = await listItems(conversationId, {
      limit: 100,
      order: 'asc',
      after
    });

    allItems.push(...response.data);

    if (!response.has_more || !response.last_id) {
      break;
    }

    after = response.last_id;
  }

  return allItems;
}
```

## Migration Steps

### Step 1: Update Store Types

Modify `frontend/src/lib/stores/types.ts`:

```typescript
// Add server ID tracking
export interface Conversation {
  id: string;                    // Local UUID (for URL routing)
  serverId: string | null;       // Server conv_* ID (null until created)
  title: string;
  createdAt: number;
  updatedAt: number;
  messages: Message[];           // For UI display
  // Remove: lastResponseId - no longer needed with conversation API
}

export interface Message {
  id: string;                    // Local UUID
  serverId: string | null;       // Server item ID (msg_*, etc.)
  role: 'user' | 'assistant';
  content: string;
  rawOutput?: ResponseOutputItem[];
  attachments?: Attachment[];
  createdAt: number;
  isStreaming?: boolean;
  isEdited?: boolean;
}
```

### Step 2: Update ConversationStore

Key changes to `frontend/src/lib/stores/conversations.svelte.ts`:

```typescript
import * as conversationsApi from '$lib/api/conversations';

class ConversationStore {
  // ... existing state ...

  /**
   * Create a new conversation (syncs with server).
   */
  async create(metadata?: Record<string, string>): Promise<Conversation> {
    // Create on server first
    const serverConv = await conversationsApi.createConversation({ metadata });

    // Create local representation
    const localConv: Conversation = {
      id: crypto.randomUUID(),
      serverId: serverConv.id,
      title: metadata?.title || 'New Chat',
      createdAt: Date.now(),
      updatedAt: Date.now(),
      messages: []
    };

    this.conversations.push(localConv);
    this.activeId = localConv.id;

    return localConv;
  }

  /**
   * Load conversation from server.
   */
  async loadFromServer(serverId: string): Promise<Conversation | null> {
    try {
      const serverConv = await conversationsApi.getConversation(serverId);
      const items = await conversationsApi.fetchAllItems(serverId);

      // Convert server items to local messages
      const messages = this.itemsToMessages(items);

      const localConv: Conversation = {
        id: crypto.randomUUID(),
        serverId: serverConv.id,
        title: serverConv.metadata?.title || 'Untitled',
        createdAt: serverConv.created_at * 1000,
        updatedAt: Date.now(),
        messages
      };

      // Check if already loaded
      const existing = this.conversations.find(c => c.serverId === serverId);
      if (existing) {
        Object.assign(existing, localConv);
        return existing;
      }

      this.conversations.push(localConv);
      return localConv;
    } catch (error) {
      console.error('Failed to load conversation:', error);
      return null;
    }
  }

  /**
   * Delete conversation (syncs with server).
   */
  async delete(id: string): Promise<boolean> {
    const conv = this.conversations.find(c => c.id === id);
    if (!conv) return false;

    // Delete from server if it exists there
    if (conv.serverId) {
      try {
        await conversationsApi.deleteConversation(conv.serverId);
      } catch (error) {
        console.error('Failed to delete from server:', error);
        // Continue with local deletion anyway
      }
    }

    // Remove locally
    const index = this.conversations.findIndex(c => c.id === id);
    if (index >= 0) {
      this.conversations.splice(index, 1);
    }

    if (this.activeId === id) {
      this.activeId = this.sorted[0]?.id ?? null;
    }

    return true;
  }

  /**
   * Convert server items to local message format.
   */
  private itemsToMessages(items: ConversationItem[]): Message[] {
    const messages: Message[] = [];
    let currentAssistant: Message | null = null;

    for (const item of items) {
      if (item.type === 'message') {
        const msg: Message = {
          id: crypto.randomUUID(),
          serverId: item.id,
          role: item.role as 'user' | 'assistant',
          content: this.extractContent(item),
          createdAt: Date.now(),
          rawOutput: item.role === 'assistant' ? [item] : undefined
        };

        if (item.role === 'assistant') {
          currentAssistant = msg;
        } else {
          currentAssistant = null;
        }

        messages.push(msg);
      } else if (currentAssistant) {
        // Append tool outputs to current assistant message
        currentAssistant.rawOutput = currentAssistant.rawOutput || [];
        currentAssistant.rawOutput.push(item);
      }
    }

    return messages;
  }

  private extractContent(item: ConversationItem): string {
    // Handle different content formats
    if (typeof item.content === 'string') {
      return item.content;
    }
    if (Array.isArray(item.content)) {
      return item.content
        .filter((c: any) => c.type === 'output_text' || c.type === 'text')
        .map((c: any) => c.text)
        .join('');
    }
    return '';
  }
}
```

### Step 3: Update Response Sending

Modify `frontend/src/lib/api/responses.ts`:

```typescript
export interface SendMessageOptions {
  conversationId: string;        // Server conv_* ID
  // Remove: previousResponseId
  model?: string;
  tools?: Tool[];
  instructions?: string;
  temperature?: number;
  onDelta?: (delta: string) => void;
  onOutputItem?: (item: ResponseOutputItem) => void;
  onComplete?: (response: Response) => void;
  onError?: (error: Error) => void;
  signal?: AbortSignal;
}

export async function sendMessageStreaming(
  input: string | InputItem[],
  options: SendMessageOptions
): Promise<void> {
  const {
    conversationId,
    model = 'gpt-4o',
    tools = [],
    instructions,
    temperature,
    onDelta,
    onOutputItem,
    onComplete,
    onError,
    signal
  } = options;

  const response = await fetch(`${BASE_URL}/v1/responses`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model,
      input,
      conversation: { id: conversationId },  // Use conversation instead
      stream: true,
      store: true,
      tools,
      instructions,
      temperature
    }),
    signal
  });

  // ... rest of streaming logic unchanged ...
}
```

### Step 4: Update Page Components

Update `frontend/src/routes/c/[id]/+page.svelte`:

```svelte
<script lang="ts">
  import { page } from '$app/stores';
  import { conversationStore } from '$lib/stores/conversations.svelte';
  import { sendMessageStreaming } from '$lib/api/responses';

  // Get conversation by local ID from URL
  let conversation = $derived(conversationStore.get($page.params.id));

  async function handleSubmit(text: string, attachments: Attachment[]) {
    if (!conversation?.serverId) {
      // Create server conversation if needed
      const newConv = await conversationStore.create();
      conversation = newConv;
    }

    // Add user message locally for immediate UI feedback
    const userMsg = conversationStore.addMessage(
      conversation.id,
      'user',
      text,
      attachments
    );

    // Add placeholder assistant message
    const assistantMsg = conversationStore.addMessage(
      conversation.id,
      'assistant',
      ''
    );
    conversationStore.setMessageStreaming(conversation.id, assistantMsg.id, true);

    try {
      await sendMessageStreaming(buildInput(text, attachments), {
        conversationId: conversation.serverId!,
        model: settings.model,
        tools: getEnabledTools(),
        instructions: settings.systemPrompt,
        temperature: settings.temperature,
        onDelta: (delta) => {
          conversationStore.updateMessageContent(
            conversation!.id,
            assistantMsg.id,
            (prev) => prev + delta
          );
        },
        onOutputItem: (item) => {
          conversationStore.setOutputItem(
            conversation!.id,
            assistantMsg.id,
            item
          );
        },
        onComplete: (response) => {
          // Response items are automatically stored in conversation
          conversationStore.setMessageStreaming(
            conversation!.id,
            assistantMsg.id,
            false
          );
        },
        onError: (error) => {
          toasts.error(error.message);
          conversationStore.removeLastAssistantMessage(conversation!.id);
        },
        signal: abortController.signal
      });
    } catch (error) {
      if (error.name !== 'AbortError') {
        toasts.error('Failed to send message');
      }
    }
  }
</script>
```

### Step 5: Handle Message Editing

When a user edits a message, delete subsequent items from the server:

```typescript
async function handleEdit(messageId: string, newContent: string) {
  const conv = conversation;
  if (!conv?.serverId) return;

  // Find the message and all messages after it
  const msgIndex = conv.messages.findIndex(m => m.id === messageId);
  if (msgIndex < 0) return;

  // Delete subsequent items from server
  const subsequentMessages = conv.messages.slice(msgIndex + 1);
  for (const msg of subsequentMessages) {
    if (msg.serverId) {
      try {
        await conversationsApi.deleteItem(conv.serverId, msg.serverId);
      } catch (error) {
        console.error('Failed to delete item:', error);
      }
    }
  }

  // Update local state
  conversationStore.updateMessage(conv.id, messageId, newContent);
  conversationStore.removeMessagesAfter(conv.id, messageId);

  // Re-send the edited message
  // ... create new assistant message and stream response ...
}
```

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────────┐      ┌──────────────────────────────┐   │
│   │  ConversationStore│      │       UI Components          │   │
│   │                   │◄────►│  (MessageList, ChatInput)    │   │
│   │  - conversations  │      └──────────────────────────────┘   │
│   │  - activeId       │                                         │
│   └────────┬──────────┘                                         │
│            │                                                     │
│            │ sync                                                │
│            ▼                                                     │
│   ┌──────────────────┐                                          │
│   │  Conversations    │                                          │
│   │  API Client       │                                          │
│   └────────┬──────────┘                                         │
│            │                                                     │
└────────────┼─────────────────────────────────────────────────────┘
             │ HTTP
             ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Backend                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────────┐      ┌──────────────────────────────┐   │
│   │ Conversations API │      │      Responses API           │   │
│   │                   │      │                               │   │
│   │ POST /v1/convers- │      │ POST /v1/responses           │   │
│   │   ations          │      │   { conversation: {id} }     │   │
│   │ GET  /v1/convers- │      │                               │   │
│   │   ations/{id}     │◄────►│ - Reads conversation items   │   │
│   │ POST /v1/convers- │      │ - Appends output items       │   │
│   │   ations/{id}/    │      │                               │   │
│   │   items           │      └──────────────────────────────┘   │
│   └────────┬──────────┘                                         │
│            │                                                     │
│            ▼                                                     │
│   ┌──────────────────┐                                          │
│   │ ConversationStore │                                          │
│   │   (In-Memory)     │                                          │
│   └──────────────────┘                                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Benefits of Migration

1. **Server-side state**: Conversations persist across sessions without localStorage
2. **Simpler context management**: No need to track `previous_response_id` chains
3. **Item-level operations**: Edit/delete individual messages
4. **Metadata support**: Store conversation title, tags, etc. on server
5. **Pagination**: Handle long conversations efficiently
6. **Multi-device sync**: Same conversation accessible from anywhere

## Migration Checklist

- [ ] Add TypeScript types for Conversations API
- [ ] Create `conversations.ts` API client
- [ ] Update `ConversationStore` with server sync methods
- [ ] Update `sendMessageStreaming` to use `conversation` param
- [ ] Update conversation creation flow
- [ ] Update message editing to delete server items
- [ ] Update regenerate flow
- [ ] Update conversation deletion
- [ ] Add loading states for server operations
- [ ] Add error handling for network failures
- [ ] Add offline fallback (optional)
- [ ] Remove localStorage persistence (or keep as cache)
- [ ] Update URL routing if using server IDs
