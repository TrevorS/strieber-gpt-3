-- ABOUTME: Initial database schema for responses-api storage.
-- Creates tables for responses, conversations, and generic key-value storage.

-- Responses table (for responses-api internal use)
CREATE TABLE IF NOT EXISTS responses (
    id TEXT PRIMARY KEY,
    response_json TEXT NOT NULL,
    request_json TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    expires_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_responses_expires ON responses(expires_at);

-- Conversations table (for responses-api internal use)
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    metadata_json TEXT,
    created_at INTEGER NOT NULL,
    expires_at INTEGER
);
CREATE INDEX IF NOT EXISTS idx_conversations_expires ON conversations(expires_at);

-- Conversation Items table
CREATE TABLE IF NOT EXISTS conversation_items (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    item_order INTEGER NOT NULL,
    content_json TEXT NOT NULL,
    created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_items_conv ON conversation_items(conversation_id, item_order);

-- Generic Storage table (for MCP servers via /v1/storage/{collection} API)
-- Collections are simple names: "jobs", "datasets", "tasks", etc.
CREATE TABLE IF NOT EXISTS generic_storage (
    collection TEXT NOT NULL,        -- e.g., "jobs", "datasets"
    id TEXT NOT NULL,                -- record ID within collection
    data_json TEXT NOT NULL,         -- arbitrary JSON blob
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    expires_at INTEGER,              -- optional TTL
    PRIMARY KEY (collection, id)
);
CREATE INDEX IF NOT EXISTS idx_storage_collection ON generic_storage(collection);
CREATE INDEX IF NOT EXISTS idx_storage_expires ON generic_storage(expires_at) WHERE expires_at IS NOT NULL;
