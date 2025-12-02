/**
 * API Module Barrel Export
 *
 * Re-exports client, types, and streaming utilities for convenient imports:
 *   import { client, createClient, parseSSEStream, type StreamEvent } from '$lib/api';
 */

export { client, createClient, getApiBaseUrl } from './client';
export {
	type StreamingCallbacks,
	type StreamingOptions,
	sendMessage,
	sendMessageStreaming
} from './responses';
export {
	isCompletedEvent,
	isErrorEvent,
	isFailedEvent,
	isTextDeltaEvent,
	parseSSEData,
	parseSSEStream,
	type StreamEvent,
	StreamParseError
} from './streaming';
export * from './types';
