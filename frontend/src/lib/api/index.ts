/**
 * API Module Barrel Export
 *
 * Re-exports client, types, and streaming utilities for convenient imports:
 *   import { client, createClient, parseSSEStream, type StreamEvent } from '$lib/api';
 */

export { client, createClient, getApiBaseUrl } from './client';
export * from './types';
export {
	parseSSEStream,
	parseSSEData,
	StreamParseError,
	isTextDeltaEvent,
	isCompletedEvent,
	isFailedEvent,
	isErrorEvent,
	type StreamEvent
} from './streaming';
