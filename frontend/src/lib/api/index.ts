/**
 * API Module Barrel Export
 *
 * Re-exports client and types for convenient imports:
 *   import { client, createClient, type Response } from '$lib/api';
 */

export { client, createClient, getApiBaseUrl } from './client';
export * from './types';
