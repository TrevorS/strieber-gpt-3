/**
 * Playwright Logger Helper
 *
 * Utilities for capturing and analyzing structured logs in E2E tests.
 *
 * Usage:
 *   import { setupLogCapture, getStructuredLogs, waitForLog } from './helpers/logger';
 *
 *   test('example', async ({ page }) => {
 *     const logs = setupLogCapture(page);
 *
 *     await page.goto('/');
 *     // ... do stuff ...
 *
 *     // Get all captured logs
 *     const allLogs = getStructuredLogs(logs);
 *
 *     // Filter by category
 *     const storeLogs = allLogs.filter(l => l.category === 'store');
 *
 *     // Wait for a specific log
 *     await waitForLog(logs, { category: 'persistence', message: 'Conversations loaded' });
 *   });
 */

import type { Page, ConsoleMessage } from '@playwright/test';

export interface LogEntry {
	timestamp: string;
	level: 'debug' | 'info' | 'warn' | 'error';
	category: string;
	message: string;
	data?: Record<string, unknown>;
}

export interface LogCapture {
	messages: ConsoleMessage[];
	structuredLogs: LogEntry[];
}

/**
 * Set up log capture for a page.
 * Call this before navigating to capture all console logs.
 */
export function setupLogCapture(page: Page): LogCapture {
	const capture: LogCapture = {
		messages: [],
		structuredLogs: []
	};

	page.on('console', (msg) => {
		capture.messages.push(msg);

		// Parse structured logs (marked with __STRUCTURED_LOG__)
		const text = msg.text();
		if (text.startsWith('__STRUCTURED_LOG__')) {
			try {
				const jsonStr = text.replace('__STRUCTURED_LOG__ ', '');
				const entry = JSON.parse(jsonStr) as LogEntry;
				capture.structuredLogs.push(entry);
			} catch {
				// Ignore parse errors
			}
		}
	});

	return capture;
}

/**
 * Get all captured structured logs.
 */
export function getStructuredLogs(capture: LogCapture): LogEntry[] {
	return [...capture.structuredLogs];
}

/**
 * Filter logs by criteria.
 */
export function filterLogs(
	capture: LogCapture,
	criteria: {
		level?: LogEntry['level'];
		category?: string;
		message?: string | RegExp;
	}
): LogEntry[] {
	return capture.structuredLogs.filter((log) => {
		if (criteria.level && log.level !== criteria.level) return false;
		if (criteria.category && log.category !== criteria.category) return false;
		if (criteria.message) {
			if (criteria.message instanceof RegExp) {
				if (!criteria.message.test(log.message)) return false;
			} else if (!log.message.includes(criteria.message)) {
				return false;
			}
		}
		return true;
	});
}

/**
 * Wait for a specific log to appear.
 */
export async function waitForLog(
	page: Page,
	capture: LogCapture,
	criteria: {
		level?: LogEntry['level'];
		category?: string;
		message?: string | RegExp;
	},
	timeout = 5000
): Promise<LogEntry> {
	const start = Date.now();

	while (Date.now() - start < timeout) {
		const matches = filterLogs(capture, criteria);
		if (matches.length > 0) {
			return matches[matches.length - 1];
		}
		await page.waitForTimeout(100);
	}

	throw new Error(
		`Timed out waiting for log: ${JSON.stringify(criteria)}\n` +
			`Available logs: ${JSON.stringify(capture.structuredLogs.map((l) => `[${l.category}] ${l.message}`))}`
	);
}

/**
 * Assert that a log exists.
 */
export function expectLog(
	capture: LogCapture,
	criteria: {
		level?: LogEntry['level'];
		category?: string;
		message?: string | RegExp;
	}
): LogEntry {
	const matches = filterLogs(capture, criteria);
	if (matches.length === 0) {
		throw new Error(
			`Expected log not found: ${JSON.stringify(criteria)}\n` +
				`Available logs:\n${capture.structuredLogs.map((l) => `  [${l.level}] [${l.category}] ${l.message}`).join('\n')}`
		);
	}
	return matches[matches.length - 1];
}

/**
 * Print all captured logs (for debugging).
 */
export function printLogs(capture: LogCapture): void {
	console.log('\n=== Captured Logs ===');
	for (const log of capture.structuredLogs) {
		const data = log.data ? ` ${JSON.stringify(log.data)}` : '';
		console.log(`[${log.timestamp}] [${log.level.toUpperCase()}] [${log.category}] ${log.message}${data}`);
	}
	console.log('=== End Logs ===\n');
}
