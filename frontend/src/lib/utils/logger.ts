/**
 * Structured Logger
 *
 * Provides structured JSON logging visible in browser console for Playwright debugging.
 * All logs include timestamps, categories, and structured data.
 *
 * Usage:
 *   import { logger } from '$lib/utils/logger';
 *   logger.info('store', 'Conversation created', { id: conv.id, title: conv.title });
 *   logger.debug('api', 'Request sent', { endpoint: '/chat', method: 'POST' });
 */

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';
export type LogCategory =
	| 'store'
	| 'api'
	| 'navigation'
	| 'ui'
	| 'streaming'
	| 'persistence'
	| 'lifecycle';

interface LogEntry {
	timestamp: string;
	level: LogLevel;
	category: LogCategory;
	message: string;
	data?: Record<string, unknown>;
	trace?: string[];
}

interface LoggerConfig {
	enabled: boolean;
	minLevel: LogLevel;
	includeTrace: boolean;
}

const LOG_LEVELS: Record<LogLevel, number> = {
	debug: 0,
	info: 1,
	warn: 2,
	error: 3
};

class Logger {
	private config: LoggerConfig = {
		enabled: true,
		minLevel: 'debug',
		includeTrace: false
	};

	// Store recent logs for inspection
	private recentLogs: LogEntry[] = [];
	private maxRecentLogs = 100;

	/**
	 * Configure the logger
	 */
	configure(config: Partial<LoggerConfig>): void {
		this.config = { ...this.config, ...config };
	}

	/**
	 * Get recent logs (useful for debugging in Playwright)
	 */
	getRecentLogs(): LogEntry[] {
		return [...this.recentLogs];
	}

	/**
	 * Clear recent logs
	 */
	clearLogs(): void {
		this.recentLogs = [];
	}

	/**
	 * Core logging method
	 */
	private log(
		level: LogLevel,
		category: LogCategory,
		message: string,
		data?: Record<string, unknown>
	): void {
		if (!this.config.enabled) return;
		if (LOG_LEVELS[level] < LOG_LEVELS[this.config.minLevel]) return;

		const entry: LogEntry = {
			timestamp: new Date().toISOString(),
			level,
			category,
			message,
			data
		};

		if (this.config.includeTrace) {
			entry.trace = this.getStackTrace();
		}

		// Store in recent logs
		this.recentLogs.push(entry);
		if (this.recentLogs.length > this.maxRecentLogs) {
			this.recentLogs.shift();
		}

		// Output to console with structured format
		const prefix = `[${entry.timestamp}] [${level.toUpperCase()}] [${category}]`;
		const consoleMethod = level === 'error' ? 'error' : level === 'warn' ? 'warn' : 'log';

		// Log as both formatted string AND JSON for easy Playwright capture
		console[consoleMethod](`${prefix} ${message}`, data ? JSON.stringify(data) : '');

		// Also log pure JSON for programmatic parsing
		console[consoleMethod]('__STRUCTURED_LOG__', JSON.stringify(entry));
	}

	private getStackTrace(): string[] {
		const stack = new Error().stack;
		if (!stack) return [];
		return stack
			.split('\n')
			.slice(4)
			.map((line) => line.trim())
			.filter((line) => !line.includes('logger.ts'));
	}

	// Convenience methods
	debug(category: LogCategory, message: string, data?: Record<string, unknown>): void {
		this.log('debug', category, message, data);
	}

	info(category: LogCategory, message: string, data?: Record<string, unknown>): void {
		this.log('info', category, message, data);
	}

	warn(category: LogCategory, message: string, data?: Record<string, unknown>): void {
		this.log('warn', category, message, data);
	}

	error(category: LogCategory, message: string, data?: Record<string, unknown>): void {
		this.log('error', category, message, data);
	}

	// Category-specific helpers for common patterns
	store = {
		action: (action: string, data?: Record<string, unknown>) =>
			this.info('store', `Action: ${action}`, data),
		stateChange: (field: string, oldValue: unknown, newValue: unknown) =>
			this.debug('store', `State change: ${field}`, { oldValue, newValue })
	};

	api = {
		request: (method: string, url: string, data?: Record<string, unknown>) =>
			this.info('api', `Request: ${method} ${url}`, data),
		response: (method: string, url: string, status: number, data?: Record<string, unknown>) =>
			this.info('api', `Response: ${method} ${url} [${status}]`, data),
		streamChunk: (conversationId: string, chunkSize: number) =>
			this.debug('streaming', 'Stream chunk received', { conversationId, chunkSize }),
		streamComplete: (conversationId: string, totalLength: number) =>
			this.info('streaming', 'Stream complete', { conversationId, totalLength })
	};

	nav = {
		navigate: (from: string, to: string, data?: Record<string, unknown>) =>
			this.info('navigation', `Navigate: ${from} -> ${to}`, data),
		beforeNavigate: (from: string, to: string) =>
			this.debug('navigation', `Before navigate: ${from} -> ${to}`)
	};

	ui = {
		event: (component: string, event: string, data?: Record<string, unknown>) =>
			this.debug('ui', `${component}: ${event}`, data),
		render: (component: string, data?: Record<string, unknown>) =>
			this.debug('ui', `Render: ${component}`, data)
	};

	lifecycle = {
		mount: (component: string, data?: Record<string, unknown>) =>
			this.debug('lifecycle', `Mount: ${component}`, data),
		effect: (component: string, effect: string, data?: Record<string, unknown>) =>
			this.debug('lifecycle', `Effect: ${component}.${effect}`, data)
	};
}

// Singleton instance
export const logger = new Logger();

// Expose to window for Playwright access
if (typeof window !== 'undefined') {
	(window as unknown as { __strieber_logger__: Logger }).__strieber_logger__ = logger;
}
