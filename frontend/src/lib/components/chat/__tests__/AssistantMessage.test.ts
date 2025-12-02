/**
 * Unit tests for AssistantMessage component
 *
 * Tests cover:
 * - Content rendering with markdown
 * - Left alignment styling
 * - Code block rendering
 */

import { render, screen } from '@testing-library/svelte';
import { describe, expect, it } from 'vitest';
import type { Message } from '$lib/stores/types';
import AssistantMessage from '../AssistantMessage.svelte';

function createTestMessage(content: string, id?: string): Message {
	return {
		id: id || crypto.randomUUID(),
		role: 'assistant',
		content,
		createdAt: Date.now()
	};
}

describe('AssistantMessage', () => {
	// ============================================================================
	// Content Rendering
	// ============================================================================

	describe('content rendering', () => {
		it('should render plain text content', () => {
			const message = createTestMessage('Hello, world!');

			render(AssistantMessage, { props: { message } });

			expect(screen.getByText('Hello, world!')).toBeInTheDocument();
		});

		it('should render markdown headers', () => {
			const message = createTestMessage('# Main Header');

			const { container } = render(AssistantMessage, { props: { message } });

			const header = container.querySelector('h1');
			expect(header).toBeInTheDocument();
			expect(header?.textContent).toContain('Main Header');
		});

		it('should render markdown bold text', () => {
			const message = createTestMessage('This is **bold** text');

			const { container } = render(AssistantMessage, { props: { message } });

			const strong = container.querySelector('strong');
			expect(strong).toBeInTheDocument();
			expect(strong?.textContent).toBe('bold');
		});

		it('should render markdown lists', () => {
			const message = createTestMessage('- Item 1\n- Item 2\n- Item 3');

			const { container } = render(AssistantMessage, { props: { message } });

			const list = container.querySelector('ul');
			expect(list).toBeInTheDocument();
			const items = container.querySelectorAll('li');
			expect(items).toHaveLength(3);
		});

		it('should render inline code', () => {
			const message = createTestMessage('Use `console.log()` for debugging');

			const { container } = render(AssistantMessage, { props: { message } });

			const code = container.querySelector('code');
			expect(code).toBeInTheDocument();
			expect(code?.textContent).toBe('console.log()');
		});
	});

	// ============================================================================
	// Code Blocks
	// ============================================================================

	describe('code blocks', () => {
		it('should render fenced code blocks', () => {
			const message = createTestMessage('```\nconst x = 1;\n```');

			const { container } = render(AssistantMessage, { props: { message } });

			const pre = container.querySelector('pre');
			expect(pre).toBeInTheDocument();
			const code = container.querySelector('code');
			expect(code).toBeInTheDocument();
		});

		it('should render code blocks with language class', () => {
			const message = createTestMessage('```javascript\nconst x = 1;\n```');

			const { container } = render(AssistantMessage, { props: { message } });

			const code = container.querySelector('code');
			expect(code?.className).toContain('language-javascript');
		});

		it('should syntax highlight code', () => {
			const message = createTestMessage('```typescript\nconst name: string = "test";\n```');

			const { container } = render(AssistantMessage, { props: { message } });

			// highlight.js adds spans with hljs classes
			const highlightedSpans = container.querySelectorAll(
				'.hljs-keyword, .hljs-string, .hljs-attr'
			);
			expect(highlightedSpans.length).toBeGreaterThan(0);
		});
	});

	// ============================================================================
	// Styling
	// ============================================================================

	describe('styling', () => {
		it('should have left alignment', () => {
			const message = createTestMessage('Test message');

			const { container } = render(AssistantMessage, { props: { message } });

			const flexContainer = container.querySelector('.justify-start');
			expect(flexContainer).toBeInTheDocument();
		});

		it('should have muted background styling', () => {
			const message = createTestMessage('Test message');

			const { container } = render(AssistantMessage, { props: { message } });

			const messageBox = container.querySelector('.bg-muted');
			expect(messageBox).toBeInTheDocument();
		});

		it('should have max-width constraint', () => {
			const message = createTestMessage('Test message');

			const { container } = render(AssistantMessage, { props: { message } });

			const messageBox = container.querySelector('.max-w-\\[80\\%\\]');
			expect(messageBox).toBeInTheDocument();
		});

		it('should have prose styling for markdown', () => {
			const message = createTestMessage('# Header');

			const { container } = render(AssistantMessage, { props: { message } });

			const prose = container.querySelector('.prose');
			expect(prose).toBeInTheDocument();
		});
	});
});
