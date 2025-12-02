/**
 * Unit tests for UserMessage component
 *
 * Tests cover:
 * - Content rendering
 * - Right alignment styling
 * - Whitespace preservation
 */

import { render, screen } from '@testing-library/svelte';
import { describe, expect, it } from 'vitest';
import type { Message } from '$lib/stores/types';
import UserMessage from '../UserMessage.svelte';

function createTestMessage(content: string, id?: string): Message {
	return {
		id: id || crypto.randomUUID(),
		role: 'user',
		content,
		createdAt: Date.now()
	};
}

describe('UserMessage', () => {
	// ============================================================================
	// Content Rendering
	// ============================================================================

	describe('content rendering', () => {
		it('should render message content', () => {
			const message = createTestMessage('Hello, world!');

			render(UserMessage, { props: { message } });

			expect(screen.getByText('Hello, world!')).toBeInTheDocument();
		});

		it('should render long content', () => {
			const longContent = 'This is a very long message '.repeat(20).trim();
			const message = createTestMessage(longContent);

			const { container } = render(UserMessage, { props: { message } });

			const paragraph = container.querySelector('p');
			expect(paragraph?.textContent).toBe(longContent);
		});

		it('should render special characters', () => {
			const message = createTestMessage('Hello <script>alert("xss")</script>');

			render(UserMessage, { props: { message } });

			expect(screen.getByText('Hello <script>alert("xss")</script>')).toBeInTheDocument();
		});

		it('should render unicode characters', () => {
			const message = createTestMessage('Hello 👋 世界 🌍');

			render(UserMessage, { props: { message } });

			expect(screen.getByText('Hello 👋 世界 🌍')).toBeInTheDocument();
		});
	});

	// ============================================================================
	// Whitespace Preservation
	// ============================================================================

	describe('whitespace preservation', () => {
		it('should preserve line breaks', () => {
			const message = createTestMessage('Line 1\nLine 2\nLine 3');

			const { container } = render(UserMessage, { props: { message } });

			const paragraph = container.querySelector('p');
			expect(paragraph).toHaveClass('whitespace-pre-wrap');
			expect(paragraph?.textContent).toBe('Line 1\nLine 2\nLine 3');
		});

		it('should preserve multiple spaces', () => {
			const message = createTestMessage('Word    with    spaces');

			const { container } = render(UserMessage, { props: { message } });

			const paragraph = container.querySelector('p');
			expect(paragraph?.textContent).toBe('Word    with    spaces');
		});
	});

	// ============================================================================
	// Styling
	// ============================================================================

	describe('styling', () => {
		it('should have right alignment', () => {
			const message = createTestMessage('Test message');

			const { container } = render(UserMessage, { props: { message } });

			const flexContainer = container.querySelector('.justify-end');
			expect(flexContainer).toBeInTheDocument();
		});

		it('should have primary background styling', () => {
			const message = createTestMessage('Test message');

			const { container } = render(UserMessage, { props: { message } });

			const messageBox = container.querySelector('.bg-primary');
			expect(messageBox).toBeInTheDocument();
		});

		it('should have max-width constraint', () => {
			const message = createTestMessage('Test message');

			const { container } = render(UserMessage, { props: { message } });

			const messageBox = container.querySelector('.max-w-\\[80\\%\\]');
			expect(messageBox).toBeInTheDocument();
		});

		it('should have rounded corners', () => {
			const message = createTestMessage('Test message');

			const { container } = render(UserMessage, { props: { message } });

			const messageBox = container.querySelector('.rounded-lg');
			expect(messageBox).toBeInTheDocument();
		});
	});
});
