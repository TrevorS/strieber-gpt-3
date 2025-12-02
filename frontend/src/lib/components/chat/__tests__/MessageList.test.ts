/**
 * Unit tests for MessageList component
 *
 * Tests cover:
 * - Empty state rendering
 * - User message rendering
 * - Assistant message rendering
 * - Mixed messages in order
 * - Auto-scroll behavior
 */

import { render, screen } from '@testing-library/svelte';
import { describe, expect, it } from 'vitest';
import type { Message } from '$lib/stores/types';
import MessageList from '../MessageList.svelte';

// Helper to create test messages
function createTestMessage(role: 'user' | 'assistant', content: string, id?: string): Message {
	return {
		id: id || crypto.randomUUID(),
		role,
		content,
		createdAt: Date.now()
	};
}

describe('MessageList', () => {
	// ============================================================================
	// Empty State
	// ============================================================================

	describe('empty state', () => {
		it('should render empty container when no messages', () => {
			const { container } = render(MessageList, { props: { messages: [] } });

			// Container should exist but have no message children
			const messageContainer = container.querySelector('.space-y-4');
			expect(messageContainer).toBeInTheDocument();
			expect(messageContainer?.children.length).toBe(0);
		});
	});

	// ============================================================================
	// User Messages
	// ============================================================================

	describe('user messages', () => {
		it('should render user message content', () => {
			const messages: Message[] = [createTestMessage('user', 'Hello, world!')];

			render(MessageList, { props: { messages } });

			expect(screen.getByText('Hello, world!')).toBeInTheDocument();
		});

		it('should render user message with right alignment', () => {
			const messages: Message[] = [createTestMessage('user', 'Test message')];

			const { container } = render(MessageList, { props: { messages } });

			const userMessageContainer = container.querySelector('.justify-end');
			expect(userMessageContainer).toBeInTheDocument();
		});
	});

	// ============================================================================
	// Assistant Messages
	// ============================================================================

	describe('assistant messages', () => {
		it('should render assistant message content', () => {
			const messages: Message[] = [createTestMessage('assistant', 'I am an AI assistant.')];

			render(MessageList, { props: { messages } });

			expect(screen.getByText('I am an AI assistant.')).toBeInTheDocument();
		});

		it('should render assistant message with left alignment', () => {
			const messages: Message[] = [createTestMessage('assistant', 'Test response')];

			const { container } = render(MessageList, { props: { messages } });

			const assistantMessageContainer = container.querySelector('.justify-start');
			expect(assistantMessageContainer).toBeInTheDocument();
		});
	});

	// ============================================================================
	// Mixed Messages
	// ============================================================================

	describe('mixed messages', () => {
		it('should render messages in order', () => {
			const messages: Message[] = [
				createTestMessage('user', 'First message', 'msg-1'),
				createTestMessage('assistant', 'Second message', 'msg-2'),
				createTestMessage('user', 'Third message', 'msg-3')
			];

			render(MessageList, { props: { messages } });

			const firstMsg = screen.getByText('First message');
			const secondMsg = screen.getByText('Second message');
			const thirdMsg = screen.getByText('Third message');

			expect(firstMsg).toBeInTheDocument();
			expect(secondMsg).toBeInTheDocument();
			expect(thirdMsg).toBeInTheDocument();

			// Verify order by checking DOM position
			const allText = document.body.textContent;
			const firstIndex = allText?.indexOf('First message') ?? -1;
			const secondIndex = allText?.indexOf('Second message') ?? -1;
			const thirdIndex = allText?.indexOf('Third message') ?? -1;

			expect(firstIndex).toBeLessThan(secondIndex);
			expect(secondIndex).toBeLessThan(thirdIndex);
		});

		it('should render multiple messages with correct styling', () => {
			const messages: Message[] = [
				createTestMessage('user', 'User says hi'),
				createTestMessage('assistant', 'Assistant responds')
			];

			const { container } = render(MessageList, { props: { messages } });

			// Should have both alignment styles
			expect(container.querySelector('.justify-end')).toBeInTheDocument();
			expect(container.querySelector('.justify-start')).toBeInTheDocument();
		});
	});

	// ============================================================================
	// Scrollable Container
	// ============================================================================

	describe('scrollable container', () => {
		it('should have overflow-y-auto class', () => {
			const { container } = render(MessageList, { props: { messages: [] } });

			const scrollContainer = container.querySelector('.overflow-y-auto');
			expect(scrollContainer).toBeInTheDocument();
		});
	});
});
