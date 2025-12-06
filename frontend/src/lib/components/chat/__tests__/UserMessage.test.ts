/**
 * Unit tests for UserMessage component
 *
 * Tests cover:
 * - Content rendering
 * - Right alignment styling
 * - Whitespace preservation
 * - Edit functionality
 */

import { render, screen, fireEvent } from '@testing-library/svelte';
import { describe, expect, it, vi } from 'vitest';
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

	// ============================================================================
	// Edit Functionality
	// ============================================================================

	describe('edit functionality', () => {
		it('should not show edit button when editable is false', () => {
			const message = createTestMessage('Test message');

			render(UserMessage, { props: { message, editable: false } });

			expect(screen.queryByTestId('edit-button')).not.toBeInTheDocument();
		});

		it('should not show edit button when editable is not provided', () => {
			const message = createTestMessage('Test message');

			render(UserMessage, { props: { message } });

			expect(screen.queryByTestId('edit-button')).not.toBeInTheDocument();
		});

		it('should show edit button when editable is true', () => {
			const message = createTestMessage('Test message');

			render(UserMessage, { props: { message, editable: true } });

			expect(screen.getByTestId('edit-button')).toBeInTheDocument();
		});

		it('should enter edit mode when edit button clicked', async () => {
			const message = createTestMessage('Original content');

			render(UserMessage, { props: { message, editable: true } });

			const editButton = screen.getByTestId('edit-button');
			await fireEvent.click(editButton);

			const textarea = screen.getByTestId('edit-textarea');
			expect(textarea).toBeInTheDocument();
			expect(textarea).toHaveValue('Original content');
		});

		it('should call onedit with new content on save', async () => {
			const message = createTestMessage('Original content');
			const onedit = vi.fn();

			render(UserMessage, { props: { message, editable: true, onedit } });

			// Enter edit mode
			await fireEvent.click(screen.getByTestId('edit-button'));

			// Change content
			const textarea = screen.getByTestId('edit-textarea');
			await fireEvent.input(textarea, { target: { value: 'Updated content' } });

			// Save
			await fireEvent.click(screen.getByTestId('save-button'));

			expect(onedit).toHaveBeenCalledWith('Updated content');
		});

		it('should cancel edit on Escape key', async () => {
			const message = createTestMessage('Original content');
			const onedit = vi.fn();

			render(UserMessage, { props: { message, editable: true, onedit } });

			// Enter edit mode
			await fireEvent.click(screen.getByTestId('edit-button'));

			// Change content
			const textarea = screen.getByTestId('edit-textarea');
			await fireEvent.input(textarea, { target: { value: 'Changed content' } });

			// Press Escape
			await fireEvent.keyDown(textarea, { key: 'Escape' });

			// Should exit edit mode without calling onedit
			expect(onedit).not.toHaveBeenCalled();
			expect(screen.queryByTestId('edit-textarea')).not.toBeInTheDocument();
		});

		it('should cancel edit on cancel button click', async () => {
			const message = createTestMessage('Original content');
			const onedit = vi.fn();

			render(UserMessage, { props: { message, editable: true, onedit } });

			// Enter edit mode
			await fireEvent.click(screen.getByTestId('edit-button'));

			// Click cancel
			await fireEvent.click(screen.getByTestId('cancel-button'));

			// Should exit edit mode without calling onedit
			expect(onedit).not.toHaveBeenCalled();
			expect(screen.queryByTestId('edit-textarea')).not.toBeInTheDocument();
		});

		it('should show edited indicator when message.isEdited is true', () => {
			const message: Message = {
				...createTestMessage('Test message'),
				isEdited: true
			};

			render(UserMessage, { props: { message } });

			expect(screen.getByText('(edited)')).toBeInTheDocument();
		});

		it('should not show edited indicator when message.isEdited is false', () => {
			const message = createTestMessage('Test message');

			render(UserMessage, { props: { message } });

			expect(screen.queryByText('(edited)')).not.toBeInTheDocument();
		});
	});
});
