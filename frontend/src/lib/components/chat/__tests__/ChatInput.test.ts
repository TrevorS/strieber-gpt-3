/**
 * Unit tests for ChatInput component
 *
 * Tests cover:
 * - Rendering with placeholder
 * - Typing updates value
 * - Enter key submits
 * - Shift+Enter doesn't submit
 * - Button disabled states
 * - Input clearing after submit
 */

import { fireEvent, render, screen } from '@testing-library/svelte';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import ChatInput from '../ChatInput.svelte';

describe('ChatInput', () => {
	let mockOnSubmit: ReturnType<typeof vi.fn>;

	beforeEach(() => {
		mockOnSubmit = vi.fn();
	});

	// ============================================================================
	// Rendering
	// ============================================================================

	describe('rendering', () => {
		it('should render with placeholder text', () => {
			render(ChatInput, { props: { onsubmit: mockOnSubmit } });

			const textarea = screen.getByPlaceholderText('Send a message...');
			expect(textarea).toBeInTheDocument();
		});

		it('should render send button', () => {
			render(ChatInput, { props: { onsubmit: mockOnSubmit } });

			const button = screen.getByRole('button');
			expect(button).toBeInTheDocument();
		});
	});

	// ============================================================================
	// Input Behavior
	// ============================================================================

	describe('input behavior', () => {
		it('should update value when typing', async () => {
			render(ChatInput, { props: { onsubmit: mockOnSubmit } });

			const textarea = screen.getByPlaceholderText('Send a message...') as HTMLTextAreaElement;
			await fireEvent.input(textarea, { target: { value: 'Hello world' } });

			expect(textarea.value).toBe('Hello world');
		});
	});

	// ============================================================================
	// Submit Behavior
	// ============================================================================

	describe('submit behavior', () => {
		it('should call onsubmit with trimmed text when Enter is pressed', async () => {
			render(ChatInput, { props: { onsubmit: mockOnSubmit } });

			const textarea = screen.getByPlaceholderText('Send a message...');
			await fireEvent.input(textarea, { target: { value: '  Hello world  ' } });
			await fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: false });

			expect(mockOnSubmit).toHaveBeenCalledWith('Hello world');
		});

		it('should not submit when Shift+Enter is pressed', async () => {
			render(ChatInput, { props: { onsubmit: mockOnSubmit } });

			const textarea = screen.getByPlaceholderText('Send a message...');
			await fireEvent.input(textarea, { target: { value: 'Hello' } });
			await fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: true });

			expect(mockOnSubmit).not.toHaveBeenCalled();
		});

		it('should call onsubmit when button is clicked', async () => {
			render(ChatInput, { props: { onsubmit: mockOnSubmit } });

			const textarea = screen.getByPlaceholderText('Send a message...');
			await fireEvent.input(textarea, { target: { value: 'Hello' } });

			const button = screen.getByRole('button');
			await fireEvent.click(button);

			expect(mockOnSubmit).toHaveBeenCalledWith('Hello');
		});

		it('should clear input after successful submit', async () => {
			render(ChatInput, { props: { onsubmit: mockOnSubmit } });

			const textarea = screen.getByPlaceholderText('Send a message...') as HTMLTextAreaElement;
			await fireEvent.input(textarea, { target: { value: 'Hello' } });
			await fireEvent.keyDown(textarea, { key: 'Enter' });

			expect(textarea.value).toBe('');
		});

		it('should not submit when input is empty', async () => {
			render(ChatInput, { props: { onsubmit: mockOnSubmit } });

			const textarea = screen.getByPlaceholderText('Send a message...');
			await fireEvent.keyDown(textarea, { key: 'Enter' });

			expect(mockOnSubmit).not.toHaveBeenCalled();
		});

		it('should not submit when input is only whitespace', async () => {
			render(ChatInput, { props: { onsubmit: mockOnSubmit } });

			const textarea = screen.getByPlaceholderText('Send a message...');
			await fireEvent.input(textarea, { target: { value: '   ' } });
			await fireEvent.keyDown(textarea, { key: 'Enter' });

			expect(mockOnSubmit).not.toHaveBeenCalled();
		});
	});

	// ============================================================================
	// Disabled State
	// ============================================================================

	describe('disabled state', () => {
		it('should disable textarea when disabled prop is true', () => {
			render(ChatInput, { props: { onsubmit: mockOnSubmit, disabled: true } });

			const textarea = screen.getByPlaceholderText('Send a message...');
			expect(textarea).toBeDisabled();
		});

		it('should disable button when disabled prop is true', () => {
			render(ChatInput, { props: { onsubmit: mockOnSubmit, disabled: true } });

			const button = screen.getByRole('button');
			expect(button).toBeDisabled();
		});

		it('should disable button when input is empty', () => {
			render(ChatInput, { props: { onsubmit: mockOnSubmit } });

			const button = screen.getByRole('button');
			expect(button).toBeDisabled();
		});

		it('should enable button when input has text', async () => {
			render(ChatInput, { props: { onsubmit: mockOnSubmit } });

			const textarea = screen.getByPlaceholderText('Send a message...');
			await fireEvent.input(textarea, { target: { value: 'Hello' } });

			const button = screen.getByRole('button');
			expect(button).not.toBeDisabled();
		});

		it('should not submit when disabled even with text', async () => {
			render(ChatInput, { props: { onsubmit: mockOnSubmit, disabled: true } });

			const textarea = screen.getByPlaceholderText('Send a message...');
			await fireEvent.input(textarea, { target: { value: 'Hello' } });
			await fireEvent.keyDown(textarea, { key: 'Enter' });

			expect(mockOnSubmit).not.toHaveBeenCalled();
		});
	});
});
