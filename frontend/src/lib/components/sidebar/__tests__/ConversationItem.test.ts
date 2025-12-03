/**
 * Unit tests for ConversationItem component
 */
import { fireEvent, render, screen } from '@testing-library/svelte';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import ConversationItem from '../ConversationItem.svelte';
import { createConversation } from '$lib/stores/types';

describe('ConversationItem', () => {
	let mockOnSelect: () => void;
	let mockOnDelete: () => void;

	beforeEach(() => {
		mockOnSelect = vi.fn();
		mockOnDelete = vi.fn();
	});

	it('should render conversation title', () => {
		const conversation = createConversation({ title: 'Test Chat' });
		render(ConversationItem, {
			props: {
				conversation,
				isActive: false,
				onselect: mockOnSelect,
				ondelete: mockOnDelete
			}
		});

		expect(screen.getByText('Test Chat')).toBeInTheDocument();
	});

	it('should truncate long titles', () => {
		const conversation = createConversation({
			title: 'This is a very long conversation title that should be truncated'
		});
		render(ConversationItem, {
			props: {
				conversation,
				isActive: false,
				onselect: mockOnSelect,
				ondelete: mockOnDelete
			}
		});

		const titleElement = screen.getByText(conversation.title);
		expect(titleElement.classList.contains('truncate')).toBe(true);
	});

	it('should highlight when active', () => {
		const conversation = createConversation({ title: 'Active Chat' });
		render(ConversationItem, {
			props: {
				conversation,
				isActive: true,
				onselect: mockOnSelect,
				ondelete: mockOnDelete
			}
		});

		const button = screen.getByText('Active Chat').closest('button');
		// Check for the active class (not the hover variant)
		expect(button?.className).toMatch(/\bbg-sidebar-accent\b(?!\/)/);
	});

	it('should not highlight when not active', () => {
		const conversation = createConversation({ title: 'Inactive Chat' });
		render(ConversationItem, {
			props: {
				conversation,
				isActive: false,
				onselect: mockOnSelect,
				ondelete: mockOnDelete
			}
		});

		const button = screen.getByText('Inactive Chat').closest('button');
		// Should have hover class but not the solid background class
		expect(button?.className).toContain('hover:bg-sidebar-accent/50');
		expect(button?.className).not.toMatch(/\bbg-sidebar-accent\s/);
	});

	it('should call onselect when clicked', async () => {
		const conversation = createConversation({ title: 'Click Me' });
		render(ConversationItem, {
			props: {
				conversation,
				isActive: false,
				onselect: mockOnSelect,
				ondelete: mockOnDelete
			}
		});

		await fireEvent.click(screen.getByText('Click Me'));
		expect(mockOnSelect).toHaveBeenCalledTimes(1);
	});

	it('should call ondelete when delete button clicked', async () => {
		const conversation = createConversation({ title: 'Delete Me' });
		const { container } = render(ConversationItem, {
			props: {
				conversation,
				isActive: false,
				onselect: mockOnSelect,
				ondelete: mockOnDelete
			}
		});

		// Find and click the delete button (it's always rendered but visually hidden)
		const deleteButton = container.querySelector('[data-testid="delete-button"]');
		expect(deleteButton).toBeTruthy();
		await fireEvent.click(deleteButton!);

		expect(mockOnDelete).toHaveBeenCalledTimes(1);
		// Should NOT trigger onselect due to stopPropagation
		expect(mockOnSelect).not.toHaveBeenCalled();
	});

	it('should not call onselect when delete is clicked', async () => {
		const conversation = createConversation();
		const { container } = render(ConversationItem, {
			props: {
				conversation,
				isActive: false,
				onselect: mockOnSelect,
				ondelete: mockOnDelete
			}
		});

		const deleteButton = container.querySelector('[data-testid="delete-button"]');
		await fireEvent.click(deleteButton!);

		expect(mockOnDelete).toHaveBeenCalled();
		expect(mockOnSelect).not.toHaveBeenCalled();
	});
});
