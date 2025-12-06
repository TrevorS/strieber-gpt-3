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
	let mockOnRename: (newTitle: string) => void;

	beforeEach(() => {
		mockOnSelect = vi.fn();
		mockOnDelete = vi.fn();
		mockOnRename = vi.fn();
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

	describe('rename functionality', () => {
		it('should show edit button on hover', () => {
			const conversation = createConversation({ title: 'Rename Me' });
			const { container } = render(ConversationItem, {
				props: {
					conversation,
					isActive: false,
					onselect: mockOnSelect,
					ondelete: mockOnDelete,
					onrename: mockOnRename
				}
			});

			const editButton = container.querySelector('[data-testid="edit-button"]');
			expect(editButton).toBeTruthy();
		});

		it('should enter edit mode when edit button clicked', async () => {
			const conversation = createConversation({ title: 'Rename Me' });
			const { container } = render(ConversationItem, {
				props: {
					conversation,
					isActive: false,
					onselect: mockOnSelect,
					ondelete: mockOnDelete,
					onrename: mockOnRename
				}
			});

			const editButton = container.querySelector('[data-testid="edit-button"]');
			await fireEvent.click(editButton!);

			const input = container.querySelector('input[data-testid="rename-input"]');
			expect(input).toBeTruthy();
			expect((input as HTMLInputElement).value).toBe('Rename Me');
		});

		it('should enter edit mode on double-click', async () => {
			const conversation = createConversation({ title: 'Double Click Me' });
			const { container } = render(ConversationItem, {
				props: {
					conversation,
					isActive: false,
					onselect: mockOnSelect,
					ondelete: mockOnDelete,
					onrename: mockOnRename
				}
			});

			const titleSpan = screen.getByText('Double Click Me');
			await fireEvent.dblClick(titleSpan);

			const input = container.querySelector('input[data-testid="rename-input"]');
			expect(input).toBeTruthy();
		});

		it('should save on Enter key and call onrename', async () => {
			const conversation = createConversation({ title: 'Original Title' });
			const { container } = render(ConversationItem, {
				props: {
					conversation,
					isActive: false,
					onselect: mockOnSelect,
					ondelete: mockOnDelete,
					onrename: mockOnRename
				}
			});

			// Enter edit mode
			const editButton = container.querySelector('[data-testid="edit-button"]');
			await fireEvent.click(editButton!);

			// Change the input value
			const input = container.querySelector(
				'input[data-testid="rename-input"]'
			) as HTMLInputElement;
			await fireEvent.input(input, { target: { value: 'New Title' } });

			// Press Enter
			await fireEvent.keyDown(input, { key: 'Enter' });

			expect(mockOnRename).toHaveBeenCalledWith('New Title');
		});

		it('should cancel on Escape key and restore original title', async () => {
			const conversation = createConversation({ title: 'Original Title' });
			const { container } = render(ConversationItem, {
				props: {
					conversation,
					isActive: false,
					onselect: mockOnSelect,
					ondelete: mockOnDelete,
					onrename: mockOnRename
				}
			});

			// Enter edit mode
			const editButton = container.querySelector('[data-testid="edit-button"]');
			await fireEvent.click(editButton!);

			// Change the input value
			const input = container.querySelector(
				'input[data-testid="rename-input"]'
			) as HTMLInputElement;
			await fireEvent.input(input, { target: { value: 'Changed Title' } });

			// Press Escape
			await fireEvent.keyDown(input, { key: 'Escape' });

			// Should exit edit mode without calling onrename
			expect(mockOnRename).not.toHaveBeenCalled();
			expect(screen.getByText('Original Title')).toBeInTheDocument();
		});

		it('should not call onrename if title is unchanged', async () => {
			const conversation = createConversation({ title: 'Same Title' });
			const { container } = render(ConversationItem, {
				props: {
					conversation,
					isActive: false,
					onselect: mockOnSelect,
					ondelete: mockOnDelete,
					onrename: mockOnRename
				}
			});

			// Enter edit mode
			const editButton = container.querySelector('[data-testid="edit-button"]');
			await fireEvent.click(editButton!);

			// Don't change the value, just press Enter
			const input = container.querySelector(
				'input[data-testid="rename-input"]'
			) as HTMLInputElement;
			await fireEvent.keyDown(input, { key: 'Enter' });

			expect(mockOnRename).not.toHaveBeenCalled();
		});

		it('should not call onrename if title is empty', async () => {
			const conversation = createConversation({ title: 'Original Title' });
			const { container } = render(ConversationItem, {
				props: {
					conversation,
					isActive: false,
					onselect: mockOnSelect,
					ondelete: mockOnDelete,
					onrename: mockOnRename
				}
			});

			// Enter edit mode
			const editButton = container.querySelector('[data-testid="edit-button"]');
			await fireEvent.click(editButton!);

			// Clear the input
			const input = container.querySelector(
				'input[data-testid="rename-input"]'
			) as HTMLInputElement;
			await fireEvent.input(input, { target: { value: '' } });
			await fireEvent.keyDown(input, { key: 'Enter' });

			expect(mockOnRename).not.toHaveBeenCalled();
		});

		it('should not prevent selection when onrename is not provided', async () => {
			const conversation = createConversation({ title: 'No Rename' });
			const { container } = render(ConversationItem, {
				props: {
					conversation,
					isActive: false,
					onselect: mockOnSelect,
					ondelete: mockOnDelete
					// onrename not provided
				}
			});

			// Edit button should not be rendered
			const editButton = container.querySelector('[data-testid="edit-button"]');
			expect(editButton).toBeNull();
		});
	});
});
