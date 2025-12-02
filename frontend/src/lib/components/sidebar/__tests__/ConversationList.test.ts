/**
 * Unit tests for ConversationList component
 */
import { fireEvent, render, screen } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import ConversationList from '../ConversationList.svelte';
import { createConversation } from '$lib/stores/types';

describe('ConversationList', () => {
	let mockOnSelect: ReturnType<typeof vi.fn>;
	let mockOnNew: ReturnType<typeof vi.fn>;
	let mockOnDelete: ReturnType<typeof vi.fn>;

	// Fix "now" to a specific date for predictable tests
	const NOW = new Date('2024-06-15T14:00:00Z').getTime();

	beforeEach(() => {
		mockOnSelect = vi.fn();
		mockOnNew = vi.fn();
		mockOnDelete = vi.fn();
		vi.useFakeTimers();
		vi.setSystemTime(NOW);
	});

	afterEach(() => {
		vi.useRealTimers();
	});

	it('should render "New Chat" button', () => {
		render(ConversationList, {
			props: {
				conversations: [],
				activeId: null,
				onselect: mockOnSelect,
				onnew: mockOnNew,
				ondelete: mockOnDelete
			}
		});

		expect(screen.getByText('New Chat')).toBeInTheDocument();
	});

	it('should call onnew when "New Chat" button clicked', async () => {
		render(ConversationList, {
			props: {
				conversations: [],
				activeId: null,
				onselect: mockOnSelect,
				onnew: mockOnNew,
				ondelete: mockOnDelete
			}
		});

		await fireEvent.click(screen.getByText('New Chat'));
		expect(mockOnNew).toHaveBeenCalledTimes(1);
	});

	it('should render empty state when no conversations', () => {
		render(ConversationList, {
			props: {
				conversations: [],
				activeId: null,
				onselect: mockOnSelect,
				onnew: mockOnNew,
				ondelete: mockOnDelete
			}
		});

		expect(screen.getByText('No conversations yet')).toBeInTheDocument();
	});

	it('should render all conversations', () => {
		const conversations = [
			createConversation({ title: 'Chat 1', updatedAt: NOW - 1000 }),
			createConversation({ title: 'Chat 2', updatedAt: NOW - 2000 })
		];

		render(ConversationList, {
			props: {
				conversations,
				activeId: null,
				onselect: mockOnSelect,
				onnew: mockOnNew,
				ondelete: mockOnDelete
			}
		});

		expect(screen.getByText('Chat 1')).toBeInTheDocument();
		expect(screen.getByText('Chat 2')).toBeInTheDocument();
	});

	it('should group conversations by date', () => {
		const conversations = [
			createConversation({ title: 'Today Chat', updatedAt: NOW - 1000 }),
			createConversation({
				title: 'Old Chat',
				updatedAt: NOW - 14 * 24 * 60 * 60 * 1000
			})
		];

		render(ConversationList, {
			props: {
				conversations,
				activeId: null,
				onselect: mockOnSelect,
				onnew: mockOnNew,
				ondelete: mockOnDelete
			}
		});

		// Should have date group headers
		expect(screen.getByText('Today')).toBeInTheDocument();
		expect(screen.getByText('Older')).toBeInTheDocument();
	});

	it('should call onselect with conversation id when clicked', async () => {
		const conversation = createConversation({
			id: 'test-123',
			title: 'Click Me',
			updatedAt: NOW
		});

		render(ConversationList, {
			props: {
				conversations: [conversation],
				activeId: null,
				onselect: mockOnSelect,
				onnew: mockOnNew,
				ondelete: mockOnDelete
			}
		});

		await fireEvent.click(screen.getByText('Click Me'));
		expect(mockOnSelect).toHaveBeenCalledWith('test-123');
	});

	it('should call ondelete with conversation id when delete clicked', async () => {
		const conversation = createConversation({
			id: 'delete-123',
			title: 'Delete Me',
			updatedAt: NOW
		});

		const { container } = render(ConversationList, {
			props: {
				conversations: [conversation],
				activeId: null,
				onselect: mockOnSelect,
				onnew: mockOnNew,
				ondelete: mockOnDelete
			}
		});

		const deleteButton = container.querySelector('[data-testid="delete-button"]');
		await fireEvent.click(deleteButton!);
		expect(mockOnDelete).toHaveBeenCalledWith('delete-123');
	});

	it('should pass isActive to conversation items', () => {
		const conversations = [
			createConversation({ id: 'active-id', title: 'Active', updatedAt: NOW }),
			createConversation({ id: 'other-id', title: 'Other', updatedAt: NOW - 1000 })
		];

		render(ConversationList, {
			props: {
				conversations,
				activeId: 'active-id',
				onselect: mockOnSelect,
				onnew: mockOnNew,
				ondelete: mockOnDelete
			}
		});

		const activeButton = screen.getByText('Active').closest('button');
		const otherButton = screen.getByText('Other').closest('button');

		// Active should have the bg-sidebar-accent class
		expect(activeButton?.className).toMatch(/\bbg-sidebar-accent\b(?!\/)/);
		// Other should not
		expect(otherButton?.className).not.toMatch(/\bbg-sidebar-accent\s/);
	});
});
