/**
 * Unit tests for ConversationList component
 */
import { fireEvent, render, screen } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import ConversationList from '../ConversationList.svelte';
import type { Conversation } from '$lib/stores/types';

// Helper to create test conversations
function createTestConversation(overrides: Partial<Conversation> = {}): Conversation {
	const now = Date.now();
	return {
		id: `conv_test_${Math.random().toString(36).slice(2)}`,
		title: 'New Chat',
		createdAt: now,
		updatedAt: now,
		messages: [],
		...overrides
	};
}

describe('ConversationList', () => {
	let mockOnSelect: (id: string) => void;
	let mockOnNew: () => void;
	let mockOnDelete: (id: string) => void;

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
			createTestConversation({ title: 'Chat 1', updatedAt: NOW - 1000 }),
			createTestConversation({ title: 'Chat 2', updatedAt: NOW - 2000 })
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
			createTestConversation({ title: 'Today Chat', updatedAt: NOW - 1000 }),
			createTestConversation({
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
		const conversation = createTestConversation({
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
		const conversation = createTestConversation({
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
			createTestConversation({ id: 'active-id', title: 'Active', updatedAt: NOW }),
			createTestConversation({ id: 'other-id', title: 'Other', updatedAt: NOW - 1000 })
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

	describe('search functionality', () => {
		it('should show search input', () => {
			render(ConversationList, {
				props: {
					conversations: [],
					activeId: null,
					onselect: mockOnSelect,
					onnew: mockOnNew,
					ondelete: mockOnDelete
				}
			});

			const searchInput = screen.getByPlaceholderText('Search conversations...');
			expect(searchInput).toBeInTheDocument();
		});

		it('should filter conversations by title', async () => {
			const conversations = [
				createTestConversation({ title: 'Python tutorial', updatedAt: NOW - 1000 }),
				createTestConversation({ title: 'JavaScript guide', updatedAt: NOW - 2000 }),
				createTestConversation({ title: 'Python advanced', updatedAt: NOW - 3000 })
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

			const searchInput = screen.getByPlaceholderText('Search conversations...');
			await fireEvent.input(searchInput, { target: { value: 'Python' } });

			expect(screen.getByText('Python tutorial')).toBeInTheDocument();
			expect(screen.getByText('Python advanced')).toBeInTheDocument();
			expect(screen.queryByText('JavaScript guide')).not.toBeInTheDocument();
		});

		it('should be case-insensitive', async () => {
			const conversations = [
				createTestConversation({ title: 'UPPERCASE CHAT', updatedAt: NOW - 1000 }),
				createTestConversation({ title: 'lowercase chat', updatedAt: NOW - 2000 })
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

			const searchInput = screen.getByPlaceholderText('Search conversations...');
			await fireEvent.input(searchInput, { target: { value: 'chat' } });

			expect(screen.getByText('UPPERCASE CHAT')).toBeInTheDocument();
			expect(screen.getByText('lowercase chat')).toBeInTheDocument();
		});

		it('should show all conversations when search is empty', async () => {
			const conversations = [
				createTestConversation({ title: 'Chat 1', updatedAt: NOW - 1000 }),
				createTestConversation({ title: 'Chat 2', updatedAt: NOW - 2000 })
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

			const searchInput = screen.getByPlaceholderText('Search conversations...');

			// Type something
			await fireEvent.input(searchInput, { target: { value: 'Chat 1' } });
			expect(screen.queryByText('Chat 2')).not.toBeInTheDocument();

			// Clear search
			await fireEvent.input(searchInput, { target: { value: '' } });
			expect(screen.getByText('Chat 1')).toBeInTheDocument();
			expect(screen.getByText('Chat 2')).toBeInTheDocument();
		});

		it('should maintain date grouping on filtered results', async () => {
			const conversations = [
				createTestConversation({ title: 'Today Python', updatedAt: NOW - 1000 }),
				createTestConversation({
					title: 'Old Python',
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

			const searchInput = screen.getByPlaceholderText('Search conversations...');
			await fireEvent.input(searchInput, { target: { value: 'Python' } });

			// Both should still be visible with their date groups
			expect(screen.getByText('Today')).toBeInTheDocument();
			expect(screen.getByText('Older')).toBeInTheDocument();
			expect(screen.getByText('Today Python')).toBeInTheDocument();
			expect(screen.getByText('Old Python')).toBeInTheDocument();
		});

		it('should show empty state when search has no matches', async () => {
			const conversations = [
				createTestConversation({ title: 'Python chat', updatedAt: NOW - 1000 })
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

			const searchInput = screen.getByPlaceholderText('Search conversations...');
			await fireEvent.input(searchInput, { target: { value: 'xyz123' } });

			expect(screen.getByText('No matching conversations')).toBeInTheDocument();
		});
	});
});
