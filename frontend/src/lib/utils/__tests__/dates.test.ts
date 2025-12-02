/**
 * Unit tests for date grouping utilities
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
	DATE_GROUP_LABELS,
	getDateGroup,
	groupConversationsByDate,
	type DateGroup
} from '../dates';
import { createConversation } from '$lib/stores/types';

describe('date utilities', () => {
	// Fix "now" to a specific date for predictable tests
	const NOW = new Date('2024-06-15T14:00:00Z').getTime();

	beforeEach(() => {
		vi.useFakeTimers();
		vi.setSystemTime(NOW);
	});

	afterEach(() => {
		vi.useRealTimers();
	});

	describe('getDateGroup', () => {
		it('should return "today" for timestamps from today', () => {
			// 2 hours ago
			const twoHoursAgo = NOW - 2 * 60 * 60 * 1000;
			expect(getDateGroup(twoHoursAgo)).toBe('today');

			// 12 hours ago (still today)
			const twelveHoursAgo = NOW - 12 * 60 * 60 * 1000;
			expect(getDateGroup(twelveHoursAgo)).toBe('today');
		});

		it('should return "today" for current timestamp', () => {
			expect(getDateGroup(NOW)).toBe('today');
		});

		it('should return "yesterday" for timestamps from yesterday', () => {
			// Yesterday at same time
			const yesterday = NOW - 24 * 60 * 60 * 1000;
			expect(getDateGroup(yesterday)).toBe('yesterday');
		});

		it('should return "previous7days" for timestamps 2-6 days ago', () => {
			// 2 days ago
			const twoDaysAgo = NOW - 2 * 24 * 60 * 60 * 1000;
			expect(getDateGroup(twoDaysAgo)).toBe('previous7days');

			// 6 days ago
			const sixDaysAgo = NOW - 6 * 24 * 60 * 60 * 1000;
			expect(getDateGroup(sixDaysAgo)).toBe('previous7days');
		});

		it('should return "older" for timestamps 7+ days ago', () => {
			// 7 days ago
			const sevenDaysAgo = NOW - 7 * 24 * 60 * 60 * 1000;
			expect(getDateGroup(sevenDaysAgo)).toBe('older');

			// 30 days ago
			const thirtyDaysAgo = NOW - 30 * 24 * 60 * 60 * 1000;
			expect(getDateGroup(thirtyDaysAgo)).toBe('older');
		});

		it('should handle midnight boundary correctly', () => {
			// Set time to just after midnight (00:30 on June 15)
			const justAfterMidnight = new Date('2024-06-15T00:30:00Z').getTime();
			vi.setSystemTime(justAfterMidnight);

			// 1 hour ago crosses into yesterday (23:30 on June 14)
			const beforeMidnight = justAfterMidnight - 60 * 60 * 1000;
			expect(getDateGroup(beforeMidnight)).toBe('yesterday');
		});
	});

	describe('groupConversationsByDate', () => {
		it('should group conversations by date', () => {
			const todayConv = createConversation({
				title: 'Today Chat',
				updatedAt: NOW - 1000
			});
			const yesterdayConv = createConversation({
				title: 'Yesterday Chat',
				updatedAt: NOW - 24 * 60 * 60 * 1000
			});
			const weekConv = createConversation({
				title: 'Week Chat',
				updatedAt: NOW - 3 * 24 * 60 * 60 * 1000
			});
			const oldConv = createConversation({
				title: 'Old Chat',
				updatedAt: NOW - 14 * 24 * 60 * 60 * 1000
			});

			const grouped = groupConversationsByDate([todayConv, yesterdayConv, weekConv, oldConv]);

			expect(grouped.get('today')).toHaveLength(1);
			expect(grouped.get('today')?.[0].title).toBe('Today Chat');

			expect(grouped.get('yesterday')).toHaveLength(1);
			expect(grouped.get('yesterday')?.[0].title).toBe('Yesterday Chat');

			expect(grouped.get('previous7days')).toHaveLength(1);
			expect(grouped.get('previous7days')?.[0].title).toBe('Week Chat');

			expect(grouped.get('older')).toHaveLength(1);
			expect(grouped.get('older')?.[0].title).toBe('Old Chat');
		});

		it('should handle empty conversations array', () => {
			const grouped = groupConversationsByDate([]);

			expect(grouped.get('today')).toBeUndefined();
			expect(grouped.get('yesterday')).toBeUndefined();
			expect(grouped.get('previous7days')).toBeUndefined();
			expect(grouped.get('older')).toBeUndefined();
		});

		it('should handle multiple conversations in same group', () => {
			const conv1 = createConversation({
				title: 'Chat 1',
				updatedAt: NOW - 1000
			});
			const conv2 = createConversation({
				title: 'Chat 2',
				updatedAt: NOW - 2000
			});
			const conv3 = createConversation({
				title: 'Chat 3',
				updatedAt: NOW - 3000
			});

			const grouped = groupConversationsByDate([conv1, conv2, conv3]);

			expect(grouped.get('today')).toHaveLength(3);
		});

		it('should maintain insertion order within groups', () => {
			const conv1 = createConversation({
				title: 'First',
				updatedAt: NOW - 1000
			});
			const conv2 = createConversation({
				title: 'Second',
				updatedAt: NOW - 2000
			});

			const grouped = groupConversationsByDate([conv1, conv2]);

			const todayGroup = grouped.get('today')!;
			expect(todayGroup[0].title).toBe('First');
			expect(todayGroup[1].title).toBe('Second');
		});

		it('should only include non-empty groups', () => {
			const todayConv = createConversation({
				title: 'Today',
				updatedAt: NOW
			});

			const grouped = groupConversationsByDate([todayConv]);

			// Only 'today' should exist
			expect(grouped.size).toBe(1);
			expect(grouped.has('today')).toBe(true);
			expect(grouped.has('yesterday')).toBe(false);
		});
	});

	describe('DATE_GROUP_LABELS', () => {
		it('should have human-readable labels for all groups', () => {
			const groups: DateGroup[] = ['today', 'yesterday', 'previous7days', 'older'];

			for (const group of groups) {
				expect(DATE_GROUP_LABELS[group]).toBeDefined();
				expect(typeof DATE_GROUP_LABELS[group]).toBe('string');
			}
		});

		it('should have expected label values', () => {
			expect(DATE_GROUP_LABELS.today).toBe('Today');
			expect(DATE_GROUP_LABELS.yesterday).toBe('Yesterday');
			expect(DATE_GROUP_LABELS.previous7days).toBe('Previous 7 Days');
			expect(DATE_GROUP_LABELS.older).toBe('Older');
		});
	});
});
