/**
 * Date grouping utilities for conversation sidebar
 *
 * Groups conversations into time-based categories for display.
 */
import type { Conversation } from '$lib/stores/types';

/**
 * Date group categories
 */
export type DateGroup = 'today' | 'yesterday' | 'previous7days' | 'older';

/**
 * Human-readable labels for date groups
 */
export const DATE_GROUP_LABELS: Record<DateGroup, string> = {
	today: 'Today',
	yesterday: 'Yesterday',
	previous7days: 'Previous 7 Days',
	older: 'Older'
};

/**
 * Order for displaying date groups (most recent first)
 */
export const DATE_GROUP_ORDER: DateGroup[] = ['today', 'yesterday', 'previous7days', 'older'];

/**
 * Determine which date group a timestamp belongs to
 *
 * @param timestamp - Unix timestamp in milliseconds
 * @returns The date group category
 */
export function getDateGroup(timestamp: number): DateGroup {
	const now = new Date();
	const date = new Date(timestamp);

	// Get start of today (midnight)
	const startOfToday = new Date(now);
	startOfToday.setHours(0, 0, 0, 0);

	// Get start of yesterday
	const startOfYesterday = new Date(startOfToday);
	startOfYesterday.setDate(startOfYesterday.getDate() - 1);

	// Get start of 7 days ago (exclusive - so "Previous 7 Days" is days 2-6)
	const startOf7DaysAgo = new Date(startOfToday);
	startOf7DaysAgo.setDate(startOf7DaysAgo.getDate() - 6);

	if (date >= startOfToday) {
		return 'today';
	} else if (date >= startOfYesterday) {
		return 'yesterday';
	} else if (date >= startOf7DaysAgo) {
		return 'previous7days';
	} else {
		return 'older';
	}
}

/**
 * Group conversations by date category
 *
 * @param conversations - Array of conversations to group
 * @returns Map of date groups to conversations (only non-empty groups)
 */
export function groupConversationsByDate(
	conversations: Conversation[]
): Map<DateGroup, Conversation[]> {
	const groups = new Map<DateGroup, Conversation[]>();

	for (const conversation of conversations) {
		const group = getDateGroup(conversation.updatedAt);

		if (!groups.has(group)) {
			groups.set(group, []);
		}
		groups.get(group)!.push(conversation);
	}

	return groups;
}
