/**
 * Unit tests for ReasoningDisplay component
 *
 * Tests cover:
 * - Rendering reasoning content
 * - Streaming vs completed states
 * - Summary display
 * - Collapsible behavior
 */

import { render, screen } from '@testing-library/svelte';
import { describe, expect, it } from 'vitest';
import type { ResponseReasoningItem } from '$lib/stores/types';
import ReasoningDisplay from '../ReasoningDisplay.svelte';

function createReasoningItem(
	text: string,
	options?: { summary?: string; id?: string }
): ResponseReasoningItem {
	return {
		id: options?.id || 'reasoning-1',
		type: 'reasoning',
		content: [{ type: 'reasoning_text', text }],
		summary: options?.summary ? [{ type: 'summary_text', text: options.summary }] : []
	};
}

describe('ReasoningDisplay', () => {
	describe('rendering', () => {
		it('should render the reasoning label', () => {
			const item = createReasoningItem('Thinking about the problem...');

			render(ReasoningDisplay, { props: { item } });

			expect(screen.getByText('Reasoning')).toBeInTheDocument();
		});

		it('should show "Thinking..." when streaming', () => {
			const item = createReasoningItem('Thinking about the problem...');

			render(ReasoningDisplay, { props: { item, isStreaming: true } });

			expect(screen.getByText('Thinking...')).toBeInTheDocument();
		});

		it('should show "Reasoning" when not streaming', () => {
			const item = createReasoningItem('Completed reasoning');

			render(ReasoningDisplay, { props: { item, isStreaming: false } });

			expect(screen.getByText('Reasoning')).toBeInTheDocument();
		});

		it('should render the Brain icon', () => {
			const item = createReasoningItem('Some reasoning');

			const { container } = render(ReasoningDisplay, { props: { item } });

			// Brain icon from lucide-svelte renders as SVG
			const svg = container.querySelector('svg');
			expect(svg).toBeInTheDocument();
		});
	});

	describe('summary display', () => {
		it('should show summary text when available', () => {
			const item = createReasoningItem('Full reasoning content', {
				summary: 'Brief summary'
			});

			render(ReasoningDisplay, { props: { item } });

			expect(screen.getByText(/Brief summary/)).toBeInTheDocument();
		});

		it('should not show summary prefix when no summary', () => {
			const item = createReasoningItem('Reasoning without summary');

			const { container } = render(ReasoningDisplay, { props: { item } });

			// Should not have the summary span with truncate class
			const summarySpan = container.querySelector('.truncate');
			expect(summarySpan?.textContent || '').not.toContain('-');
		});
	});

	describe('collapsible behavior', () => {
		it('should render as a collapsible element', () => {
			const item = createReasoningItem('Hidden reasoning content');

			const { container } = render(ReasoningDisplay, { props: { item } });

			// Should have a button trigger for collapsible
			const trigger = container.querySelector('button');
			expect(trigger).toBeInTheDocument();
		});

		it('should have chevron icon for expand/collapse indication', () => {
			const item = createReasoningItem('Some reasoning');

			const { container } = render(ReasoningDisplay, { props: { item } });

			// Should have multiple SVG icons (Brain and ChevronDown)
			const svgs = container.querySelectorAll('svg');
			expect(svgs.length).toBeGreaterThanOrEqual(2);
		});
	});

	describe('content handling', () => {
		it('should handle empty content gracefully', () => {
			const item: ResponseReasoningItem = {
				id: 'reasoning-empty',
				type: 'reasoning',
				content: [],
				summary: []
			};

			const { container } = render(ReasoningDisplay, { props: { item } });

			// Should still render the component structure
			expect(container.querySelector('button')).toBeInTheDocument();
		});

		it('should handle undefined content gracefully', () => {
			const item: ResponseReasoningItem = {
				id: 'reasoning-undefined',
				type: 'reasoning',
				summary: []
			};

			const { container } = render(ReasoningDisplay, { props: { item } });

			// Should still render the component structure
			expect(container.querySelector('button')).toBeInTheDocument();
		});
	});
});
