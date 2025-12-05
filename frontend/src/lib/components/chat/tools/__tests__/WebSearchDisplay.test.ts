/**
 * Unit tests for WebSearchDisplay component
 *
 * Tests cover:
 * - Rendering search states
 * - Query display
 * - Source links
 * - Loading indicators
 */

import { render, screen } from '@testing-library/svelte';
import { describe, expect, it } from 'vitest';
import type { ResponseFunctionWebSearch } from '$lib/stores/types';
import WebSearchDisplay from '../WebSearchDisplay.svelte';

type WebSearchWithAction = ResponseFunctionWebSearch & {
	action?: {
		type: 'search';
		query?: string;
		sources?: Array<{ url: string; title?: string }>;
	};
};

function createWebSearchItem(
	status: 'in_progress' | 'completed' | 'searching',
	options?: { query?: string; sources?: Array<{ url: string; title?: string }> }
): WebSearchWithAction {
	return {
		id: 'web-search-1',
		type: 'web_search_call',
		status,
		action: options
			? {
					type: 'search',
					query: options.query,
					sources: options.sources
				}
			: undefined
	};
}

describe('WebSearchDisplay', () => {
	describe('loading state', () => {
		it('should show loading indicator when in_progress', () => {
			const item = createWebSearchItem('in_progress');

			const { container } = render(WebSearchDisplay, { props: { item } });

			const spinner = container.querySelector('.animate-spin');
			expect(spinner).toBeInTheDocument();
		});

		it('should show loading indicator when searching', () => {
			const item = createWebSearchItem('searching');

			const { container } = render(WebSearchDisplay, { props: { item } });

			const spinner = container.querySelector('.animate-spin');
			expect(spinner).toBeInTheDocument();
		});

		it('should show "Web Search" title when loading without query', () => {
			const item = createWebSearchItem('in_progress');

			render(WebSearchDisplay, { props: { item } });

			expect(screen.getByText('Web Search')).toBeInTheDocument();
		});
	});

	describe('completed state', () => {
		it('should show search query when completed', () => {
			const item = createWebSearchItem('completed', {
				query: 'test query'
			});

			render(WebSearchDisplay, { props: { item } });

			expect(screen.getByText('Searched: "test query"')).toBeInTheDocument();
		});

		it('should show fallback title when no query', () => {
			const item = createWebSearchItem('completed');

			render(WebSearchDisplay, { props: { item } });

			expect(screen.getByText('Web Search')).toBeInTheDocument();
		});

		it('should show checkmark when completed', () => {
			const item = createWebSearchItem('completed');

			const { container } = render(WebSearchDisplay, { props: { item } });

			const greenIcon = container.querySelector('.text-green-600');
			expect(greenIcon).toBeInTheDocument();
		});
	});

	describe('source links', () => {
		it('should render source links when available', () => {
			const item = createWebSearchItem('completed', {
				query: 'test',
				sources: [
					{ url: 'https://example.com', title: 'Example Site' },
					{ url: 'https://test.com', title: 'Test Site' }
				]
			});

			render(WebSearchDisplay, { props: { item } });

			expect(screen.getByText('Example Site')).toBeInTheDocument();
			expect(screen.getByText('Test Site')).toBeInTheDocument();
		});

		it('should use URL as fallback when no title', () => {
			const item = createWebSearchItem('completed', {
				query: 'test',
				sources: [{ url: 'https://example.com' }]
			});

			render(WebSearchDisplay, { props: { item } });

			expect(screen.getByText('https://example.com')).toBeInTheDocument();
		});

		it('should render links with target="_blank"', () => {
			const item = createWebSearchItem('completed', {
				query: 'test',
				sources: [{ url: 'https://example.com', title: 'Example' }]
			});

			render(WebSearchDisplay, { props: { item } });

			const link = screen.getByText('Example').closest('a');
			expect(link).toHaveAttribute('target', '_blank');
		});

		it('should render links with rel="noopener noreferrer"', () => {
			const item = createWebSearchItem('completed', {
				query: 'test',
				sources: [{ url: 'https://example.com', title: 'Example' }]
			});

			render(WebSearchDisplay, { props: { item } });

			const link = screen.getByText('Example').closest('a');
			expect(link).toHaveAttribute('rel', 'noopener noreferrer');
		});

		it('should render ExternalLink icon for each source', () => {
			const item = createWebSearchItem('completed', {
				query: 'test',
				sources: [
					{ url: 'https://example.com', title: 'Example' },
					{ url: 'https://test.com', title: 'Test' }
				]
			});

			const { container } = render(WebSearchDisplay, { props: { item } });

			// Should have multiple SVGs (Search icon + status icon + chevron + 2 ExternalLink icons)
			const svgs = container.querySelectorAll('svg');
			expect(svgs.length).toBeGreaterThanOrEqual(4);
		});
	});

	describe('empty sources', () => {
		it('should show "No results" when sources empty', () => {
			const item = createWebSearchItem('completed', {
				query: 'test',
				sources: []
			});

			render(WebSearchDisplay, { props: { item } });

			expect(screen.getByText('No results')).toBeInTheDocument();
		});
	});

	describe('collapsible behavior', () => {
		it('should render as a collapsible element', () => {
			const item = createWebSearchItem('completed', { query: 'test' });

			const { container } = render(WebSearchDisplay, { props: { item } });

			const trigger = container.querySelector('button');
			expect(trigger).toBeInTheDocument();
		});

		it('should have chevron icon', () => {
			const item = createWebSearchItem('completed');

			const { container } = render(WebSearchDisplay, { props: { item } });

			// Should have SVG icons including chevron
			const svgs = container.querySelectorAll('svg');
			expect(svgs.length).toBeGreaterThanOrEqual(2);
		});
	});
});
