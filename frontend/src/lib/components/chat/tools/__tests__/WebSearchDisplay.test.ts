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
	status: 'in_progress' | 'searching' | 'completed' | 'failed',
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

			render(WebSearchDisplay, { props: { item } });

			expect(screen.getByText('Searching the web...')).toBeInTheDocument();
		});

		it('should show loading indicator when searching', () => {
			const item = createWebSearchItem('searching');

			render(WebSearchDisplay, { props: { item } });

			expect(screen.getByText('Searching the web...')).toBeInTheDocument();
		});

		it('should render spinner icon when loading', () => {
			const item = createWebSearchItem('in_progress');

			const { container } = render(WebSearchDisplay, { props: { item } });

			// Loader2 icon has animate-spin class
			const spinner = container.querySelector('.animate-spin');
			expect(spinner).toBeInTheDocument();
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

		it('should show fallback text when no query', () => {
			const item = createWebSearchItem('completed');

			render(WebSearchDisplay, { props: { item } });

			expect(screen.getByText('Web search completed')).toBeInTheDocument();
		});

		it('should render Search icon when completed', () => {
			const item = createWebSearchItem('completed');

			const { container } = render(WebSearchDisplay, { props: { item } });

			// Should have SVG icon
			const svg = container.querySelector('svg');
			expect(svg).toBeInTheDocument();
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

			const link = screen.getByRole('link', { name: /Example/i });
			expect(link).toHaveAttribute('target', '_blank');
		});

		it('should render links with rel="noopener noreferrer"', () => {
			const item = createWebSearchItem('completed', {
				query: 'test',
				sources: [{ url: 'https://example.com', title: 'Example' }]
			});

			render(WebSearchDisplay, { props: { item } });

			const link = screen.getByRole('link', { name: /Example/i });
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

			// Should have multiple SVGs (Search icon + 2 ExternalLink icons)
			const svgs = container.querySelectorAll('svg');
			expect(svgs.length).toBeGreaterThanOrEqual(3);
		});
	});

	describe('empty sources', () => {
		it('should not render source list when empty', () => {
			const item = createWebSearchItem('completed', {
				query: 'test',
				sources: []
			});

			const { container } = render(WebSearchDisplay, { props: { item } });

			const list = container.querySelector('ul');
			expect(list).not.toBeInTheDocument();
		});
	});
});
