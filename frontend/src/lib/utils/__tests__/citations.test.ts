/**
 * Tests for citation utilities
 */

import { describe, it, expect } from 'vitest';
import {
	extractCitations,
	transformCitationMarkers,
	getUniqueCitations,
	type Citation
} from '../citations';
import type { ResponseOutputItem, ResponseOutputMessage } from '$lib/stores/types';

describe('extractCitations', () => {
	it('returns empty array when no message items', () => {
		const rawOutput: ResponseOutputItem[] = [];
		expect(extractCitations(rawOutput)).toEqual([]);
	});

	it('returns empty array when message has no annotations', () => {
		const rawOutput: ResponseOutputItem[] = [
			{
				type: 'message',
				id: 'msg_1',
				status: 'completed',
				role: 'assistant',
				content: [
					{
						type: 'output_text',
						text: 'Some text without citations'
					}
				]
			} as ResponseOutputMessage
		];
		expect(extractCitations(rawOutput)).toEqual([]);
	});

	it('extracts url_citation annotations', () => {
		const rawOutput: ResponseOutputItem[] = [
			{
				type: 'message',
				id: 'msg_1',
				status: 'completed',
				role: 'assistant',
				content: [
					{
						type: 'output_text',
						text: 'According to [1], the answer is clear.',
						annotations: [
							{
								type: 'url_citation',
								url: 'https://example.com/1',
								title: 'Example Source',
								start_index: 13,
								end_index: 16
							}
						]
					}
				]
			} as unknown as ResponseOutputMessage
		];

		const citations = extractCitations(rawOutput);
		expect(citations).toHaveLength(1);
		expect(citations[0]).toEqual({
			index: 1,
			url: 'https://example.com/1',
			title: 'Example Source',
			startIndex: 13,
			endIndex: 16
		});
	});

	it('handles multiple citations in order', () => {
		const rawOutput: ResponseOutputItem[] = [
			{
				type: 'message',
				id: 'msg_1',
				status: 'completed',
				role: 'assistant',
				content: [
					{
						type: 'output_text',
						text: 'Sources [1] and [2] confirm this.',
						annotations: [
							{
								type: 'url_citation',
								url: 'https://example.com/2',
								title: 'Second Source',
								start_index: 16,
								end_index: 19
							},
							{
								type: 'url_citation',
								url: 'https://example.com/1',
								title: 'First Source',
								start_index: 8,
								end_index: 11
							}
						]
					}
				]
			} as unknown as ResponseOutputMessage
		];

		const citations = extractCitations(rawOutput);
		expect(citations).toHaveLength(2);
		// Should be sorted by start_index
		expect(citations[0].title).toBe('First Source');
		expect(citations[0].index).toBe(1);
		expect(citations[1].title).toBe('Second Source');
		expect(citations[1].index).toBe(2);
	});

	it('ignores non-url_citation annotations', () => {
		const rawOutput: ResponseOutputItem[] = [
			{
				type: 'message',
				id: 'msg_1',
				status: 'completed',
				role: 'assistant',
				content: [
					{
						type: 'output_text',
						text: 'Some text',
						annotations: [
							{
								type: 'file_citation',
								file_id: 'file_123',
								filename: 'test.txt',
								index: 0
							}
						]
					}
				]
			} as unknown as ResponseOutputMessage
		];

		const citations = extractCitations(rawOutput);
		expect(citations).toHaveLength(0);
	});
});

describe('transformCitationMarkers', () => {
	it('returns original text when no citations', () => {
		const text = 'Some text with [1] marker.';
		const citations: Citation[] = [];
		expect(transformCitationMarkers(text, citations)).toBe(text);
	});

	it('returns original text when no markers', () => {
		const text = 'Some text without markers.';
		const citations: Citation[] = [
			{
				index: 1,
				url: 'https://example.com',
				title: 'Example',
				startIndex: 0,
				endIndex: 3
			}
		];
		expect(transformCitationMarkers(text, citations)).toBe(text);
	});

	it('transforms citation markers to links', () => {
		const text = 'According to [1], this is true.';
		const citations: Citation[] = [
			{
				index: 1,
				url: 'https://example.com',
				title: 'Example Source',
				startIndex: 13,
				endIndex: 16
			}
		];

		const result = transformCitationMarkers(text, citations);
		expect(result).toContain('href="https://example.com"');
		expect(result).toContain('title="Example Source"');
		expect(result).toContain('class="citation-link"');
		expect(result).toContain('[1]</a>');
	});

	it('transforms multiple markers', () => {
		const text = 'Sources [1] and [2] both agree.';
		const citations: Citation[] = [
			{
				index: 1,
				url: 'https://example.com/1',
				title: 'Source 1',
				startIndex: 8,
				endIndex: 11
			},
			{
				index: 2,
				url: 'https://example.com/2',
				title: 'Source 2',
				startIndex: 16,
				endIndex: 19
			}
		];

		const result = transformCitationMarkers(text, citations);
		expect(result).toContain('href="https://example.com/1"');
		expect(result).toContain('href="https://example.com/2"');
		expect(result).toContain('[1]</a>');
		expect(result).toContain('[2]</a>');
	});

	it('leaves unmatched markers unchanged', () => {
		const text = 'Citation [1] is valid, but [2] has no source.';
		const citations: Citation[] = [
			{
				index: 1,
				url: 'https://example.com',
				title: 'Example',
				startIndex: 9,
				endIndex: 12
			}
		];

		const result = transformCitationMarkers(text, citations);
		expect(result).toContain('href="https://example.com"');
		expect(result).toContain('[2]'); // Still present as plain text
	});

	it('escapes HTML in URLs and titles', () => {
		const text = 'See [1] for details.';
		const citations: Citation[] = [
			{
				index: 1,
				url: 'https://example.com?foo=bar&baz=1',
				title: 'Title with "quotes" & <angle brackets>',
				startIndex: 4,
				endIndex: 7
			}
		];

		const result = transformCitationMarkers(text, citations);
		expect(result).toContain('&amp;');
		expect(result).toContain('&quot;');
		expect(result).toContain('&lt;');
		expect(result).toContain('&gt;');
	});
});

describe('getUniqueCitations', () => {
	it('returns empty array for empty input', () => {
		expect(getUniqueCitations([])).toEqual([]);
	});

	it('returns all citations when unique', () => {
		const citations: Citation[] = [
			{ index: 1, url: 'https://a.com', title: 'A', startIndex: 0, endIndex: 3 },
			{ index: 2, url: 'https://b.com', title: 'B', startIndex: 4, endIndex: 7 }
		];

		const unique = getUniqueCitations(citations);
		expect(unique).toHaveLength(2);
	});

	it('removes duplicate URLs', () => {
		const citations: Citation[] = [
			{ index: 1, url: 'https://a.com', title: 'A', startIndex: 0, endIndex: 3 },
			{ index: 2, url: 'https://a.com', title: 'A duplicate', startIndex: 4, endIndex: 7 },
			{ index: 3, url: 'https://b.com', title: 'B', startIndex: 8, endIndex: 11 }
		];

		const unique = getUniqueCitations(citations);
		expect(unique).toHaveLength(2);
		expect(unique[0].url).toBe('https://a.com');
		expect(unique[1].url).toBe('https://b.com');
	});
});
