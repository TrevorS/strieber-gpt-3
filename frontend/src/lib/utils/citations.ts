/**
 * Citation utilities for parsing and transforming inline citations.
 *
 * Handles [1], [2] style citation markers in text and links them to
 * URL citations from the API response.
 */

import type { ResponseOutputItem, ResponseOutputMessage } from '$lib/stores/types';

/**
 * A URL citation from the API response.
 */
export interface Citation {
	/** 1-based citation index (from [1], [2], etc.) */
	index: number;
	/** URL of the cited source */
	url: string;
	/** Title of the cited source */
	title: string;
	/** Start character position of the citation marker in the text */
	startIndex: number;
	/** End character position of the citation marker in the text */
	endIndex: number;
}

/**
 * Extract URL citations from raw output items.
 *
 * Looks through message items for output_text content that contains
 * annotations of type 'url_citation'.
 */
export function extractCitations(rawOutput: ResponseOutputItem[]): Citation[] {
	const citations: Citation[] = [];

	for (const item of rawOutput) {
		if (item.type !== 'message') continue;

		const message = item as ResponseOutputMessage;
		for (const content of message.content) {
			if (content.type !== 'output_text') continue;

			// Type assertion for annotations which may not be perfectly typed
			const annotations = (content as { annotations?: unknown[] }).annotations;
			if (!annotations) continue;

			for (const annotation of annotations) {
				const ann = annotation as {
					type?: string;
					url?: string;
					title?: string;
					start_index?: number;
					end_index?: number;
				};

				if (ann.type === 'url_citation' && ann.url && ann.title !== undefined) {
					const startIdx = ann.start_index ?? 0;
					const endIdx = ann.end_index ?? startIdx;

					citations.push({
						index: citations.length + 1, // Will be recalculated
						url: ann.url,
						title: ann.title,
						startIndex: startIdx,
						endIndex: endIdx
					});
				}
			}
		}
	}

	// Sort by start position and reassign indices
	citations.sort((a, b) => a.startIndex - b.startIndex);
	citations.forEach((c, i) => {
		c.index = i + 1;
	});

	return citations;
}

/**
 * A parsed citation marker from text.
 */
interface CitationMarker {
	/** 1-based citation number */
	number: number;
	/** Start position in text */
	start: number;
	/** End position in text */
	end: number;
}

/**
 * Parse citation markers like [1], [2] from text.
 */
function parseCitationMarkers(text: string): CitationMarker[] {
	const markers: CitationMarker[] = [];
	const regex = /\[(\d+)\]/g;

	for (const match of text.matchAll(regex)) {
		const num = parseInt(match[1], 10);
		if (num > 0 && match.index !== undefined) {
			markers.push({
				number: num,
				start: match.index,
				end: match.index + match[0].length
			});
		}
	}

	return markers;
}

/**
 * Transform citation markers [N] in text to clickable superscript links.
 *
 * Takes the original text and citations array, and returns HTML with
 * citation markers transformed to <sup><a> elements linking to the source.
 *
 * @param text - The text containing [1], [2] style markers
 * @param citations - The citations extracted from the response
 * @returns HTML string with markers transformed to links
 */
export function transformCitationMarkers(text: string, citations: Citation[]): string {
	if (citations.length === 0) {
		return text;
	}

	// Build a map of citation index to citation
	const citationMap = new Map<number, Citation>();
	for (const c of citations) {
		citationMap.set(c.index, c);
	}

	// Find all markers and replace them
	const markers = parseCitationMarkers(text);
	if (markers.length === 0) {
		return text;
	}

	// Process markers in reverse order to preserve positions
	let result = text;
	for (let i = markers.length - 1; i >= 0; i--) {
		const marker = markers[i];
		const citation = citationMap.get(marker.number);

		if (citation) {
			// Create a superscript link
			const link = `<sup class="citation-link"><a href="${escapeHtml(citation.url)}" target="_blank" rel="noopener noreferrer" title="${escapeHtml(citation.title)}">[${marker.number}]</a></sup>`;
			result = result.slice(0, marker.start) + link + result.slice(marker.end);
		}
	}

	return result;
}

/**
 * Escape HTML special characters to prevent XSS.
 */
function escapeHtml(str: string): string {
	return str
		.replace(/&/g, '&amp;')
		.replace(/</g, '&lt;')
		.replace(/>/g, '&gt;')
		.replace(/"/g, '&quot;')
		.replace(/'/g, '&#039;');
}

/**
 * Get unique citations (deduplicated by URL).
 * Useful for rendering the citation list at the bottom.
 */
export function getUniqueCitations(citations: Citation[]): Citation[] {
	const seen = new Set<string>();
	const unique: Citation[] = [];

	for (const c of citations) {
		if (!seen.has(c.url)) {
			seen.add(c.url);
			unique.push(c);
		}
	}

	return unique;
}
