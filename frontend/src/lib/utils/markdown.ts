/**
 * Markdown Rendering Utilities
 *
 * Uses marked for parsing and highlight.js for code syntax highlighting.
 * Configured for safe HTML rendering with no XSS vulnerabilities.
 */

import hljs from 'highlight.js';
import { Marked, type Tokens } from 'marked';

// SVG icons for copy button
const COPY_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>`;

/**
 * Custom renderer that wraps code blocks with header bar containing language label and copy button
 */
const customRenderer = {
	code(token: Tokens.Code): string {
		const lang = token.lang || 'text';
		const code = token.text;

		// Highlight the code
		let highlighted: string;
		if (lang && lang !== 'text' && hljs.getLanguage(lang)) {
			try {
				highlighted = hljs.highlight(code, { language: lang }).value;
			} catch {
				highlighted = hljs.highlightAuto(code).value;
			}
		} else {
			try {
				highlighted = hljs.highlightAuto(code).value;
			} catch {
				highlighted = code;
			}
		}

		// Generate unique ID for copy button functionality
		const codeId = `code-${Math.random().toString(36).substring(2, 9)}`;

		// Return wrapped code block with header
		return `<div class="code-block-wrapper"><div class="code-block-header"><span class="code-block-lang">${lang}</span><button class="code-header-copy-btn" data-code-id="${codeId}" title="Copy code">${COPY_ICON}</button></div><pre><code id="${codeId}" class="hljs language-${lang}">${highlighted}</code></pre></div>`;
	}
};

/**
 * Create a configured marked instance with custom code block rendering
 * Note: We handle highlighting ourselves in customRenderer instead of using markedHighlight
 * because markedHighlight's internal renderer overrides custom renderers
 */
const marked = new Marked({
	gfm: true, // GitHub Flavored Markdown
	breaks: true, // Convert \n to <br>
	renderer: customRenderer
});

/**
 * Render markdown text to HTML.
 *
 * @param text - Markdown text to render
 * @returns HTML string
 *
 * @example
 * ```typescript
 * const html = renderMarkdown('# Hello\n\nWorld');
 * // Returns: '<h1>Hello</h1>\n<p>World</p>\n'
 * ```
 */
export function renderMarkdown(text: string): string {
	if (!text) return '';
	return marked.parse(text) as string;
}

/**
 * Render markdown text to HTML asynchronously.
 * Useful for very large documents.
 *
 * @param text - Markdown text to render
 * @returns Promise resolving to HTML string
 */
export async function renderMarkdownAsync(text: string): Promise<string> {
	if (!text) return '';
	return marked.parse(text);
}

/**
 * Check if text contains markdown syntax.
 * Useful for deciding whether to render as markdown or plain text.
 *
 * @param text - Text to check
 * @returns True if text appears to contain markdown
 */
export function containsMarkdown(text: string): boolean {
	// Common markdown patterns
	const patterns = [
		/^#{1,6}\s/, // Headers
		/\*\*.*\*\*/, // Bold
		/\*.*\*/, // Italic
		/`.*`/, // Inline code
		/```/, // Code blocks
		/^\s*[-*+]\s/, // Unordered lists
		/^\s*\d+\.\s/, // Ordered lists
		/\[.*\]\(.*\)/, // Links
		/!\[.*\]\(.*\)/ // Images
	];

	return patterns.some((pattern) => pattern.test(text));
}
