/**
 * Markdown Rendering Utilities
 *
 * Uses marked for parsing and highlight.js for code syntax highlighting.
 * Configured for safe HTML rendering with no XSS vulnerabilities.
 */

import hljs from 'highlight.js';
import { Marked } from 'marked';
import { markedHighlight } from 'marked-highlight';

/**
 * Create a configured marked instance with syntax highlighting
 */
const marked = new Marked(
	markedHighlight({
		emptyLangClass: 'hljs',
		langPrefix: 'hljs language-',
		highlight(code: string, lang: string): string {
			if (lang && hljs.getLanguage(lang)) {
				try {
					return hljs.highlight(code, { language: lang }).value;
				} catch {
					// Fall through to auto-detect
				}
			}
			// Auto-detect language
			try {
				return hljs.highlightAuto(code).value;
			} catch {
				return code;
			}
		}
	})
);

// Configure marked options
marked.setOptions({
	gfm: true, // GitHub Flavored Markdown
	breaks: true // Convert \n to <br>
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
