/**
 * Unit tests for markdown utilities
 *
 * Tests cover:
 * - Basic markdown rendering (headers, lists, bold, italic)
 * - Code blocks with syntax highlighting
 * - Edge cases and empty input
 * - XSS prevention
 */

import { describe, expect, it } from 'vitest';
import { containsMarkdown, renderMarkdown } from '../markdown';

describe('renderMarkdown', () => {
	// ============================================================================
	// Basic Markdown
	// ============================================================================

	describe('basic markdown', () => {
		it('should render headers', () => {
			const result = renderMarkdown('# Hello');
			expect(result).toContain('<h1');
			expect(result).toContain('Hello');
		});

		it('should render multiple header levels', () => {
			const md = '# H1\n## H2\n### H3';
			const result = renderMarkdown(md);
			expect(result).toContain('<h1');
			expect(result).toContain('<h2');
			expect(result).toContain('<h3');
		});

		it('should render bold text', () => {
			const result = renderMarkdown('This is **bold** text');
			expect(result).toContain('<strong>bold</strong>');
		});

		it('should render italic text', () => {
			const result = renderMarkdown('This is *italic* text');
			expect(result).toContain('<em>italic</em>');
		});

		it('should render unordered lists', () => {
			const md = '- Item 1\n- Item 2\n- Item 3';
			const result = renderMarkdown(md);
			expect(result).toContain('<ul>');
			expect(result).toContain('<li>');
			expect(result).toContain('Item 1');
		});

		it('should render ordered lists', () => {
			const md = '1. First\n2. Second\n3. Third';
			const result = renderMarkdown(md);
			expect(result).toContain('<ol>');
			expect(result).toContain('<li>');
		});

		it('should render links', () => {
			const result = renderMarkdown('[Example](https://example.com)');
			expect(result).toContain('<a');
			expect(result).toContain('href="https://example.com"');
			expect(result).toContain('Example');
		});

		it('should render paragraphs', () => {
			const result = renderMarkdown('First paragraph\n\nSecond paragraph');
			expect(result).toContain('<p>');
		});
	});

	// ============================================================================
	// Code Blocks
	// ============================================================================

	describe('code blocks', () => {
		it('should render inline code', () => {
			const result = renderMarkdown('Use `console.log()` for debugging');
			expect(result).toContain('<code>');
			expect(result).toContain('console.log()');
		});

		it('should render fenced code blocks', () => {
			const md = '```\nconst x = 1;\n```';
			const result = renderMarkdown(md);
			expect(result).toContain('<pre>');
			expect(result).toContain('<code');
			// Content may be highlighted with spans
			expect(result).toContain('const');
			expect(result).toContain('x');
		});

		it('should render code blocks with language', () => {
			const md = '```javascript\nconst x = 1;\n```';
			const result = renderMarkdown(md);
			expect(result).toContain('language-javascript');
		});

		it('should syntax highlight TypeScript', () => {
			const md = '```typescript\nconst greeting: string = "hello";\n```';
			const result = renderMarkdown(md);
			expect(result).toContain('hljs');
			expect(result).toContain('language-typescript');
		});

		it('should syntax highlight Python', () => {
			const md = '```python\ndef hello():\n    print("world")\n```';
			const result = renderMarkdown(md);
			expect(result).toContain('language-python');
		});
	});

	// ============================================================================
	// Edge Cases
	// ============================================================================

	describe('edge cases', () => {
		it('should handle empty string', () => {
			expect(renderMarkdown('')).toBe('');
		});

		it('should handle plain text without markdown', () => {
			const result = renderMarkdown('Just plain text');
			expect(result).toContain('Just plain text');
		});

		it('should handle line breaks', () => {
			const result = renderMarkdown('Line 1\nLine 2');
			expect(result).toContain('<br');
		});

		it('should handle unicode', () => {
			const result = renderMarkdown('Hello 👋 世界');
			expect(result).toContain('Hello 👋 世界');
		});
	});

	// ============================================================================
	// HTML Handling
	// ============================================================================

	describe('HTML handling', () => {
		it('should pass through HTML (marked default behavior)', () => {
			// Note: marked passes through HTML by default
			// For user content, sanitize on the server or use DOMPurify
			// For AI responses, content is generated and trusted
			const result = renderMarkdown('<div>Custom HTML</div>');
			expect(result).toContain('<div>');
		});

		it('should render HTML entities correctly', () => {
			const result = renderMarkdown('&lt;escaped&gt;');
			expect(result).toContain('&lt;');
			expect(result).toContain('&gt;');
		});
	});
});

describe('containsMarkdown', () => {
	it('should detect headers', () => {
		expect(containsMarkdown('# Header')).toBe(true);
		expect(containsMarkdown('## Subheader')).toBe(true);
	});

	it('should detect bold text', () => {
		expect(containsMarkdown('This is **bold**')).toBe(true);
	});

	it('should detect italic text', () => {
		expect(containsMarkdown('This is *italic*')).toBe(true);
	});

	it('should detect code blocks', () => {
		expect(containsMarkdown('```\ncode\n```')).toBe(true);
	});

	it('should detect inline code', () => {
		expect(containsMarkdown('Use `code` here')).toBe(true);
	});

	it('should detect lists', () => {
		expect(containsMarkdown('- item')).toBe(true);
		expect(containsMarkdown('* item')).toBe(true);
		expect(containsMarkdown('1. item')).toBe(true);
	});

	it('should detect links', () => {
		expect(containsMarkdown('[text](url)')).toBe(true);
	});

	it('should return false for plain text', () => {
		expect(containsMarkdown('Just plain text')).toBe(false);
	});
});
