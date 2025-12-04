import { describe, it, expect, vi } from 'vitest';
import {
	getAttachmentType,
	isValidFileSize,
	getFileExtension,
	fileToDataUrl,
	fileToText,
	createAttachment,
	getLanguageFromExtension,
	formatTextAttachmentsForPrompt,
	IMAGE_TYPES,
	TEXT_TYPES,
	TEXT_EXTENSIONS,
	MAX_FILE_SIZE,
	type Attachment
} from '../files';

// Mock generateUUID
vi.mock('$lib/stores/types', () => ({
	generateUUID: () => 'test-uuid-123'
}));

/**
 * Create a mock File object.
 * When size is specified and larger than content, pads with zeros.
 */
function createMockFile(
	name: string,
	type: string,
	size?: number,
	content: string = 'test content'
): File {
	let data: BlobPart;
	if (size !== undefined && size > content.length) {
		// Create a buffer of the exact size
		const buffer = new ArrayBuffer(size);
		const view = new Uint8Array(buffer);
		// Copy content at the start
		for (let i = 0; i < content.length; i++) {
			view[i] = content.charCodeAt(i);
		}
		data = buffer;
	} else {
		data = content;
	}
	const blob = new Blob([data], { type });
	return new File([blob], name, { type });
}

describe('getAttachmentType', () => {
	it('returns "image" for jpeg files', () => {
		const file = createMockFile('photo.jpg', 'image/jpeg');
		expect(getAttachmentType(file)).toBe('image');
	});

	it('returns "image" for png files', () => {
		const file = createMockFile('image.png', 'image/png');
		expect(getAttachmentType(file)).toBe('image');
	});

	it('returns "image" for gif files', () => {
		const file = createMockFile('animation.gif', 'image/gif');
		expect(getAttachmentType(file)).toBe('image');
	});

	it('returns null for webp files (not supported)', () => {
		const file = createMockFile('image.webp', 'image/webp');
		expect(getAttachmentType(file)).toBeNull();
	});

	it('returns "text" for plain text files', () => {
		const file = createMockFile('readme.txt', 'text/plain');
		expect(getAttachmentType(file)).toBe('text');
	});

	it('returns "text" for json files', () => {
		const file = createMockFile('data.json', 'application/json');
		expect(getAttachmentType(file)).toBe('text');
	});

	it('returns "text" for markdown files', () => {
		const file = createMockFile('README.md', 'text/markdown');
		expect(getAttachmentType(file)).toBe('text');
	});

	it('returns "text" for files with text extension but generic MIME', () => {
		const file = createMockFile('script.py', 'application/octet-stream');
		expect(getAttachmentType(file)).toBe('text');
	});

	it('returns null for unsupported types', () => {
		const file = createMockFile('archive.zip', 'application/zip');
		expect(getAttachmentType(file)).toBe(null);
	});

	it('returns null for PDF files', () => {
		const file = createMockFile('document.pdf', 'application/pdf');
		expect(getAttachmentType(file)).toBe(null);
	});
});

describe('isValidFileSize', () => {
	it('returns true for files under the limit', () => {
		const file = createMockFile('small.txt', 'text/plain', 1000);
		expect(isValidFileSize(file)).toBe(true);
	});

	it('returns true for files exactly at the limit', () => {
		const file = createMockFile('exact.txt', 'text/plain', MAX_FILE_SIZE);
		expect(isValidFileSize(file)).toBe(true);
	});

	it('returns false for files over the limit', () => {
		const file = createMockFile('large.txt', 'text/plain', MAX_FILE_SIZE + 1);
		expect(isValidFileSize(file)).toBe(false);
	});
});

describe('getFileExtension', () => {
	it('returns lowercase extension with dot', () => {
		expect(getFileExtension('file.TXT')).toBe('.txt');
		expect(getFileExtension('image.PNG')).toBe('.png');
	});

	it('returns null for files without extension', () => {
		expect(getFileExtension('Makefile')).toBe(null);
	});

	it('returns null for files ending with dot', () => {
		expect(getFileExtension('file.')).toBe(null);
	});

	it('handles multiple dots correctly', () => {
		expect(getFileExtension('archive.tar.gz')).toBe('.gz');
	});
});

describe('fileToDataUrl', () => {
	it('converts file to data URL', async () => {
		const content = 'Hello, World!';
		const file = createMockFile('test.txt', 'text/plain', content.length, content);

		const result = await fileToDataUrl(file);

		expect(result).toMatch(/^data:text\/plain;base64,/);
	});

	it('converts image to data URL', async () => {
		// Create a minimal PNG (1x1 transparent pixel)
		const pngData = new Uint8Array([
			0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44,
			0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1f,
			0x15, 0xc4, 0x89, 0x00, 0x00, 0x00, 0x0a, 0x49, 0x44, 0x41, 0x54, 0x08, 0xd7, 0x63, 0x00,
			0x01, 0x00, 0x00, 0x05, 0x00, 0x01, 0x0d, 0x0a, 0x2d, 0xb4, 0x00, 0x00, 0x00, 0x00, 0x49,
			0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82
		]);
		const blob = new Blob([pngData], { type: 'image/png' });
		const file = new File([blob], 'pixel.png', { type: 'image/png' });

		const result = await fileToDataUrl(file);

		expect(result).toMatch(/^data:image\/png;base64,/);
	});
});

describe('fileToText', () => {
	it('reads file as text', async () => {
		const content = 'Hello, World!';
		const file = createMockFile('test.txt', 'text/plain', content.length, content);

		const result = await fileToText(file);

		expect(result).toBe(content);
	});

	it('handles unicode content', async () => {
		const content = 'Hello, World!';
		const file = createMockFile('unicode.txt', 'text/plain', undefined, content);

		const result = await fileToText(file);

		expect(result).toBe(content);
	});
});

describe('createAttachment', () => {
	it('creates image attachment with data URL', async () => {
		const pngData = new Uint8Array([
			0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44,
			0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1f,
			0x15, 0xc4, 0x89, 0x00, 0x00, 0x00, 0x0a, 0x49, 0x44, 0x41, 0x54, 0x08, 0xd7, 0x63, 0x00,
			0x01, 0x00, 0x00, 0x05, 0x00, 0x01, 0x0d, 0x0a, 0x2d, 0xb4, 0x00, 0x00, 0x00, 0x00, 0x49,
			0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82
		]);
		const blob = new Blob([pngData], { type: 'image/png' });
		const file = new File([blob], 'image.png', { type: 'image/png' });

		const result = await createAttachment(file);

		expect(result).not.toBeNull();
		expect(result?.type).toBe('image');
		expect(result?.name).toBe('image.png');
		expect(result?.mimeType).toBe('image/png');
		expect(result?.content).toMatch(/^data:image\/png;base64,/);
		expect(result?.id).toBe('test-uuid-123');
	});

	it('creates text attachment with raw content', async () => {
		const content = 'console.log("hello");';
		const file = createMockFile('script.js', 'text/javascript', content.length, content);

		const result = await createAttachment(file);

		expect(result).not.toBeNull();
		expect(result?.type).toBe('text');
		expect(result?.name).toBe('script.js');
		expect(result?.content).toBe(content);
	});

	it('returns null for unsupported file types', async () => {
		const file = createMockFile('archive.zip', 'application/zip');

		const result = await createAttachment(file);

		expect(result).toBeNull();
	});

	it('returns null for files exceeding size limit', async () => {
		// Create a file that would exceed the limit
		const largeContent = 'x'.repeat(MAX_FILE_SIZE + 1);
		const file = createMockFile('large.txt', 'text/plain', MAX_FILE_SIZE + 1, largeContent);

		const result = await createAttachment(file);

		expect(result).toBeNull();
	});
});

describe('getLanguageFromExtension', () => {
	it('returns correct language for common extensions', () => {
		expect(getLanguageFromExtension('app.js')).toBe('javascript');
		expect(getLanguageFromExtension('app.ts')).toBe('typescript');
		expect(getLanguageFromExtension('script.py')).toBe('python');
		expect(getLanguageFromExtension('main.rs')).toBe('rust');
		expect(getLanguageFromExtension('main.go')).toBe('go');
		expect(getLanguageFromExtension('App.java')).toBe('java');
	});

	it('returns "text" for unknown extensions', () => {
		expect(getLanguageFromExtension('file.xyz')).toBe('text');
	});

	it('returns "text" for files without extension', () => {
		expect(getLanguageFromExtension('Makefile')).toBe('text');
	});
});

describe('formatTextAttachmentsForPrompt', () => {
	it('formats single text attachment', () => {
		const attachments: Attachment[] = [
			{
				id: '1',
				name: 'script.py',
				mimeType: 'text/x-python',
				type: 'text',
				content: 'print("hello")'
			}
		];

		const result = formatTextAttachmentsForPrompt(attachments);

		expect(result).toBe('Here is script.py:\n```python\nprint("hello")\n```');
	});

	it('formats multiple text attachments', () => {
		const attachments: Attachment[] = [
			{
				id: '1',
				name: 'index.js',
				mimeType: 'text/javascript',
				type: 'text',
				content: 'const x = 1;'
			},
			{
				id: '2',
				name: 'style.css',
				mimeType: 'text/css',
				type: 'text',
				content: 'body { margin: 0; }'
			}
		];

		const result = formatTextAttachmentsForPrompt(attachments);

		expect(result).toContain('Here is index.js:\n```javascript\nconst x = 1;\n```');
		expect(result).toContain('Here is style.css:\n```css\nbody { margin: 0; }\n```');
	});

	it('ignores image attachments', () => {
		const attachments: Attachment[] = [
			{
				id: '1',
				name: 'photo.png',
				mimeType: 'image/png',
				type: 'image',
				content: 'data:image/png;base64,...'
			},
			{
				id: '2',
				name: 'script.py',
				mimeType: 'text/x-python',
				type: 'text',
				content: 'print("hello")'
			}
		];

		const result = formatTextAttachmentsForPrompt(attachments);

		expect(result).not.toContain('photo.png');
		expect(result).toContain('script.py');
	});

	it('returns empty string for no text attachments', () => {
		const attachments: Attachment[] = [
			{
				id: '1',
				name: 'photo.png',
				mimeType: 'image/png',
				type: 'image',
				content: 'data:image/png;base64,...'
			}
		];

		const result = formatTextAttachmentsForPrompt(attachments);

		expect(result).toBe('');
	});

	it('returns empty string for empty array', () => {
		const result = formatTextAttachmentsForPrompt([]);

		expect(result).toBe('');
	});
});

describe('constant exports', () => {
	it('exports IMAGE_TYPES array', () => {
		expect(IMAGE_TYPES).toContain('image/jpeg');
		expect(IMAGE_TYPES).toContain('image/png');
		expect(IMAGE_TYPES).toContain('image/gif');
		// WebP intentionally not supported due to model compatibility
		expect(IMAGE_TYPES).not.toContain('image/webp');
	});

	it('exports TEXT_TYPES array', () => {
		expect(TEXT_TYPES).toContain('text/plain');
		expect(TEXT_TYPES).toContain('application/json');
		expect(TEXT_TYPES).toContain('text/javascript');
	});

	it('exports TEXT_EXTENSIONS array', () => {
		expect(TEXT_EXTENSIONS).toContain('.txt');
		expect(TEXT_EXTENSIONS).toContain('.js');
		expect(TEXT_EXTENSIONS).toContain('.py');
		expect(TEXT_EXTENSIONS).toContain('.rs');
	});

	it('exports MAX_FILE_SIZE constant', () => {
		expect(MAX_FILE_SIZE).toBe(20 * 1024 * 1024);
	});
});
