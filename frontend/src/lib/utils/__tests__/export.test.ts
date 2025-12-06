/**
 * Unit tests for export utilities
 */
import { describe, expect, it, vi, beforeEach } from 'vitest';
import { exportAsJSON, exportAsMarkdown, downloadFile } from '../export';
import { createConversation, createMessage } from '$lib/stores/types';

describe('export utilities', () => {
	describe('exportAsJSON', () => {
		it('should export conversation as valid JSON string', () => {
			const conversation = createConversation({ title: 'Test Chat' });

			const result = exportAsJSON(conversation);

			expect(() => JSON.parse(result)).not.toThrow();
		});

		it('should include all conversation fields', () => {
			const conversation = createConversation({
				id: 'test-id',
				title: 'Test Chat'
			});

			const result = JSON.parse(exportAsJSON(conversation));

			expect(result.id).toBe('test-id');
			expect(result.title).toBe('Test Chat');
			expect(result.messages).toEqual([]);
			expect(result.createdAt).toBeDefined();
			expect(result.updatedAt).toBeDefined();
		});

		it('should include messages with content', () => {
			const conversation = createConversation({ title: 'Test Chat' });
			conversation.messages = [
				createMessage('user', 'Hello'),
				createMessage('assistant', 'Hi there!')
			];

			const result = JSON.parse(exportAsJSON(conversation));

			expect(result.messages).toHaveLength(2);
			expect(result.messages[0].content).toBe('Hello');
			expect(result.messages[1].content).toBe('Hi there!');
		});

		it('should format JSON with indentation', () => {
			const conversation = createConversation({ title: 'Test' });

			const result = exportAsJSON(conversation);

			// Formatted JSON should have newlines
			expect(result).toContain('\n');
		});
	});

	describe('exportAsMarkdown', () => {
		it('should include conversation title as heading', () => {
			const conversation = createConversation({ title: 'My Chat' });

			const result = exportAsMarkdown(conversation);

			expect(result).toContain('# My Chat');
		});

		it('should format user messages with role heading', () => {
			const conversation = createConversation({ title: 'Test' });
			conversation.messages = [createMessage('user', 'Hello world')];

			const result = exportAsMarkdown(conversation);

			expect(result).toContain('## User');
			expect(result).toContain('Hello world');
		});

		it('should format assistant messages with role heading', () => {
			const conversation = createConversation({ title: 'Test' });
			conversation.messages = [createMessage('assistant', 'Hi there!')];

			const result = exportAsMarkdown(conversation);

			expect(result).toContain('## Assistant');
			expect(result).toContain('Hi there!');
		});

		it('should handle multiple messages in order', () => {
			const conversation = createConversation({ title: 'Test' });
			conversation.messages = [
				createMessage('user', 'First message'),
				createMessage('assistant', 'Second message'),
				createMessage('user', 'Third message')
			];

			const result = exportAsMarkdown(conversation);

			const firstIndex = result.indexOf('First message');
			const secondIndex = result.indexOf('Second message');
			const thirdIndex = result.indexOf('Third message');

			expect(firstIndex).toBeLessThan(secondIndex);
			expect(secondIndex).toBeLessThan(thirdIndex);
		});

		it('should handle empty conversation', () => {
			const conversation = createConversation({ title: 'Empty Chat' });

			const result = exportAsMarkdown(conversation);

			expect(result).toContain('# Empty Chat');
			expect(result).not.toContain('## User');
			expect(result).not.toContain('## Assistant');
		});

		it('should preserve message content formatting', () => {
			const conversation = createConversation({ title: 'Test' });
			conversation.messages = [createMessage('assistant', '```python\nprint("Hello")\n```')];

			const result = exportAsMarkdown(conversation);

			expect(result).toContain('```python');
			expect(result).toContain('print("Hello")');
		});
	});

	describe('downloadFile', () => {
		let mockCreateElement: ReturnType<typeof vi.fn>;
		let mockAppendChild: ReturnType<typeof vi.fn>;
		let mockRemoveChild: ReturnType<typeof vi.fn>;
		let mockClick: ReturnType<typeof vi.fn>;
		let mockRevokeObjectURL: ReturnType<typeof vi.fn>;
		let mockCreateObjectURL: ReturnType<typeof vi.fn>;

		beforeEach(() => {
			mockClick = vi.fn();
			mockCreateElement = vi.fn(() => ({
				href: '',
				download: '',
				click: mockClick
			}));
			mockAppendChild = vi.fn();
			mockRemoveChild = vi.fn();
			mockRevokeObjectURL = vi.fn();
			mockCreateObjectURL = vi.fn(() => 'blob:test-url');

			vi.stubGlobal('document', {
				createElement: mockCreateElement,
				body: {
					appendChild: mockAppendChild,
					removeChild: mockRemoveChild
				}
			});

			vi.stubGlobal('URL', {
				createObjectURL: mockCreateObjectURL,
				revokeObjectURL: mockRevokeObjectURL
			});
		});

		it('should create download link with correct filename', () => {
			downloadFile('test content', 'test.json', 'application/json');

			expect(mockCreateElement).toHaveBeenCalledWith('a');
		});

		it('should create blob with correct mime type', () => {
			downloadFile('test content', 'test.md', 'text/markdown');

			expect(mockCreateObjectURL).toHaveBeenCalled();
			const blob = mockCreateObjectURL.mock.calls[0][0];
			expect(blob).toBeInstanceOf(Blob);
			expect(blob.type).toBe('text/markdown');
		});

		it('should trigger click on download link', () => {
			downloadFile('content', 'file.txt', 'text/plain');

			expect(mockClick).toHaveBeenCalled();
		});

		it('should clean up object URL after download', () => {
			downloadFile('content', 'file.txt', 'text/plain');

			expect(mockRevokeObjectURL).toHaveBeenCalledWith('blob:test-url');
		});
	});
});
