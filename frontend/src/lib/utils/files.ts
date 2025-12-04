/**
 * File Processing Utilities
 *
 * Functions for handling file uploads - validation, conversion, and attachment creation.
 */

import { generateUUID } from '$lib/stores/types';

/** Type of attachment based on file content */
export type AttachmentType = 'image' | 'text';

/** An attachment ready to be sent with a message */
export interface Attachment {
	id: string;
	name: string;
	mimeType: string;
	type: AttachmentType;
	/** For images: data URL; for text: raw file content */
	content: string;
}

/** Supported image MIME types */
export const IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/gif'];

/** Supported text MIME types */
export const TEXT_TYPES = [
	'text/plain',
	'text/markdown',
	'text/x-markdown',
	'application/json',
	'text/javascript',
	'application/javascript',
	'text/typescript',
	'text/x-python',
	'text/x-rust',
	'text/x-go',
	'text/x-java',
	'text/x-c',
	'text/x-c++',
	'text/css',
	'text/html',
	'text/xml',
	'application/xml',
	'text/yaml',
	'application/x-yaml',
	'text/csv',
	'application/toml'
];

/** Common text file extensions (fallback when MIME type is generic) */
export const TEXT_EXTENSIONS = [
	'.txt',
	'.md',
	'.json',
	'.js',
	'.ts',
	'.jsx',
	'.tsx',
	'.py',
	'.rs',
	'.go',
	'.java',
	'.c',
	'.cpp',
	'.h',
	'.hpp',
	'.css',
	'.html',
	'.xml',
	'.yaml',
	'.yml',
	'.csv',
	'.toml',
	'.sh',
	'.bash',
	'.zsh',
	'.sql',
	'.svelte',
	'.vue',
	'.rb',
	'.php',
	'.swift',
	'.kt',
	'.scala',
	'.r',
	'.lua',
	'.pl',
	'.pm'
];

/** Maximum file size in bytes (20MB) */
export const MAX_FILE_SIZE = 20 * 1024 * 1024;

/**
 * Determine the attachment type for a file.
 * Returns null if the file type is not supported.
 */
export function getAttachmentType(file: File): AttachmentType | null {
	// Check image types
	if (IMAGE_TYPES.includes(file.type)) {
		return 'image';
	}

	// Check text types by MIME
	if (TEXT_TYPES.includes(file.type)) {
		return 'text';
	}

	// Fallback: check by extension for text files
	// (browsers often report 'application/octet-stream' for unknown types)
	const ext = getFileExtension(file.name);
	if (ext && TEXT_EXTENSIONS.includes(ext)) {
		return 'text';
	}

	return null;
}

/**
 * Check if a file is within the size limit.
 */
export function isValidFileSize(file: File): boolean {
	return file.size <= MAX_FILE_SIZE;
}

/**
 * Get file extension including the dot, lowercase.
 */
export function getFileExtension(filename: string): string | null {
	const lastDot = filename.lastIndexOf('.');
	if (lastDot === -1 || lastDot === filename.length - 1) {
		return null;
	}
	return filename.slice(lastDot).toLowerCase();
}

/**
 * Convert a file to a data URL (base64).
 */
export function fileToDataUrl(file: File): Promise<string> {
	return new Promise((resolve, reject) => {
		const reader = new FileReader();
		reader.onload = () => {
			if (typeof reader.result === 'string') {
				resolve(reader.result);
			} else {
				reject(new Error('Failed to read file as data URL'));
			}
		};
		reader.onerror = () => reject(new Error('Failed to read file'));
		reader.readAsDataURL(file);
	});
}

/**
 * Read a file as text.
 */
export function fileToText(file: File): Promise<string> {
	return new Promise((resolve, reject) => {
		const reader = new FileReader();
		reader.onload = () => {
			if (typeof reader.result === 'string') {
				resolve(reader.result);
			} else {
				reject(new Error('Failed to read file as text'));
			}
		};
		reader.onerror = () => reject(new Error('Failed to read file'));
		reader.readAsText(file);
	});
}

/**
 * Create an attachment from a file.
 * Returns null if the file type is not supported or exceeds size limit.
 */
export async function createAttachment(file: File): Promise<Attachment | null> {
	const type = getAttachmentType(file);
	if (!type) {
		return null;
	}

	if (!isValidFileSize(file)) {
		return null;
	}

	let content: string;
	if (type === 'image') {
		content = await fileToDataUrl(file);
	} else {
		content = await fileToText(file);
	}

	return {
		id: generateUUID(),
		name: file.name,
		mimeType: file.type || 'application/octet-stream',
		type,
		content
	};
}

/**
 * Get a language identifier for syntax highlighting based on file extension.
 */
export function getLanguageFromExtension(filename: string): string {
	const ext = getFileExtension(filename);
	if (!ext) return 'text';

	const languageMap: Record<string, string> = {
		'.js': 'javascript',
		'.jsx': 'javascript',
		'.ts': 'typescript',
		'.tsx': 'typescript',
		'.py': 'python',
		'.rs': 'rust',
		'.go': 'go',
		'.java': 'java',
		'.c': 'c',
		'.cpp': 'cpp',
		'.h': 'c',
		'.hpp': 'cpp',
		'.css': 'css',
		'.html': 'html',
		'.xml': 'xml',
		'.json': 'json',
		'.yaml': 'yaml',
		'.yml': 'yaml',
		'.md': 'markdown',
		'.sql': 'sql',
		'.sh': 'bash',
		'.bash': 'bash',
		'.zsh': 'bash',
		'.svelte': 'svelte',
		'.vue': 'vue',
		'.rb': 'ruby',
		'.php': 'php',
		'.swift': 'swift',
		'.kt': 'kotlin',
		'.scala': 'scala',
		'.r': 'r',
		'.lua': 'lua',
		'.toml': 'toml'
	};

	return languageMap[ext] || 'text';
}

/**
 * Format text attachments as demarcated content for the prompt.
 */
export function formatTextAttachmentsForPrompt(attachments: Attachment[]): string {
	const textAttachments = attachments.filter((a) => a.type === 'text');
	if (textAttachments.length === 0) return '';

	return textAttachments
		.map((a) => {
			const lang = getLanguageFromExtension(a.name);
			return `Here is ${a.name}:\n\`\`\`${lang}\n${a.content}\n\`\`\``;
		})
		.join('\n\n');
}
