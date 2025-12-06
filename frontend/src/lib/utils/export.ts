/**
 * Export utilities for conversations
 */
import type { Conversation } from '$lib/stores/types';

/**
 * Export a conversation as a formatted JSON string
 */
export function exportAsJSON(conversation: Conversation): string {
	return JSON.stringify(conversation, null, 2);
}

/**
 * Export a conversation as a Markdown document
 */
export function exportAsMarkdown(conversation: Conversation): string {
	const lines: string[] = [];

	// Title
	lines.push(`# ${conversation.title}`);
	lines.push('');

	// Messages
	for (const message of conversation.messages) {
		const roleLabel = message.role === 'user' ? 'User' : 'Assistant';
		lines.push(`## ${roleLabel}`);
		lines.push('');
		lines.push(message.content);
		lines.push('');
	}

	return lines.join('\n');
}

/**
 * Trigger a file download in the browser
 */
export function downloadFile(content: string, filename: string, mimeType: string): void {
	const blob = new Blob([content], { type: mimeType });
	const url = URL.createObjectURL(blob);

	const link = document.createElement('a');
	link.href = url;
	link.download = filename;

	document.body.appendChild(link);
	link.click();
	document.body.removeChild(link);

	URL.revokeObjectURL(url);
}

/**
 * Export and download a conversation as JSON
 */
export function downloadConversationAsJSON(conversation: Conversation): void {
	const content = exportAsJSON(conversation);
	const filename = `${sanitizeFilename(conversation.title)}.json`;
	downloadFile(content, filename, 'application/json');
}

/**
 * Export and download a conversation as Markdown
 */
export function downloadConversationAsMarkdown(conversation: Conversation): void {
	const content = exportAsMarkdown(conversation);
	const filename = `${sanitizeFilename(conversation.title)}.md`;
	downloadFile(content, filename, 'text/markdown');
}

/**
 * Sanitize a string for use as a filename
 */
function sanitizeFilename(name: string): string {
	return name
		.replace(/[<>:"/\\|?*]/g, '') // Remove invalid characters
		.replace(/\s+/g, '_') // Replace spaces with underscores
		.slice(0, 100); // Limit length
}
