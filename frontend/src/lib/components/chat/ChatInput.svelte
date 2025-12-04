<script lang="ts">
	import { untrack } from 'svelte';
	import { Button } from '$lib/components/ui/button';
	import { Send, Square, Paperclip } from 'lucide-svelte';
	import { logger } from '$lib/utils/logger';
	import { settingsStore, toastStore } from '$lib/stores';
	import { createAttachment, getAttachmentType, isValidFileSize, type Attachment } from '$lib/utils/files';
	import { AttachmentStrip } from '$lib/components/chat';

	let {
		onsubmit,
		onstop,
		disabled = false,
		streaming = false
	}: {
		onsubmit: (text: string, attachments: Attachment[]) => void;
		onstop?: () => void;
		disabled?: boolean;
		streaming?: boolean;
	} = $props();

	let value = $state('');
	let textarea: HTMLTextAreaElement;
	let fileInput: HTMLInputElement;
	let attachments = $state<Attachment[]>([]);
	let isDragging = $state(false);

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter' && !e.shiftKey) {
			e.preventDefault();
			logger.ui.event('ChatInput', 'Enter key pressed', { disabled, hasValue: value.trim().length > 0 });
			submit();
		}
	}

	function submit() {
		const text = value.trim();
		const hasContent = text || attachments.length > 0;
		if (hasContent && !disabled) {
			logger.ui.event('ChatInput', 'Message submitted', {
				textLength: text.length,
				attachmentCount: attachments.length
			});
			onsubmit(text, attachments);
			value = '';
			attachments = [];
			if (textarea) {
				textarea.style.height = 'auto';
				textarea.focus();
			}
		} else {
			logger.debug('ui', 'Submit blocked', { disabled, textLength: text.length });
		}
	}

	function autoResize() {
		if (textarea) {
			textarea.style.height = 'auto';
			textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
		}
	}

	let canSubmit = $derived((value.trim().length > 0 || attachments.length > 0) && !disabled);

	// Track previous streaming state to detect when streaming ends
	let wasStreaming = $state(false);
	let hasMounted = $state(false);

	$effect(() => {
		const currentlyStreaming = streaming; // Track this dependency
		const prevStreaming = untrack(() => wasStreaming); // Don't track the read
		const mounted = untrack(() => hasMounted);

		// Focus textarea when:
		// 1. Streaming ends (was streaming, now not)
		// 2. First mount when not streaming
		if (textarea) {
			if (prevStreaming && !currentlyStreaming) {
				textarea.focus();
				logger.debug('ui', 'Refocused textarea after streaming ended');
			} else if (!mounted && !currentlyStreaming) {
				textarea.focus();
				logger.debug('ui', 'Focused textarea on mount');
			}
		}

		wasStreaming = currentlyStreaming;
		hasMounted = true;
	});

	function handleStop() {
		logger.ui.event('ChatInput', 'Stop streaming clicked', {});
		onstop?.();
	}

	// File upload handling
	async function processFiles(files: FileList | File[]) {
		const fileArray = Array.from(files);

		for (const file of fileArray) {
			// Check file size first
			if (!isValidFileSize(file)) {
				toastStore.warning(`File "${file.name}" is too large (max 20MB)`);
				continue;
			}

			// Check file type
			const type = getAttachmentType(file);
			if (!type) {
				toastStore.warning(`File type not supported: ${file.name}`);
				continue;
			}

			// Check vision capability for images
			if (type === 'image' && !settingsStore.supportsVision()) {
				toastStore.warning(`Current model doesn't support images. Select a vision-capable model.`);
				continue;
			}

			// Create attachment
			const attachment = await createAttachment(file);
			if (attachment) {
				attachments = [...attachments, attachment];
				logger.ui.event('ChatInput', 'File attached', {
					name: file.name,
					type: attachment.type
				});
			}
		}
	}

	function handleFileSelect(e: Event) {
		const input = e.target as HTMLInputElement;
		if (input.files?.length) {
			processFiles(input.files);
			// Reset input so same file can be selected again
			input.value = '';
		}
	}

	function openFilePicker() {
		fileInput?.click();
	}

	function removeAttachment(id: string) {
		attachments = attachments.filter(a => a.id !== id);
		logger.ui.event('ChatInput', 'Attachment removed', { id });
	}

	// Paste handling for images
	function handlePaste(e: ClipboardEvent) {
		const items = e.clipboardData?.items;
		if (!items) return;

		const imageItems = Array.from(items).filter(item => item.type.startsWith('image/'));
		if (imageItems.length === 0) return;

		// Check vision capability
		if (!settingsStore.supportsVision()) {
			toastStore.warning(`Current model doesn't support images. Select a vision-capable model.`);
			return;
		}

		e.preventDefault();
		const files = imageItems
			.map(item => item.getAsFile())
			.filter((f): f is File => f !== null);

		if (files.length > 0) {
			processFiles(files);
		}
	}

	// Drag and drop handling
	function handleDragOver(e: DragEvent) {
		e.preventDefault();
		isDragging = true;
	}

	function handleDragLeave(e: DragEvent) {
		e.preventDefault();
		// Only set false if leaving the container, not entering a child
		const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
		const x = e.clientX;
		const y = e.clientY;
		if (x < rect.left || x > rect.right || y < rect.top || y > rect.bottom) {
			isDragging = false;
		}
	}

	function handleDrop(e: DragEvent) {
		e.preventDefault();
		isDragging = false;

		if (e.dataTransfer?.files.length) {
			processFiles(e.dataTransfer.files);
		}
	}

	// Build accept string based on model capabilities
	let acceptTypes = $derived.by(() => {
		const types = [
			// Text types - always allowed
			'.txt,.md,.json,.js,.ts,.jsx,.tsx,.py,.rs,.go,.java,.c,.cpp,.h,.hpp',
			'.css,.html,.xml,.yaml,.yml,.csv,.toml,.sh,.bash,.zsh,.sql',
			'.svelte,.vue,.rb,.php,.swift,.kt,.scala,.r,.lua,.pl,.pm'
		];

		if (settingsStore.supportsVision()) {
			types.push('image/jpeg,image/png,image/gif');
		}

		return types.join(',');
	});
</script>

<div
	class="border-t p-4 {isDragging ? 'bg-accent/30' : ''} transition-colors"
	ondragover={handleDragOver}
	ondragleave={handleDragLeave}
	ondrop={handleDrop}
	role="region"
	aria-label="Message input"
>
	<div class="max-w-3xl mx-auto">
		{#if attachments.length > 0}
			<AttachmentStrip {attachments} onremove={removeAttachment} />
		{/if}
		<div class="flex gap-2 items-end">
			<!-- Hidden file input -->
			<input
				bind:this={fileInput}
				type="file"
				multiple
				accept={acceptTypes}
				onchange={handleFileSelect}
				class="hidden"
				aria-hidden="true"
			/>

			<!-- File picker button -->
			<Button
				onclick={openFilePicker}
				variant="ghost"
				size="icon"
				class="shrink-0"
				aria-label="Attach files"
				data-testid="attach-button"
			>
				<Paperclip class="h-4 w-4" />
			</Button>

			<textarea
				bind:this={textarea}
				bind:value
				onkeydown={handleKeydown}
				oninput={autoResize}
				onpaste={handlePaste}
				placeholder="Send a message..."
				rows="1"
				class="flex-1 resize-none rounded-lg border p-3 focus:outline-none focus:ring-2 focus:ring-ring"
			></textarea>
			{#if streaming}
				<Button onclick={handleStop} variant="destructive" size="icon" data-testid="stop-button">
					<Square class="h-4 w-4" />
				</Button>
			{:else}
				<Button onclick={submit} disabled={!canSubmit} size="icon" data-testid="send-button">
					<Send class="h-4 w-4" />
				</Button>
			{/if}
		</div>

		{#if isDragging}
			<div class="mt-2 text-center text-sm text-muted-foreground">
				Drop files here to attach
			</div>
		{/if}
	</div>
</div>
