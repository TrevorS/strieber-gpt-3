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

			// Note: Images are always allowed - tools like zimage_controlnet can handle them
			// even when the main model doesn't support vision

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

		// Images always allowed - tools can handle them even without vision model
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

	// Build accept string - always include images (tools can handle them)
	let acceptTypes = $derived.by(() => {
		const types = [
			// Text types
			'.txt,.md,.json,.js,.ts,.jsx,.tsx,.py,.rs,.go,.java,.c,.cpp,.h,.hpp',
			'.css,.html,.xml,.yaml,.yml,.csv,.toml,.sh,.bash,.zsh,.sql',
			'.svelte,.vue,.rb,.php,.swift,.kt,.scala,.r,.lua,.pl,.pm',
			// Image types - always allowed for tool use (zimage_controlnet, etc.)
			'image/jpeg,image/png,image/gif,image/webp'
		];

		return types.join(',');
	});
</script>

<div
	class="p-4 {isDragging ? 'bg-accent/30' : ''} transition-colors"
	ondragover={handleDragOver}
	ondragleave={handleDragLeave}
	ondrop={handleDrop}
	role="region"
	aria-label="Message input"
>
	<div class="max-w-3xl mx-auto">
		{#if attachments.length > 0}
			<div class="mb-3">
				<AttachmentStrip {attachments} onremove={removeAttachment} />
			</div>
		{/if}

		<!-- Floating input container -->
		<div class="relative flex items-center gap-1 rounded-2xl border bg-background
					shadow-sm ring-1 ring-border/50
					focus-within:ring-2 focus-within:ring-ring/50 focus-within:shadow-md
					transition-all duration-200 py-2 px-3">
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

			<!-- Attach button (left, inside) -->
			<button
				onclick={openFilePicker}
				type="button"
				class="shrink-0 p-2 rounded-xl text-muted-foreground
					   hover:text-foreground hover:bg-accent transition-colors"
				aria-label="Attach files"
				data-testid="attach-button"
			>
				<Paperclip class="h-5 w-5" />
			</button>

			<!-- Textarea (center) -->
			<textarea
				bind:this={textarea}
				bind:value
				onkeydown={handleKeydown}
				oninput={autoResize}
				onpaste={handlePaste}
				placeholder="Message Strieber GPT..."
				rows="1"
				class="flex-1 resize-none bg-transparent py-2 px-1 text-base leading-relaxed
					   placeholder:text-muted-foreground/70 focus:outline-none
					   max-h-[200px] overflow-y-auto"
			></textarea>

			<!-- Send/Stop button (right, inside) -->
			{#if streaming}
				<button
					onclick={handleStop}
					type="button"
					class="shrink-0 p-2 rounded-xl bg-destructive text-destructive-foreground
						   hover:bg-destructive/90 transition-colors"
					data-testid="stop-button"
					aria-label="Stop generating"
				>
					<Square class="h-5 w-5 fill-current" />
				</button>
			{:else}
				<button
					onclick={submit}
					type="button"
					disabled={!canSubmit}
					class="shrink-0 p-2 rounded-xl transition-colors
						   {canSubmit
							 ? 'bg-primary text-primary-foreground hover:bg-primary/90'
							 : 'text-muted-foreground/50 cursor-not-allowed'}"
					data-testid="send-button"
					aria-label="Send message"
				>
					<Send class="h-5 w-5" />
				</button>
			{/if}
		</div>

		{#if isDragging}
			<div class="mt-3 text-center text-sm text-muted-foreground animate-pulse">
				Drop files here to attach
			</div>
		{/if}
	</div>
</div>
