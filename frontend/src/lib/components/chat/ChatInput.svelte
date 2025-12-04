<script lang="ts">
	import { untrack } from 'svelte';
	import { Button } from '$lib/components/ui/button';
	import { Send, Square } from 'lucide-svelte';
	import { logger } from '$lib/utils/logger';

	let {
		onsubmit,
		onstop,
		disabled = false,
		streaming = false
	}: {
		onsubmit: (text: string) => void;
		onstop?: () => void;
		disabled?: boolean;
		streaming?: boolean;
	} = $props();

	let value = $state('');
	let textarea: HTMLTextAreaElement;

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter' && !e.shiftKey) {
			e.preventDefault();
			logger.ui.event('ChatInput', 'Enter key pressed', { disabled, hasValue: value.trim().length > 0 });
			submit();
		}
	}

	function submit() {
		const text = value.trim();
		if (text && !disabled) {
			logger.ui.event('ChatInput', 'Message submitted', { textLength: text.length });
			onsubmit(text);
			value = '';
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

	let canSubmit = $derived(value.trim().length > 0 && !disabled);

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
</script>

<div class="border-t p-4">
	<div class="flex gap-2 items-end max-w-3xl mx-auto">
		<textarea
			bind:this={textarea}
			bind:value
			onkeydown={handleKeydown}
			oninput={autoResize}
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
</div>
