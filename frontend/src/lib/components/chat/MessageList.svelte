<script lang="ts">
	import { untrack } from 'svelte';
	import type { Message } from '$lib/stores/types';
	import UserMessage from './UserMessage.svelte';
	import AssistantMessage from './AssistantMessage.svelte';

	let {
		messages,
		canRegenerate = false,
		onregenerate
	}: {
		messages: Message[];
		canRegenerate?: boolean;
		onregenerate?: () => void;
	} = $props();

	let container: HTMLDivElement;
	let isAtBottom = $state(true);
	const SCROLL_THRESHOLD = 100;

	function handleScroll() {
		if (!container) return;
		const { scrollTop, scrollHeight, clientHeight } = container;
		isAtBottom = scrollHeight - scrollTop - clientHeight < SCROLL_THRESHOLD;
	}

	$effect(() => {
		// Track values that actually change (props don't auto-track)
		const len = messages.length;
		const lastMsg = messages[len - 1];
		const _lastContent = lastMsg?.content;
		const _lastStreaming = lastMsg?.isStreaming;

		if (container && untrack(() => isAtBottom)) {
			// Use requestAnimationFrame to ensure DOM has updated
			requestAnimationFrame(() => {
				if (container) {
					container.scrollTop = container.scrollHeight;
				}
			});
		}
	});

	// Find the last assistant message index
	function isLastAssistantMessage(index: number): boolean {
		// Find the last assistant message in the list
		for (let i = messages.length - 1; i >= 0; i--) {
			if (messages[i].role === 'assistant') {
				return i === index;
			}
		}
		return false;
	}
</script>

<div bind:this={container} onscroll={handleScroll} class="flex-1 overflow-y-auto p-4">
	<div class="max-w-3xl mx-auto space-y-4">
		{#each messages as message, index (message.id)}
			{#if message.role === 'user'}
				<UserMessage {message} />
			{:else}
				<AssistantMessage
					{message}
					isLast={isLastAssistantMessage(index)}
					{canRegenerate}
					{onregenerate}
				/>
			{/if}
		{/each}
	</div>
</div>
