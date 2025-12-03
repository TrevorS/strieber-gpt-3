<script lang="ts">
	import type { Message } from '$lib/stores/types';
	import { isMessageItem } from '$lib/stores/types';
	import MarkdownContent from './MarkdownContent.svelte';
	import { OutputItemRenderer } from './tools';

	let { message }: { message: Message } = $props();

	// Filter out message items (those are rendered via content/MarkdownContent)
	let toolItems = $derived(
		(message.rawOutput ?? []).filter((item) => !isMessageItem(item))
	);

	let hasContent = $derived(message.content.trim().length > 0);
	let hasToolItems = $derived(toolItems.length > 0);
</script>

<div class="flex justify-start">
	<div class="max-w-[80%] space-y-3">
		<!-- Render tool outputs first (reasoning, web search, code interpreter, etc.) -->
		{#if hasToolItems}
			{#each toolItems as item ('id' in item ? item.id : Math.random())}
				<OutputItemRenderer {item} isStreaming={message.isStreaming} />
			{/each}
		{/if}

		<!-- Render text content -->
		{#if hasContent}
			<div class="rounded-lg bg-muted px-4 py-2">
				<MarkdownContent content={message.content} />
			</div>
		{/if}
	</div>
</div>
