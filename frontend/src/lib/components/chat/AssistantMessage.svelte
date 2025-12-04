<script lang="ts">
	import { RefreshCw } from 'lucide-svelte';
	import { Button } from '$lib/components/ui/button';
	import type { Message } from '$lib/stores/types';
	import { isMessageItem } from '$lib/stores/types';
	import MarkdownContent from './MarkdownContent.svelte';
	import { OutputItemRenderer } from './tools';

	let {
		message,
		isLast = false,
		canRegenerate = false,
		onregenerate
	}: {
		message: Message;
		isLast?: boolean;
		canRegenerate?: boolean;
		onregenerate?: () => void;
	} = $props();

	// Filter out message items (those are rendered via content/MarkdownContent)
	let toolItems = $derived((message.rawOutput ?? []).filter((item) => !isMessageItem(item)));

	let hasContent = $derived(message.content.trim().length > 0);
	let hasToolItems = $derived(toolItems.length > 0);
	let showRegenerate = $derived(isLast && canRegenerate && !message.isStreaming);

	// Show content container if there's content OR if streaming (to show loading state)
	let showContentContainer = $derived(hasContent || message.isStreaming);
</script>

<div class="group flex justify-start">
	<div class="max-w-[80%] space-y-3">
		<!-- Render tool outputs first (reasoning, web search, code interpreter, etc.) -->
		{#if hasToolItems}
			{#each toolItems as item ('id' in item ? item.id : Math.random())}
				<OutputItemRenderer {item} isStreaming={message.isStreaming} />
			{/each}
		{/if}

		<!-- Render text content or streaming placeholder -->
		{#if showContentContainer}
			<div class="rounded-lg bg-muted px-4 py-2">
				{#if hasContent}
					<MarkdownContent content={message.content} />
				{:else}
					<span class="inline-block w-2 h-4 bg-foreground/30 animate-pulse rounded-sm"></span>
				{/if}
			</div>
		{/if}

		<!-- Regenerate button (shown on hover for last message) -->
		{#if showRegenerate}
			<div class="opacity-0 group-hover:opacity-100 transition-opacity">
				<Button
					variant="ghost"
					size="sm"
					onclick={() => onregenerate?.()}
					class="text-muted-foreground hover:text-foreground"
					data-testid="regenerate-button"
				>
					<RefreshCw class="h-4 w-4 mr-1" />
					Regenerate
				</Button>
			</div>
		{/if}
	</div>
</div>
