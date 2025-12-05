<script lang="ts">
	import * as Collapsible from '$lib/components/ui/collapsible';
	import { Brain, ChevronDown } from 'lucide-svelte';
	import type { ResponseReasoningItem } from '$lib/stores/types';

	let { item, isStreaming = false }: { item: ResponseReasoningItem; isStreaming?: boolean } =
		$props();

	let open = $state(false);

	// Extract text content from the reasoning item
	let reasoningText = $derived(() => {
		if (!item.content) return '';
		return item.content.map((c) => c.text).join('');
	});

	// Extract summary text if available
	let summaryText = $derived(() => {
		if (!item.summary) return '';
		return item.summary.map((s) => s.text).join('');
	});

	// Determine status label
	let statusLabel = $derived(isStreaming ? 'Thinking...' : 'Reasoning');
</script>

<Collapsible.Root bind:open class="rounded-lg border bg-muted/30 p-1">
	<Collapsible.Trigger
		class="flex w-full items-center gap-2 p-4 text-sm text-muted-foreground hover:bg-muted/50 rounded-lg"
	>
		<Brain class="h-4 w-4" />
		<span>{statusLabel}</span>
		{#if summaryText()}
			<span class="ml-1 truncate text-xs opacity-70">- {summaryText()}</span>
		{/if}
		<ChevronDown class="ml-auto h-4 w-4 transition-transform {open ? 'rotate-180' : ''}" />
	</Collapsible.Trigger>
	<Collapsible.Content class="px-4 pb-4 pt-2">
		<p class="whitespace-pre-wrap text-sm text-muted-foreground">{reasoningText()}</p>
	</Collapsible.Content>
</Collapsible.Root>
