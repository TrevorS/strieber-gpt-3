<script lang="ts">
	import { RefreshCw, Copy, Check } from 'lucide-svelte';
	import { Button } from '$lib/components/ui/button';
	import type { Message } from '$lib/stores/types';
	import { isMessageItem } from '$lib/stores/types';
	import MarkdownContent from './MarkdownContent.svelte';
	import CitationList from './CitationList.svelte';
	import { OutputItemRenderer } from './tools';
	import { extractCitations, getUniqueCitations } from '$lib/utils/citations';

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

	// Extract citations from raw output (only when not streaming for stability)
	let citations = $derived(
		!message.isStreaming && message.rawOutput ? extractCitations(message.rawOutput) : []
	);
	let uniqueCitations = $derived(getUniqueCitations(citations));

	let hasContent = $derived(message.content.trim().length > 0);
	let hasToolItems = $derived(toolItems.length > 0);
	let showRegenerate = $derived(isLast && canRegenerate && !message.isStreaming);
	let showActions = $derived(hasContent && !message.isStreaming);

	// Show content container if there's content OR if streaming (to show loading state)
	let showContentContainer = $derived(hasContent || message.isStreaming);

	// Copy message state
	let copied = $state(false);

	async function copyMessage() {
		await navigator.clipboard.writeText(message.content);
		copied = true;
		setTimeout(() => (copied = false), 2000);
	}
</script>

<div class="group flex flex-col items-start space-y-3">
	<!-- Render tool outputs first (reasoning, web search, code interpreter, etc.) -->
	<!-- Tools get full width of the container -->
	{#if hasToolItems}
		<div class="w-full space-y-3">
			{#each toolItems as item ('id' in item ? item.id : Math.random())}
				<OutputItemRenderer {item} isStreaming={message.isStreaming} />
			{/each}
		</div>
	{/if}

	<!-- Render text content or streaming placeholder -->
	<!-- Text content is limited to 80% width -->
	{#if showContentContainer}
		<div class="max-w-[80%] xl:max-w-[85%] 2xl:max-w-[90%] rounded-lg bg-muted px-4 py-2">
			{#if hasContent}
				<MarkdownContent content={message.content} {citations} />
				<!-- Citation list appears after content, inside the container -->
				{#if uniqueCitations.length > 0}
					<CitationList citations={uniqueCitations} />
				{/if}
			{:else}
				<span class="inline-block w-2 h-4 bg-foreground/30 animate-pulse rounded-sm"></span>
			{/if}
		</div>
	{/if}

	<!-- Action buttons (shown on hover) -->
	{#if showActions}
		<div class="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
			<Button
				variant="ghost"
				size="sm"
				onclick={copyMessage}
				class="text-muted-foreground hover:text-foreground"
				data-testid="copy-button"
			>
				{#if copied}
					<Check class="h-4 w-4 mr-1" />
					Copied
				{:else}
					<Copy class="h-4 w-4 mr-1" />
					Copy
				{/if}
			</Button>
			{#if showRegenerate}
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
			{/if}
		</div>
	{/if}
</div>
