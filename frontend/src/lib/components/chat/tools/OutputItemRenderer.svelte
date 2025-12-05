<script lang="ts">
	import {
		type ResponseOutputItem,
		isReasoningItem,
		isWebSearchItem,
		isCodeInterpreterItem,
		isFunctionCallItem,
		isMessageItem
	} from '$lib/stores/types';
	import ReasoningDisplay from './ReasoningDisplay.svelte';
	import WebSearchDisplay from './WebSearchDisplay.svelte';
	import CodeInterpreterDisplay from './CodeInterpreterDisplay.svelte';
	import FunctionCallDisplay from './FunctionCallDisplay.svelte';

	let { item, isStreaming = false }: { item: ResponseOutputItem; isStreaming?: boolean } =
		$props();
</script>

{#if isReasoningItem(item)}
	<ReasoningDisplay {item} {isStreaming} />
{:else if isWebSearchItem(item)}
	<WebSearchDisplay {item} />
{:else if isCodeInterpreterItem(item)}
	<CodeInterpreterDisplay {item} />
{:else if isFunctionCallItem(item)}
	<FunctionCallDisplay {item} />
{:else if isMessageItem(item)}
	<!-- Message items are handled by MarkdownContent in the parent, skip here -->
{:else}
	<!-- Unknown item type - show debug info in development -->
	{#if import.meta.env.DEV}
		<div class="rounded border border-dashed border-yellow-500 bg-yellow-50 p-2 text-xs dark:bg-yellow-950">
			<span class="font-medium">Unknown output item type:</span>
			<code class="ml-1">{item.type}</code>
		</div>
	{/if}
{/if}
