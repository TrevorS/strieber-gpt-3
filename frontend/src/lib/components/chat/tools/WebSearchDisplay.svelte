<script lang="ts">
	import { Search, ExternalLink } from 'lucide-svelte';
	import ToolCallWrapper from './ToolCallWrapper.svelte';
	import type { ResponseFunctionWebSearch } from '$lib/stores/types';

	let { item }: { item: ResponseFunctionWebSearch } = $props();

	// Try to extract action data if available (may come from extended event data)
	let action = $derived(() => {
		// The action property may be present on web search calls with results
		const itemWithAction = item as ResponseFunctionWebSearch & {
			action?: {
				type: 'search';
				query?: string;
				sources?: Array<{ url: string; title?: string }>;
			};
		};
		return itemWithAction.action;
	});

	let query = $derived(action()?.query ?? '');
	let sources = $derived(action()?.sources ?? []);
	let title = $derived(query ? `Searched: "${query}"` : 'Web Search');
</script>

<ToolCallWrapper {title} status={item.status} icon={Search}>
	{#if sources.length > 0}
		<ul class="p-3 space-y-2">
			{#each sources as source, i}
				<li class="flex items-start gap-2 text-sm">
					<span class="text-muted-foreground font-mono">[{i + 1}]</span>
					<a
						href={source.url}
						target="_blank"
						rel="noopener noreferrer"
						class="text-blue-600 hover:underline dark:text-blue-400 flex items-center gap-1"
					>
						{source.title || source.url}
						<ExternalLink class="h-3 w-3" />
					</a>
				</li>
			{/each}
		</ul>
	{:else}
		<div class="p-3 text-sm text-muted-foreground">No results</div>
	{/if}
</ToolCallWrapper>
