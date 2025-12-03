<script lang="ts">
	import { Search, ExternalLink, Loader2 } from 'lucide-svelte';
	import type { ResponseFunctionWebSearch } from '$lib/stores/types';

	let { item }: { item: ResponseFunctionWebSearch } = $props();

	let isSearching = $derived(item.status === 'in_progress' || item.status === 'searching');

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
</script>

<div class="rounded-lg border p-3 space-y-2">
	<div class="flex items-center gap-2 text-sm">
		{#if isSearching}
			<Loader2 class="h-4 w-4 animate-spin" />
			<span class="text-muted-foreground">Searching the web...</span>
		{:else}
			<Search class="h-4 w-4" />
			{#if query}
				<span>Searched: "{query}"</span>
			{:else}
				<span>Web search completed</span>
			{/if}
		{/if}
	</div>

	{#if sources.length > 0}
		<ul class="space-y-1 pl-6 text-sm">
			{#each sources as source}
				<li>
					<a
						href={source.url}
						target="_blank"
						rel="noopener noreferrer"
						class="inline-flex items-center gap-1 text-blue-600 hover:underline dark:text-blue-400"
					>
						{source.title || source.url}
						<ExternalLink class="h-3 w-3" />
					</a>
				</li>
			{/each}
		</ul>
	{/if}
</div>
