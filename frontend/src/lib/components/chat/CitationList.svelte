<script lang="ts">
	import type { Citation } from '$lib/utils/citations';

	let { citations }: { citations: Citation[] } = $props();
</script>

{#if citations.length > 0}
	<div class="mt-4 pt-3 border-t border-border">
		<h4 class="text-xs font-medium text-muted-foreground mb-2">Sources</h4>
		<ol class="list-none space-y-1.5 text-sm">
			{#each citations as citation (citation.url)}
				<li class="flex items-start gap-2">
					<span class="text-muted-foreground font-mono text-xs mt-0.5 shrink-0">
						[{citation.index}]
					</span>
					<a
						href={citation.url}
						target="_blank"
						rel="noopener noreferrer"
						class="text-primary hover:underline break-words min-w-0"
						title={citation.url}
					>
						{citation.title || citation.url}
					</a>
				</li>
			{/each}
		</ol>
	</div>
{/if}
