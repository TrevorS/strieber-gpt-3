<script lang="ts">
	import { renderMarkdown } from '$lib/utils/markdown';
	import { transformCitationMarkers, type Citation } from '$lib/utils/citations';

	let { content, citations = [] }: { content: string; citations?: Citation[] } = $props();

	// Transform citation markers before rendering markdown
	let transformedContent = $derived(
		citations.length > 0 ? transformCitationMarkers(content, citations) : content
	);

	let html = $derived(renderMarkdown(transformedContent));
</script>

<div class="prose prose-sm dark:prose-invert max-w-none prose-pre:bg-muted prose-pre:overflow-x-auto prose-pre:border prose-pre:border-border prose-pre:rounded-lg">
	{@html html}
</div>

<style>
	/* Styling for inline citation links */
	:global(.citation-link) {
		font-size: 0.75rem;
		vertical-align: super;
		line-height: 0;
	}

	:global(.citation-link a) {
		color: var(--primary);
		text-decoration: none;
		font-weight: 500;
	}

	:global(.citation-link a:hover) {
		text-decoration: underline;
	}
</style>
