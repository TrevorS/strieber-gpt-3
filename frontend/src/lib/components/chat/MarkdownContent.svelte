<script lang="ts">
	import { renderMarkdown } from '$lib/utils/markdown';

	let { content }: { content: string } = $props();

	let html = $derived(renderMarkdown(content));
</script>

<div class="prose prose-sm dark:prose-invert max-w-none prose-pre:bg-muted prose-pre:overflow-x-auto prose-pre:border prose-pre:border-border prose-pre:rounded-lg">
	{@html html}
</div>
