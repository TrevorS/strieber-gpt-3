<script lang="ts">
	import { renderMarkdown } from '$lib/utils/markdown';

	let { content }: { content: string } = $props();

	let html = $derived(renderMarkdown(content));
</script>

<div class="prose prose-sm dark:prose-invert max-w-none prose-pre:bg-zinc-900 prose-pre:text-zinc-100">
	{@html html}
</div>
