<script lang="ts">
	import { Wrench } from 'lucide-svelte';
	import hljs from 'highlight.js/lib/core';
	import json from 'highlight.js/lib/languages/json';
	import ToolCallWrapper from './ToolCallWrapper.svelte';
	import type { ResponseFunctionToolCall } from '$lib/stores/types';

	// Register JSON language for highlighting
	hljs.registerLanguage('json', json);

	let { item }: { item: ResponseFunctionToolCall } = $props();

	// Parse and format arguments
	let formattedArgs = $derived(() => {
		try {
			const parsed = JSON.parse(item.arguments);
			return JSON.stringify(parsed, null, 2);
		} catch {
			return item.arguments;
		}
	});

	// Highlight the formatted JSON
	let highlightedArgs = $derived(() => {
		const formatted = formattedArgs();
		try {
			return hljs.highlight(formatted, { language: 'json' }).value;
		} catch {
			return formatted;
		}
	});
</script>

<ToolCallWrapper title={item.name} status={item.status} icon={Wrench}>
	<div class="p-3">
		<div class="mb-1 text-xs text-muted-foreground">Arguments:</div>
		<pre class="overflow-x-auto rounded bg-muted p-2 text-xs hljs"><code>{@html highlightedArgs()}</code></pre>
	</div>
</ToolCallWrapper>
