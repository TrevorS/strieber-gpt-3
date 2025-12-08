<script lang="ts">
	import { Wrench } from 'lucide-svelte';
	import hljs from 'highlight.js/lib/core';
	import json from 'highlight.js/lib/languages/json';
	import ToolCallWrapper from './ToolCallWrapper.svelte';
	import type { ResponseFunctionToolCall } from '$lib/stores/types';

	// Register JSON language for highlighting
	hljs.registerLanguage('json', json);

	// Extend the type to include optional output field (added by backend)
	type FunctionCallWithOutput = ResponseFunctionToolCall & { output?: string };

	let { item }: { item: FunctionCallWithOutput } = $props();

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

	// Parse and format output (if available)
	let formattedOutput = $derived(() => {
		if (!item.output) return null;
		try {
			const parsed = JSON.parse(item.output);
			return JSON.stringify(parsed, null, 2);
		} catch {
			return item.output;
		}
	});

	// Highlight the formatted output
	let highlightedOutput = $derived(() => {
		const formatted = formattedOutput();
		if (!formatted) return null;
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
		{#if highlightedOutput()}
			<div class="mt-3 mb-1 text-xs text-muted-foreground">Output:</div>
			<pre class="overflow-x-auto rounded bg-muted p-2 text-xs hljs max-h-60 overflow-y-auto"><code>{@html highlightedOutput()}</code></pre>
		{/if}
	</div>
</ToolCallWrapper>
