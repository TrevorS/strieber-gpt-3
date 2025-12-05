<script lang="ts">
	import { Code } from 'lucide-svelte';
	import ToolCallWrapper from './ToolCallWrapper.svelte';
	import type { ResponseCodeInterpreterToolCall } from '$lib/stores/types';

	let { item }: { item: ResponseCodeInterpreterToolCall } = $props();

	// Extract outputs
	let outputs = $derived(item.outputs ?? []);
</script>

<ToolCallWrapper title="Code Interpreter" status={item.status} icon={Code}>
	{#if item.code}
		<pre
			class="overflow-x-auto bg-zinc-900 p-3 text-sm text-zinc-100"><code>{item.code}</code></pre>
	{/if}

	{#if outputs.length > 0}
		<div class="space-y-2 border-t p-3">
			{#each outputs as output}
				{#if output.type === 'logs' && 'logs' in output}
					<pre class="rounded bg-muted p-2 text-sm">{output.logs}</pre>
				{:else if output.type === 'image' && 'url' in output}
					<img
						src={output.url as string}
						alt="Generated output"
						class="max-w-full rounded"
					/>
				{/if}
			{/each}
		</div>
	{/if}
</ToolCallWrapper>
