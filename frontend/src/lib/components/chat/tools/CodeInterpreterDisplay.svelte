<script lang="ts">
	import { Code, Play, CheckCircle, XCircle, Loader2 } from 'lucide-svelte';
	import type { ResponseCodeInterpreterToolCall } from '$lib/stores/types';

	let { item }: { item: ResponseCodeInterpreterToolCall } = $props();

	let isRunning = $derived(item.status === 'in_progress' || item.status === 'interpreting');
	let isCompleted = $derived(item.status === 'completed');
	let isFailed = $derived(item.status === 'failed');

	// Extract outputs
	let outputs = $derived(item.outputs ?? []);
</script>

<div class="overflow-hidden rounded-lg border">
	<!-- Header -->
	<div class="flex items-center gap-2 border-b bg-muted p-2 text-sm">
		<Code class="h-4 w-4" />
		<span>Code Interpreter</span>
		<div class="ml-auto">
			{#if isRunning}
				<Loader2 class="h-4 w-4 animate-spin text-blue-500" />
			{:else if isCompleted}
				<CheckCircle class="h-4 w-4 text-green-600" />
			{:else if isFailed}
				<XCircle class="h-4 w-4 text-red-600" />
			{:else}
				<Play class="h-4 w-4 text-muted-foreground" />
			{/if}
		</div>
	</div>

	<!-- Code block -->
	{#if item.code}
		<pre
			class="overflow-x-auto bg-zinc-900 p-3 text-sm text-zinc-100"><code>{item.code}</code></pre>
	{/if}

	<!-- Outputs -->
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
</div>
