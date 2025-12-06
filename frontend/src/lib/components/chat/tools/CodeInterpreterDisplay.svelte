<script lang="ts">
	import { Code, Copy, Check } from 'lucide-svelte';
	import ToolCallWrapper from './ToolCallWrapper.svelte';
	import type { ResponseCodeInterpreterToolCall } from '$lib/stores/types';

	let { item }: { item: ResponseCodeInterpreterToolCall } = $props();

	// Extract outputs
	let outputs = $derived(item.outputs ?? []);

	// Copy button state
	let copied = $state(false);

	async function copyCode() {
		if (!item.code) return;
		await navigator.clipboard.writeText(item.code);
		copied = true;
		setTimeout(() => (copied = false), 2000);
	}
</script>

<ToolCallWrapper title="Code Interpreter" status={item.status} icon={Code}>
	{#if item.code}
		<div class="code-interpreter-block">
			<div class="code-interpreter-header">
				<span class="code-interpreter-lang">python</span>
				<button
					class="code-interpreter-copy"
					class:copied
					onclick={copyCode}
					title="Copy code"
				>
					{#if copied}
						<Check class="h-3.5 w-3.5" />
					{:else}
						<Copy class="h-3.5 w-3.5" />
					{/if}
				</button>
			</div>
			<pre class="overflow-x-auto bg-zinc-900 p-3 text-sm text-zinc-100"><code>{item.code}</code></pre>
		</div>
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

<style>
	.code-interpreter-block {
		overflow: hidden;
		border-radius: 0.5rem;
	}

	.code-interpreter-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 0.5rem 0.75rem;
		background: #27272a; /* zinc-800 */
		border-bottom: 1px solid #3f3f46; /* zinc-700 */
	}

	.code-interpreter-lang {
		font-size: 0.75rem;
		font-weight: 500;
		color: #a1a1aa; /* zinc-400 */
		text-transform: lowercase;
	}

	.code-interpreter-copy {
		padding: 0.25rem;
		border-radius: 0.25rem;
		background: transparent;
		border: none;
		cursor: pointer;
		opacity: 0.6;
		transition: opacity 0.2s, background-color 0.2s;
		color: #a1a1aa; /* zinc-400 */
		display: flex;
		align-items: center;
		justify-content: center;
	}

	.code-interpreter-copy:hover {
		opacity: 1;
		background: #3f3f46; /* zinc-700 */
		color: #fafafa; /* zinc-50 */
	}

	.code-interpreter-copy.copied {
		opacity: 1;
		color: #22c55e; /* green-500 */
	}

	.code-interpreter-block pre {
		margin: 0;
		border-radius: 0 0 0.5rem 0.5rem;
	}
</style>
