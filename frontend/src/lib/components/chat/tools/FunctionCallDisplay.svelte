<script lang="ts">
	import * as Collapsible from '$lib/components/ui/collapsible';
	import { Wrench, ChevronDown, Loader2, CheckCircle, XCircle } from 'lucide-svelte';
	import type { ResponseFunctionToolCall } from '$lib/stores/types';

	let { item, isStreaming = false }: { item: ResponseFunctionToolCall; isStreaming?: boolean } =
		$props();

	let open = $state(false);

	// Parse arguments for display
	let formattedArgs = $derived(() => {
		try {
			const parsed = JSON.parse(item.arguments);
			return JSON.stringify(parsed, null, 2);
		} catch {
			return item.arguments;
		}
	});

	// Determine status (function calls don't have explicit status, infer from streaming)
	let statusIcon = $derived(() => {
		if (isStreaming) return 'loading';
		return 'completed';
	});
</script>

<Collapsible.Root bind:open class="rounded-lg border">
	<!-- Header -->
	<Collapsible.Trigger class="flex w-full items-center gap-2 p-3 text-sm hover:bg-muted/50">
		<Wrench class="h-4 w-4" />
		<span class="font-medium">{item.name}</span>
		<div class="ml-auto flex items-center gap-2">
			{#if statusIcon() === 'loading'}
				<Loader2 class="h-4 w-4 animate-spin text-blue-500" />
			{:else}
				<CheckCircle class="h-4 w-4 text-green-600" />
			{/if}
			<ChevronDown class="h-4 w-4 transition-transform {open ? 'rotate-180' : ''}" />
		</div>
	</Collapsible.Trigger>

	<!-- Arguments -->
	<Collapsible.Content class="border-t">
		<div class="p-3">
			<div class="mb-1 text-xs text-muted-foreground">Arguments:</div>
			<pre
				class="overflow-x-auto rounded bg-muted p-2 text-xs">{formattedArgs()}</pre>
		</div>
	</Collapsible.Content>
</Collapsible.Root>
