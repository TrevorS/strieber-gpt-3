<script lang="ts">
	import * as Collapsible from '$lib/components/ui/collapsible';
	import { ChevronDown, Loader2, CheckCircle, XCircle } from 'lucide-svelte';
	import type { Snippet } from 'svelte';

	let {
		title,
		status = 'in_progress',
		icon: Icon,
		defaultOpen = false,
		children
	}: {
		title: string;
		status?: 'in_progress' | 'completed' | 'failed' | 'searching' | 'interpreting' | string;
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		icon: any;
		defaultOpen?: boolean;
		children: Snippet;
	} = $props();

	let open = $state(defaultOpen);

	let isLoading = $derived(
		status === 'in_progress' || status === 'searching' || status === 'interpreting'
	);
	let isCompleted = $derived(status === 'completed');
	let isFailed = $derived(status === 'failed' || status === 'incomplete');
</script>

<Collapsible.Root bind:open class="w-full rounded-lg border">
	<Collapsible.Trigger
		class="flex w-full items-center gap-2 p-3 text-sm cursor-pointer hover:bg-muted/50"
	>
		<Icon class="h-4 w-4" />
		<span class="font-medium truncate">{title}</span>
		<div class="ml-auto flex items-center gap-2">
			{#if isLoading}
				<Loader2 class="h-4 w-4 animate-spin text-blue-500" />
			{:else if isCompleted}
				<CheckCircle class="h-4 w-4 text-green-600" />
			{:else if isFailed}
				<XCircle class="h-4 w-4 text-red-600" />
			{/if}
			<ChevronDown class="h-4 w-4 transition-transform {open ? 'rotate-180' : ''}" />
		</div>
	</Collapsible.Trigger>
	<Collapsible.Content class="border-t">
		{@render children()}
	</Collapsible.Content>
</Collapsible.Root>
