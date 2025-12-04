<script lang="ts">
	import { onMount } from 'svelte';
	import { ChevronDown, Loader2 } from 'lucide-svelte';
	import { fetchModels, type Model } from '$lib/api/models';
	import { settingsStore } from '$lib/stores/settings.svelte';

	let models = $state<Model[]>([]);
	let loading = $state(true);
	let open = $state(false);

	onMount(async () => {
		const start = Date.now();
		models = await fetchModels();
		// Sync to settings store for capability checks
		settingsStore.setModels(models);
		// Minimum loading time to prevent flash
		const elapsed = Date.now() - start;
		const remaining = Math.max(0, 350 - elapsed);
		setTimeout(() => {
			loading = false;
		}, remaining);
	});

	function selectModel(modelId: string) {
		settingsStore.setModel(modelId);
		open = false;
	}

	function handleKeydown(event: KeyboardEvent) {
		if (event.key === 'Escape') {
			open = false;
		}
	}
</script>

<svelte:window onkeydown={handleKeydown} />

<div class="relative" data-testid="model-selector">
	<button
		onclick={() => (open = !open)}
		class="flex items-center gap-2 px-3 py-1.5 text-sm rounded-md border bg-background hover:bg-accent transition-colors"
		aria-haspopup="listbox"
		aria-expanded={open}
		disabled={loading}
		data-testid="model-selector-trigger"
	>
		{#if loading}
			<Loader2 class="h-4 w-4 animate-spin" />
			<span class="text-muted-foreground">Loading...</span>
		{:else}
			<span class="max-w-[150px] truncate">{settingsStore.selectedModel}</span>
			<ChevronDown class="h-4 w-4 text-muted-foreground" />
		{/if}
	</button>

	{#if open && !loading}
		<div
			class="absolute top-full left-0 mt-1 w-64 max-h-60 overflow-y-auto bg-popover border rounded-md shadow-lg z-50"
			role="listbox"
			data-testid="model-selector-dropdown"
		>
			{#if models.length === 0}
				<div class="px-3 py-2 text-sm text-muted-foreground">No models available</div>
			{:else}
				{#each models as model (model.id)}
					<button
						onclick={() => selectModel(model.id)}
						class="w-full px-3 py-2 text-left text-sm hover:bg-accent transition-colors
							{model.id === settingsStore.selectedModel ? 'bg-accent/50 font-medium' : ''}"
						role="option"
						aria-selected={model.id === settingsStore.selectedModel}
						data-testid="model-option"
					>
						{model.id}
					</button>
				{/each}
			{/if}
		</div>
	{/if}
</div>

{#if open}
	<!-- Backdrop to close dropdown when clicking outside -->
	<div
		class="fixed inset-0 z-40"
		onclick={() => (open = false)}
		onkeydown={(e) => e.key === 'Enter' && (open = false)}
		role="button"
		tabindex="-1"
		aria-label="Close model selector"
	></div>
{/if}
