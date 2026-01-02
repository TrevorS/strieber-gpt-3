<script lang="ts">
	import { onMount } from 'svelte';
	import { Plus, FolderOpen, Image, ChevronRight } from 'lucide-svelte';

	interface Dataset {
		name: string;
		path: string;
		image_count: number;
		has_captions: boolean;
	}

	let datasets = $state<Dataset[]>([]);
	let loading = $state(true);
	let error = $state<string | null>(null);
	let showCreateModal = $state(false);
	let newDatasetName = $state('');
	let creating = $state(false);

	async function loadDatasets() {
		loading = true;
		error = null;
		try {
			const res = await fetch('/api/datasets');
			if (!res.ok) throw new Error('Failed to load datasets');
			datasets = await res.json();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load datasets';
		} finally {
			loading = false;
		}
	}

	async function createDataset() {
		if (!newDatasetName.trim()) return;
		creating = true;
		try {
			const res = await fetch('/api/datasets', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ name: newDatasetName.trim() })
			});
			if (!res.ok) {
				const data = await res.json();
				throw new Error(data.message || 'Failed to create dataset');
			}
			showCreateModal = false;
			newDatasetName = '';
			await loadDatasets();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to create dataset';
		} finally {
			creating = false;
		}
	}

	onMount(loadDatasets);
</script>

<div class="container mx-auto px-4 py-8">
	<header class="flex items-center justify-between mb-8">
		<div>
			<h1 class="text-3xl font-bold text-foreground">Datasets</h1>
			<p class="text-muted-foreground mt-1">Manage training datasets for LoRA training</p>
		</div>
		<button
			onclick={() => (showCreateModal = true)}
			class="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors"
		>
			<Plus class="w-5 h-5" />
			New Dataset
		</button>
	</header>

	{#if loading}
		<div class="flex items-center justify-center py-12">
			<div class="text-muted-foreground">Loading datasets...</div>
		</div>
	{:else if error}
		<div class="bg-destructive/10 text-destructive px-4 py-3 rounded-lg mb-4">
			{error}
			<button onclick={loadDatasets} class="underline ml-2">Retry</button>
		</div>
	{:else if datasets.length === 0}
		<div class="text-center py-12 border border-dashed border-border rounded-lg">
			<FolderOpen class="w-12 h-12 text-muted-foreground mx-auto mb-4" />
			<p class="text-muted-foreground mb-4">No datasets yet</p>
			<button
				onclick={() => (showCreateModal = true)}
				class="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90"
			>
				Create your first dataset
			</button>
		</div>
	{:else}
		<div class="grid gap-4">
			{#each datasets as dataset}
				<a
					href="/datasets/{dataset.name}"
					class="flex items-center justify-between p-4 bg-card border border-border rounded-lg hover:bg-accent transition-colors group"
				>
					<div class="flex items-center gap-4">
						<FolderOpen class="w-8 h-8 text-primary" />
						<div>
							<h2 class="text-lg font-semibold text-card-foreground group-hover:text-accent-foreground">
								{dataset.name}
							</h2>
							<div class="flex items-center gap-3 text-sm text-muted-foreground">
								<span class="flex items-center gap-1">
									<Image class="w-4 h-4" />
									{dataset.image_count} images
								</span>
								{#if dataset.has_captions}
									<span class="text-green-600">Captioned</span>
								{:else if dataset.image_count > 0}
									<span class="text-yellow-600">Needs captions</span>
								{/if}
							</div>
						</div>
					</div>
					<ChevronRight class="w-5 h-5 text-muted-foreground group-hover:text-accent-foreground" />
				</a>
			{/each}
		</div>
	{/if}
</div>

<!-- Create Dataset Modal -->
{#if showCreateModal}
	<div class="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
		<div class="bg-card border border-border rounded-lg p-6 w-full max-w-md mx-4">
			<h2 class="text-xl font-semibold text-card-foreground mb-4">Create Dataset</h2>
			<input
				type="text"
				bind:value={newDatasetName}
				placeholder="Dataset name"
				class="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring mb-4"
				onkeydown={(e) => e.key === 'Enter' && createDataset()}
			/>
			<div class="flex justify-end gap-3">
				<button
					onclick={() => {
						showCreateModal = false;
						newDatasetName = '';
					}}
					class="px-4 py-2 text-muted-foreground hover:text-foreground transition-colors"
				>
					Cancel
				</button>
				<button
					onclick={createDataset}
					disabled={creating || !newDatasetName.trim()}
					class="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 disabled:opacity-50 transition-colors"
				>
					{creating ? 'Creating...' : 'Create'}
				</button>
			</div>
		</div>
	</div>
{/if}
