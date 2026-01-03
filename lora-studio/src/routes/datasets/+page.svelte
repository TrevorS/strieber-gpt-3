<script lang="ts">
	import { onMount } from 'svelte';
	import { Plus, FolderOpen, Image, ChevronRight, ArrowLeft } from 'lucide-svelte';
	import { Button } from '$lib/components/ui/button';
	import * as Card from '$lib/components/ui/card';
	import { Badge } from '$lib/components/ui/badge';
	import * as Dialog from '$lib/components/ui/dialog';
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';

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
	let newTriggerToken = $state('');
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
				body: JSON.stringify({
					name: newDatasetName.trim(),
					trigger_token: newTriggerToken.trim() || undefined
				})
			});
			if (!res.ok) {
				const data = await res.json();
				throw new Error(data.message || 'Failed to create dataset');
			}
			showCreateModal = false;
			newDatasetName = '';
			newTriggerToken = '';
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
	<a
		href="/"
		class="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground mb-4"
	>
		<ArrowLeft class="w-4 h-4" />
		LoRA Studio
	</a>

	<header class="flex items-center justify-between mb-8">
		<div>
			<h1 class="text-3xl font-bold text-foreground">Datasets</h1>
			<p class="text-muted-foreground mt-1">Manage training datasets for LoRA training</p>
		</div>
		<Button onclick={() => (showCreateModal = true)}>
			<Plus class="w-4 h-4 mr-2" />
			New Dataset
		</Button>
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
		<Card.Root class="border-dashed">
			<Card.Content class="flex flex-col items-center justify-center py-12">
				<FolderOpen class="w-12 h-12 text-muted-foreground mb-4" />
				<p class="text-muted-foreground mb-4">No datasets yet</p>
				<Button onclick={() => (showCreateModal = true)}>
					Create your first dataset
				</Button>
			</Card.Content>
		</Card.Root>
	{:else}
		<div class="grid gap-4">
			{#each datasets as dataset}
				<a href="/datasets/{dataset.name}" class="block group">
					<Card.Root class="hover:bg-accent transition-colors">
						<Card.Content class="flex items-center justify-between p-4">
							<div class="flex items-center gap-4">
								<div class="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
									<FolderOpen class="w-5 h-5 text-primary" />
								</div>
								<div>
									<h2 class="text-lg font-semibold text-card-foreground group-hover:text-accent-foreground">
										{dataset.name}
									</h2>
									<div class="flex items-center gap-3 text-sm text-muted-foreground mt-1">
										<span class="flex items-center gap-1">
											<Image class="w-4 h-4" />
											{dataset.image_count} images
										</span>
										{#if dataset.has_captions}
											<Badge variant="default" class="bg-green-600 hover:bg-green-600">Captioned</Badge>
										{:else if dataset.image_count > 0}
											<Badge variant="secondary">Needs captions</Badge>
										{/if}
									</div>
								</div>
							</div>
							<ChevronRight class="w-5 h-5 text-muted-foreground group-hover:text-accent-foreground transition-transform group-hover:translate-x-1" />
						</Card.Content>
					</Card.Root>
				</a>
			{/each}
		</div>
	{/if}
</div>

<!-- Create Dataset Modal -->
<Dialog.Root bind:open={showCreateModal}>
	<Dialog.Content class="max-w-md">
		<Dialog.Header>
			<Dialog.Title>Create Dataset</Dialog.Title>
			<Dialog.Description>Create a new dataset for LoRA training</Dialog.Description>
		</Dialog.Header>

		<div class="space-y-4 py-4">
			<div class="space-y-2">
				<Label for="dataset-name">Dataset name</Label>
				<Input
					id="dataset-name"
					bind:value={newDatasetName}
					placeholder="e.g., my_character"
				/>
			</div>
			<div class="space-y-2">
				<Label for="trigger-token">
					Trigger token <span class="text-muted-foreground text-xs">(optional)</span>
				</Label>
				<Input
					id="trigger-token"
					bind:value={newTriggerToken}
					placeholder="e.g., sks, ohwx (defaults to dataset name)"
					onkeydown={(e: KeyboardEvent) => e.key === 'Enter' && createDataset()}
				/>
				<p class="text-xs text-muted-foreground">
					Word prepended to captions. Use something unique that won't appear naturally.
				</p>
			</div>
		</div>

		<Dialog.Footer>
			<Button
				variant="outline"
				onclick={() => {
					showCreateModal = false;
					newDatasetName = '';
					newTriggerToken = '';
				}}
			>
				Cancel
			</Button>
			<Button onclick={createDataset} disabled={creating || !newDatasetName.trim()}>
				{creating ? 'Creating...' : 'Create'}
			</Button>
		</Dialog.Footer>
	</Dialog.Content>
</Dialog.Root>
