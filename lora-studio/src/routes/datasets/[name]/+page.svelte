<script lang="ts">
	import { onMount } from 'svelte';
	import { page } from '$app/stores';
	import { goto } from '$app/navigation';
	import { ArrowLeft, Plus, Sparkles, Trash2, X } from 'lucide-svelte';

	interface DatasetImage {
		filename: string;
		path: string;
		caption: string | null;
		caption_file: string | null;
	}

	interface DatasetInfo {
		name: string;
		path: string;
		images: DatasetImage[];
	}

	let dataset = $state<DatasetInfo | null>(null);
	let loading = $state(true);
	let error = $state<string | null>(null);

	// Add images state
	let showAddModal = $state(false);
	let imageUrls = $state('');
	let adding = $state(false);

	// Caption state
	let captioning = $state(false);
	let captionStyle = $state<'tags' | 'natural' | 'booru'>('tags');

	// Delete state
	let showDeleteConfirm = $state(false);
	let deleting = $state(false);

	$effect(() => {
		const name = $page.params.name;
		if (name) loadDataset(name);
	});

	async function loadDataset(name: string) {
		loading = true;
		error = null;
		try {
			const res = await fetch(`/api/datasets/${name}`);
			if (!res.ok) throw new Error('Failed to load dataset');
			dataset = await res.json();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load dataset';
		} finally {
			loading = false;
		}
	}

	async function addImages() {
		if (!imageUrls.trim() || !dataset) return;
		adding = true;
		error = null;
		try {
			const urls = imageUrls
				.split('\n')
				.map((u) => u.trim())
				.filter((u) => u);
			const res = await fetch(`/api/datasets/${dataset.name}/images`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ urls })
			});
			if (!res.ok) throw new Error('Failed to add images');
			showAddModal = false;
			imageUrls = '';
			await loadDataset(dataset.name);
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to add images';
		} finally {
			adding = false;
		}
	}

	async function captionImages() {
		if (!dataset) return;
		captioning = true;
		error = null;
		try {
			const res = await fetch(`/api/datasets/${dataset.name}/caption`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ style: captionStyle, overwrite: false })
			});
			if (!res.ok) throw new Error('Failed to caption images');
			await loadDataset(dataset.name);
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to caption images';
		} finally {
			captioning = false;
		}
	}

	async function deleteDataset() {
		if (!dataset) return;
		deleting = true;
		try {
			const res = await fetch(`/api/datasets/${dataset.name}`, { method: 'DELETE' });
			if (!res.ok) throw new Error('Failed to delete dataset');
			goto('/datasets');
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to delete dataset';
			deleting = false;
		}
	}

	function getImageUrl(image: DatasetImage): string {
		// TODO: Add image serving endpoint
		return `/api/datasets/${dataset?.name}/images/${image.filename}`;
	}

	onMount(() => {});
</script>

<div class="container mx-auto px-4 py-8">
	<a
		href="/datasets"
		class="inline-flex items-center gap-2 text-muted-foreground hover:text-foreground mb-6"
	>
		<ArrowLeft class="w-4 h-4" />
		Back to datasets
	</a>

	{#if loading}
		<div class="flex items-center justify-center py-12">
			<div class="text-muted-foreground">Loading dataset...</div>
		</div>
	{:else if error}
		<div class="bg-destructive/10 text-destructive px-4 py-3 rounded-lg mb-4">
			{error}
			<button onclick={() => dataset && loadDataset(dataset.name)} class="underline ml-2"
				>Retry</button
			>
		</div>
	{:else if dataset}
		<header class="flex items-center justify-between mb-8">
			<div>
				<h1 class="text-3xl font-bold text-foreground">{dataset.name}</h1>
				<p class="text-muted-foreground mt-1">{dataset.images.length} images</p>
			</div>
			<div class="flex items-center gap-3">
				<button
					onclick={() => (showAddModal = true)}
					class="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90"
				>
					<Plus class="w-5 h-5" />
					Add Images
				</button>
				<button
					onclick={captionImages}
					disabled={captioning || dataset.images.length === 0}
					class="flex items-center gap-2 px-4 py-2 bg-secondary text-secondary-foreground rounded-lg hover:bg-secondary/80 disabled:opacity-50"
				>
					<Sparkles class="w-5 h-5" />
					{captioning ? 'Captioning...' : 'Auto-Caption'}
				</button>
				<button
					onclick={() => (showDeleteConfirm = true)}
					class="flex items-center gap-2 px-4 py-2 text-destructive hover:bg-destructive/10 rounded-lg"
				>
					<Trash2 class="w-5 h-5" />
				</button>
			</div>
		</header>

		{#if dataset.images.length === 0}
			<div class="text-center py-12 border border-dashed border-border rounded-lg">
				<p class="text-muted-foreground mb-4">No images in this dataset</p>
				<button
					onclick={() => (showAddModal = true)}
					class="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90"
				>
					Add images
				</button>
			</div>
		{:else}
			<div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
				{#each dataset.images as image}
					<div class="border border-border rounded-lg overflow-hidden bg-card">
						<div class="aspect-square bg-muted">
							<img
								src="/api/datasets/{dataset.name}/images/{image.filename}"
								alt={image.caption || image.filename}
								class="w-full h-full object-cover"
								loading="lazy"
							/>
						</div>
						<div class="p-3">
							{#if image.caption}
								<p class="text-sm text-foreground line-clamp-2">{image.caption}</p>
							{:else}
								<p class="text-sm text-muted-foreground italic">No caption</p>
							{/if}
						</div>
					</div>
				{/each}
			</div>
		{/if}
	{/if}
</div>

<!-- Add Images Modal -->
{#if showAddModal}
	<div class="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
		<div class="bg-card border border-border rounded-lg p-6 w-full max-w-lg mx-4">
			<div class="flex items-center justify-between mb-4">
				<h2 class="text-xl font-semibold text-card-foreground">Add Images</h2>
				<button onclick={() => (showAddModal = false)} class="text-muted-foreground hover:text-foreground">
					<X class="w-5 h-5" />
				</button>
			</div>
			<p class="text-sm text-muted-foreground mb-3">Enter image URLs, one per line:</p>
			<textarea
				bind:value={imageUrls}
				placeholder="https://example.com/image1.jpg&#10;https://example.com/image2.jpg"
				rows="6"
				class="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring mb-4 font-mono text-sm"
			></textarea>
			<div class="flex justify-end gap-3">
				<button
					onclick={() => {
						showAddModal = false;
						imageUrls = '';
					}}
					class="px-4 py-2 text-muted-foreground hover:text-foreground"
				>
					Cancel
				</button>
				<button
					onclick={addImages}
					disabled={adding || !imageUrls.trim()}
					class="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 disabled:opacity-50"
				>
					{adding ? 'Adding...' : 'Add Images'}
				</button>
			</div>
		</div>
	</div>
{/if}

<!-- Delete Confirmation Modal -->
{#if showDeleteConfirm}
	<div class="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
		<div class="bg-card border border-border rounded-lg p-6 w-full max-w-md mx-4">
			<h2 class="text-xl font-semibold text-card-foreground mb-4">Delete Dataset?</h2>
			<p class="text-muted-foreground mb-6">
				This will permanently delete <strong>{dataset?.name}</strong> and all its images.
			</p>
			<div class="flex justify-end gap-3">
				<button
					onclick={() => (showDeleteConfirm = false)}
					class="px-4 py-2 text-muted-foreground hover:text-foreground"
				>
					Cancel
				</button>
				<button
					onclick={deleteDataset}
					disabled={deleting}
					class="px-4 py-2 bg-destructive text-white rounded-lg hover:bg-destructive/90 disabled:opacity-50"
				>
					{deleting ? 'Deleting...' : 'Delete'}
				</button>
			</div>
		</div>
	</div>
{/if}
