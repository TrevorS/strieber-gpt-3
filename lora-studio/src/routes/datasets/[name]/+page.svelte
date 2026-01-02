<script lang="ts">
	import { onMount } from 'svelte';
	import { page } from '$app/stores';
	import { goto } from '$app/navigation';
	import { ArrowLeft, Plus, Sparkles, Trash2, X, Check, Pencil, RefreshCw, Eraser } from 'lucide-svelte';

	interface DatasetImage {
		filename: string;
		path: string;
		caption: string | null;
		caption_file: string | null;
	}

	interface DatasetInfo {
		name: string;
		path: string;
		trigger_token: string;
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
	let captionStyle = $state<'detailed' | 'simple' | 'tags'>('detailed');
	let clearingAllCaptions = $state(false);

	// Delete dataset state
	let showDeleteConfirm = $state(false);
	let deleting = $state(false);

	// Inline edit state
	let editingFilename = $state<string | null>(null);
	let editingCaption = $state('');
	let savingCaption = $state(false);

	// Delete image state
	let deletingImage = $state<string | null>(null);
	let showDeleteImageConfirm = $state<string | null>(null);

	// Lightbox state
	let lightboxImage = $state<DatasetImage | null>(null);
	let lightboxEditMode = $state(false);
	let lightboxCaption = $state('');
	let regeneratingCaption = $state(false);

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

	async function clearAllCaptions() {
		if (!dataset) return;
		const datasetName = dataset.name;
		clearingAllCaptions = true;
		error = null;
		try {
			// Clear each caption by setting to empty string
			const promises = dataset.images
				.filter((img) => img.caption)
				.map((img) =>
					fetch(`/api/datasets/${datasetName}/images/${img.filename}`, {
						method: 'PUT',
						headers: { 'Content-Type': 'application/json' },
						body: JSON.stringify({ caption: '' })
					})
				);
			await Promise.all(promises);
			await loadDataset(datasetName);
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to clear captions';
		} finally {
			clearingAllCaptions = false;
		}
	}

	async function clearCaption(filename: string) {
		if (!dataset) return;
		try {
			const res = await fetch(`/api/datasets/${dataset.name}/images/${filename}`, {
				method: 'PUT',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ caption: '' })
			});
			if (!res.ok) throw new Error('Failed to clear caption');
			await loadDataset(dataset.name);
			// Update lightbox if open
			if (lightboxImage?.filename === filename) {
				lightboxImage = { ...lightboxImage, caption: null };
				lightboxCaption = '';
			}
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to clear caption';
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

	// Inline caption editing
	function startEditing(image: DatasetImage) {
		editingFilename = image.filename;
		editingCaption = image.caption || '';
	}

	function cancelEditing() {
		editingFilename = null;
		editingCaption = '';
	}

	async function saveCaption(filename: string) {
		if (!dataset) return;
		savingCaption = true;
		try {
			const res = await fetch(`/api/datasets/${dataset.name}/images/${filename}`, {
				method: 'PUT',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ caption: editingCaption })
			});
			if (!res.ok) throw new Error('Failed to save caption');
			editingFilename = null;
			await loadDataset(dataset.name);
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to save caption';
		} finally {
			savingCaption = false;
		}
	}

	function handleCaptionKeydown(e: KeyboardEvent, filename: string) {
		if (e.key === 'Enter' && !e.shiftKey) {
			e.preventDefault();
			saveCaption(filename);
		} else if (e.key === 'Escape') {
			cancelEditing();
		}
	}

	// Delete image
	async function deleteImage(filename: string) {
		if (!dataset) return;
		deletingImage = filename;
		try {
			const res = await fetch(`/api/datasets/${dataset.name}/images/${filename}`, {
				method: 'DELETE'
			});
			if (!res.ok) throw new Error('Failed to delete image');
			showDeleteImageConfirm = null;
			if (lightboxImage?.filename === filename) {
				lightboxImage = null;
			}
			await loadDataset(dataset.name);
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to delete image';
		} finally {
			deletingImage = null;
		}
	}

	// Lightbox functions
	function openLightbox(image: DatasetImage) {
		lightboxImage = image;
		lightboxCaption = image.caption || '';
		lightboxEditMode = false;
	}

	function closeLightbox() {
		lightboxImage = null;
		lightboxEditMode = false;
	}

	async function saveLightboxCaption() {
		if (!dataset || !lightboxImage) return;
		savingCaption = true;
		try {
			const res = await fetch(`/api/datasets/${dataset.name}/images/${lightboxImage.filename}`, {
				method: 'PUT',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ caption: lightboxCaption })
			});
			if (!res.ok) throw new Error('Failed to save caption');
			lightboxEditMode = false;
			await loadDataset(dataset.name);
			// Update lightbox with new data
			const updated = dataset?.images.find((i) => i.filename === lightboxImage?.filename);
			if (updated) lightboxImage = updated;
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to save caption';
		} finally {
			savingCaption = false;
		}
	}

	async function regenerateLightboxCaption() {
		if (!dataset || !lightboxImage) return;
		regeneratingCaption = true;
		try {
			const res = await fetch(`/api/datasets/${dataset.name}/caption`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					style: captionStyle,
					overwrite: true,
					image_name: lightboxImage.filename
				})
			});
			if (!res.ok) throw new Error('Failed to regenerate caption');
			await loadDataset(dataset.name);
			const updated = dataset?.images.find((i) => i.filename === lightboxImage?.filename);
			if (updated) {
				lightboxImage = updated;
				lightboxCaption = updated.caption || '';
			}
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to regenerate caption';
		} finally {
			regeneratingCaption = false;
		}
	}

	function handleLightboxKeydown(e: KeyboardEvent) {
		if (e.key === 'Escape') {
			if (lightboxEditMode) {
				lightboxEditMode = false;
				lightboxCaption = lightboxImage?.caption || '';
			} else {
				closeLightbox();
			}
		}
	}

	onMount(() => {});
</script>

<svelte:window onkeydown={lightboxImage ? handleLightboxKeydown : undefined} />

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
				<p class="text-muted-foreground mt-1">
					{dataset.images.length} images
					<span class="mx-2">·</span>
					<span class="font-mono text-sm">trigger: {dataset.trigger_token}</span>
				</p>
			</div>
			<div class="flex items-center gap-3">
				<button
					onclick={() => (showAddModal = true)}
					class="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90"
				>
					<Plus class="w-5 h-5" />
					Add Images
				</button>

				<!-- Caption style selector -->
				<select
					bind:value={captionStyle}
					class="px-3 py-2 bg-secondary text-secondary-foreground rounded-lg border-none focus:outline-none focus:ring-2 focus:ring-ring"
				>
					<option value="detailed">Detailed</option>
					<option value="simple">Simple</option>
					<option value="tags">Tags</option>
				</select>

				<button
					onclick={captionImages}
					disabled={captioning || dataset.images.length === 0}
					class="flex items-center gap-2 px-4 py-2 bg-secondary text-secondary-foreground rounded-lg hover:bg-secondary/80 disabled:opacity-50"
				>
					<Sparkles class="w-5 h-5" />
					{captioning ? 'Captioning...' : 'Auto-Caption'}
				</button>
				<button
					onclick={clearAllCaptions}
					disabled={clearingAllCaptions || !dataset.images.some((img) => img.caption)}
					class="flex items-center gap-2 px-4 py-2 bg-secondary text-secondary-foreground rounded-lg hover:bg-secondary/80 disabled:opacity-50"
					title="Clear all captions"
				>
					<Eraser class="w-5 h-5" />
					{clearingAllCaptions ? 'Clearing...' : 'Clear All'}
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
					<div class="border border-border rounded-lg overflow-hidden bg-card group relative">
						<!-- Delete button overlay -->
						<button
							onclick={(e) => {
								e.stopPropagation();
								showDeleteImageConfirm = image.filename;
							}}
							class="absolute top-2 right-2 z-10 p-1.5 bg-black/60 hover:bg-destructive rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
						>
							<Trash2 class="w-4 h-4 text-white" />
						</button>

						<!-- Clickable image for lightbox -->
						<button
							onclick={() => openLightbox(image)}
							class="w-full aspect-square bg-muted cursor-pointer"
						>
							<img
								src="/api/datasets/{dataset.name}/images/{image.filename}"
								alt={image.caption || image.filename}
								class="w-full h-full object-cover"
								loading="lazy"
							/>
						</button>

						<div class="p-3">
							{#if editingFilename === image.filename}
								<!-- Inline edit mode -->
								<div class="flex flex-col gap-2">
									<textarea
										bind:value={editingCaption}
										onkeydown={(e) => handleCaptionKeydown(e, image.filename)}
										class="w-full px-2 py-1 text-sm bg-background border border-input rounded resize-none focus:outline-none focus:ring-2 focus:ring-ring"
										rows="3"
									></textarea>
									<div class="flex justify-end gap-1">
										<button
											onclick={cancelEditing}
											class="p-1 text-muted-foreground hover:text-foreground"
										>
											<X class="w-4 h-4" />
										</button>
										<button
											onclick={() => saveCaption(image.filename)}
											disabled={savingCaption}
											class="p-1 text-primary hover:text-primary/80 disabled:opacity-50"
										>
											<Check class="w-4 h-4" />
										</button>
									</div>
								</div>
							{:else}
								<!-- Display mode with edit button -->
								<div
									class="group/caption flex items-start justify-between gap-2 cursor-pointer"
									onclick={() => startEditing(image)}
									onkeydown={(e) => e.key === 'Enter' && startEditing(image)}
									role="button"
									tabindex="0"
								>
									{#if image.caption}
										<p class="text-sm text-foreground line-clamp-2 flex-1">{image.caption}</p>
									{:else}
										<p class="text-sm text-muted-foreground italic flex-1">No caption</p>
									{/if}
									<Pencil
										class="w-3 h-3 text-muted-foreground opacity-0 group-hover/caption:opacity-100 flex-shrink-0 mt-0.5"
									/>
								</div>
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
				<button
					onclick={() => (showAddModal = false)}
					class="text-muted-foreground hover:text-foreground"
				>
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

<!-- Delete Dataset Confirmation Modal -->
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

<!-- Delete Image Confirmation Modal -->
{#if showDeleteImageConfirm}
	<div class="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
		<div class="bg-card border border-border rounded-lg p-6 w-full max-w-md mx-4">
			<h2 class="text-xl font-semibold text-card-foreground mb-4">Delete Image?</h2>
			<p class="text-muted-foreground mb-6">
				This will permanently delete this image and its caption.
			</p>
			<div class="flex justify-end gap-3">
				<button
					onclick={() => (showDeleteImageConfirm = null)}
					class="px-4 py-2 text-muted-foreground hover:text-foreground"
				>
					Cancel
				</button>
				<button
					onclick={() => showDeleteImageConfirm && deleteImage(showDeleteImageConfirm)}
					disabled={deletingImage !== null}
					class="px-4 py-2 bg-destructive text-white rounded-lg hover:bg-destructive/90 disabled:opacity-50"
				>
					{deletingImage ? 'Deleting...' : 'Delete'}
				</button>
			</div>
		</div>
	</div>
{/if}

<!-- Image Lightbox -->
{#if lightboxImage && dataset}
	<div
		class="fixed inset-0 bg-black/90 flex items-center justify-center z-50"
		onclick={closeLightbox}
		onkeydown={(e) => e.key === 'Escape' && closeLightbox()}
		role="dialog"
		tabindex="-1"
	>
		<div
			class="flex flex-col lg:flex-row max-w-6xl w-full max-h-[90vh] mx-4 gap-4"
			onclick={(e) => e.stopPropagation()}
			onkeydown={() => {}}
			role="presentation"
		>
			<!-- Image -->
			<div class="flex-1 flex items-center justify-center min-h-0">
				<img
					src="/api/datasets/{dataset.name}/images/{lightboxImage.filename}"
					alt={lightboxImage.caption || lightboxImage.filename}
					class="max-w-full max-h-[70vh] lg:max-h-[85vh] object-contain rounded-lg"
				/>
			</div>

			<!-- Caption panel -->
			<div class="lg:w-80 bg-card rounded-lg p-4 flex flex-col gap-4">
				<div class="flex items-center justify-between">
					<h3 class="font-medium text-foreground truncate">{lightboxImage.filename}</h3>
					<button onclick={closeLightbox} class="text-muted-foreground hover:text-foreground">
						<X class="w-5 h-5" />
					</button>
				</div>

				<div class="flex-1 flex flex-col gap-3">
					<div class="flex items-center justify-between">
						<span class="text-sm font-medium text-muted-foreground">Caption</span>
						<div class="flex items-center gap-2">
							{#if !lightboxEditMode}
								<button
									onclick={() => {
										lightboxEditMode = true;
										lightboxCaption = lightboxImage?.caption || '';
									}}
									class="p-1 text-muted-foreground hover:text-foreground"
									title="Edit caption"
								>
									<Pencil class="w-4 h-4" />
								</button>
							{/if}
							<button
								onclick={regenerateLightboxCaption}
								disabled={regeneratingCaption}
								class="p-1 text-muted-foreground hover:text-foreground disabled:opacity-50"
								title="Regenerate caption"
							>
								<RefreshCw class="w-4 h-4 {regeneratingCaption ? 'animate-spin' : ''}" />
							</button>
							{#if lightboxImage?.caption}
								<button
									onclick={() => lightboxImage && clearCaption(lightboxImage.filename)}
									class="p-1 text-muted-foreground hover:text-foreground"
									title="Clear caption"
								>
									<Eraser class="w-4 h-4" />
								</button>
							{/if}
						</div>
					</div>

					{#if lightboxEditMode}
						<textarea
							bind:value={lightboxCaption}
							class="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground resize-none focus:outline-none focus:ring-2 focus:ring-ring"
							rows="6"
						></textarea>
						<div class="flex justify-end gap-2">
							<button
								onclick={() => {
									lightboxEditMode = false;
									lightboxCaption = lightboxImage?.caption || '';
								}}
								class="px-3 py-1.5 text-sm text-muted-foreground hover:text-foreground"
							>
								Cancel
							</button>
							<button
								onclick={saveLightboxCaption}
								disabled={savingCaption}
								class="px-3 py-1.5 text-sm bg-primary text-primary-foreground rounded hover:bg-primary/90 disabled:opacity-50"
							>
								{savingCaption ? 'Saving...' : 'Save'}
							</button>
						</div>
					{:else if lightboxImage.caption}
						<p class="text-sm text-foreground whitespace-pre-wrap">{lightboxImage.caption}</p>
					{:else}
						<p class="text-sm text-muted-foreground italic">No caption</p>
					{/if}
				</div>

				<!-- Delete button -->
				<button
					onclick={() => (showDeleteImageConfirm = lightboxImage?.filename || null)}
					class="flex items-center justify-center gap-2 px-4 py-2 text-destructive hover:bg-destructive/10 rounded-lg"
				>
					<Trash2 class="w-4 h-4" />
					Delete Image
				</button>
			</div>
		</div>
	</div>
{/if}
