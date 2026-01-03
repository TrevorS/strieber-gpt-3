<script lang="ts">
	import { onMount } from 'svelte';
	import { page } from '$app/stores';
	import { goto } from '$app/navigation';
	import { ArrowLeft, Plus, Sparkles, Trash2, Pencil, RefreshCw, Eraser } from 'lucide-svelte';
	import { Button } from '$lib/components/ui/button';
	import * as Card from '$lib/components/ui/card';
	import { AspectRatio } from '$lib/components/ui/aspect-ratio';
	import { Badge } from '$lib/components/ui/badge';
	import * as Dialog from '$lib/components/ui/dialog';
	import * as Select from '$lib/components/ui/select';
	import { Textarea } from '$lib/components/ui/textarea';
	import { Separator } from '$lib/components/ui/separator';

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

	// Delete image state
	let deletingImage = $state<string | null>(null);
	let showDeleteImageConfirm = $state<string | null>(null);

	// Lightbox state
	let lightboxImage = $state<DatasetImage | null>(null);
	let lightboxEditMode = $state(false);
	let lightboxCaption = $state('');
	let savingCaption = $state(false);
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
				<div class="flex items-center gap-3 mt-2">
					<Badge variant="outline">{dataset.images.length} images</Badge>
					<Badge variant="secondary" class="font-mono">{dataset.trigger_token}</Badge>
				</div>
			</div>
			<div class="flex items-center gap-2">
				<Button onclick={() => (showAddModal = true)}>
					<Plus class="w-4 h-4 mr-2" />
					Add Images
				</Button>

				<Separator orientation="vertical" class="h-8" />

				<!-- Caption style selector -->
				<Select.Root type="single" bind:value={captionStyle}>
					<Select.Trigger class="w-[130px]">
						{captionStyle.charAt(0).toUpperCase() + captionStyle.slice(1)}
					</Select.Trigger>
					<Select.Content>
						<Select.Item value="detailed">Detailed</Select.Item>
						<Select.Item value="simple">Simple</Select.Item>
						<Select.Item value="tags">Tags</Select.Item>
					</Select.Content>
				</Select.Root>

				<Button
					variant="secondary"
					onclick={captionImages}
					disabled={captioning || dataset.images.length === 0}
				>
					<Sparkles class="w-4 h-4 mr-2" />
					{captioning ? 'Captioning...' : 'Auto-Caption'}
				</Button>

				<Button
					variant="secondary"
					onclick={clearAllCaptions}
					disabled={clearingAllCaptions || !dataset.images.some((img) => img.caption)}
				>
					<Eraser class="w-4 h-4 mr-2" />
					{clearingAllCaptions ? 'Clearing...' : 'Clear All'}
				</Button>

				<Separator orientation="vertical" class="h-8" />

				<Button variant="ghost" size="icon" onclick={() => (showDeleteConfirm = true)}>
					<Trash2 class="w-4 h-4 text-destructive" />
				</Button>
			</div>
		</header>

		{#if dataset.images.length === 0}
			<Card.Root class="border-dashed">
				<Card.Content class="flex flex-col items-center justify-center py-12">
					<p class="text-muted-foreground mb-4">No images in this dataset</p>
					<Button onclick={() => (showAddModal = true)}>Add images</Button>
				</Card.Content>
			</Card.Root>
		{:else}
			<div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
				{#each dataset.images as image}
					<Card.Root class="overflow-hidden group relative">
						<!-- Delete button overlay -->
						<Button
							variant="destructive"
							size="icon"
							class="absolute top-2 right-2 z-10 h-7 w-7 opacity-0 group-hover:opacity-100 transition-opacity"
							onclick={(e: MouseEvent) => {
								e.stopPropagation();
								showDeleteImageConfirm = image.filename;
							}}
						>
							<Trash2 class="w-3.5 h-3.5" />
						</Button>

						<!-- Clickable image for lightbox -->
						<AspectRatio ratio={1} class="bg-muted">
							<button
								onclick={() => openLightbox(image)}
								class="block w-full h-full cursor-pointer"
							>
								<img
									src="/api/datasets/{dataset.name}/images/{image.filename}"
									alt={image.caption || image.filename}
									class="w-full h-full object-cover"
									loading="lazy"
								/>
							</button>
						</AspectRatio>

						<Card.Content class="p-3">
							{#if image.caption}
								<p class="text-sm text-muted-foreground line-clamp-3">{image.caption}</p>
							{:else}
								<p class="text-sm text-muted-foreground/60 italic">No caption</p>
							{/if}
						</Card.Content>
					</Card.Root>
				{/each}
			</div>
		{/if}
	{/if}
</div>

<!-- Add Images Modal -->
<Dialog.Root bind:open={showAddModal}>
	<Dialog.Content class="max-w-lg">
		<Dialog.Header>
			<Dialog.Title>Add Images</Dialog.Title>
			<Dialog.Description>Enter image URLs, one per line</Dialog.Description>
		</Dialog.Header>

		<Textarea
			bind:value={imageUrls}
			placeholder="https://example.com/image1.jpg&#10;https://example.com/image2.jpg"
			rows={6}
			class="font-mono text-sm"
		/>

		<Dialog.Footer>
			<Button
				variant="outline"
				onclick={() => {
					showAddModal = false;
					imageUrls = '';
				}}
			>
				Cancel
			</Button>
			<Button onclick={addImages} disabled={adding || !imageUrls.trim()}>
				{adding ? 'Adding...' : 'Add Images'}
			</Button>
		</Dialog.Footer>
	</Dialog.Content>
</Dialog.Root>

<!-- Delete Dataset Confirmation Modal -->
<Dialog.Root bind:open={showDeleteConfirm}>
	<Dialog.Content class="max-w-md">
		<Dialog.Header>
			<Dialog.Title>Delete Dataset?</Dialog.Title>
			<Dialog.Description>
				This will permanently delete <strong>{dataset?.name}</strong> and all its images.
				This action cannot be undone.
			</Dialog.Description>
		</Dialog.Header>
		<Dialog.Footer>
			<Button variant="outline" onclick={() => (showDeleteConfirm = false)}>Cancel</Button>
			<Button variant="destructive" onclick={deleteDataset} disabled={deleting}>
				{deleting ? 'Deleting...' : 'Delete'}
			</Button>
		</Dialog.Footer>
	</Dialog.Content>
</Dialog.Root>

<!-- Delete Image Confirmation Modal -->
<Dialog.Root open={showDeleteImageConfirm !== null} onOpenChange={(open) => !open && (showDeleteImageConfirm = null)}>
	<Dialog.Content class="max-w-md">
		<Dialog.Header>
			<Dialog.Title>Delete Image?</Dialog.Title>
			<Dialog.Description>
				This will permanently delete this image and its caption.
			</Dialog.Description>
		</Dialog.Header>
		<Dialog.Footer>
			<Button variant="outline" onclick={() => (showDeleteImageConfirm = null)}>Cancel</Button>
			<Button
				variant="destructive"
				onclick={() => showDeleteImageConfirm && deleteImage(showDeleteImageConfirm)}
				disabled={deletingImage !== null}
			>
				{deletingImage ? 'Deleting...' : 'Delete'}
			</Button>
		</Dialog.Footer>
	</Dialog.Content>
</Dialog.Root>

<!-- Image Lightbox -->
<Dialog.Root open={lightboxImage !== null && dataset !== null} onOpenChange={(open) => !open && closeLightbox()}>
	<Dialog.Content class="max-w-5xl w-[95vw] p-0 gap-0 overflow-hidden">
		{#if lightboxImage && dataset}
			<div class="flex flex-col">
				<!-- Image - full width, generous height -->
				<div class="bg-black flex items-center justify-center p-4">
					<img
						src="/api/datasets/{dataset.name}/images/{lightboxImage.filename}"
						alt={lightboxImage.caption || lightboxImage.filename}
						class="max-h-[70vh] max-w-full object-contain"
					/>
				</div>

				<!-- Caption panel below -->
				<div class="p-6 border-t bg-background">
					<div class="flex items-center justify-between mb-4">
						<p class="text-sm font-medium text-muted-foreground truncate">{lightboxImage.filename}</p>
						<div class="flex items-center gap-1">
							{#if !lightboxEditMode}
								<Button
									variant="ghost"
									size="icon"
									class="h-7 w-7"
									onclick={() => {
										lightboxEditMode = true;
										lightboxCaption = lightboxImage?.caption || '';
									}}
								>
									<Pencil class="w-4 h-4" />
								</Button>
							{/if}
							<Button
								variant="ghost"
								size="icon"
								class="h-7 w-7"
								onclick={regenerateLightboxCaption}
								disabled={regeneratingCaption}
							>
								<RefreshCw class="w-4 h-4 {regeneratingCaption ? 'animate-spin' : ''}" />
							</Button>
							{#if lightboxImage?.caption}
								<Button
									variant="ghost"
									size="icon"
									class="h-7 w-7"
									onclick={() => lightboxImage && clearCaption(lightboxImage.filename)}
								>
									<Eraser class="w-4 h-4" />
								</Button>
							{/if}
							<Button
								variant="ghost"
								size="icon"
								class="h-7 w-7 text-destructive hover:text-destructive"
								onclick={() => (showDeleteImageConfirm = lightboxImage?.filename || null)}
							>
								<Trash2 class="w-4 h-4" />
							</Button>
						</div>
					</div>

					{#if lightboxEditMode}
						<Textarea bind:value={lightboxCaption} rows={4} class="resize-none mb-3" />
						<div class="flex justify-end gap-2">
							<Button
								variant="outline"
								size="sm"
								onclick={() => {
									lightboxEditMode = false;
									lightboxCaption = lightboxImage?.caption || '';
								}}
							>
								Cancel
							</Button>
							<Button size="sm" onclick={saveLightboxCaption} disabled={savingCaption}>
								{savingCaption ? 'Saving...' : 'Save'}
							</Button>
						</div>
					{:else if lightboxImage.caption}
						<p class="text-sm text-foreground whitespace-pre-wrap">{lightboxImage.caption}</p>
					{:else}
						<p class="text-sm text-muted-foreground italic">No caption</p>
					{/if}
				</div>
			</div>
		{/if}
	</Dialog.Content>
</Dialog.Root>
