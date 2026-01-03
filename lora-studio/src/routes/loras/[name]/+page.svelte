<script lang="ts">
	import { onMount } from 'svelte';
	import { page } from '$app/stores';
	import { goto } from '$app/navigation';
	import { ArrowLeft, Pencil, Trash2, Sparkles, Loader2, ExternalLink } from 'lucide-svelte';
	import { Button } from '$lib/components/ui/button';
	import * as Card from '$lib/components/ui/card';
	import { Badge } from '$lib/components/ui/badge';
	import * as Dialog from '$lib/components/ui/dialog';
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';
	import { Textarea } from '$lib/components/ui/textarea';
	import { Separator } from '$lib/components/ui/separator';
	import { AspectRatio } from '$lib/components/ui/aspect-ratio';

	interface SourceJob {
		job_id: string;
		dataset_name: string;
		trigger_token: string;
		completed_at: string | null;
	}

	interface LoRADetails {
		name: string;
		filename: string;
		file_size: number;
		created_at: number;
		source_job: SourceJob | null;
		training_samples?: string[];
	}

	let lora = $state<LoRADetails | null>(null);
	let loading = $state(true);
	let error = $state<string | null>(null);

	// Rename state
	let showRenameModal = $state(false);
	let newName = $state('');
	let renaming = $state(false);

	// Delete state
	let showDeleteConfirm = $state(false);
	let deleting = $state(false);

	// Test generation state
	let testPrompt = $state('');
	let generating = $state(false);
	let generatedImage = $state<string | null>(null);
	let generateError = $state<string | null>(null);

	function formatFileSize(bytes: number): string {
		if (bytes < 1024) return `${bytes} B`;
		if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
		if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
		return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
	}

	function formatDate(timestamp: number): string {
		return new Date(timestamp * 1000).toLocaleDateString('en-US', {
			month: 'long',
			day: 'numeric',
			year: 'numeric',
			hour: 'numeric',
			minute: '2-digit'
		});
	}

	async function loadLoRA(name: string) {
		loading = true;
		error = null;
		try {
			const res = await fetch(`/api/loras/${name}`);
			if (!res.ok) throw new Error('Failed to load LoRA');
			lora = await res.json();
			// Set default prompt with trigger token if available
			if (lora?.source_job?.trigger_token && !testPrompt) {
				testPrompt = `${lora.source_job.trigger_token}, portrait, studio lighting`;
			} else if (lora && !testPrompt) {
				testPrompt = `${lora.name}, portrait, studio lighting`;
			}
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load LoRA';
		} finally {
			loading = false;
		}
	}

	async function renameLora() {
		if (!lora || !newName.trim()) return;
		renaming = true;
		try {
			const res = await fetch(`/api/loras/${lora.name}`, {
				method: 'PUT',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ new_name: newName.trim() })
			});
			if (!res.ok) {
				const data = await res.json();
				throw new Error(data.message || 'Failed to rename LoRA');
			}
			showRenameModal = false;
			// Navigate to new URL
			goto(`/loras/${newName.trim()}`);
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to rename LoRA';
		} finally {
			renaming = false;
		}
	}

	async function deleteLora() {
		if (!lora) return;
		deleting = true;
		try {
			const res = await fetch(`/api/loras/${lora.name}`, { method: 'DELETE' });
			if (!res.ok) throw new Error('Failed to delete LoRA');
			goto('/loras');
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to delete LoRA';
			deleting = false;
		}
	}

	async function generateTestImage() {
		if (!lora || !testPrompt.trim()) return;
		generating = true;
		generateError = null;
		generatedImage = null;

		try {
			const res = await fetch(`/api/loras/${lora.name}/test`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ prompt: testPrompt.trim() })
			});
			if (!res.ok) {
				const data = await res.json();
				throw new Error(data.message || 'Failed to generate image');
			}
			const result = await res.json();
			// Expect result.image_url or result.image_base64
			if (result.image_url) {
				generatedImage = result.image_url;
			} else if (result.image_base64) {
				generatedImage = `data:image/png;base64,${result.image_base64}`;
			} else {
				throw new Error('No image returned');
			}
		} catch (e) {
			generateError = e instanceof Error ? e.message : 'Failed to generate image';
		} finally {
			generating = false;
		}
	}

	$effect(() => {
		const name = $page.params.name;
		if (name) loadLoRA(name);
	});
</script>

<div class="container mx-auto px-4 py-8">
	<a
		href="/loras"
		class="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground mb-4"
	>
		<ArrowLeft class="w-4 h-4" />
		LoRAs
	</a>

	{#if loading}
		<div class="flex items-center justify-center py-12">
			<div class="text-muted-foreground">Loading LoRA...</div>
		</div>
	{:else if error}
		<div class="bg-destructive/10 text-destructive px-4 py-3 rounded-lg mb-4">
			{error}
			<button onclick={() => lora && loadLoRA(lora.name)} class="underline ml-2">Retry</button>
		</div>
	{:else if lora}
		<header class="flex items-center justify-between mb-8">
			<div>
				<h1 class="text-3xl font-bold text-foreground">{lora.name}</h1>
				{#if lora.source_job}
					<div class="flex items-center gap-2 mt-2">
						<Badge variant="secondary">{lora.source_job.dataset_name}</Badge>
						<Badge variant="outline" class="font-mono">{lora.source_job.trigger_token}</Badge>
					</div>
				{/if}
			</div>
			<div class="flex items-center gap-2">
				<Button
					variant="outline"
					onclick={() => {
						newName = lora?.name || '';
						showRenameModal = true;
					}}
				>
					<Pencil class="w-4 h-4 mr-2" />
					Rename
				</Button>
				<Button variant="destructive" onclick={() => (showDeleteConfirm = true)}>
					<Trash2 class="w-4 h-4 mr-2" />
					Delete
				</Button>
			</div>
		</header>

		<div class="space-y-6">
			<!-- Details Card -->
			<Card.Root>
				<Card.Header>
					<Card.Title>Details</Card.Title>
				</Card.Header>
				<Card.Content>
					<dl class="grid grid-cols-1 md:grid-cols-2 gap-4">
						<div>
							<dt class="text-sm font-medium text-muted-foreground">Filename</dt>
							<dd class="text-sm text-foreground font-mono mt-1">{lora.filename}</dd>
						</div>
						<div>
							<dt class="text-sm font-medium text-muted-foreground">Size</dt>
							<dd class="text-sm text-foreground mt-1">{formatFileSize(lora.file_size)}</dd>
						</div>
						<div>
							<dt class="text-sm font-medium text-muted-foreground">Created</dt>
							<dd class="text-sm text-foreground mt-1">{formatDate(lora.created_at)}</dd>
						</div>
						{#if lora.source_job}
							<div>
								<dt class="text-sm font-medium text-muted-foreground">Trigger Token</dt>
								<dd class="text-sm text-foreground font-mono mt-1">
									{lora.source_job.trigger_token}
								</dd>
							</div>
							<div>
								<dt class="text-sm font-medium text-muted-foreground">Source Dataset</dt>
								<dd class="mt-1">
									<a
										href="/datasets/{lora.source_job.dataset_name}"
										class="text-sm text-primary hover:underline inline-flex items-center gap-1"
									>
										{lora.source_job.dataset_name}
										<ExternalLink class="w-3 h-3" />
									</a>
								</dd>
							</div>
							<div>
								<dt class="text-sm font-medium text-muted-foreground">Training Job</dt>
								<dd class="mt-1">
									<a
										href="/training?job={lora.source_job.job_id}"
										class="text-sm text-primary hover:underline inline-flex items-center gap-1"
									>
										{lora.source_job.job_id.slice(0, 8)}
										<ExternalLink class="w-3 h-3" />
									</a>
								</dd>
							</div>
						{/if}
					</dl>
				</Card.Content>
			</Card.Root>

			<!-- Test Generation Card -->
			<Card.Root>
				<Card.Header>
					<Card.Title>Test Generation</Card.Title>
					<Card.Description>Generate a test image using this LoRA</Card.Description>
				</Card.Header>
				<Card.Content class="space-y-4">
					<div class="space-y-2">
						<Label for="test-prompt">Prompt</Label>
						<Textarea
							id="test-prompt"
							bind:value={testPrompt}
							placeholder="Enter a prompt with the trigger token..."
							rows={3}
						/>
					</div>

					<Button onclick={generateTestImage} disabled={generating || !testPrompt.trim()}>
						{#if generating}
							<Loader2 class="w-4 h-4 mr-2 animate-spin" />
							Generating...
						{:else}
							<Sparkles class="w-4 h-4 mr-2" />
							Generate Image
						{/if}
					</Button>

					{#if generateError}
						<div class="bg-destructive/10 text-destructive px-4 py-3 rounded-lg text-sm">
							{generateError}
						</div>
					{/if}

					{#if generatedImage}
						<Separator />
						<div class="max-w-md">
							<AspectRatio ratio={1} class="bg-muted rounded-lg overflow-hidden">
								<img
									src={generatedImage}
									alt="Generated test image"
									class="w-full h-full object-contain"
								/>
							</AspectRatio>
						</div>
					{/if}
				</Card.Content>
			</Card.Root>

			<!-- Training Samples Card (if available) -->
			{#if lora.source_job && lora.training_samples && lora.training_samples.length > 0}
				<Card.Root>
					<Card.Header>
						<Card.Title>Training Samples</Card.Title>
						<Card.Description>Sample images generated during training</Card.Description>
					</Card.Header>
					<Card.Content>
						<div class="grid grid-cols-2 md:grid-cols-4 gap-4">
							{#each lora.training_samples as sample}
								<AspectRatio ratio={1} class="bg-muted rounded-lg overflow-hidden">
									<img
										src="/api/jobs/{lora.source_job.job_id}/samples/{sample}"
										alt="Training sample"
										class="w-full h-full object-cover"
										loading="lazy"
									/>
								</AspectRatio>
							{/each}
						</div>
					</Card.Content>
				</Card.Root>
			{/if}
		</div>
	{/if}
</div>

<!-- Rename Modal -->
<Dialog.Root bind:open={showRenameModal}>
	<Dialog.Content class="max-w-md">
		<Dialog.Header>
			<Dialog.Title>Rename LoRA</Dialog.Title>
			<Dialog.Description>Enter a new name for this LoRA</Dialog.Description>
		</Dialog.Header>

		<div class="space-y-4 py-4">
			<div class="space-y-2">
				<Label for="new-name">New name</Label>
				<Input
					id="new-name"
					bind:value={newName}
					placeholder="e.g., my_character_v2"
					onkeydown={(e: KeyboardEvent) => e.key === 'Enter' && renameLora()}
				/>
				<p class="text-xs text-muted-foreground">
					Use letters, numbers, underscores, and hyphens. Must start with a letter.
				</p>
			</div>
		</div>

		<Dialog.Footer>
			<Button
				variant="outline"
				onclick={() => {
					showRenameModal = false;
					newName = '';
				}}
			>
				Cancel
			</Button>
			<Button onclick={renameLora} disabled={renaming || !newName.trim()}>
				{renaming ? 'Renaming...' : 'Rename'}
			</Button>
		</Dialog.Footer>
	</Dialog.Content>
</Dialog.Root>

<!-- Delete Confirmation Modal -->
<Dialog.Root bind:open={showDeleteConfirm}>
	<Dialog.Content class="max-w-md">
		<Dialog.Header>
			<Dialog.Title>Delete LoRA?</Dialog.Title>
			<Dialog.Description>
				This will permanently delete <strong>{lora?.name}</strong>. This action cannot be undone.
			</Dialog.Description>
		</Dialog.Header>
		<Dialog.Footer>
			<Button variant="outline" onclick={() => (showDeleteConfirm = false)}>Cancel</Button>
			<Button variant="destructive" onclick={deleteLora} disabled={deleting}>
				{deleting ? 'Deleting...' : 'Delete'}
			</Button>
		</Dialog.Footer>
	</Dialog.Content>
</Dialog.Root>
