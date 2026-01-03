<script lang="ts">
	import { onMount } from 'svelte';
	import { ArrowLeft, Package, ChevronRight } from 'lucide-svelte';
	import { Button } from '$lib/components/ui/button';
	import * as Card from '$lib/components/ui/card';
	import { Badge } from '$lib/components/ui/badge';

	interface SourceJob {
		job_id: string;
		dataset_name: string;
		trigger_token: string;
		completed_at: string | null;
	}

	interface LoRA {
		name: string;
		filename: string;
		file_size: number;
		created_at: number;
		source_job: SourceJob | null;
	}

	let loras = $state<LoRA[]>([]);
	let loading = $state(true);
	let error = $state<string | null>(null);

	function formatFileSize(bytes: number): string {
		if (bytes < 1024) return `${bytes} B`;
		if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
		if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
		return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
	}

	function formatDate(timestamp: number): string {
		return new Date(timestamp * 1000).toLocaleDateString('en-US', {
			month: 'short',
			day: 'numeric',
			year: 'numeric'
		});
	}

	async function loadLoRAs() {
		loading = true;
		error = null;
		try {
			const res = await fetch('/api/loras');
			if (!res.ok) throw new Error('Failed to load LoRAs');
			loras = await res.json();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load LoRAs';
		} finally {
			loading = false;
		}
	}

	onMount(loadLoRAs);
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
			<h1 class="text-3xl font-bold text-foreground">LoRAs</h1>
			<p class="text-muted-foreground mt-1">Trained LoRA models ready for image generation</p>
		</div>
	</header>

	{#if loading}
		<div class="flex items-center justify-center py-12">
			<div class="text-muted-foreground">Loading LoRAs...</div>
		</div>
	{:else if error}
		<div class="bg-destructive/10 text-destructive px-4 py-3 rounded-lg mb-4">
			{error}
			<button onclick={loadLoRAs} class="underline ml-2">Retry</button>
		</div>
	{:else if loras.length === 0}
		<Card.Root class="border-dashed">
			<Card.Content class="flex flex-col items-center justify-center py-12">
				<Package class="w-12 h-12 text-muted-foreground mb-4" />
				<p class="text-muted-foreground mb-2">No LoRAs yet</p>
				<p class="text-sm text-muted-foreground/70 text-center max-w-md">
					Train a dataset and promote a checkpoint to create your first LoRA
				</p>
			</Card.Content>
		</Card.Root>
	{:else}
		<div class="grid gap-4">
			{#each loras as lora}
				<a href="/loras/{lora.name}" class="block group">
					<Card.Root class="hover:bg-accent transition-colors">
						<Card.Content class="flex items-center justify-between p-4">
							<div class="flex items-center gap-4">
								<div
									class="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center"
								>
									<Package class="w-5 h-5 text-primary" />
								</div>
								<div>
									<h2
										class="text-lg font-semibold text-card-foreground group-hover:text-accent-foreground"
									>
										{lora.name}
									</h2>
									<div class="flex items-center gap-3 text-sm text-muted-foreground mt-1">
										<span>{formatFileSize(lora.file_size)}</span>
										<span>•</span>
										<span>Created {formatDate(lora.created_at)}</span>
									</div>
									{#if lora.source_job}
										<div class="flex items-center gap-2 mt-1">
											<span class="text-xs text-muted-foreground">Source:</span>
											<Badge variant="secondary" class="text-xs"
												>{lora.source_job.dataset_name}</Badge
											>
											<Badge variant="outline" class="font-mono text-xs"
												>{lora.source_job.trigger_token}</Badge
											>
										</div>
									{/if}
								</div>
							</div>
							<ChevronRight
								class="w-5 h-5 text-muted-foreground group-hover:text-accent-foreground transition-transform group-hover:translate-x-1"
							/>
						</Card.Content>
					</Card.Root>
				</a>
			{/each}
		</div>
	{/if}
</div>
