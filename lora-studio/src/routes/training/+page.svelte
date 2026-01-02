<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { ArrowLeft, Plus, Play, Square, ChevronDown, ChevronUp, Upload } from 'lucide-svelte';
	import * as Tabs from '$lib/components/ui/tabs';
	import * as Card from '$lib/components/ui/card';
	import { Progress } from '$lib/components/ui/progress';
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';
	import * as Collapsible from '$lib/components/ui/collapsible';
	import * as Dialog from '$lib/components/ui/dialog';
	import * as Select from '$lib/components/ui/select';
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';

	interface TrainingConfig {
		steps: number;
		lr: number;
		lora_rank: number;
		batch_size: number;
		checkpoint_every: number;
		sample_every: number;
		image_size: number;
	}

	interface TrainingJob {
		job_id: string;
		dataset_name: string;
		trigger_token: string;
		status: 'pending' | 'running' | 'completed' | 'failed' | 'stopped';
		current_step: number;
		total_steps: number;
		latest_loss: number | null;
		checkpoints: string[];
		sample_images: string[];
		started_at: string | null;
		completed_at: string | null;
		error_message: string | null;
		config?: TrainingConfig;
	}

	interface Dataset {
		name: string;
		trigger_token: string;
		image_count: number;
		has_captions: boolean;
	}

	let jobs = $state<TrainingJob[]>([]);
	let datasets = $state<Dataset[]>([]);
	let loading = $state(true);
	let error = $state<string | null>(null);
	let activeTab = $state('all');
	let expandedJobId = $state<string | null>(null);
	let pollInterval: ReturnType<typeof setInterval> | null = null;

	// New training modal state
	let showNewTrainingModal = $state(false);
	let selectedDatasetName = $state<string | undefined>(undefined);
	let trainingSteps = $state(3000);
	let learningRate = $state(0.0001);
	let loraRank = $state(8);
	let checkpointEvery = $state(500);
	let sampleEvery = $state(250);
	let samplePrompts = $state('');
	let startingTraining = $state(false);

	// Promote checkpoint state
	let promotingCheckpoint = $state<string | null>(null);
	let selectedCheckpoint = $state<string | null>(null);
	let promoting = $state(false);

	// Stopping job state
	let stoppingJobId = $state<string | null>(null);

	// Collapsible open states
	let jobOpenStates = $state<Record<string, boolean>>({});

	// Sample image metadata
	interface SampleImage {
		filename: string;
		step?: number;
		prompt_index?: number;
		timestamp?: number;
		prompt?: string;
	}

	// Fetched sample images per job (from filesystem)
	let jobSamples = $state<Record<string, SampleImage[]>>({});

	// Selected sample for modal view
	let selectedSample = $state<{ jobId: string; sample: SampleImage } | null>(null);

	const statusColors: Record<string, string> = {
		pending: 'bg-yellow-500',
		running: 'bg-green-500',
		completed: 'bg-blue-500',
		failed: 'bg-red-500',
		stopped: 'bg-gray-500'
	};

	const statusBadgeVariant: Record<string, 'default' | 'secondary' | 'destructive' | 'outline'> = {
		pending: 'secondary',
		running: 'default',
		completed: 'default',
		failed: 'destructive',
		stopped: 'outline'
	};

	// Derived state for selected dataset
	let selectedDataset = $derived(selectedDatasetName ? datasets.find((d) => d.name === selectedDatasetName) ?? null : null);
	let eligibleDatasets = $derived(datasets.filter((d) => d.has_captions && d.image_count >= 5));

	$effect(() => {
		// Update polling based on whether there are running jobs
		const hasRunningJobs = jobs.some((j) => j.status === 'running' || j.status === 'pending');
		if (hasRunningJobs && !pollInterval) {
			pollInterval = setInterval(loadJobs, 5000);
		} else if (!hasRunningJobs && pollInterval) {
			clearInterval(pollInterval);
			pollInterval = null;
		}
	});

	// Load samples when job is expanded
	$effect(() => {
		for (const [jobId, isOpen] of Object.entries(jobOpenStates)) {
			if (isOpen && !jobSamples[jobId]) {
				loadJobSamples(jobId);
			}
		}
	});

	// Load samples for running jobs on poll (force refresh)
	$effect(() => {
		for (const job of jobs) {
			if (job.status === 'running') {
				loadJobSamples(job.job_id, true);
			}
		}
	});

	async function loadJobs() {
		try {
			const res = await fetch('/api/jobs');
			if (!res.ok) throw new Error('Failed to load jobs');
			jobs = await res.json();
			error = null;
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load jobs';
		} finally {
			loading = false;
		}
	}

	async function loadDatasets() {
		try {
			const res = await fetch('/api/datasets');
			if (!res.ok) throw new Error('Failed to load datasets');
			datasets = await res.json();
		} catch (e) {
			console.error('Failed to load datasets:', e);
		}
	}

	async function startTraining() {
		if (!selectedDataset) return;
		startingTraining = true;
		error = null;

		try {
			const prompts = samplePrompts
				.split('\n')
				.map((p) => p.trim())
				.filter((p) => p.length > 0);

			const res = await fetch('/api/jobs', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					dataset_name: selectedDataset.name,
					steps: trainingSteps,
					learning_rate: learningRate,
					lora_rank: loraRank,
					checkpoint_every: checkpointEvery,
					sample_every: sampleEvery,
					sample_prompts: prompts.length > 0 ? prompts : undefined
				})
			});

			if (!res.ok) {
				const data = await res.json();
				throw new Error(data.message || 'Failed to start training');
			}

			showNewTrainingModal = false;
			resetTrainingForm();
			await loadJobs();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to start training';
		} finally {
			startingTraining = false;
		}
	}

	async function stopJob(jobId: string) {
		stoppingJobId = jobId;
		try {
			const res = await fetch(`/api/jobs/${jobId}/stop`, { method: 'POST' });
			if (!res.ok) throw new Error('Failed to stop job');
			await loadJobs();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to stop job';
		} finally {
			stoppingJobId = null;
		}
	}

	async function promoteCheckpoint(jobId: string, checkpointName: string) {
		promoting = true;
		try {
			const res = await fetch(`/api/jobs/${jobId}/promote`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ checkpoint_name: checkpointName })
			});

			if (!res.ok) {
				const data = await res.json();
				throw new Error(data.message || 'Failed to promote checkpoint');
			}

			promotingCheckpoint = null;
			selectedCheckpoint = null;
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to promote checkpoint';
		} finally {
			promoting = false;
		}
	}

	function resetTrainingForm() {
		selectedDatasetName = undefined;
		trainingSteps = 3000;
		learningRate = 0.0001;
		loraRank = 8;
		checkpointEvery = 500;
		sampleEvery = 250;
		samplePrompts = '';
	}

	function getFilteredJobs(status: string): TrainingJob[] {
		if (status === 'all') return jobs;
		return jobs.filter((j) => j.status === status);
	}

	function formatTime(isoString: string | null): string {
		if (!isoString) return '-';
		return new Date(isoString).toLocaleString();
	}

	function getProgressPercent(job: TrainingJob): number {
		if (job.total_steps === 0) return 0;
		return Math.round((job.current_step / job.total_steps) * 100);
	}

	function formatDuration(startedAt: string | null, completedAt: string | null): string {
		if (!startedAt) return '-';
		const start = new Date(startedAt);
		const end = completedAt ? new Date(completedAt) : new Date();
		const diffMs = end.getTime() - start.getTime();
		const diffMins = Math.floor(diffMs / 60000);
		const hours = Math.floor(diffMins / 60);
		const mins = diffMins % 60;
		if (hours > 0) {
			return `${hours}h ${mins}m`;
		}
		return `${mins}m`;
	}

	async function loadJobSamples(jobId: string, forceRefresh = false) {
		if (jobSamples[jobId] && !forceRefresh) return; // Already loaded
		try {
			const res = await fetch(`/api/jobs/${jobId}/samples`);
			if (res.ok) {
				const data = await res.json();
				jobSamples[jobId] = data.samples || [];
			}
		} catch (e) {
			console.error('Failed to load samples:', e);
		}
	}

	function getSamplesForJob(job: TrainingJob): SampleImage[] {
		// Prefer fetched samples with metadata
		if (jobSamples[job.job_id]?.length > 0) {
			return jobSamples[job.job_id];
		}
		// Fallback: extract just filenames from full paths (no metadata)
		return job.sample_images.map((p) => ({
			filename: p.split('/').pop() || p
		}));
	}

	function formatSampleTimestamp(timestamp?: number): string {
		if (!timestamp) return '-';
		return new Date(timestamp).toLocaleString();
	}

	function toggleJob(jobId: string) {
		jobOpenStates[jobId] = !jobOpenStates[jobId];
	}

	function handleDatasetChange(value: string | undefined) {
		selectedDatasetName = value;
		const dataset = datasets.find((d) => d.name === value);
		if (dataset) {
			samplePrompts = `${dataset.trigger_token}, portrait, studio lighting\n${dataset.trigger_token}, outdoor, natural light`;
		}
	}

	onMount(() => {
		loadJobs();
		loadDatasets();
	});

	onDestroy(() => {
		if (pollInterval) {
			clearInterval(pollInterval);
		}
	});
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
			<h1 class="text-3xl font-bold text-foreground">Training</h1>
			<p class="text-muted-foreground mt-1">Train and monitor LoRA training jobs</p>
		</div>
		<Button onclick={() => (showNewTrainingModal = true)}>
			<Plus class="w-4 h-4 mr-2" />
			New Training Job
		</Button>
	</header>

	{#if error}
		<div class="bg-destructive/10 text-destructive px-4 py-3 rounded-lg mb-4">
			{error}
			<button onclick={() => (error = null)} class="underline ml-2">Dismiss</button>
		</div>
	{/if}

	{#if loading}
		<div class="flex items-center justify-center py-12">
			<div class="text-muted-foreground">Loading jobs...</div>
		</div>
	{:else}
		<Tabs.Root bind:value={activeTab} class="w-full">
			<Tabs.List class="mb-6">
				<Tabs.Trigger value="all">All ({jobs.length})</Tabs.Trigger>
				<Tabs.Trigger value="running">
					Running ({jobs.filter((j) => j.status === 'running').length})
				</Tabs.Trigger>
				<Tabs.Trigger value="completed">
					Completed ({jobs.filter((j) => j.status === 'completed').length})
				</Tabs.Trigger>
				<Tabs.Trigger value="failed">
					Failed ({jobs.filter((j) => j.status === 'failed').length})
				</Tabs.Trigger>
			</Tabs.List>

			{#each ['all', 'running', 'completed', 'failed'] as tabValue}
				<Tabs.Content value={tabValue}>
					{@const filteredJobs = getFilteredJobs(tabValue)}
					{#if filteredJobs.length === 0}
						<div class="text-center py-12 border border-dashed border-border rounded-lg">
							<Play class="w-12 h-12 text-muted-foreground mx-auto mb-4" />
							<p class="text-muted-foreground mb-4">
								{#if tabValue === 'all'}
									No training jobs yet
								{:else}
									No {tabValue} jobs
								{/if}
							</p>
							{#if tabValue === 'all'}
								<Button onclick={() => (showNewTrainingModal = true)}>
									Start your first training job
								</Button>
							{/if}
						</div>
					{:else}
						<div class="space-y-4">
							{#each filteredJobs as job (job.job_id)}
								<Collapsible.Root open={jobOpenStates[job.job_id] ?? false} onOpenChange={(open) => (jobOpenStates[job.job_id] = open)}>
									<Card.Root>
										<Collapsible.Trigger class="w-full text-left">
											<Card.Header class="pb-2">
												<div class="flex items-center justify-between">
													<div class="flex items-center gap-3">
														<div class="w-3 h-3 rounded-full {statusColors[job.status]}"></div>
														<span class="font-mono text-lg">{job.job_id}</span>
														<Badge variant={statusBadgeVariant[job.status]}>
															{job.status}
														</Badge>
													</div>
													<div class="flex items-center gap-2">
														{#if job.status === 'running' || job.status === 'pending'}
															<Button
																variant="outline"
																size="sm"
																onclick={(e: MouseEvent) => {
																	e.stopPropagation();
																	stopJob(job.job_id);
																}}
																disabled={stoppingJobId === job.job_id}
															>
																<Square class="w-4 h-4 mr-1" />
																{stoppingJobId === job.job_id ? 'Stopping...' : 'Stop'}
															</Button>
														{/if}
														{#if jobOpenStates[job.job_id]}
															<ChevronUp class="w-5 h-5 text-muted-foreground" />
														{:else}
															<ChevronDown class="w-5 h-5 text-muted-foreground" />
														{/if}
													</div>
												</div>
											</Card.Header>
											<Card.Content class="pt-0">
												<div class="text-sm text-muted-foreground mb-3">
													Dataset: {job.dataset_name}
													<span class="mx-2">·</span>
													Trigger: <span class="font-mono">{job.trigger_token}</span>
													<span class="mx-2">·</span>
													Duration: {formatDuration(job.started_at, job.completed_at)}
												</div>
												<div class="flex items-center gap-4">
													<div class="flex-1">
														<Progress value={getProgressPercent(job)} class="h-2" />
													</div>
													<div class="text-sm text-muted-foreground min-w-[120px] text-right">
														{job.current_step}/{job.total_steps} ({getProgressPercent(job)}%)
													</div>
												</div>
												<div class="flex gap-4 mt-2 text-sm text-muted-foreground">
													{#if job.latest_loss !== null}
														<span>Loss: {job.latest_loss.toFixed(4)}</span>
													{/if}
													{#if job.checkpoints.length > 0}
														<span>Checkpoints: {job.checkpoints.length}</span>
													{/if}
													{#if getSamplesForJob(job).length > 0}
														<span>Samples: {getSamplesForJob(job).length}</span>
													{/if}
												</div>
												<!-- Preview thumbnails in collapsed view -->
												{@const samples = getSamplesForJob(job)}
												{#if samples.length > 0}
													<div class="flex gap-1 mt-3">
														{#each samples.slice(-4) as sample}
															<button
																type="button"
																onclick={(e: MouseEvent) => {
																	e.stopPropagation();
																	selectedSample = { jobId: job.job_id, sample };
																}}
																class="w-12 h-12 rounded border border-border overflow-hidden hover:ring-2 hover:ring-primary"
															>
																<img
																	src="/api/jobs/{job.job_id}/samples/{sample.filename}"
																	alt="Sample"
																	class="w-full h-full object-cover"
																/>
															</button>
														{/each}
													</div>
												{/if}
											</Card.Content>
										</Collapsible.Trigger>

										<Collapsible.Content>
											<Card.Content class="border-t border-border pt-4">
												<!-- Timing info -->
												<div class="grid grid-cols-3 gap-4 mb-6">
													<div>
														<div class="text-sm text-muted-foreground">Started</div>
														<div class="font-medium">{formatTime(job.started_at)}</div>
													</div>
													<div>
														<div class="text-sm text-muted-foreground">Completed</div>
														<div class="font-medium">{formatTime(job.completed_at)}</div>
													</div>
													<div>
														<div class="text-sm text-muted-foreground">Duration</div>
														<div class="font-medium">{formatDuration(job.started_at, job.completed_at)}</div>
													</div>
												</div>

												<!-- Training config -->
												{#if job.config}
													<div class="mb-6">
														<h3 class="text-sm font-medium text-muted-foreground mb-3">Training Config</h3>
														<div class="grid grid-cols-4 gap-3 text-sm">
															<div class="bg-muted/50 rounded px-3 py-2">
																<div class="text-muted-foreground text-xs">Steps</div>
																<div class="font-medium">{job.config.steps}</div>
															</div>
															<div class="bg-muted/50 rounded px-3 py-2">
																<div class="text-muted-foreground text-xs">Learning Rate</div>
																<div class="font-medium">{job.config.lr}</div>
															</div>
															<div class="bg-muted/50 rounded px-3 py-2">
																<div class="text-muted-foreground text-xs">LoRA Rank</div>
																<div class="font-medium">{job.config.lora_rank}</div>
															</div>
															<div class="bg-muted/50 rounded px-3 py-2">
																<div class="text-muted-foreground text-xs">Image Size</div>
																<div class="font-medium">{job.config.image_size}px</div>
															</div>
														</div>
													</div>
												{/if}

												{#if job.error_message}
													<div class="bg-destructive/10 text-destructive px-4 py-3 rounded-lg mb-6">
														<div class="font-medium mb-1">Error</div>
														<div class="text-sm whitespace-pre-wrap">
															{job.error_message.slice(0, 500)}
														</div>
													</div>
												{/if}

												{@const expandedSamples = getSamplesForJob(job)}
												{#if expandedSamples.length > 0}
													<div class="mb-6">
														<h3 class="text-sm font-medium text-muted-foreground mb-3">
															Sample Images ({expandedSamples.length})
														</h3>
														<div class="grid grid-cols-4 gap-2">
															{#each expandedSamples as sample}
																<button
																	type="button"
																	onclick={() => (selectedSample = { jobId: job.job_id, sample })}
																	class="w-full aspect-square rounded-lg border border-border overflow-hidden hover:ring-2 hover:ring-primary transition-all"
																>
																	<img
																		src="/api/jobs/{job.job_id}/samples/{sample.filename}"
																		alt="Training sample"
																		class="w-full h-full object-cover"
																	/>
																</button>
															{/each}
														</div>
													</div>
												{/if}

												{#if job.checkpoints.length > 0}
													<div>
														<h3 class="text-sm font-medium text-muted-foreground mb-3">
															Checkpoints
														</h3>
														<div class="space-y-2">
															{#each job.checkpoints as checkpoint, i}
																<div
																	class="flex items-center justify-between p-2 rounded-lg bg-muted/50"
																>
																	<div class="flex items-center gap-2">
																		<input
																			type="radio"
																			name="checkpoint-{job.job_id}"
																			value={checkpoint}
																			checked={selectedCheckpoint === checkpoint &&
																				promotingCheckpoint === job.job_id}
																			onchange={() => {
																				selectedCheckpoint = checkpoint;
																				promotingCheckpoint = job.job_id;
																			}}
																			class="w-4 h-4"
																		/>
																		<span class="font-mono text-sm">{checkpoint}</span>
																		{#if i === job.checkpoints.length - 1}
																			<Badge variant="secondary" class="text-xs">latest</Badge>
																		{/if}
																	</div>
																</div>
															{/each}
														</div>
														{#if promotingCheckpoint === job.job_id && selectedCheckpoint}
															<div class="mt-4 flex justify-end">
																<Button
																	onclick={() =>
																		promoteCheckpoint(job.job_id, selectedCheckpoint!)}
																	disabled={promoting}
																>
																	<Upload class="w-4 h-4 mr-2" />
																	{promoting ? 'Promoting...' : 'Promote to LoRAs'}
																</Button>
															</div>
														{/if}
													</div>
												{/if}
											</Card.Content>
										</Collapsible.Content>
									</Card.Root>
								</Collapsible.Root>
							{/each}
						</div>
					{/if}
				</Tabs.Content>
			{/each}
		</Tabs.Root>
	{/if}
</div>

<!-- New Training Dialog -->
<Dialog.Root bind:open={showNewTrainingModal}>
	<Dialog.Content class="max-w-lg">
		<Dialog.Header>
			<Dialog.Title>Start Training</Dialog.Title>
			<Dialog.Description>Configure and start a new LoRA training job</Dialog.Description>
		</Dialog.Header>

		<div class="space-y-4 py-4">
			<div class="space-y-2">
				<Label for="dataset">Dataset</Label>
				<Select.Root type="single" bind:value={selectedDatasetName} onValueChange={handleDatasetChange}>
					<Select.Trigger id="dataset">
						{selectedDataset ? `${selectedDataset.name} (${selectedDataset.image_count} images)` : 'Select a dataset'}
					</Select.Trigger>
					<Select.Content>
						{#each eligibleDatasets as dataset}
							<Select.Item value={dataset.name}>
								{dataset.name} ({dataset.image_count} images)
							</Select.Item>
						{/each}
					</Select.Content>
				</Select.Root>
				{#if eligibleDatasets.length === 0}
					<p class="text-sm text-muted-foreground">
						No datasets ready. Datasets need at least 5 images with captions.
					</p>
				{/if}
			</div>

			<div class="grid grid-cols-3 gap-4">
				<div class="space-y-2">
					<Label for="steps">Steps</Label>
					<Input id="steps" type="number" bind:value={trainingSteps} min={100} max={50000} />
				</div>
				<div class="space-y-2">
					<Label for="lr">Learning Rate</Label>
					<Input
						id="lr"
						type="number"
						bind:value={learningRate}
						min={0.00001}
						max={0.01}
						step={0.00001}
					/>
				</div>
				<div class="space-y-2">
					<Label for="rank">LoRA Rank</Label>
					<Input id="rank" type="number" bind:value={loraRank} min={4} max={128} />
				</div>
			</div>

			<div class="grid grid-cols-2 gap-4">
				<div class="space-y-2">
					<Label for="checkpoint">Checkpoint Every</Label>
					<Input id="checkpoint" type="number" bind:value={checkpointEvery} min={100} max={5000} />
				</div>
				<div class="space-y-2">
					<Label for="sample">Sample Every</Label>
					<Input id="sample" type="number" bind:value={sampleEvery} min={50} max={2000} />
				</div>
			</div>

			<div class="space-y-2">
				<Label for="prompts">Sample Prompts (one per line)</Label>
				<textarea
					id="prompts"
					bind:value={samplePrompts}
					rows={3}
					class="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring resize-none"
					placeholder="ohwx, portrait, studio lighting"
				></textarea>
			</div>
		</div>

		<Dialog.Footer>
			<Button variant="outline" onclick={() => (showNewTrainingModal = false)}>Cancel</Button>
			<Button onclick={startTraining} disabled={!selectedDataset || startingTraining}>
				{startingTraining ? 'Starting...' : 'Start Training'}
			</Button>
		</Dialog.Footer>
	</Dialog.Content>
</Dialog.Root>

<!-- Sample Image Detail Dialog -->
<Dialog.Root open={selectedSample !== null} onOpenChange={(open) => !open && (selectedSample = null)}>
	<Dialog.Content class="max-w-3xl">
		{#if selectedSample}
			<Dialog.Header>
				<Dialog.Title>Sample Image</Dialog.Title>
				<Dialog.Description>
					Step {selectedSample.sample.step ?? 'Unknown'}
				</Dialog.Description>
			</Dialog.Header>

			<div class="py-4 space-y-4">
				<div class="flex justify-center">
					<img
						src="/api/jobs/{selectedSample.jobId}/samples/{selectedSample.sample.filename}"
						alt="Training sample"
						class="max-h-[60vh] rounded-lg border border-border"
					/>
				</div>

				<div class="grid grid-cols-2 gap-4 text-sm">
					<div class="bg-muted/50 rounded px-3 py-2">
						<div class="text-muted-foreground text-xs">Step</div>
						<div class="font-medium">{selectedSample.sample.step ?? '-'}</div>
					</div>
					<div class="bg-muted/50 rounded px-3 py-2">
						<div class="text-muted-foreground text-xs">Generated At</div>
						<div class="font-medium">{formatSampleTimestamp(selectedSample.sample.timestamp)}</div>
					</div>
					{#if selectedSample.sample.prompt}
						<div class="col-span-2 bg-muted/50 rounded px-3 py-2">
							<div class="text-muted-foreground text-xs">Prompt</div>
							<div class="font-medium font-mono text-sm">{selectedSample.sample.prompt}</div>
						</div>
					{/if}
				</div>
			</div>

			<Dialog.Footer>
				<Button
					variant="outline"
					onclick={() => window.open(`/api/jobs/${selectedSample!.jobId}/samples/${selectedSample!.sample.filename}`, '_blank')}
				>
					Open Full Size
				</Button>
				<Button onclick={() => (selectedSample = null)}>Close</Button>
			</Dialog.Footer>
		{/if}
	</Dialog.Content>
</Dialog.Root>
