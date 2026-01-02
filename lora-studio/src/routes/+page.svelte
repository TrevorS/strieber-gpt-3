<script lang="ts">
	import { Layers, FolderOpen, Sparkles, Play, Activity } from 'lucide-svelte';
</script>

<div class="container mx-auto px-4 py-8">
	<header class="mb-8">
		<h1 class="text-3xl font-bold text-foreground">LoRA Studio</h1>
		<p class="text-muted-foreground mt-2">
			Manage datasets, train LoRAs, and generate images
		</p>
	</header>

	<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
		<a
			href="/datasets"
			class="group p-6 rounded-lg border border-border bg-card hover:bg-accent transition-colors"
		>
			<FolderOpen class="w-8 h-8 text-primary mb-3" />
			<h2 class="text-lg font-semibold text-card-foreground group-hover:text-accent-foreground">
				Datasets
			</h2>
			<p class="text-sm text-muted-foreground mt-1">
				Create and manage training datasets
			</p>
		</a>

		<a
			href="/training"
			class="group p-6 rounded-lg border border-border bg-card hover:bg-accent transition-colors"
		>
			<Play class="w-8 h-8 text-primary mb-3" />
			<h2 class="text-lg font-semibold text-card-foreground group-hover:text-accent-foreground">
				Training
			</h2>
			<p class="text-sm text-muted-foreground mt-1">
				Train and monitor LoRA jobs
			</p>
		</a>

		<a
			href="/loras"
			class="group p-6 rounded-lg border border-border bg-card hover:bg-accent transition-colors"
		>
			<Layers class="w-8 h-8 text-primary mb-3" />
			<h2 class="text-lg font-semibold text-card-foreground group-hover:text-accent-foreground">
				LoRAs
			</h2>
			<p class="text-sm text-muted-foreground mt-1">
				Browse and manage trained LoRAs
			</p>
		</a>

		<a
			href="/generate"
			class="group p-6 rounded-lg border border-border bg-card hover:bg-accent transition-colors"
		>
			<Sparkles class="w-8 h-8 text-primary mb-3" />
			<h2 class="text-lg font-semibold text-card-foreground group-hover:text-accent-foreground">
				Generate
			</h2>
			<p class="text-sm text-muted-foreground mt-1">
				Generate images with LoRAs
			</p>
		</a>

		<a
			href="/status"
			class="group p-6 rounded-lg border border-border bg-card hover:bg-accent transition-colors"
		>
			<Activity class="w-8 h-8 text-primary mb-3" />
			<h2 class="text-lg font-semibold text-card-foreground group-hover:text-accent-foreground">
				Status
			</h2>
			<p class="text-sm text-muted-foreground mt-1">
				Manage running containers
			</p>
		</a>
	</div>
</div>
