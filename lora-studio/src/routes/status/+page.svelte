<script lang="ts">
	import { ArrowLeft, RefreshCw, Play, Square } from 'lucide-svelte';
	import * as Table from '$lib/components/ui/table';
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';

	interface Container {
		id: string;
		name: string;
		status: string;
		image: string;
		ports: { container_port: string; host_port: string }[];
	}

	let containers = $state<Container[]>([]);
	let loading = $state(true);
	let error = $state<string | null>(null);
	let actionInProgress = $state<string | null>(null);

	async function fetchContainers() {
		loading = true;
		error = null;
		try {
			const res = await fetch('/api/containers');
			if (!res.ok) {
				const data = await res.json();
				throw new Error(data.error || 'Failed to fetch containers');
			}
			containers = await res.json();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Unknown error';
		} finally {
			loading = false;
		}
	}

	async function stopContainer(name: string) {
		actionInProgress = name;
		try {
			const res = await fetch(`/api/containers/${name}/stop`, { method: 'POST' });
			if (!res.ok) {
				const data = await res.json();
				throw new Error(data.message || 'Failed to stop container');
			}
			await fetchContainers();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Unknown error';
		} finally {
			actionInProgress = null;
		}
	}

	async function startContainer(name: string) {
		actionInProgress = name;
		try {
			const res = await fetch(`/api/containers/${name}/start`, { method: 'POST' });
			if (!res.ok) {
				const data = await res.json();
				throw new Error(data.message || 'Failed to start container');
			}
			await fetchContainers();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Unknown error';
		} finally {
			actionInProgress = null;
		}
	}

	function getStatusVariant(status: string): 'default' | 'secondary' | 'destructive' | 'outline' {
		switch (status) {
			case 'running':
				return 'default';
			case 'exited':
			case 'dead':
				return 'destructive';
			default:
				return 'secondary';
		}
	}

	function formatPorts(ports: Container['ports']): string {
		if (!ports || ports.length === 0) return '-';
		return ports.map((p) => `${p.host_port}:${p.container_port.split('/')[0]}`).join(', ');
	}

	// Initial fetch
	$effect(() => {
		fetchContainers();
	});
</script>

<div class="container mx-auto py-6 px-4">
	<a
		href="/"
		class="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground mb-4"
	>
		<ArrowLeft class="w-4 h-4" />
		LoRA Studio
	</a>

	<div class="flex items-center justify-between mb-6">
		<div>
			<h1 class="text-2xl font-bold">System Status</h1>
			<p class="text-muted-foreground">Manage running containers</p>
		</div>
		<Button variant="outline" size="sm" onclick={fetchContainers} disabled={loading}>
			<RefreshCw class="w-4 h-4 mr-2 {loading ? 'animate-spin' : ''}" />
			Refresh
		</Button>
	</div>

	{#if error}
		<div class="bg-destructive/10 text-destructive px-4 py-3 rounded-md mb-4">
			{error}
		</div>
	{/if}

	{#if loading && containers.length === 0}
		<div class="text-center py-12 text-muted-foreground">Loading containers...</div>
	{:else if containers.length === 0}
		<div class="text-center py-12 text-muted-foreground">No containers found</div>
	{:else}
		<Table.Root>
			<Table.Header>
				<Table.Row>
					<Table.Head class="w-[250px]">Container</Table.Head>
					<Table.Head>Image</Table.Head>
					<Table.Head>Ports</Table.Head>
					<Table.Head>Status</Table.Head>
					<Table.Head class="text-right">Actions</Table.Head>
				</Table.Row>
			</Table.Header>
			<Table.Body>
				{#each containers as container (container.id)}
					<Table.Row>
						<Table.Cell class="font-mono text-sm">
							{container.name.replace('strieber-', '')}
						</Table.Cell>
						<Table.Cell class="text-muted-foreground text-sm">
							{container.image.split(':')[0].replace('strieber-', '')}
						</Table.Cell>
						<Table.Cell class="font-mono text-sm">
							{formatPorts(container.ports)}
						</Table.Cell>
						<Table.Cell>
							<Badge variant={getStatusVariant(container.status)}>
								{container.status}
							</Badge>
						</Table.Cell>
						<Table.Cell class="text-right">
							{#if container.status === 'running'}
								<Button
									variant="outline"
									size="sm"
									onclick={() => stopContainer(container.name)}
									disabled={actionInProgress === container.name}
								>
									<Square class="w-3 h-3 mr-1" />
									Stop
								</Button>
							{:else}
								<Button
									variant="outline"
									size="sm"
									onclick={() => startContainer(container.name)}
									disabled={actionInProgress === container.name}
								>
									<Play class="w-3 h-3 mr-1" />
									Start
								</Button>
							{/if}
						</Table.Cell>
					</Table.Row>
				{/each}
			</Table.Body>
		</Table.Root>

		<p class="text-sm text-muted-foreground mt-4">
			Tip: Stop <code class="px-1 bg-muted rounded">llama-server</code> before training to free GPU
			memory.
		</p>
	{/if}
</div>
