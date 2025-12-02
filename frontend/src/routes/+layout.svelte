<script lang="ts">
	import '../app.css';
	import favicon from '$lib/assets/favicon.svg';

	let { children } = $props();
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
</svelte:head>

<div class="flex h-screen">
	<!-- Sidebar placeholder -->
	<aside class="w-64 border-r bg-sidebar text-sidebar-foreground hidden md:flex flex-col">
		<div class="p-4 border-b">
			<h1 class="font-semibold">Strieber</h1>
		</div>
		<div class="flex-1 p-4">
			<p class="text-sm text-muted-foreground">Conversations</p>
		</div>
	</aside>

	<!-- Main content -->
	<main class="flex-1 flex flex-col min-w-0">
		{@render children()}
	</main>
</div>
