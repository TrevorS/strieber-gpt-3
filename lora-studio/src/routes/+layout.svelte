<script lang="ts">
	import '../app.css';
	import favicon from '$lib/assets/favicon.svg';

	let { children } = $props();
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
	<title>LoRA Studio</title>
</svelte:head>

<div class="min-h-screen bg-background">
	{@render children()}
</div>
