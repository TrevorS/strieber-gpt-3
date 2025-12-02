<script lang="ts">
	import type { Message } from '$lib/stores/types';
	import UserMessage from './UserMessage.svelte';
	import AssistantMessage from './AssistantMessage.svelte';

	let { messages }: { messages: Message[] } = $props();

	let container: HTMLDivElement;

	$effect(() => {
		messages; // track dependency
		if (container) {
			container.scrollTop = container.scrollHeight;
		}
	});
</script>

<div bind:this={container} class="flex-1 overflow-y-auto p-4">
	<div class="max-w-3xl mx-auto space-y-4">
		{#each messages as message (message.id)}
			{#if message.role === 'user'}
				<UserMessage {message} />
			{:else}
				<AssistantMessage {message} />
			{/if}
		{/each}
	</div>
</div>
