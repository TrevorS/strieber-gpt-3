<script lang="ts">
	import type { Message } from '$lib/stores/types';
	import MarkdownContent from './MarkdownContent.svelte';

	let { message }: { message: Message } = $props();
</script>

<div class="flex justify-start">
	<div class="bg-muted rounded-lg px-4 py-2 max-w-[80%]">
		<MarkdownContent content={message.content} />
	</div>
</div>
