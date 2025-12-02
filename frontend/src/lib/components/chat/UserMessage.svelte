<script lang="ts">
	import type { Message } from '$lib/stores/types';

	let { message }: { message: Message } = $props();
</script>

<div class="flex justify-end">
	<div class="bg-primary text-primary-foreground rounded-lg px-4 py-2 max-w-[80%]">
		<p class="whitespace-pre-wrap">{message.content}</p>
	</div>
</div>
