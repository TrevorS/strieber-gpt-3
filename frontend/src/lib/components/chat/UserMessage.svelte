<script lang="ts">
	import type { Message } from '$lib/stores/types';

	let { message }: { message: Message } = $props();
</script>

<div class="flex justify-end">
	<div class="max-w-[80%] space-y-2">
		{#if message.attachments?.length}
			<div class="flex flex-wrap justify-end gap-2">
				{#each message.attachments as attachment (attachment.id)}
					{#if attachment.type === 'image'}
						<img
							src={attachment.content}
							alt={attachment.name}
							class="rounded-lg max-h-64 object-contain"
						/>
					{/if}
				{/each}
			</div>
		{/if}
		<div class="bg-primary text-primary-foreground rounded-lg px-4 py-2">
			<p class="whitespace-pre-wrap">{message.content}</p>
		</div>
	</div>
</div>
