<script lang="ts">
	import { X, FileText } from 'lucide-svelte';
	import type { Attachment } from '$lib/utils/files';

	let {
		attachments,
		onremove
	}: {
		attachments: Attachment[];
		onremove: (id: string) => void;
	} = $props();
</script>

{#if attachments.length > 0}
	<div class="flex flex-wrap gap-2 p-2 border-b bg-muted/30">
		{#each attachments as attachment (attachment.id)}
			<div
				class="relative group flex items-center gap-2 px-2 py-1 bg-background border rounded-md text-sm"
			>
				{#if attachment.type === 'image'}
					<img
						src={attachment.content}
						alt={attachment.name}
						class="h-10 w-10 object-cover rounded"
					/>
				{:else}
					<FileText class="h-5 w-5 text-muted-foreground" />
				{/if}
				<span class="max-w-[120px] truncate text-muted-foreground">{attachment.name}</span>
				<button
					onclick={() => onremove(attachment.id)}
					class="absolute -top-1 -right-1 p-0.5 bg-destructive text-destructive-foreground rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
					aria-label="Remove {attachment.name}"
				>
					<X class="h-3 w-3" />
				</button>
			</div>
		{/each}
	</div>
{/if}
