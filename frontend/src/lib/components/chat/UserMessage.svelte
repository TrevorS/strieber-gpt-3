<script lang="ts">
	import type { Message } from '$lib/stores/types';
	import { Button } from '$lib/components/ui/button';
	import { Pencil, Check, X } from 'lucide-svelte';

	let {
		message,
		editable = false,
		onedit
	}: {
		message: Message;
		editable?: boolean;
		onedit?: (newContent: string) => void;
	} = $props();

	let isEditing = $state(false);
	let editValue = $state('');
	let textareaRef: HTMLTextAreaElement | undefined = $state();

	function startEditing() {
		editValue = message.content;
		isEditing = true;
		// Focus the textarea after it's rendered
		setTimeout(() => textareaRef?.focus(), 0);
	}

	function saveEdit() {
		const trimmed = editValue.trim();
		if (trimmed && trimmed !== message.content && onedit) {
			onedit(trimmed);
		}
		isEditing = false;
	}

	function cancelEdit() {
		isEditing = false;
		editValue = '';
	}

	function handleKeyDown(e: KeyboardEvent) {
		if (e.key === 'Escape') {
			e.preventDefault();
			cancelEdit();
		}
	}
</script>

<div class="flex justify-end group">
	<div class="max-w-[80%] xl:max-w-[85%] 2xl:max-w-[90%] space-y-2">
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
		<div class="flex items-start gap-2">
			{#if editable && !isEditing}
				<Button
					variant="ghost"
					size="icon"
					class="h-8 w-8 opacity-0 group-hover:opacity-100 transition-opacity shrink-0"
					onclick={startEditing}
					data-testid="edit-button"
				>
					<Pencil class="h-4 w-4" />
				</Button>
			{/if}
			<div class="bg-primary text-primary-foreground rounded-lg px-4 py-2 flex-1">
				{#if isEditing}
					<textarea
						bind:this={textareaRef}
						bind:value={editValue}
						onkeydown={handleKeyDown}
						data-testid="edit-textarea"
						class="w-full min-h-[60px] bg-primary-foreground text-primary rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-ring resize-y"
						rows="3"
					></textarea>
					<div class="flex justify-end gap-2 mt-2">
						<Button
							variant="ghost"
							size="sm"
							onclick={cancelEdit}
							data-testid="cancel-button"
							class="text-primary-foreground hover:bg-primary-foreground/20"
						>
							<X class="h-4 w-4 mr-1" />
							Cancel
						</Button>
						<Button
							variant="ghost"
							size="sm"
							onclick={saveEdit}
							data-testid="save-button"
							class="text-primary-foreground hover:bg-primary-foreground/20"
						>
							<Check class="h-4 w-4 mr-1" />
							Save
						</Button>
					</div>
				{:else}
					<p class="whitespace-pre-wrap">{message.content}</p>
					{#if message.isEdited}
						<span class="text-xs opacity-70">(edited)</span>
					{/if}
				{/if}
			</div>
		</div>
	</div>
</div>
