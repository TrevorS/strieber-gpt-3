<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { Download, Pencil, Trash2 } from 'lucide-svelte';
	import type { Conversation } from '$lib/stores/types';

	let {
		conversation,
		isActive,
		onselect,
		ondelete,
		onrename,
		onexport
	}: {
		conversation: Conversation;
		isActive: boolean;
		onselect: () => void;
		ondelete: () => void;
		onrename?: (newTitle: string) => void;
		onexport?: () => void;
	} = $props();

	let isEditing = $state(false);
	let editValue = $state('');
	let inputRef: HTMLInputElement | undefined = $state();

	function handleDelete(e: MouseEvent) {
		e.stopPropagation();
		ondelete();
	}

	function handleEdit(e: MouseEvent) {
		e.stopPropagation();
		startEditing();
	}

	function handleDoubleClick(e: MouseEvent) {
		if (onrename) {
			e.stopPropagation();
			startEditing();
		}
	}

	function startEditing() {
		editValue = conversation.title;
		isEditing = true;
		// Focus the input after it's rendered
		setTimeout(() => inputRef?.focus(), 0);
	}

	function saveEdit() {
		const trimmed = editValue.trim();
		if (trimmed && trimmed !== conversation.title && onrename) {
			onrename(trimmed);
		}
		isEditing = false;
	}

	function cancelEdit() {
		isEditing = false;
		editValue = '';
	}

	function handleKeyDown(e: KeyboardEvent) {
		if (e.key === 'Enter') {
			e.preventDefault();
			saveEdit();
		} else if (e.key === 'Escape') {
			e.preventDefault();
			cancelEdit();
		}
	}

	function handleInputClick(e: MouseEvent) {
		e.stopPropagation();
	}

	function handleExport(e: MouseEvent) {
		e.stopPropagation();
		onexport?.();
	}
</script>

<button
	type="button"
	class="w-full text-left px-3 py-2 rounded-md flex items-center gap-2 group transition-colors
		{isActive ? 'bg-sidebar-accent text-sidebar-accent-foreground' : 'hover:bg-sidebar-accent/50'}"
	onclick={onselect}
	aria-current={isActive ? 'page' : undefined}
	data-testid="conversation-item"
>
	{#if isEditing}
		<input
			bind:this={inputRef}
			type="text"
			bind:value={editValue}
			onkeydown={handleKeyDown}
			onblur={saveEdit}
			onclick={handleInputClick}
			data-testid="rename-input"
			class="flex-1 text-sm bg-background border border-input rounded px-2 py-0.5 focus:outline-none focus:ring-1 focus:ring-ring"
		/>
	{:else}
		<!-- svelte-ignore a11y_no_static_element_interactions -->
		<span
			class="flex-1 truncate text-sm"
			ondblclick={handleDoubleClick}
		>
			{conversation.title}
		</span>
	{/if}
	{#if !isEditing}
		{#if onrename}
			<Button
				variant="ghost"
				size="icon"
				class="h-6 w-6 opacity-0 group-hover:opacity-100 transition-opacity shrink-0"
				onclick={handleEdit}
				data-testid="edit-button"
			>
				<Pencil class="h-3 w-3" />
			</Button>
		{/if}
		{#if onexport}
			<Button
				variant="ghost"
				size="icon"
				class="h-6 w-6 opacity-0 group-hover:opacity-100 transition-opacity shrink-0"
				onclick={handleExport}
				data-testid="export-button"
			>
				<Download class="h-3 w-3" />
			</Button>
		{/if}
		<Button
			variant="ghost"
			size="icon"
			class="h-6 w-6 opacity-0 group-hover:opacity-100 transition-opacity shrink-0"
			onclick={handleDelete}
			data-testid="delete-button"
		>
			<Trash2 class="h-3 w-3" />
		</Button>
	{/if}
</button>
