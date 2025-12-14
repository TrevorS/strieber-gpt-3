<script lang="ts">
	import { fade } from 'svelte/transition';
	import { Button } from '$lib/components/ui/button';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { Download, Pencil, Trash2 } from 'lucide-svelte';
	import type { Conversation } from '$lib/stores/types';

	let {
		conversation,
		isActive,
		isTitlePending = false,
		onselect,
		ondelete,
		onrename,
		onexport
	}: {
		conversation: Conversation;
		isActive: boolean;
		isTitlePending?: boolean;
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
	class="w-full text-left px-3 py-2 rounded-md flex items-center gap-2 group transition-colors relative overflow-hidden
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
		<Tooltip.Root delayDuration={400}>
			<Tooltip.Trigger>
				{#snippet child({ props })}
					<!-- svelte-ignore a11y_no_static_element_interactions -->
					<span
						{...props}
						class="flex-1 truncate text-sm min-w-0"
						ondblclick={handleDoubleClick}
					>
						{#if isTitlePending}
							<span
								class="inline-block bg-muted animate-pulse rounded h-4 w-24"
								in:fade={{ duration: 150 }}
								out:fade={{ duration: 100 }}
							></span>
						{:else}
							<span in:fade={{ duration: 200, delay: 50 }}>
								{conversation.title}
							</span>
						{/if}
					</span>
				{/snippet}
			</Tooltip.Trigger>
			{#if !isTitlePending}
				<Tooltip.Content side="right" class="max-w-xs">
					{conversation.title}
				</Tooltip.Content>
			{/if}
		</Tooltip.Root>

		<!-- Action buttons overlay - absolutely positioned so title gets full width -->
		<div
			class="absolute right-0 inset-y-0 flex items-center gap-0.5
				opacity-0 group-hover:opacity-100 transition-opacity duration-200
				bg-gradient-to-l from-sidebar-accent via-sidebar-accent to-transparent
				pl-6 pr-2"
		>
			{#if onrename}
				<Button
					variant="ghost"
					size="icon"
					class="h-6 w-6"
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
					class="h-6 w-6"
					onclick={handleExport}
					data-testid="export-button"
				>
					<Download class="h-3 w-3" />
				</Button>
			{/if}
			<Button
				variant="ghost"
				size="icon"
				class="h-6 w-6"
				onclick={handleDelete}
				data-testid="delete-button"
			>
				<Trash2 class="h-3 w-3" />
			</Button>
		</div>
	{/if}
</button>
