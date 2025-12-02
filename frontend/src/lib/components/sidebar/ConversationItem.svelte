<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { Trash2 } from 'lucide-svelte';
	import type { Conversation } from '$lib/stores/types';

	let {
		conversation,
		isActive,
		onselect,
		ondelete
	}: {
		conversation: Conversation;
		isActive: boolean;
		onselect: () => void;
		ondelete: () => void;
	} = $props();

	function handleDelete(e: MouseEvent) {
		e.stopPropagation();
		ondelete();
	}
</script>

<button
	type="button"
	class="w-full text-left px-3 py-2 rounded-md flex items-center gap-2 group transition-colors
		{isActive ? 'bg-sidebar-accent text-sidebar-accent-foreground' : 'hover:bg-sidebar-accent/50'}"
	onclick={onselect}
>
	<span class="flex-1 truncate text-sm">{conversation.title}</span>
	<Button
		variant="ghost"
		size="icon"
		class="h-6 w-6 opacity-0 group-hover:opacity-100 transition-opacity shrink-0"
		onclick={handleDelete}
		data-testid="delete-button"
	>
		<Trash2 class="h-3 w-3" />
	</Button>
</button>
