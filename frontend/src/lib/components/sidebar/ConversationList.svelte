<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { ScrollArea } from '$lib/components/ui/scroll-area';
	import { Plus } from 'lucide-svelte';
	import type { Conversation } from '$lib/stores/types';
	import {
		DATE_GROUP_LABELS,
		DATE_GROUP_ORDER,
		groupConversationsByDate,
		type DateGroup
	} from '$lib/utils/dates';
	import ConversationItem from './ConversationItem.svelte';

	let {
		conversations,
		activeId,
		onselect,
		onnew,
		ondelete
	}: {
		conversations: Conversation[];
		activeId: string | null;
		onselect: (id: string) => void;
		onnew: () => void;
		ondelete: (id: string) => void;
	} = $props();

	// Group conversations by date
	let groupedConversations = $derived(groupConversationsByDate(conversations));

	// Get groups in order, only including non-empty ones
	let orderedGroups = $derived(
		DATE_GROUP_ORDER.filter((group) => groupedConversations.has(group)) as DateGroup[]
	);
</script>

<div class="flex flex-col h-full">
	<div class="p-3">
		<Button onclick={onnew} class="w-full" variant="outline">
			<Plus class="h-4 w-4 mr-2" />
			New Chat
		</Button>
	</div>

	<ScrollArea class="flex-1">
		<div class="px-2 pb-4">
			{#if conversations.length === 0}
				<p class="text-sm text-muted-foreground px-3 py-2">No conversations yet</p>
			{:else}
				{#each orderedGroups as group (group)}
					<div class="mb-4">
						<h3 class="px-3 py-1 text-xs font-medium text-muted-foreground">
							{DATE_GROUP_LABELS[group]}
						</h3>
						<div class="space-y-1">
							{#each groupedConversations.get(group) ?? [] as conversation (conversation.id)}
								<ConversationItem
									{conversation}
									isActive={conversation.id === activeId}
									onselect={() => onselect(conversation.id)}
									ondelete={() => ondelete(conversation.id)}
								/>
							{/each}
						</div>
					</div>
				{/each}
			{/if}
		</div>
	</ScrollArea>
</div>
