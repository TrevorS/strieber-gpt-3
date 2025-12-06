<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { ScrollArea } from '$lib/components/ui/scroll-area';
	import { Plus, Search } from 'lucide-svelte';
	import type { Conversation } from '$lib/stores/types';
	import {
		DATE_GROUP_LABELS,
		DATE_GROUP_ORDER,
		groupConversationsByDate,
		type DateGroup
	} from '$lib/utils/dates';
	import ConversationItem from './ConversationItem.svelte';

	let {
		loading = false,
		conversations,
		activeId,
		onselect,
		onnew,
		ondelete,
		onrename,
		onexport
	}: {
		loading?: boolean;
		conversations: Conversation[];
		activeId: string | null;
		onselect: (id: string) => void;
		onnew: () => void;
		ondelete: (id: string) => void;
		onrename?: (id: string, title: string) => void;
		onexport?: (id: string) => void;
	} = $props();

	// Search state
	let searchQuery = $state('');

	// Filter conversations by search query (case-insensitive)
	let filteredConversations = $derived(
		searchQuery.trim()
			? conversations.filter((c) =>
					c.title.toLowerCase().includes(searchQuery.trim().toLowerCase())
				)
			: conversations
	);

	// Group filtered conversations by date
	let groupedConversations = $derived(groupConversationsByDate(filteredConversations));

	// Get groups in order, only including non-empty ones
	let orderedGroups = $derived(
		DATE_GROUP_ORDER.filter((group) => groupedConversations.has(group)) as DateGroup[]
	);

	// Check if search has no results
	let noSearchResults = $derived(
		searchQuery.trim() && filteredConversations.length === 0
	);
</script>

<div class="flex flex-col h-full">
	<div class="p-3 space-y-2">
		<Button onclick={onnew} class="w-full" variant="outline">
			<Plus class="h-4 w-4 mr-2" />
			New Chat
		</Button>
		<div class="relative">
			<Search class="absolute left-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground pointer-events-none" />
			<input
				type="text"
				bind:value={searchQuery}
				placeholder="Search conversations..."
				class="w-full pl-8 pr-3 py-1.5 text-sm bg-background border border-input rounded-md focus:outline-none focus:ring-1 focus:ring-ring"
			/>
		</div>
	</div>

	<ScrollArea class="flex-1">
		<div class="px-2 pb-4">
			{#if loading}
				<div class="px-1 py-2 space-y-2">
					{#each [1, 2, 3] as _}
						<div class="h-9 bg-muted animate-pulse rounded-md"></div>
					{/each}
				</div>
			{:else if conversations.length === 0}
				<p class="text-sm text-muted-foreground px-3 py-2">No conversations yet</p>
			{:else if noSearchResults}
				<p class="text-sm text-muted-foreground px-3 py-2">No matching conversations</p>
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
									onrename={onrename ? (title) => onrename(conversation.id, title) : undefined}
									onexport={onexport ? () => onexport(conversation.id) : undefined}
								/>
							{/each}
						</div>
					</div>
				{/each}
			{/if}
		</div>
	</ScrollArea>
</div>
