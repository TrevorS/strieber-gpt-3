<script lang="ts">
	import * as Tooltip from '$lib/components/ui/tooltip';
	import ConversationItem from '../ConversationItem.svelte';
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
</script>

<Tooltip.Provider>
	<ConversationItem
		{conversation}
		{isActive}
		{isTitlePending}
		{onselect}
		{ondelete}
		{onrename}
		{onexport}
	/>
</Tooltip.Provider>
