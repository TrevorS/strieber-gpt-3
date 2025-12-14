<script lang="ts">
	import * as Tooltip from '$lib/components/ui/tooltip';
	import ConversationList from '../ConversationList.svelte';
	import type { Conversation } from '$lib/stores/types';

	let {
		conversations = [],
		activeId = null,
		loading = false,
		onselect = () => {},
		onnew = () => {},
		ondelete = () => {},
		onrename,
		onexport
	}: {
		conversations?: Conversation[];
		activeId?: string | null;
		loading?: boolean;
		onselect?: (id: string) => void;
		onnew?: () => void;
		ondelete?: (id: string) => void;
		onrename?: (id: string, title: string) => void;
		onexport?: (id: string) => void;
	} = $props();
</script>

<Tooltip.Provider>
	<ConversationList
		{conversations}
		{activeId}
		{loading}
		{onselect}
		{onnew}
		{ondelete}
		{onrename}
		{onexport}
	/>
</Tooltip.Provider>
