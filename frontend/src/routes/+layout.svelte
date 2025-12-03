<script lang="ts">
	import '../app.css';
	import favicon from '$lib/assets/favicon.svg';
	import { goto } from '$app/navigation';
	import { browser } from '$app/environment';
	import { conversationStore } from '$lib/stores';
	import { loadConversations, saveConversations } from '$lib/utils/storage';
	import { ConversationList } from '$lib/components/sidebar';

	let { children } = $props();

	// Load conversations from localStorage on mount (browser only)
	// Always start with New Chat state (activeId = null) on refresh
	if (browser) {
		const saved = loadConversations();
		if (saved) {
			conversationStore.load(saved.conversations, null);
		}
	}

	// Persist on every change
	$effect(() => {
		if (browser) {
			saveConversations(conversationStore.conversations, conversationStore.activeId);
		}
	});

	function handleSelect(id: string) {
		goto(`/c/${id}`);
	}

	function handleNew() {
		conversationStore.setActive(null);
		goto('/');
	}

	function handleDelete(id: string) {
		const wasActive = conversationStore.activeId === id;
		conversationStore.delete(id);

		// If we deleted the active conversation, navigate appropriately
		if (wasActive) {
			if (conversationStore.activeId) {
				// Switched to another conversation
				goto(`/c/${conversationStore.activeId}`);
			} else {
				// No conversations left
				goto('/');
			}
		}
	}
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
</svelte:head>

<div class="flex h-screen">
	<!-- Sidebar -->
	<aside class="w-64 border-r bg-sidebar text-sidebar-foreground hidden md:flex flex-col">
		<div class="p-4 border-b">
			<h1 class="font-semibold">Strieber</h1>
		</div>
		<ConversationList
			conversations={conversationStore.sorted}
			activeId={conversationStore.activeId}
			onselect={handleSelect}
			onnew={handleNew}
			ondelete={handleDelete}
		/>
	</aside>

	<!-- Main content -->
	<main class="flex-1 flex flex-col min-w-0">
		{@render children()}
	</main>
</div>
