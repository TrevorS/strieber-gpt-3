<script lang="ts">
	import '../app.css';
	import favicon from '$lib/assets/favicon.svg';
	import { goto } from '$app/navigation';
	import { browser } from '$app/environment';
	import { tick } from 'svelte';
	import { fade } from 'svelte/transition';
	import { conversationStore, settingsStore } from '$lib/stores';
	import { loadConversations, saveConversations } from '$lib/utils/storage';
	import { ConversationList } from '$lib/components/sidebar';
	import { Button } from '$lib/components/ui/button';
	import { ToastContainer } from '$lib/components/ui/toast';
	import { Menu } from 'lucide-svelte';
	import { logger } from '$lib/utils/logger';

	let { children } = $props();

	// Mobile sidebar state
	let sidebarOpen = $state(false);
	let loaded = $state(false);

	function toggleSidebar() {
		sidebarOpen = !sidebarOpen;
		logger.ui.event('Sidebar', 'Toggle', { open: sidebarOpen });
	}

	function closeSidebar() {
		if (sidebarOpen) {
			sidebarOpen = false;
			logger.ui.event('Sidebar', 'Close', {});
		}
	}

	// Load conversations from localStorage on mount (browser only)
	// Always start with New Chat state on refresh (activeId = null)
	if (browser) {
		logger.lifecycle.mount('Layout', { browser: true });
		const saved = loadConversations();
		if (saved) {
			logger.info('persistence', 'Loading from localStorage', {
				conversationCount: saved.conversations.length,
				savedActiveId: saved.activeId,
				settingActiveIdTo: null
			});
			conversationStore.load(saved.conversations, null);
		} else {
			logger.info('persistence', 'No saved conversations found');
		}
		// Minimum loading time to prevent skeleton flash
		setTimeout(() => {
			loaded = true;
		}, 350);
	}

	// Persist on every change
	$effect(() => {
		if (browser) {
			logger.debug('persistence', 'Saving to localStorage', {
				conversationCount: conversationStore.conversations.length,
				activeId: conversationStore.activeId
			});
			saveConversations(conversationStore.conversations, conversationStore.activeId);
		}
	});

	// Apply theme to document
	$effect(() => {
		if (browser) {
			const theme = settingsStore.theme;
			let isDark = false;

			if (theme === 'dark') {
				isDark = true;
			} else if (theme === 'system') {
				isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
			}

			if (isDark) {
				document.documentElement.classList.add('dark');
			} else {
				document.documentElement.classList.remove('dark');
			}

			logger.debug('ui', 'Theme applied', { theme, isDark });
		}
	});

	// Listen for system theme changes when in system mode
	$effect(() => {
		if (browser && settingsStore.theme === 'system') {
			const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
			const handler = (e: MediaQueryListEvent) => {
				if (e.matches) {
					document.documentElement.classList.add('dark');
				} else {
					document.documentElement.classList.remove('dark');
				}
				logger.debug('ui', 'System theme changed', { isDark: e.matches });
			};

			mediaQuery.addEventListener('change', handler);
			return () => mediaQuery.removeEventListener('change', handler);
		}
	});

	function handleSelect(id: string) {
		logger.ui.event('Sidebar', 'Conversation selected', { id });
		closeSidebar();
		goto(`/c/${id}`);
	}

	async function handleNew() {
		logger.ui.event('Sidebar', 'New Chat clicked', { previousActiveId: conversationStore.activeId });
		closeSidebar();
		conversationStore.setActive(null);
		await tick();
		await goto('/');
	}

	function handleDelete(id: string) {
		logger.ui.event('Sidebar', 'Delete clicked', { id, wasActive: conversationStore.activeId === id });
		const wasActive = conversationStore.activeId === id;
		conversationStore.delete(id);

		// If we deleted the active conversation, navigate appropriately
		if (wasActive) {
			if (conversationStore.activeId) {
				// Switched to another conversation
				logger.nav.navigate('current', `/c/${conversationStore.activeId}`, { reason: 'deleted-active' });
				goto(`/c/${conversationStore.activeId}`);
			} else {
				// No conversations left
				logger.nav.navigate('current', '/', { reason: 'deleted-last' });
				goto('/');
			}
		}
	}
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
</svelte:head>

<!-- Escape key handler -->
<svelte:window onkeydown={(e) => e.key === 'Escape' && sidebarOpen && closeSidebar()} />

<div class="flex h-screen">
	<!-- Mobile header -->
	<header
		class="md:hidden fixed top-0 left-0 right-0 h-14 bg-background border-b flex items-center px-4 z-40"
	>
		<Button
			variant="ghost"
			size="icon"
			onclick={toggleSidebar}
			aria-label="Toggle sidebar"
			data-testid="sidebar-toggle"
		>
			<Menu class="h-5 w-5" />
		</Button>
		<h1 class="font-semibold ml-3">Strieber</h1>
	</header>

	<!-- Backdrop overlay (mobile only) -->
	{#if sidebarOpen}
		<div
			class="md:hidden fixed inset-0 bg-black/50 z-40"
			onclick={closeSidebar}
			onkeydown={(e) => e.key === 'Enter' && closeSidebar()}
			role="button"
			tabindex="-1"
			aria-label="Close sidebar"
			transition:fade={{ duration: 200 }}
			data-testid="sidebar-backdrop"
		></div>
	{/if}

	<!-- Sidebar -->
	<aside
		class="w-64 border-r bg-sidebar text-sidebar-foreground flex flex-col
			fixed md:static inset-y-0 left-0 z-50
			transform transition-transform duration-300 ease-in-out
			{sidebarOpen ? 'translate-x-0' : '-translate-x-full'}
			md:translate-x-0"
		data-testid="sidebar"
	>
		<div class="p-4 border-b">
			<h1 class="font-semibold">Strieber</h1>
		</div>
		<ConversationList
			loading={!loaded}
			conversations={conversationStore.sorted}
			activeId={conversationStore.activeId}
			onselect={handleSelect}
			onnew={handleNew}
			ondelete={handleDelete}
		/>
	</aside>

	<!-- Main content -->
	<main class="flex-1 flex flex-col min-w-0 pt-14 md:pt-0">
		{@render children()}
	</main>
</div>

<ToastContainer />
