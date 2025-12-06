<script lang="ts">
	import '../app.css';
	import favicon from '$lib/assets/favicon.svg';
	import { goto } from '$app/navigation';
	import { browser } from '$app/environment';
	import { tick } from 'svelte';
	import { fade } from 'svelte/transition';
	import { conversationStore, settingsStore } from '$lib/stores';
	import { loadConversations, saveConversations } from '$lib/utils/storage';
	import { createShortcutHandler, type ShortcutAction } from '$lib/utils/shortcuts';
	import { ConversationList } from '$lib/components/sidebar';
	import { Button } from '$lib/components/ui/button';
	import { ToastContainer } from '$lib/components/ui/toast';
	import { Menu, PanelLeftClose, PanelLeft, Plus, MessageSquare } from 'lucide-svelte';
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

	function toggleCollapse() {
		settingsStore.toggleSidebarCollapsed();
		logger.ui.event('Sidebar', 'Toggle Collapse', { collapsed: settingsStore.sidebarCollapsed });
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

	function handleRename(id: string, title: string) {
		logger.ui.event('Sidebar', 'Rename', { id, title });
		conversationStore.updateTitle(id, title);
	}

	function handleExport(id: string) {
		const conversation = conversationStore.get(id);
		if (conversation) {
			logger.ui.event('Sidebar', 'Export', { id, title: conversation.title });
			// Import dynamically to avoid SSR issues
			import('$lib/utils/export').then(({ downloadConversationAsMarkdown }) => {
				downloadConversationAsMarkdown(conversation);
			});
		}
	}

	// Global keyboard shortcuts
	const shortcuts: ShortcutAction[] = [
		{
			key: 'n',
			cmdOrCtrl: true,
			handler: () => {
				logger.ui.event('Shortcut', 'New Chat', {});
				handleNew();
			},
			description: 'New chat'
		},
		{
			key: '/',
			cmdOrCtrl: true,
			handler: () => {
				logger.ui.event('Shortcut', 'Toggle Sidebar', {});
				toggleSidebar();
			},
			description: 'Toggle sidebar'
		},
		{
			key: 'Escape',
			handler: () => {
				// Close sidebar if open, otherwise let the event propagate to pages
				// for stopping streaming
				if (sidebarOpen) {
					logger.ui.event('Shortcut', 'Close Sidebar', {});
					closeSidebar();
				}
			},
			description: 'Close sidebar / Stop streaming'
		}
	];

	const handleKeydown = createShortcutHandler(shortcuts);
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
</svelte:head>

<!-- Global keyboard shortcuts -->
<svelte:window onkeydown={handleKeydown} />

<!-- Skip to main content link for keyboard users -->
<a
	href="#main-content"
	class="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-[100] focus:bg-background focus:text-foreground focus:px-4 focus:py-2 focus:rounded-md focus:shadow-lg focus:ring-2 focus:ring-ring"
>
	Skip to main content
</a>

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
		<h1 class="font-semibold ml-3 text-lg tracking-tight">Strieber GPT</h1>
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
		class="border-r bg-sidebar text-sidebar-foreground flex flex-col
			fixed md:static inset-y-0 left-0 z-50
			transform transition-all duration-300 ease-in-out
			{sidebarOpen ? 'translate-x-0' : '-translate-x-full'}
			md:translate-x-0
			{settingsStore.sidebarCollapsed ? 'md:w-16' : 'w-64'}"
		data-testid="sidebar"
	>
		<!-- Header -->
		<div class="h-14 px-4 border-b flex items-center shrink-0 {settingsStore.sidebarCollapsed ? 'justify-center' : 'justify-between'}">
			{#if !settingsStore.sidebarCollapsed}
				<h1 class="font-semibold text-lg tracking-tight whitespace-nowrap">Strieber GPT</h1>
			{/if}
			<Button
				variant="ghost"
				size="icon"
				onclick={toggleCollapse}
				class="hidden md:flex shrink-0"
				aria-label={settingsStore.sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
				data-testid="sidebar-collapse-toggle"
			>
				{#if settingsStore.sidebarCollapsed}
					<PanelLeft class="h-5 w-5" />
				{:else}
					<PanelLeftClose class="h-5 w-5" />
				{/if}
			</Button>
		</div>

		<!-- Collapsed: Icon rail -->
		{#if settingsStore.sidebarCollapsed}
			<div class="flex-1 flex flex-col items-center py-3 gap-2 overflow-hidden">
				<Button
					variant="ghost"
					size="icon"
					onclick={handleNew}
					aria-label="New chat"
					class="w-10 h-10"
					data-testid="new-chat-icon"
				>
					<Plus class="h-5 w-5" />
				</Button>
				<!-- Recent conversation indicators -->
				{#each conversationStore.sorted.slice(0, 5) as conv (conv.id)}
					<Button
						variant={conversationStore.activeId === conv.id ? 'secondary' : 'ghost'}
						size="icon"
						onclick={() => handleSelect(conv.id)}
						aria-label={conv.title}
						class="w-10 h-10"
					>
						<MessageSquare class="h-4 w-4" />
					</Button>
				{/each}
			</div>
		{:else}
			<!-- Expanded: Full conversation list -->
			<ConversationList
				loading={!loaded}
				conversations={conversationStore.sorted}
				activeId={conversationStore.activeId}
				onselect={handleSelect}
				onnew={handleNew}
				ondelete={handleDelete}
				onrename={handleRename}
				onexport={handleExport}
			/>
		{/if}
	</aside>

	<!-- Main content -->
	<main id="main-content" class="flex-1 flex flex-col min-w-0 pt-14 md:pt-0">
		{@render children()}
	</main>
</div>

<ToastContainer />
