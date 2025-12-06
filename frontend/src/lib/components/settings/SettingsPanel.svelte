<script lang="ts">
	import { fly } from 'svelte/transition';
	import { X } from 'lucide-svelte';
	import { Button } from '$lib/components/ui/button';
	import ThemeToggle from './ThemeToggle.svelte';
	import TemperatureSlider from './TemperatureSlider.svelte';
	import SystemPromptEditor from './SystemPromptEditor.svelte';
	import ToolToggles from './ToolToggles.svelte';

	let { open = false, onclose }: { open: boolean; onclose: () => void } = $props();

	let panelRef: HTMLDivElement | undefined = $state();
	let previousActiveElement: HTMLElement | null = $state(null);

	// Focusable element selector
	const FOCUSABLE_SELECTOR =
		'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])';

	function getFocusableElements(): HTMLElement[] {
		if (!panelRef) return [];
		return Array.from(panelRef.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR));
	}

	function handleKeydown(event: KeyboardEvent) {
		if (event.key === 'Escape') {
			onclose();
			return;
		}

		// Focus trap: handle Tab key
		if (event.key === 'Tab' && open && panelRef) {
			const focusable = getFocusableElements();
			if (focusable.length === 0) return;

			const first = focusable[0];
			const last = focusable[focusable.length - 1];

			if (event.shiftKey) {
				// Shift+Tab: if on first element, go to last
				if (document.activeElement === first) {
					event.preventDefault();
					last.focus();
				}
			} else {
				// Tab: if on last element, go to first
				if (document.activeElement === last) {
					event.preventDefault();
					first.focus();
				}
			}
		}
	}

	// Focus first element when panel opens, restore focus on close
	$effect(() => {
		if (open) {
			// Store the previously focused element
			previousActiveElement = document.activeElement as HTMLElement;

			// Focus the first focusable element after transition
			setTimeout(() => {
				const focusable = getFocusableElements();
				if (focusable.length > 0) {
					focusable[0].focus();
				}
			}, 50);
		} else if (previousActiveElement) {
			// Restore focus to the element that opened the panel
			previousActiveElement.focus();
			previousActiveElement = null;
		}
	});
</script>

<svelte:window onkeydown={handleKeydown} />

{#if open}
	<!-- Backdrop -->
	<div
		class="fixed inset-0 bg-black/50 z-50"
		onclick={onclose}
		onkeydown={(e) => e.key === 'Enter' && onclose()}
		role="button"
		tabindex="-1"
		aria-label="Close settings"
		data-testid="settings-backdrop"
	></div>

	<!-- Panel -->
	<div
		bind:this={panelRef}
		class="fixed right-0 top-0 h-full w-80 bg-background border-l shadow-lg z-50 flex flex-col"
		role="dialog"
		aria-modal="true"
		aria-label="Settings"
		transition:fly={{ x: 320, duration: 200 }}
		data-testid="settings-panel"
	>
		<!-- Header -->
		<div class="flex items-center justify-between p-4 border-b">
			<h2 class="font-semibold text-base">Settings</h2>
			<Button variant="ghost" size="icon" onclick={onclose} aria-label="Close settings">
				<X class="h-5 w-5" />
			</Button>
		</div>

		<!-- Content -->
		<div class="flex-1 overflow-y-auto p-4 space-y-6">
			<!-- Theme -->
			<section>
				<h3 class="font-medium text-xs mb-2">Theme</h3>
				<ThemeToggle />
			</section>

			<!-- Temperature -->
			<section>
				<h3 class="font-medium text-xs mb-2">Model Temperature</h3>
				<TemperatureSlider />
			</section>

			<!-- Separator -->
			<hr class="border-border" />

			<!-- System Prompt -->
			<section>
				<h3 class="font-medium text-xs mb-2">Custom Instructions</h3>
				<SystemPromptEditor />
			</section>

			<!-- Separator -->
			<hr class="border-border" />

			<!-- Tools -->
			<section>
				<h3 class="font-medium text-xs mb-2">Available Tools</h3>
				<ToolToggles />
			</section>
		</div>
	</div>
{/if}
