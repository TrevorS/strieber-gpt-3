<script lang="ts">
	import { Sun, Moon, Monitor } from 'lucide-svelte';
	import { Button } from '$lib/components/ui/button';
	import { settingsStore } from '$lib/stores/settings.svelte';

	type Theme = 'light' | 'dark' | 'system';

	const themes: { value: Theme; icon: typeof Sun; label: string }[] = [
		{ value: 'light', icon: Sun, label: 'Light' },
		{ value: 'dark', icon: Moon, label: 'Dark' },
		{ value: 'system', icon: Monitor, label: 'System' }
	];
</script>

<div class="flex gap-1" role="radiogroup" aria-label="Theme selection" data-testid="theme-toggle">
	{#each themes as theme (theme.value)}
		<Button
			variant={settingsStore.theme === theme.value ? 'secondary' : 'ghost'}
			size="sm"
			onclick={() => settingsStore.setTheme(theme.value)}
			aria-pressed={settingsStore.theme === theme.value}
			data-testid="theme-option-{theme.value}"
		>
			<theme.icon class="h-4 w-4 mr-1" />
			{theme.label}
		</Button>
	{/each}
</div>
