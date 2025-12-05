<script lang="ts">
	import { Globe, Code, Cloud, BookOpen } from 'lucide-svelte';
	import { Button } from '$lib/components/ui/button';
	import { settingsStore } from '$lib/stores/settings.svelte';

	const tools = [
		{ id: 'web_search', name: 'Web Search', description: 'Search the internet', icon: Globe },
		{ id: 'code_interpreter', name: 'Code Interpreter', description: 'Execute Python code', icon: Code },
		{ id: 'weather', name: 'Weather', description: 'Get weather forecasts', icon: Cloud },
		{ id: 'reader', name: 'Reader', description: 'Read web pages', icon: BookOpen }
	];

	function isSupported(toolId: string): boolean {
		const supported = settingsStore.supportedTools();
		return supported === null || supported.includes(toolId);
	}

	function isEnabled(toolId: string): boolean {
		return settingsStore.enabledTools[toolId] !== false;
	}

	function toggle(toolId: string) {
		const currentState = isEnabled(toolId);
		settingsStore.setToolEnabled(toolId, !currentState);
	}
</script>

<div class="space-y-3" data-testid="tool-toggles">
	<div class="flex items-center gap-3 p-2">
		<!-- Spacer matching icon box width -->
		<div class="flex-shrink-0 w-8"></div>
		<!-- Label takes flex-1 like tool text -->
		<span class="flex-1 text-xs font-medium">Tools</span>
		<!-- Buttons aligned with toggle switches -->
		<div class="flex-shrink-0 flex gap-1">
			<Button
				variant="ghost"
				size="sm"
				onclick={() => settingsStore.setAllToolsEnabled(true)}
				data-testid="enable-all-tools"
			>
				All On
			</Button>
			<Button
				variant="ghost"
				size="sm"
				onclick={() => settingsStore.setAllToolsEnabled(false)}
				data-testid="disable-all-tools"
			>
				All Off
			</Button>
		</div>
	</div>

	<div class="space-y-2">
		{#each tools as tool (tool.id)}
			{@const supported = isSupported(tool.id)}
			{@const enabled = isEnabled(tool.id)}
			<button
				type="button"
				onclick={() => supported && toggle(tool.id)}
				disabled={!supported}
				class="w-full flex items-center gap-3 p-2 rounded-md transition-colors
					{supported ? 'hover:bg-accent cursor-pointer' : 'opacity-50 cursor-not-allowed'}
					{enabled && supported ? 'bg-accent/50' : ''}"
				data-testid="tool-toggle-{tool.id}"
			>
				<div class="flex-shrink-0 w-8 h-8 rounded-md flex items-center justify-center
					{enabled && supported ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground'}">
					<tool.icon class="h-4 w-4" />
				</div>
				<div class="flex-1 text-left">
					<p class="text-xs font-medium" class:text-muted-foreground={!supported}>
						{tool.name}
					</p>
					<p class="text-[10px] text-muted-foreground">{tool.description}</p>
				</div>
				<div class="flex-shrink-0">
					<div
						class="w-10 h-6 rounded-full transition-colors relative
							{enabled && supported ? 'bg-primary' : 'bg-muted'}"
					>
						<div
							class="absolute top-1 w-4 h-4 rounded-full bg-white shadow transition-transform
								{enabled && supported ? 'translate-x-5' : 'translate-x-1'}"
						></div>
					</div>
				</div>
			</button>
		{/each}
	</div>

	{#if settingsStore.supportedTools() !== null && settingsStore.supportedTools()?.length === 0}
		<p class="text-[10px] text-muted-foreground italic">
			Current model doesn't support tools
		</p>
	{/if}
</div>
