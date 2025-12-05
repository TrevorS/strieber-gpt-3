<script lang="ts">
	import { RotateCcw } from 'lucide-svelte';
	import { Button } from '$lib/components/ui/button';
	import { settingsStore } from '$lib/stores/settings.svelte';

	function handleInput(e: Event) {
		const target = e.target as HTMLTextAreaElement;
		settingsStore.setSystemPrompt(target.value);
	}

	function handleClear() {
		settingsStore.setSystemPrompt('');
	}
</script>

<div class="space-y-2" data-testid="system-prompt-editor">
	<div class="flex items-center justify-between">
		<label for="system-prompt" class="text-xs font-medium">System Prompt</label>
		{#if settingsStore.systemPrompt}
			<Button variant="ghost" size="sm" onclick={handleClear} data-testid="clear-system-prompt">
				<RotateCcw class="h-3 w-3 mr-1" />
				Clear
			</Button>
		{/if}
	</div>
	<textarea
		id="system-prompt"
		value={settingsStore.systemPrompt}
		oninput={handleInput}
		placeholder="Enter custom instructions for the assistant..."
		rows={4}
		class="w-full rounded-md border border-input bg-background px-3 py-2 text-sm font-mono ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 resize-none"
		data-testid="system-prompt-textarea"
	></textarea>
	<p class="text-[10px] text-muted-foreground">
		{settingsStore.systemPrompt.length} characters
	</p>
</div>
