<script lang="ts">
	import { settingsStore } from '$lib/stores/settings.svelte';

	function handleInput(e: Event) {
		const target = e.target as HTMLInputElement;
		settingsStore.setTemperature(parseFloat(target.value));
	}
</script>

<div class="space-y-2" data-testid="temperature-slider">
	<div class="flex justify-between text-sm">
		<label for="temperature">Temperature</label>
		<span class="text-muted-foreground">{settingsStore.temperature.toFixed(1)}</span>
	</div>
	<input
		id="temperature"
		type="range"
		min="0"
		max="2"
		step="0.1"
		value={settingsStore.temperature}
		oninput={handleInput}
		class="w-full accent-primary"
	/>
	<div class="flex justify-between text-xs text-muted-foreground">
		<span>Precise</span>
		<span>Creative</span>
	</div>
</div>
