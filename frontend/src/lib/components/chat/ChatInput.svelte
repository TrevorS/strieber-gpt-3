<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { Send } from 'lucide-svelte';

	let {
		onsubmit,
		disabled = false
	}: {
		onsubmit: (text: string) => void;
		disabled?: boolean;
	} = $props();

	let value = $state('');
	let textarea: HTMLTextAreaElement;

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter' && !e.shiftKey) {
			e.preventDefault();
			submit();
		}
	}

	function submit() {
		const text = value.trim();
		if (text && !disabled) {
			onsubmit(text);
			value = '';
			if (textarea) textarea.style.height = 'auto';
		}
	}

	function autoResize() {
		if (textarea) {
			textarea.style.height = 'auto';
			textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
		}
	}

	let canSubmit = $derived(value.trim().length > 0 && !disabled);
</script>

<div class="border-t p-4">
	<div class="flex gap-2 items-end max-w-3xl mx-auto">
		<textarea
			bind:this={textarea}
			bind:value
			onkeydown={handleKeydown}
			oninput={autoResize}
			placeholder="Send a message..."
			rows="1"
			class="flex-1 resize-none rounded-lg border p-3 focus:outline-none focus:ring-2 focus:ring-ring"
		></textarea>
		<Button onclick={submit} disabled={!canSubmit} size="icon">
			<Send class="h-4 w-4" />
		</Button>
	</div>
</div>
