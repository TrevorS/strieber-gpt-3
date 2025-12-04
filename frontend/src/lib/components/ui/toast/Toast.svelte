<script lang="ts">
	import { fly } from 'svelte/transition';
	import { AlertCircle, CheckCircle2, Info, AlertTriangle, X } from 'lucide-svelte';
	import type { ToastType } from '$lib/stores/toasts.svelte';

	interface Props {
		id: string;
		message: string;
		type: ToastType;
		onclose: (id: string) => void;
	}

	let { id, message, type, onclose }: Props = $props();

	const icons = {
		error: AlertCircle,
		success: CheckCircle2,
		info: Info,
		warning: AlertTriangle
	};

	const styles = {
		error: 'bg-destructive text-white',
		success: 'bg-green-600 text-white',
		info: 'bg-primary text-primary-foreground',
		warning: 'bg-yellow-500 text-black'
	};

	let Icon = $derived(icons[type]);
</script>

<div
	class="flex items-center gap-3 px-4 py-3 rounded-lg shadow-lg min-w-[280px] max-w-[400px] {styles[type]}"
	role="alert"
	transition:fly={{ x: 100, duration: 300 }}
	data-testid="toast"
	data-toast-type={type}
>
	<Icon class="h-5 w-5 shrink-0" />
	<p class="flex-1 text-sm font-medium">{message}</p>
	<button
		onclick={() => onclose(id)}
		class="shrink-0 p-1 rounded hover:bg-white/20 transition-colors"
		aria-label="Dismiss"
		data-testid="toast-close"
	>
		<X class="h-4 w-4" />
	</button>
</div>
