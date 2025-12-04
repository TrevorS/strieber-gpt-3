<script lang="ts">
	import { toastStore } from '$lib/stores/toasts.svelte';
	import Toast from './Toast.svelte';
</script>

<div
	class="fixed top-4 right-4 z-50 flex flex-col gap-2"
	aria-live="polite"
	aria-label="Notifications"
	data-testid="toast-container"
>
	{#each toastStore.toasts as toast (toast.id)}
		<Toast
			id={toast.id}
			message={toast.message}
			type={toast.type}
			onclose={(id) => toastStore.remove(id)}
		/>
	{/each}
</div>
