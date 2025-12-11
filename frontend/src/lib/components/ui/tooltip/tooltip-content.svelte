<script lang="ts">
	import { Tooltip } from 'bits-ui';
	import type { Snippet } from 'svelte';

	let {
		class: className = '',
		side = 'right',
		sideOffset = 8,
		children,
		...rest
	}: {
		class?: string;
		side?: 'top' | 'right' | 'bottom' | 'left';
		sideOffset?: number;
		children?: Snippet;
	} = $props();
</script>

<Tooltip.Portal>
	<Tooltip.Content
		{side}
		{sideOffset}
		class="z-50 overflow-hidden rounded-md bg-popover px-3 py-1.5 text-sm text-popover-foreground shadow-md animate-in fade-in-0 zoom-in-95 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2 {className}"
		{...rest}
	>
		{@render children?.()}
	</Tooltip.Content>
</Tooltip.Portal>
