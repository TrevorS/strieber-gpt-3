<script lang="ts">
	import { renderMarkdown } from '$lib/utils/markdown';
	import { transformCitationMarkers, type Citation } from '$lib/utils/citations';

	let { content, citations = [] }: { content: string; citations?: Citation[] } = $props();

	// Transform citation markers before rendering markdown
	let transformedContent = $derived(
		citations.length > 0 ? transformCitationMarkers(content, citations) : content
	);

	let html = $derived(renderMarkdown(transformedContent));

	const COPY_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>`;
	const CHECK_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>`;

	// Handle copy button clicks via event delegation
	function handleClick(event: MouseEvent) {
		const target = event.target as HTMLElement;
		const btn = target.closest('.code-header-copy-btn') as HTMLButtonElement | null;
		if (!btn) return;

		const codeId = btn.dataset.codeId;
		if (!codeId) return;

		const codeEl = document.getElementById(codeId);
		if (!codeEl) return;

		const code = codeEl.textContent || '';
		navigator.clipboard.writeText(code).then(() => {
			btn.classList.add('copied');
			btn.innerHTML = CHECK_ICON;
			setTimeout(() => {
				btn.classList.remove('copied');
				btn.innerHTML = COPY_ICON;
			}, 2000);
		});
	}
</script>

<!-- svelte-ignore a11y_click_events_have_key_events a11y_no_static_element_interactions -->
<div onclick={handleClick} class="prose prose-sm dark:prose-invert max-w-none
			prose-headings:font-semibold prose-headings:tracking-tight
			prose-p:leading-relaxed
			prose-table:my-4
			prose-th:px-4 prose-th:py-2 prose-th:text-left
			prose-td:px-4 prose-td:py-2
			prose-pre:bg-muted prose-pre:overflow-x-auto prose-pre:border prose-pre:border-border prose-pre:rounded-lg
			prose-code:font-mono prose-pre:font-mono">
	{@html html}
</div>

<style>
	/* Styling for inline citation links */
	:global(.citation-link) {
		font-size: 0.75rem;
		vertical-align: super;
		line-height: 0;
	}

	:global(.citation-link a) {
		color: var(--primary);
		text-decoration: none;
		font-weight: 500;
	}

	:global(.citation-link a:hover) {
		text-decoration: underline;
	}

	/* Code block wrapper and header bar */
	:global(.code-block-wrapper) {
		border-radius: 0.5rem;
		overflow: hidden;
		border: 1px solid var(--border);
		margin: 1rem 0;
	}

	:global(.code-block-wrapper pre) {
		margin: 0 !important;
		border: none !important;
		border-radius: 0 !important;
	}

	:global(.code-block-header) {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 0.5rem 0.75rem;
		background: var(--muted);
		border-bottom: 1px solid var(--border);
	}

	:global(.code-block-lang) {
		font-size: 0.75rem;
		font-weight: 500;
		color: var(--muted-foreground);
		text-transform: lowercase;
		font-family: inherit;
	}

	:global(.code-header-copy-btn) {
		padding: 0.25rem;
		border-radius: 0.25rem;
		background: transparent;
		border: none;
		cursor: pointer;
		opacity: 0.6;
		transition: opacity 0.2s, background-color 0.2s;
		color: var(--muted-foreground);
		display: flex;
		align-items: center;
		justify-content: center;
	}

	:global(.code-header-copy-btn:hover) {
		opacity: 1;
		background: var(--accent);
		color: var(--foreground);
	}

	:global(.code-header-copy-btn.copied) {
		opacity: 1;
		color: var(--primary);
	}
</style>
