<script lang="ts">
	import { renderMarkdown } from '$lib/utils/markdown';
	import { transformCitationMarkers, type Citation } from '$lib/utils/citations';

	let { content, citations = [] }: { content: string; citations?: Citation[] } = $props();

	// Transform citation markers before rendering markdown
	let transformedContent = $derived(
		citations.length > 0 ? transformCitationMarkers(content, citations) : content
	);

	let html = $derived(renderMarkdown(transformedContent));

	let container: HTMLDivElement;

	// Add copy buttons to code blocks
	function setupCopyButtons() {
		if (!container) return;
		container.querySelectorAll('pre').forEach((pre) => {
			if (pre.querySelector('.code-copy-btn')) return; // Already has button

			const btn = document.createElement('button');
			btn.className = 'code-copy-btn';
			btn.title = 'Copy code';
			btn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>`;
			btn.onclick = async () => {
				const code = pre.querySelector('code')?.textContent || '';
				await navigator.clipboard.writeText(code);
				btn.classList.add('copied');
				btn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>`;
				setTimeout(() => {
					btn.classList.remove('copied');
					btn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>`;
				}, 2000);
			};
			pre.style.position = 'relative';
			pre.appendChild(btn);
		});
	}

	$effect(() => {
		if (html) {
			// Wait for DOM update
			requestAnimationFrame(() => setupCopyButtons());
		}
	});
</script>

<div bind:this={container} class="prose prose-sm dark:prose-invert max-w-none
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

	/* Copy button styling for code blocks */
	:global(.code-copy-btn) {
		position: absolute;
		top: 0.5rem;
		right: 0.5rem;
		padding: 0.375rem;
		border-radius: 0.375rem;
		background: var(--background);
		border: 1px solid var(--border);
		cursor: pointer;
		opacity: 0;
		transition: opacity 0.2s, background-color 0.2s;
		color: var(--muted-foreground);
		display: flex;
		align-items: center;
		justify-content: center;
	}

	:global(pre:hover .code-copy-btn) {
		opacity: 1;
	}

	:global(.code-copy-btn:hover) {
		background: var(--accent);
		color: var(--foreground);
	}

	:global(.code-copy-btn.copied) {
		background: var(--primary);
		color: var(--primary-foreground);
		border-color: var(--primary);
	}
</style>
