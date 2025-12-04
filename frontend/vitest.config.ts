import { defineConfig } from 'vitest/config';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import { svelteTesting } from '@testing-library/svelte/vite';

export default defineConfig({
	plugins: [svelte(), svelteTesting()],
	test: {
		include: ['src/**/*.{test,spec}.{js,ts}'],
		environment: 'jsdom',
		globals: true,
		setupFiles: ['./vitest-setup.ts'],
		// Resolve SvelteKit aliases
		alias: {
			$lib: new URL('./src/lib', import.meta.url).pathname,
			'$app/environment': new URL('./src/test-mocks/app-environment.ts', import.meta.url).pathname,
			'$app/navigation': new URL('./src/test-mocks/app-navigation.ts', import.meta.url).pathname,
			'$app/state': new URL('./src/test-mocks/app-state.ts', import.meta.url).pathname
		}
	}
});
