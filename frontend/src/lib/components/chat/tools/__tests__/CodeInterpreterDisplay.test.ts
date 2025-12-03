/**
 * Unit tests for CodeInterpreterDisplay component
 *
 * Tests cover:
 * - Code rendering
 * - Status indicators
 * - Output display (logs and images)
 */

import { render, screen } from '@testing-library/svelte';
import { describe, expect, it } from 'vitest';
import type { ResponseCodeInterpreterToolCall } from '$lib/stores/types';
import CodeInterpreterDisplay from '../CodeInterpreterDisplay.svelte';

function createCodeInterpreterItem(
	status: 'in_progress' | 'completed' | 'incomplete' | 'interpreting' | 'failed',
	options?: {
		code?: string;
		outputs?: Array<{ type: 'logs'; logs: string } | { type: 'image'; url: string }>;
	}
): ResponseCodeInterpreterToolCall {
	return {
		id: 'code-interpreter-1',
		type: 'code_interpreter_call',
		status,
		code: options?.code ?? null,
		container_id: 'container-1',
		outputs: options?.outputs ?? null
	};
}

describe('CodeInterpreterDisplay', () => {
	describe('header', () => {
		it('should render "Code Interpreter" label', () => {
			const item = createCodeInterpreterItem('completed');

			render(CodeInterpreterDisplay, { props: { item } });

			expect(screen.getByText('Code Interpreter')).toBeInTheDocument();
		});

		it('should render Code icon', () => {
			const item = createCodeInterpreterItem('completed');

			const { container } = render(CodeInterpreterDisplay, { props: { item } });

			const svg = container.querySelector('svg');
			expect(svg).toBeInTheDocument();
		});
	});

	describe('status indicators', () => {
		it('should show loading spinner when in_progress', () => {
			const item = createCodeInterpreterItem('in_progress');

			const { container } = render(CodeInterpreterDisplay, { props: { item } });

			const spinner = container.querySelector('.animate-spin');
			expect(spinner).toBeInTheDocument();
		});

		it('should show loading spinner when interpreting', () => {
			const item = createCodeInterpreterItem('interpreting');

			const { container } = render(CodeInterpreterDisplay, { props: { item } });

			const spinner = container.querySelector('.animate-spin');
			expect(spinner).toBeInTheDocument();
		});

		it('should show green checkmark when completed', () => {
			const item = createCodeInterpreterItem('completed');

			const { container } = render(CodeInterpreterDisplay, { props: { item } });

			// CheckCircle icon should have text-green-600 class
			const greenIcon = container.querySelector('.text-green-600');
			expect(greenIcon).toBeInTheDocument();
		});

		it('should show red X when failed', () => {
			const item = createCodeInterpreterItem('failed');

			const { container } = render(CodeInterpreterDisplay, { props: { item } });

			// XCircle icon should have text-red-600 class
			const redIcon = container.querySelector('.text-red-600');
			expect(redIcon).toBeInTheDocument();
		});
	});

	describe('code display', () => {
		it('should render code when provided', () => {
			const item = createCodeInterpreterItem('completed', {
				code: 'print("Hello, world!")'
			});

			render(CodeInterpreterDisplay, { props: { item } });

			expect(screen.getByText('print("Hello, world!")')).toBeInTheDocument();
		});

		it('should render code in pre/code tags', () => {
			const item = createCodeInterpreterItem('completed', {
				code: 'x = 42'
			});

			const { container } = render(CodeInterpreterDisplay, { props: { item } });

			const pre = container.querySelector('pre');
			const code = container.querySelector('code');
			expect(pre).toBeInTheDocument();
			expect(code).toBeInTheDocument();
		});

		it('should not render code block when code is null', () => {
			const item = createCodeInterpreterItem('completed', { code: undefined });

			const { container } = render(CodeInterpreterDisplay, { props: { item } });

			// Only the header pre might exist, check for code element
			const codeElements = container.querySelectorAll('pre code');
			expect(codeElements.length).toBe(0);
		});
	});

	describe('output display', () => {
		it('should render log output', () => {
			const item = createCodeInterpreterItem('completed', {
				code: 'print("test")',
				outputs: [{ type: 'logs', logs: 'test output' }]
			});

			render(CodeInterpreterDisplay, { props: { item } });

			expect(screen.getByText('test output')).toBeInTheDocument();
		});

		it('should render multiple log outputs', () => {
			const item = createCodeInterpreterItem('completed', {
				code: 'print("a"); print("b")',
				outputs: [
					{ type: 'logs', logs: 'output a' },
					{ type: 'logs', logs: 'output b' }
				]
			});

			render(CodeInterpreterDisplay, { props: { item } });

			expect(screen.getByText('output a')).toBeInTheDocument();
			expect(screen.getByText('output b')).toBeInTheDocument();
		});

		it('should render image output', () => {
			const item = createCodeInterpreterItem('completed', {
				code: 'plt.show()',
				outputs: [{ type: 'image', url: 'https://example.com/image.png' }]
			});

			const { container } = render(CodeInterpreterDisplay, { props: { item } });

			const img = container.querySelector('img');
			expect(img).toBeInTheDocument();
			expect(img).toHaveAttribute('src', 'https://example.com/image.png');
		});

		it('should not render outputs section when empty', () => {
			const item = createCodeInterpreterItem('completed', {
				code: 'x = 1',
				outputs: []
			});

			const { container } = render(CodeInterpreterDisplay, { props: { item } });

			// Should only have one border-t (the code section), not outputs
			const borderTElements = container.querySelectorAll('.border-t');
			expect(borderTElements.length).toBeLessThanOrEqual(1);
		});
	});
});
