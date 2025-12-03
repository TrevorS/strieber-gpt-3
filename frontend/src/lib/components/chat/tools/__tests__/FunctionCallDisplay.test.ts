/**
 * Unit tests for FunctionCallDisplay component
 *
 * Tests cover:
 * - Function name display
 * - Arguments formatting
 * - Streaming vs completed states
 * - Collapsible behavior
 */

import { render, screen } from '@testing-library/svelte';
import { describe, expect, it } from 'vitest';
import type { ResponseFunctionToolCall } from '$lib/stores/types';
import FunctionCallDisplay from '../FunctionCallDisplay.svelte';

function createFunctionCallItem(
	name: string,
	args: Record<string, unknown>,
	options?: { id?: string; callId?: string }
): ResponseFunctionToolCall {
	return {
		id: options?.id || 'func-call-1',
		type: 'function_call',
		name,
		arguments: JSON.stringify(args),
		call_id: options?.callId || 'call-1'
	};
}

describe('FunctionCallDisplay', () => {
	describe('function name', () => {
		it('should render the function name', () => {
			const item = createFunctionCallItem('get_weather', { location: 'NYC' });

			render(FunctionCallDisplay, { props: { item } });

			expect(screen.getByText('get_weather')).toBeInTheDocument();
		});

		it('should render Wrench icon', () => {
			const item = createFunctionCallItem('my_function', {});

			const { container } = render(FunctionCallDisplay, { props: { item } });

			const svg = container.querySelector('svg');
			expect(svg).toBeInTheDocument();
		});
	});

	describe('status indicators', () => {
		it('should show spinner when streaming', () => {
			const item = createFunctionCallItem('test_func', {});

			const { container } = render(FunctionCallDisplay, {
				props: { item, isStreaming: true }
			});

			const spinner = container.querySelector('.animate-spin');
			expect(spinner).toBeInTheDocument();
		});

		it('should show checkmark when not streaming', () => {
			const item = createFunctionCallItem('test_func', {});

			const { container } = render(FunctionCallDisplay, {
				props: { item, isStreaming: false }
			});

			const greenIcon = container.querySelector('.text-green-600');
			expect(greenIcon).toBeInTheDocument();
		});
	});

	describe('collapsible behavior', () => {
		it('should render as a collapsible element', () => {
			const item = createFunctionCallItem('my_func', { key: 'value' });

			const { container } = render(FunctionCallDisplay, { props: { item } });

			const trigger = container.querySelector('button');
			expect(trigger).toBeInTheDocument();
		});

		it('should have chevron icon', () => {
			const item = createFunctionCallItem('my_func', {});

			const { container } = render(FunctionCallDisplay, { props: { item } });

			// Should have multiple SVG icons
			const svgs = container.querySelectorAll('svg');
			expect(svgs.length).toBeGreaterThanOrEqual(2);
		});
	});

	describe('arguments display', () => {
		it('should show "Arguments:" label', () => {
			const item = createFunctionCallItem('func', { a: 1 });

			render(FunctionCallDisplay, { props: { item } });

			expect(screen.getByText('Arguments:')).toBeInTheDocument();
		});

		it('should format JSON arguments with indentation', () => {
			const item = createFunctionCallItem('test', { key: 'value', num: 42 });

			const { container } = render(FunctionCallDisplay, { props: { item } });

			const pre = container.querySelector('pre');
			expect(pre).toBeInTheDocument();
			// Should be formatted with newlines
			expect(pre?.textContent).toContain('\n');
		});

		it('should handle empty arguments object', () => {
			const item = createFunctionCallItem('empty_func', {});

			const { container } = render(FunctionCallDisplay, { props: { item } });

			const pre = container.querySelector('pre');
			expect(pre?.textContent).toBe('{}');
		});

		it('should handle complex nested arguments', () => {
			const item = createFunctionCallItem('complex_func', {
				nested: { a: 1, b: 2 },
				array: [1, 2, 3]
			});

			const { container } = render(FunctionCallDisplay, { props: { item } });

			const pre = container.querySelector('pre');
			expect(pre?.textContent).toContain('nested');
			expect(pre?.textContent).toContain('array');
		});

		it('should handle invalid JSON gracefully', () => {
			// Create item with raw invalid JSON string
			const item: ResponseFunctionToolCall = {
				id: 'func-1',
				type: 'function_call',
				name: 'bad_func',
				arguments: 'not valid json',
				call_id: 'call-1'
			};

			const { container } = render(FunctionCallDisplay, { props: { item } });

			// Should still render, showing the raw string
			const pre = container.querySelector('pre');
			expect(pre?.textContent).toBe('not valid json');
		});
	});
});
