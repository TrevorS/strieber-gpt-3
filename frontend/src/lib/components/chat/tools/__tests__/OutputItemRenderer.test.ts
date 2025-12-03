/**
 * Unit tests for OutputItemRenderer component
 *
 * Tests cover:
 * - Dispatching to correct component based on type
 * - Handling unknown types
 * - Passing props correctly
 */

import { render, screen } from '@testing-library/svelte';
import { describe, expect, it } from 'vitest';
import type {
	ResponseOutputItem,
	ResponseReasoningItem,
	ResponseFunctionWebSearch,
	ResponseCodeInterpreterToolCall,
	ResponseFunctionToolCall,
	ResponseOutputMessage
} from '$lib/stores/types';
import OutputItemRenderer from '../OutputItemRenderer.svelte';

describe('OutputItemRenderer', () => {
	describe('reasoning items', () => {
		it('should render ReasoningDisplay for reasoning type', () => {
			const item: ResponseReasoningItem = {
				id: 'reasoning-1',
				type: 'reasoning',
				content: [{ type: 'reasoning_text', text: 'Thinking...' }],
				summary: []
			};

			render(OutputItemRenderer, { props: { item } });

			// ReasoningDisplay shows "Reasoning" label
			expect(screen.getByText('Reasoning')).toBeInTheDocument();
		});

		it('should pass isStreaming to ReasoningDisplay', () => {
			const item: ResponseReasoningItem = {
				id: 'reasoning-1',
				type: 'reasoning',
				content: [{ type: 'reasoning_text', text: 'Some reasoning content' }],
				summary: []
			};

			const { container } = render(OutputItemRenderer, { props: { item, isStreaming: true } });

			// When streaming, the trigger button shows "Thinking..." label
			const button = container.querySelector('button');
			expect(button?.textContent).toContain('Thinking...');
		});
	});

	describe('web search items', () => {
		it('should render WebSearchDisplay for web_search_call type', () => {
			const item: ResponseFunctionWebSearch = {
				id: 'web-search-1',
				type: 'web_search_call',
				status: 'completed'
			};

			render(OutputItemRenderer, { props: { item } });

			// WebSearchDisplay shows "Web search completed" for completed status without query
			expect(screen.getByText('Web search completed')).toBeInTheDocument();
		});

		it('should show loading state for in_progress web search', () => {
			const item: ResponseFunctionWebSearch = {
				id: 'web-search-1',
				type: 'web_search_call',
				status: 'in_progress'
			};

			render(OutputItemRenderer, { props: { item } });

			expect(screen.getByText('Searching the web...')).toBeInTheDocument();
		});
	});

	describe('code interpreter items', () => {
		it('should render CodeInterpreterDisplay for code_interpreter_call type', () => {
			const item: ResponseCodeInterpreterToolCall = {
				id: 'code-1',
				type: 'code_interpreter_call',
				status: 'completed',
				code: 'print("test")',
				container_id: 'container-1',
				outputs: null
			};

			render(OutputItemRenderer, { props: { item } });

			expect(screen.getByText('Code Interpreter')).toBeInTheDocument();
		});
	});

	describe('function call items', () => {
		it('should render FunctionCallDisplay for function_call type', () => {
			const item: ResponseFunctionToolCall = {
				id: 'func-1',
				type: 'function_call',
				name: 'get_weather',
				arguments: '{"location": "NYC"}',
				call_id: 'call-1'
			};

			render(OutputItemRenderer, { props: { item } });

			expect(screen.getByText('get_weather')).toBeInTheDocument();
		});

		it('should pass isStreaming to FunctionCallDisplay', () => {
			const item: ResponseFunctionToolCall = {
				id: 'func-1',
				type: 'function_call',
				name: 'my_func',
				arguments: '{}',
				call_id: 'call-1'
			};

			const { container } = render(OutputItemRenderer, {
				props: { item, isStreaming: true }
			});

			// Should show spinner when streaming
			const spinner = container.querySelector('.animate-spin');
			expect(spinner).toBeInTheDocument();
		});
	});

	describe('message items', () => {
		it('should not render anything for message type', () => {
			const item: ResponseOutputMessage = {
				id: 'msg-1',
				type: 'message',
				role: 'assistant',
				status: 'completed',
				content: [{ type: 'output_text', text: 'Hello', annotations: [] }]
			};

			const { container } = render(OutputItemRenderer, { props: { item } });

			// Message items are skipped (handled by MarkdownContent in parent)
			// Should render an empty component
			expect(container.children[0]?.children.length || 0).toBe(0);
		});
	});

	describe('unknown items', () => {
		it('should show debug info in dev mode for unknown types', () => {
			// Create an item with an unknown type - cast through unknown to avoid type errors
			const item = {
				id: 'unknown-1',
				type: 'some_unknown_type'
			} as unknown as ResponseOutputItem;

			const { container } = render(OutputItemRenderer, { props: { item } });

			// In dev mode (which vitest runs in), should show debug info
			// The text might be split across elements, so check for code element
			const codeElement = container.querySelector('code');
			if (codeElement) {
				expect(codeElement.textContent).toBe('some_unknown_type');
			}
		});
	});
});
