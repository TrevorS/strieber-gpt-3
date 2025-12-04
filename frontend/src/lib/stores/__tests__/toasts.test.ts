/**
 * Unit tests for toast store
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';
import { toastStore } from '../toasts.svelte';

describe('toastStore', () => {
	beforeEach(() => {
		// Clear all toasts before each test
		toastStore.clear();
		vi.useFakeTimers();
	});

	afterEach(() => {
		vi.restoreAllMocks();
		vi.useRealTimers();
	});

	describe('add', () => {
		it('should add a toast with generated id', () => {
			const id = toastStore.add('Test message');

			expect(id).toBeTruthy();
			expect(toastStore.toasts).toHaveLength(1);
			expect(toastStore.toasts[0]).toMatchObject({
				id,
				message: 'Test message',
				type: 'info',
				duration: 5000
			});
		});

		it('should support different toast types', () => {
			toastStore.add('Error', 'error');
			toastStore.add('Success', 'success');
			toastStore.add('Warning', 'warning');
			toastStore.add('Info', 'info');

			expect(toastStore.toasts.map((t) => t.type)).toEqual(['error', 'success', 'warning', 'info']);
		});

		it('should support custom duration', () => {
			toastStore.add('Custom duration', 'info', 10000);

			expect(toastStore.toasts[0].duration).toBe(10000);
		});

		it('should auto-dismiss after duration', () => {
			toastStore.add('Auto dismiss', 'info', 3000);

			expect(toastStore.toasts).toHaveLength(1);

			vi.advanceTimersByTime(3000);

			expect(toastStore.toasts).toHaveLength(0);
		});

		it('should not auto-dismiss with duration 0', () => {
			toastStore.add('Persistent', 'info', 0);

			vi.advanceTimersByTime(10000);

			expect(toastStore.toasts).toHaveLength(1);
		});
	});

	describe('remove', () => {
		it('should remove toast by id', () => {
			const id1 = toastStore.add('First');
			const id2 = toastStore.add('Second');

			toastStore.remove(id1);

			expect(toastStore.toasts).toHaveLength(1);
			expect(toastStore.toasts[0].id).toBe(id2);
		});

		it('should handle removing non-existent id gracefully', () => {
			toastStore.add('Test');

			expect(() => toastStore.remove('non-existent')).not.toThrow();
			expect(toastStore.toasts).toHaveLength(1);
		});
	});

	describe('clear', () => {
		it('should remove all toasts', () => {
			toastStore.add('First');
			toastStore.add('Second');
			toastStore.add('Third');

			toastStore.clear();

			expect(toastStore.toasts).toHaveLength(0);
		});
	});

	describe('convenience methods', () => {
		it('error() should add error toast', () => {
			toastStore.error('Error message');

			expect(toastStore.toasts[0].type).toBe('error');
			expect(toastStore.toasts[0].message).toBe('Error message');
		});

		it('success() should add success toast', () => {
			toastStore.success('Success message');

			expect(toastStore.toasts[0].type).toBe('success');
			expect(toastStore.toasts[0].message).toBe('Success message');
		});

		it('info() should add info toast', () => {
			toastStore.info('Info message');

			expect(toastStore.toasts[0].type).toBe('info');
			expect(toastStore.toasts[0].message).toBe('Info message');
		});

		it('warning() should add warning toast', () => {
			toastStore.warning('Warning message');

			expect(toastStore.toasts[0].type).toBe('warning');
			expect(toastStore.toasts[0].message).toBe('Warning message');
		});

		it('convenience methods should support custom duration', () => {
			toastStore.error('Error', 1000);
			toastStore.success('Success', 2000);
			toastStore.info('Info', 3000);
			toastStore.warning('Warning', 4000);

			expect(toastStore.toasts.map((t) => t.duration)).toEqual([1000, 2000, 3000, 4000]);
		});
	});
});
