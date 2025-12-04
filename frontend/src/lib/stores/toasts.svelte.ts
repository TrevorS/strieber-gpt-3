/**
 * Toast Notification Store
 *
 * Svelte 5 runes-based store for managing toast notifications.
 * Supports auto-dismiss and multiple toast types.
 */

export type ToastType = 'error' | 'success' | 'info' | 'warning';

export interface Toast {
	id: string;
	message: string;
	type: ToastType;
	duration: number;
}

/**
 * Toast store class using Svelte 5 runes.
 *
 * @example
 * ```svelte
 * <script>
 *   import { toastStore } from '$lib/stores';
 *
 *   function handleError() {
 *     toastStore.error('Something went wrong');
 *   }
 * </script>
 * ```
 */
class ToastStore {
	/** Active toasts */
	toasts = $state<Toast[]>([]);

	/** Default duration in milliseconds */
	private defaultDuration = 5000;

	/** Generate unique ID */
	private generateId(): string {
		return `toast_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
	}

	/**
	 * Add a toast notification.
	 */
	add(message: string, type: ToastType = 'info', duration?: number): string {
		const id = this.generateId();
		const toast: Toast = {
			id,
			message,
			type,
			duration: duration ?? this.defaultDuration
		};

		this.toasts.push(toast);

		// Auto-dismiss after duration
		if (toast.duration > 0) {
			setTimeout(() => this.remove(id), toast.duration);
		}

		return id;
	}

	/**
	 * Remove a toast by ID.
	 */
	remove(id: string): void {
		const index = this.toasts.findIndex((t) => t.id === id);
		if (index !== -1) {
			this.toasts.splice(index, 1);
		}
	}

	/**
	 * Clear all toasts.
	 */
	clear(): void {
		this.toasts = [];
	}

	// Convenience methods
	error(message: string, duration?: number): string {
		return this.add(message, 'error', duration);
	}

	success(message: string, duration?: number): string {
		return this.add(message, 'success', duration);
	}

	info(message: string, duration?: number): string {
		return this.add(message, 'info', duration);
	}

	warning(message: string, duration?: number): string {
		return this.add(message, 'warning', duration);
	}
}

export const toastStore = new ToastStore();
