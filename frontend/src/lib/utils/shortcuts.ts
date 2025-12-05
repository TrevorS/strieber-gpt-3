/**
 * Keyboard Shortcuts Utility
 *
 * Centralized keyboard shortcut handling for the application.
 */

export interface ShortcutAction {
	/** Key to listen for (e.g., 'n', '/', 'Escape') */
	key: string;
	/** Whether Cmd (Mac) or Ctrl (Windows/Linux) must be pressed */
	cmdOrCtrl?: boolean;
	/** Whether Shift must be pressed */
	shift?: boolean;
	/** Handler function */
	handler: () => void;
	/** Description for help text */
	description: string;
}

/**
 * Check if an element is an input field where shortcuts should be ignored.
 */
function isInputElement(target: EventTarget | null): boolean {
	if (!target || !(target instanceof HTMLElement)) return false;
	const tagName = target.tagName.toLowerCase();
	return (
		tagName === 'input' ||
		tagName === 'textarea' ||
		tagName === 'select' ||
		target.isContentEditable
	);
}

/**
 * Check if the platform is Mac.
 */
function isMac(): boolean {
	return typeof navigator !== 'undefined' && /Mac|iPod|iPhone|iPad/.test(navigator.platform);
}

/**
 * Create a keyboard event handler for shortcuts.
 *
 * @param actions - Array of shortcut actions to handle
 * @returns Event handler function
 *
 * @example
 * ```typescript
 * const handler = createShortcutHandler([
 *   { key: 'n', cmdOrCtrl: true, handler: () => newChat(), description: 'New chat' },
 *   { key: '/', cmdOrCtrl: true, handler: () => toggleSidebar(), description: 'Toggle sidebar' },
 * ]);
 *
 * // In Svelte:
 * <svelte:window onkeydown={handler} />
 * ```
 */
export function createShortcutHandler(actions: ShortcutAction[]): (event: KeyboardEvent) => void {
	return (event: KeyboardEvent) => {
		// Don't handle shortcuts when typing in inputs (except Escape)
		if (event.key !== 'Escape' && isInputElement(event.target)) {
			return;
		}

		const mac = isMac();
		const cmdOrCtrlPressed = mac ? event.metaKey : event.ctrlKey;

		for (const action of actions) {
			// Check key match (case-insensitive)
			if (event.key.toLowerCase() !== action.key.toLowerCase()) {
				continue;
			}

			// Check modifier requirements
			if (action.cmdOrCtrl && !cmdOrCtrlPressed) {
				continue;
			}
			if (!action.cmdOrCtrl && cmdOrCtrlPressed) {
				continue;
			}
			if (action.shift && !event.shiftKey) {
				continue;
			}
			if (!action.shift && event.shiftKey) {
				continue;
			}

			// Match found - prevent default and execute handler
			event.preventDefault();
			action.handler();
			return;
		}
	};
}

/**
 * Format a shortcut for display.
 *
 * @param action - The shortcut action
 * @returns Formatted string like "⌘N" or "Ctrl+N"
 */
export function formatShortcut(action: ShortcutAction): string {
	const parts: string[] = [];
	const mac = isMac();

	if (action.cmdOrCtrl) {
		parts.push(mac ? '⌘' : 'Ctrl+');
	}
	if (action.shift) {
		parts.push(mac ? '⇧' : 'Shift+');
	}

	// Format the key
	let keyDisplay = action.key.toUpperCase();
	if (action.key === '/') keyDisplay = '/';
	if (action.key === 'Escape') keyDisplay = 'Esc';

	parts.push(keyDisplay);

	return parts.join('');
}
