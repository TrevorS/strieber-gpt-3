import '@testing-library/jest-dom/vitest';

// Mock ResizeObserver for components that use it (e.g., ScrollArea)
global.ResizeObserver = class ResizeObserver {
	observe() {}
	unobserve() {}
	disconnect() {}
};
