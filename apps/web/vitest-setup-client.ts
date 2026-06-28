import '@testing-library/jest-dom/vitest';

// jsdom does not implement matchMedia; stub it so components that probe color-scheme
// or responsive breakpoints don't throw during component tests.
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: (query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addEventListener: () => {},
    removeEventListener: () => {},
    addListener: () => {},
    removeListener: () => {},
    dispatchEvent: () => false
  })
});
