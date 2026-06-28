// SPA-style client rendering is fine for an internal ops console; SSR is left on
// (adapter-node) but we don't prerender since data is dynamic once the API lands.
export const prerender = false;
export const ssr = true;
