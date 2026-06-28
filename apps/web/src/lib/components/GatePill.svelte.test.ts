import { describe, expect, it } from 'vitest';
import { render } from '@testing-library/svelte';
import GatePill from './GatePill.svelte';

describe('GatePill', () => {
  it('shows PASS in the up color when the gate is passed', () => {
    const { getByText } = render(GatePill, { props: { passed: true } });
    const el = getByText('Pass');
    expect(el).toBeInTheDocument();
    expect(el.className).toContain('text-up');
  });

  it('shows FAIL in the down color when the gate is failed', () => {
    const { getByText } = render(GatePill, { props: { passed: false } });
    const el = getByText('Fail');
    expect(el).toBeInTheDocument();
    expect(el.className).toContain('text-down');
  });
});
