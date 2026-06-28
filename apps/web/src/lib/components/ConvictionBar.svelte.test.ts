import { describe, expect, it } from 'vitest';
import { render } from '@testing-library/svelte';
import ConvictionBar from './ConvictionBar.svelte';

describe('ConvictionBar', () => {
  it('renders the calibrated conviction as a percentage', () => {
    const { getByText } = render(ConvictionBar, { props: { value: 0.71 } });
    expect(getByText('71%')).toBeInTheDocument();
  });

  it('clamps out-of-range values into [0, 1]', () => {
    const { getByText } = render(ConvictionBar, { props: { value: 1.4 } });
    expect(getByText('100%')).toBeInTheDocument();
  });

  it('uses the up color for high conviction', () => {
    const { getByText } = render(ConvictionBar, { props: { value: 0.8 } });
    const label = getByText('80%');
    expect(label.className).toContain('text-up');
  });

  it('uses the caution color for low conviction', () => {
    const { getByText } = render(ConvictionBar, { props: { value: 0.3 } });
    const label = getByText('30%');
    expect(label.className).toContain('text-caution');
  });
});
