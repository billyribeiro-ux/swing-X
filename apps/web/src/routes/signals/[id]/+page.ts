import { error } from '@sveltejs/kit';
import { getSignal } from '$lib/api/client';
import { buildCandles, buildVwap } from '$lib/fixtures/candles';
import type { PageLoad } from './$types';

export const load: PageLoad = async ({ params }) => {
  const signal = await getSignal(params.id);
  if (!signal) {
    error(404, `Signal ${params.id} not found`);
  }
  const candles = buildCandles(signal.ticker, signal.decisionTs);
  const vwap = buildVwap(candles);
  return { signal, candles, vwap };
};
