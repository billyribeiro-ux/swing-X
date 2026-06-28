import { getSignals } from '$lib/api/client';
import { rangeFromSearchParams } from '$lib/date-range';
import type { PageLoad } from './$types';

export const load: PageLoad = async ({ fetch, url }) => {
  const range = rangeFromSearchParams(url.searchParams);
  const signals = await getSignals(fetch, range);
  return { signals };
};
