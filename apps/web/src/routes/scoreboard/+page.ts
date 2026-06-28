import { getSignals } from '$lib/api/client';
import type { PageLoad } from './$types';

export const load: PageLoad = async ({ fetch }) => {
  const signals = await getSignals(fetch);
  return { signals };
};
