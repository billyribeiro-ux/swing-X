import { getSignals } from '$lib/api/client';
import type { PageLoad } from './$types';

export const load: PageLoad = async () => {
  const signals = await getSignals();
  return { signals };
};
