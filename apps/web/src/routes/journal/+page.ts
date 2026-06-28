import { getJournal } from '$lib/api/client';
import type { PageLoad } from './$types';

export const load: PageLoad = async () => {
  const trades = await getJournal();
  return { trades };
};
