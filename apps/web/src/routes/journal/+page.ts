import { getJournal } from '$lib/api/client';
import type { PageLoad } from './$types';

export const load: PageLoad = async ({ fetch }) => {
  const trades = await getJournal(fetch);
  return { trades };
};
