import { getChangelog } from '$lib/api/client';
import type { PageLoad } from './$types';

export const load: PageLoad = async ({ fetch }) => {
  const weeks = await getChangelog(fetch);
  return { weeks };
};
