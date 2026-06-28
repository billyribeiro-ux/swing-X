import { getChangelog } from '$lib/api/client';
import type { PageLoad } from './$types';

export const load: PageLoad = async () => {
  const weeks = await getChangelog();
  return { weeks };
};
