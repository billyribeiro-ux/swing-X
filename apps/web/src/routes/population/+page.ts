import { getPopulation } from '$lib/api/client';
import type { PageLoad } from './$types';

export const load: PageLoad = async ({ fetch }) => {
  const population = await getPopulation(fetch);
  return { population };
};
