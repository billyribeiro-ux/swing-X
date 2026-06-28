import { getPopulation } from '$lib/api/client';
import type { PageLoad } from './$types';

export const load: PageLoad = async () => {
  const population = await getPopulation();
  return { population };
};
