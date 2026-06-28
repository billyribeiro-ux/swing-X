import { getMonitorEvents } from '$lib/api/client';
import type { PageLoad } from './$types';

export const load: PageLoad = async ({ fetch }) => {
  const events = await getMonitorEvents(fetch);
  return { events };
};
