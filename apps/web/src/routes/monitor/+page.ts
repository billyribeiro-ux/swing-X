import { getMonitorEvents } from '$lib/api/client';
import type { PageLoad } from './$types';

export const load: PageLoad = async () => {
  const events = await getMonitorEvents();
  return { events };
};
