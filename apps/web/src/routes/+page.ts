import { redirect } from '@sveltejs/kit';

/** Root path redirects to the signal scoreboard. */
export function load(): never {
  redirect(307, '/scoreboard');
}
