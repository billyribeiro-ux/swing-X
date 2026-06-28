import type { Component } from 'svelte';
import { Article, Books, ChartLine, ClipboardText, WarningOctagon } from 'phosphor-svelte';

/** Parameterless top-level routes reachable from the sidebar. */
export type NavRoute = '/scoreboard' | '/population' | '/monitor' | '/journal' | '/changelog';

/** A primary navigation entry for the sidebar. */
export interface NavItem {
  /** A static route id, so it can be passed through `resolve()` type-safely. */
  href: NavRoute;
  label: string;
  icon: Component;
  /** Short hint shown as a tooltip/title. */
  hint: string;
}

export const navItems: NavItem[] = [
  {
    href: '/scoreboard',
    label: 'Scoreboard',
    icon: ChartLine,
    hint: 'Live surfaced signals'
  },
  {
    href: '/population',
    label: 'Population',
    icon: Books,
    hint: 'Strategy genomes & OOS scores'
  },
  {
    href: '/monitor',
    label: 'Monitor',
    icon: WarningOctagon,
    hint: 'Adaptation alerts'
  },
  {
    href: '/journal',
    label: 'Journal',
    icon: ClipboardText,
    hint: 'Paper-trade journal'
  },
  {
    href: '/changelog',
    label: 'Changelog',
    icon: Article,
    hint: 'Weekly self-changelog'
  }
];
