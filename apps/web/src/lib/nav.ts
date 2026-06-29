import type { Component } from 'svelte';
import {
  Article,
  Books,
  ChartLine,
  ClipboardText,
  CurrencyDollar,
  Notebook,
  Stack,
  WarningOctagon
} from 'phosphor-svelte';

/** Parameterless top-level routes reachable from the sidebar. */
export type NavRoute =
  | '/scoreboard'
  | '/population'
  | '/monitor'
  | '/journal'
  | '/changelog'
  | '/equity'
  | '/equity/journal'
  | '/equity/population';

/** A primary navigation entry for the sidebar. */
export interface NavItem {
  /** A static route id, so it can be passed through `resolve()` type-safely. */
  href: NavRoute;
  label: string;
  icon: Component;
  /** Short hint shown as a tooltip/title. */
  hint: string;
}

/** A labeled cluster of sidebar entries (one scanner, or shared tooling). */
export interface NavGroup {
  /** Section heading shown above the group, or `null` for an unlabeled group. */
  label: string | null;
  items: NavItem[];
}

/** Sidebar navigation, grouped by scanner. */
export const navGroups: NavGroup[] = [
  {
    label: 'ETF Scanner',
    items: [
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
      }
    ]
  },
  {
    label: 'Equity Scanner',
    items: [
      {
        href: '/equity',
        label: 'Equity Scoreboard',
        icon: CurrencyDollar,
        hint: 'Live surfaced equity signals'
      },
      {
        href: '/equity/population',
        label: 'Equity Population',
        icon: Stack,
        hint: 'Equity strategy genomes & OOS scores'
      },
      {
        href: '/equity/journal',
        label: 'Equity Journal',
        icon: Notebook,
        hint: 'Equity paper-trade journal'
      }
    ]
  },
  {
    label: null,
    items: [
      {
        href: '/changelog',
        label: 'Changelog',
        icon: Article,
        hint: 'Weekly self-changelog'
      }
    ]
  }
];

/** Flat list of all sidebar entries, in display order. */
export const navItems: NavItem[] = navGroups.flatMap((group) => group.items);
