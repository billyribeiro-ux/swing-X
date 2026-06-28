import type { Snippet } from 'svelte';

/** Column definition for the generic {@link DataTable} component. */
export interface Column<R> {
  /** Stable key; also used as the default sort key. */
  id: string;
  /** Header label. */
  header: string;
  /** Whether this column is sortable. Defaults to true. */
  sortable?: boolean;
  /** Right-align (for numerics). */
  numeric?: boolean;
  /** Extra header/cell width or alignment classes. */
  class?: string;
  /** Value used for sorting; required for non-trivial cells. */
  sortValue?: (row: R) => string | number;
  /** Cell renderer snippet. Receives the row. */
  cell: Snippet<[R]>;
}
