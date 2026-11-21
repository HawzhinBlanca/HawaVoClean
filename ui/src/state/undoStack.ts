// Generic immutable bounded Undo/Redo stack designed for unit overrides and
// studio rack parameter snapshots.

export const DEFAULT_UNDO_MAX_DEPTH = 50;

export interface UndoStackOptions<T> {
  /** Maximum number of past (and future) states retained. Defaults to 50. */
  maxDepth?: number | undefined;
  /** Optional equality predicate to skip no-op pushes. Defaults to Object.is. */
  equals?: ((a: T, b: T) => boolean) | undefined;
}

/**
 * Immutable generic Undo/Redo stack.
 *
 * Every mutating operation (`push`, `undo`, `redo`, `clear`, `replacePresent`)
 * returns a fresh `UndoStack<T>` instance without mutating prior instances, so
 * it integrates cleanly with React state and Zustand stores.
 */
export class UndoStack<T> {
  readonly past: readonly T[];
  readonly present: T;
  readonly future: readonly T[];
  readonly maxDepth: number;
  private readonly equals: (a: T, b: T) => boolean;

  constructor(
    present: T,
    options?: UndoStackOptions<T>,
    past: readonly T[] = [],
    future: readonly T[] = [],
  ) {
    const maxDepth = options?.maxDepth ?? DEFAULT_UNDO_MAX_DEPTH;
    if (!Number.isInteger(maxDepth) || maxDepth < 1) {
      throw new RangeError(`UndoStack maxDepth must be a positive integer, got ${String(maxDepth)}`);
    }

    this.maxDepth = maxDepth;
    this.equals = options?.equals ?? Object.is;
    this.present = present;
    this.past = Object.freeze(past.length > maxDepth ? past.slice(past.length - maxDepth) : [...past]);
    this.future = Object.freeze(future.length > maxDepth ? future.slice(0, maxDepth) : [...future]);
    Object.freeze(this);
  }

  static create<T>(initial: T, options?: UndoStackOptions<T>): UndoStack<T> {
    return new UndoStack<T>(initial, options);
  }

  get canUndo(): boolean {
    return this.past.length > 0;
  }

  get canRedo(): boolean {
    return this.future.length > 0;
  }

  get undoCount(): number {
    return this.past.length;
  }

  get redoCount(): number {
    return this.future.length;
  }

  /** Push a new state, clearing the redo branch and enforcing `maxDepth`. */
  push(next: T): UndoStack<T> {
    if (this.equals(this.present, next)) {
      return this;
    }
    const nextPast =
      this.past.length >= this.maxDepth
        ? [...this.past.slice(this.past.length - this.maxDepth + 1), this.present]
        : [...this.past, this.present];

    return new UndoStack<T>(
      next,
      { maxDepth: this.maxDepth, equals: this.equals },
      nextPast,
      [],
    );
  }

  /** Step backward by one snapshot. Returns `this` if nothing can be undone. */
  undo(): UndoStack<T> {
    if (this.past.length === 0) {
      return this;
    }
    const previous = this.past[this.past.length - 1]!;
    const nextPast = this.past.slice(0, -1);
    const nextFuture = [this.present, ...this.future].slice(0, this.maxDepth);

    return new UndoStack<T>(
      previous,
      { maxDepth: this.maxDepth, equals: this.equals },
      nextPast,
      nextFuture,
    );
  }

  /** Step forward by one snapshot. Returns `this` if nothing can be redone. */
  redo(): UndoStack<T> {
    if (this.future.length === 0) {
      return this;
    }
    const nextPresent = this.future[0]!;
    const nextFuture = this.future.slice(1);
    const nextPast =
      this.past.length >= this.maxDepth
        ? [...this.past.slice(this.past.length - this.maxDepth + 1), this.present]
        : [...this.past, this.present];

    return new UndoStack<T>(
      nextPresent,
      { maxDepth: this.maxDepth, equals: this.equals },
      nextPast,
      nextFuture,
    );
  }

  /**
   * Replace `present` in place without recording an undo entry (useful for
   * coalescing continuous slider drags before committing a checkpoint).
   */
  replacePresent(next: T): UndoStack<T> {
    if (this.equals(this.present, next)) {
      return this;
    }
    return new UndoStack<T>(
      next,
      { maxDepth: this.maxDepth, equals: this.equals },
      this.past,
      this.future,
    );
  }

  /**
   * Drop all undo/redo history while keeping `present` (or resetting `present`
   * to `nextPresent` when supplied).
   */
  clear(nextPresent?: T): UndoStack<T> {
    const targetPresent = arguments.length > 0 ? (nextPresent as T) : this.present;
    if (
      this.past.length === 0 &&
      this.future.length === 0 &&
      this.equals(this.present, targetPresent)
    ) {
      return this;
    }
    return new UndoStack<T>(
      targetPresent,
      { maxDepth: this.maxDepth, equals: this.equals },
      [],
      [],
    );
  }
}

export function createUndoStack<T>(initial: T, options?: UndoStackOptions<T>): UndoStack<T> {
  return UndoStack.create(initial, options);
}

/**
 * Shallow value equality helper for flat parameter objects (such as
 * `StudioRackSettings` or per-unit override maps).
 */
export function shallowRecordEqual<T extends object>(a: T, b: T): boolean {
  if (Object.is(a, b)) return true;
  const keysA = Object.keys(a) as (keyof T)[];
  const keysB = Object.keys(b) as (keyof T)[];
  if (keysA.length !== keysB.length) return false;
  for (const key of keysA) {
    if (!Object.prototype.hasOwnProperty.call(b, key) || !Object.is(a[key], b[key])) {
      return false;
    }
  }
  return true;
}
