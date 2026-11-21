import { describe, expect, it } from 'vitest';
import {
  DEFAULT_UNDO_MAX_DEPTH,
  UndoStack,
  createUndoStack,
  shallowRecordEqual,
} from './undoStack';
import { DEFAULT_STUDIO_RACK_SETTINGS, type StudioRackSettings } from './presets';

describe('UndoStack<T>', () => {
  it('initializes cleanly with default maxDepth=50 and empty history', () => {
    const stack = createUndoStack(DEFAULT_STUDIO_RACK_SETTINGS);
    expect(stack.present).toBe(DEFAULT_STUDIO_RACK_SETTINGS);
    expect(stack.maxDepth).toBe(DEFAULT_UNDO_MAX_DEPTH);
    expect(stack.canUndo).toBe(false);
    expect(stack.canRedo).toBe(false);
    expect(stack.undoCount).toBe(0);
    expect(stack.redoCount).toBe(0);
    expect(Object.isFrozen(stack)).toBe(true);
    expect(Object.isFrozen(stack.past)).toBe(true);
    expect(Object.isFrozen(stack.future)).toBe(true);
  });

  it('rejects invalid maxDepth values', () => {
    expect(() => createUndoStack(0, { maxDepth: 0 })).toThrow(RangeError);
    expect(() => createUndoStack(0, { maxDepth: -5 })).toThrow(RangeError);
    expect(() => createUndoStack(0, { maxDepth: 2.5 })).toThrow(RangeError);
    expect(() => createUndoStack(0, { maxDepth: Number.NaN })).toThrow(RangeError);
  });

  it('pushes, undoes, and redoes immutably without mutating prior snapshots', () => {
    const s0 = createUndoStack<Record<number, string>>({});
    const s1 = s0.push({ 1: 'enhanced' });
    const s2 = s1.push({ 1: 'enhanced', 2: 'reverted' });

    // Prior instances remain untouched
    expect(s0.present).toEqual({});
    expect(s0.canUndo).toBe(false);
    expect(s1.present).toEqual({ 1: 'enhanced' });
    expect(s1.undoCount).toBe(1);

    expect(s2.present).toEqual({ 1: 'enhanced', 2: 'reverted' });
    expect(s2.canUndo).toBe(true);
    expect(s2.canRedo).toBe(false);

    const u1 = s2.undo();
    expect(u1.present).toEqual({ 1: 'enhanced' });
    expect(u1.canUndo).toBe(true);
    expect(u1.canRedo).toBe(true);
    expect(u1.redoCount).toBe(1);

    const u0 = u1.undo();
    expect(u0.present).toEqual({});
    expect(u0.canUndo).toBe(false);
    expect(u0.redoCount).toBe(2);

    // Extra undo is a no-op returning the same instance
    expect(u0.undo()).toBe(u0);

    const r1 = u0.redo();
    expect(r1.present).toEqual({ 1: 'enhanced' });
    const r2 = r1.redo();
    expect(r2.present).toEqual({ 1: 'enhanced', 2: 'reverted' });
    expect(r2.canRedo).toBe(false);

    // Extra redo is a no-op
    expect(r2.redo()).toBe(r2);
  });

  it('clears the redo branch when a new change is pushed after undo', () => {
    let stack = createUndoStack('a').push('b').push('c');
    stack = stack.undo(); // present='b', future=['c']
    expect(stack.canRedo).toBe(true);

    const branched = stack.push('d');
    expect(branched.present).toBe('d');
    expect(branched.canRedo).toBe(false);
    expect(branched.past).toEqual(['a', 'b']);
  });

  it('enforces bounded maxDepth=50 by dropping oldest entries', () => {
    let stack = createUndoStack(0);
    for (let i = 1; i <= 75; i++) {
      stack = stack.push(i);
    }
    expect(stack.present).toBe(75);
    expect(stack.undoCount).toBe(50);
    expect(stack.past[0]).toBe(25);
    expect(stack.past[49]).toBe(74);

    // Undoing 50 times reaches 25, then stops
    for (let i = 0; i < 50; i++) {
      stack = stack.undo();
    }
    expect(stack.present).toBe(25);
    expect(stack.canUndo).toBe(false);
    expect(stack.redoCount).toBe(50);
  });

  it('skips duplicate pushes using custom equality (shallowRecordEqual for StudioRackSettings)', () => {
    const stack = createUndoStack<StudioRackSettings>(DEFAULT_STUDIO_RACK_SETTINGS, {
      equals: shallowRecordEqual,
    });
    const sameCopy = stack.push({ ...DEFAULT_STUDIO_RACK_SETTINGS });
    expect(sameCopy).toBe(stack);
    expect(sameCopy.canUndo).toBe(false);

    const modified = stack.push({ ...DEFAULT_STUDIO_RACK_SETTINGS, strength: 0.9 });
    expect(modified).not.toBe(stack);
    expect(modified.canUndo).toBe(true);
    expect(modified.present.strength).toBe(0.9);
  });

  it('supports replacePresent for live slider coalescing and clear() with or without reset value', () => {
    const s1 = UndoStack.create(10).push(20);
    const dragged = s1.replacePresent(25);
    expect(dragged.present).toBe(25);
    expect(dragged.past).toEqual([10]);
    expect(dragged.undo().present).toBe(10);

    const clearedSame = dragged.clear();
    expect(clearedSame.present).toBe(25);
    expect(clearedSame.canUndo).toBe(false);
    expect(clearedSame.canRedo).toBe(false);
    expect(clearedSame.clear()).toBe(clearedSame);

    const clearedReset = dragged.clear(0);
    expect(clearedReset.present).toBe(0);
    expect(clearedReset.canUndo).toBe(false);
  });
});
