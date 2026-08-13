// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { describe, expect, it } from 'vitest';

import { layoutViews } from './camera-strip';

describe('layoutViews', () => {
  it('gives each view its own column', () => {
    const [left, right] = layoutViews(2, 400, 150);
    expect(left.x).toBe(0);
    expect(right.x).toBe(200);
  });

  it('caps the square at the strip height and aligns it to the top', () => {
    const [left, right] = layoutViews(2, 400, 150);
    expect(left.size).toBe(150);
    expect(right.size).toBe(150);
    // WebGL viewports are measured from the bottom, so a full-height square
    // starts at y = 0.
    expect(left.y).toBe(0);
  });

  it('caps the square at the column width when the strip is narrow', () => {
    const [left, right] = layoutViews(2, 200, 150);
    expect(left.size).toBe(100);
    expect(left.y).toBe(50);
    expect(right.x).toBe(100);
  });

  it('never returns a degenerate viewport', () => {
    expect(layoutViews(2, 0, 0)[0].size).toBe(1);
  });
});
