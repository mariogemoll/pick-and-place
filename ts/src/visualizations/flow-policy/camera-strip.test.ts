// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { describe, expect, it } from 'vitest';

import { layoutViews, VIEW_GAP } from './camera-strip';

describe('layoutViews', () => {
  it('gives each view its own column, with a gap between them', () => {
    const [left, right] = layoutViews(2, 400 + VIEW_GAP, 150);
    expect(left.x).toBe(0);
    expect(right.x).toBe(200 + VIEW_GAP);
  });

  it('caps the square at the strip height and aligns it to the top', () => {
    const [left, right] = layoutViews(2, 400, 150);
    expect(left.size).toBe(150);
    expect(right.size).toBe(150);
    // WebGL viewports are measured from the bottom, so a full-height square
    // starts at y = 0.
    expect(left.y).toBe(0);
  });

  it('fills a strip sized to hold squares of its own height', () => {
    // What the stylesheet sizes the host to: one square per view plus the gaps.
    const views = layoutViews(2, 2 * 76 + VIEW_GAP, 76);
    expect(views.map(view => view.size)).toEqual([76, 76]);
    expect(views.map(view => view.x)).toEqual([0, 76 + VIEW_GAP]);
    expect(views.every(view => view.y === 0)).toBe(true);
  });

  it('caps the square at the column width when the strip is narrow', () => {
    const [left, right] = layoutViews(2, 200 + VIEW_GAP, 150);
    expect(left.size).toBe(100);
    expect(left.y).toBe(50);
    expect(right.x).toBe(100 + VIEW_GAP);
  });

  it('never returns a degenerate viewport', () => {
    expect(layoutViews(2, 0, 0)[0].size).toBe(1);
  });
});
