// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { describe, expect, it } from 'vitest';

import {
  columnCenter,
  type FlowPanelFrame,
  HISTORY_COLUMNS,
  layoutPanel,
  movementOpacity,
  rowCenter,
  type StripColumn,
  stripColumns
} from './flow-panel';

const ACT_STEPS = 8;
const COLUMNS = HISTORY_COLUMNS + ACT_STEPS;

// One joint per column keeps the arrays readable: a column's value is its own
// entry, so where a column came from is visible in what it carries.
function frame(overrides: Partial<FlowPanelFrame> = {}): FlowPanelFrame {
  return {
    history: new Float32Array([-2, -1]),
    values: new Float32Array([0, 1, 2, 3, 4, 5, 6, 7]),
    trail: null,
    joints: 1,
    actSteps: ACT_STEPS,
    slide: 0,
    progress: 1,
    ...overrides
  };
}

describe('layoutPanel', () => {
  it('divides the plot evenly between the columns', () => {
    const geometry = layoutPanel(388, 200, COLUMNS, 6);
    expect(geometry.cellWidth * COLUMNS).toBeCloseTo(geometry.plotWidth);
  });

  it('fits the rows to the plot, overlap included', () => {
    const geometry = layoutPanel(388, 120, COLUMNS, 6);
    // The outermost half-tracks are inside the plot, so the first row's track
    // starts at its top edge and the last row's ends at its bottom edge.
    expect(rowCenter(geometry, 0) - geometry.trackHeight / 2)
      .toBeCloseTo(geometry.plotTop);
    expect(rowCenter(geometry, 5) + geometry.trackHeight / 2)
      .toBeCloseTo(geometry.plotTop + geometry.plotHeight);
  });

  it('overlaps neighbouring rows rather than giving each a lane', () => {
    const geometry = layoutPanel(388, 120, COLUMNS, 6);
    expect(geometry.trackHeight).toBeGreaterThan(geometry.rowPitch);
    // The bands still clear each other, so the rows stay readable as rows.
    const band = geometry.trackHeight / 3;
    expect(band).toBeLessThan(geometry.rowPitch);
  });

  it('puts the seam on the boundary between executed and pending columns', () => {
    const geometry = layoutPanel(388, 200, COLUMNS, 6);
    expect(geometry.seamX).toBeCloseTo(geometry.plotLeft + HISTORY_COLUMNS * geometry.cellWidth);
    // The last executed column sits left of it, the first pending one right.
    expect(columnCenter(geometry, HISTORY_COLUMNS - 1)).toBeLessThan(geometry.seamX);
    expect(columnCenter(geometry, HISTORY_COLUMNS)).toBeGreaterThan(geometry.seamX);
  });

  it('never gives the plot a negative size', () => {
    const geometry = layoutPanel(0, 0, COLUMNS, 6);
    expect(geometry.plotWidth).toBe(0);
    expect(geometry.plotHeight).toBe(0);
  });
});

describe('movementOpacity', () => {
  it('keeps the line off a horizon that is still nowhere near integrated', () => {
    expect(movementOpacity(frame({ progress: 0 }))).toBe(0);
    expect(movementOpacity(frame({ progress: 0.5 }))).toBe(0);
  });

  it('fades the line in over the last of the integration', () => {
    expect(movementOpacity(frame({ progress: 0.9 }))).toBeCloseTo(0.5);
  });

  it('has the line fully in once the values have landed', () => {
    expect(movementOpacity(frame({ progress: 1 }))).toBe(1);
  });
});

describe('stripColumns', () => {
  it('fills the strip from the seam outwards before anything has scrolled', () => {
    const columns = stripColumns(frame());
    expect(columns.map(column => column.position))
      .toEqual([...Array(COLUMNS).keys()]);
    expect(columns.map(column => column.values[column.offset]))
      .toEqual([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7]);
  });

  it('counts only the history as executed before anything has scrolled', () => {
    expect(stripColumns(frame()).map(column => column.executed))
      .toEqual([true, true, false, false, false, false, false, false, false, false]);
  });

  it('lands the last two executed steps in the history columns', () => {
    // A whole horizon executed scrolls the strip by its length, which is the
    // picture the next horizon's cycle starts from.
    const columns = stripColumns(frame({ slide: ACT_STEPS }));
    const settled = columns.filter(column => column.position >= 0);
    expect(settled.map(column => column.position)).toEqual([0, 1]);
    expect(settled.map(column => column.values[column.offset])).toEqual([6, 7]);
    expect(settled.every(column => column.executed)).toBe(true);
  });

  it('turns a column executed as its dot passes the seam', () => {
    // The dot is drawn at the column's center, so it reads as executed from the
    // moment it passes the seam: half a column into the step that runs it.
    const firstPending = (slide: number): StripColumn =>
      stripColumns(frame({ slide }))[HISTORY_COLUMNS];
    expect(firstPending(0.4).executed).toBe(false);
    expect(firstPending(0.6).executed).toBe(true);
    // By the end of that step it is a whole column into the past.
    const executed = firstPending(1);
    expect(executed.values[executed.offset]).toBe(0);
    expect(executed.position).toBe(HISTORY_COLUMNS - 1);
  });

  it('scrolls the history out through the left edge', () => {
    const [oldest] = stripColumns(frame({ slide: 3 }));
    expect(oldest.position).toBe(-3);
  });

  it('right-aligns a history too short to fill its columns', () => {
    const columns = stripColumns(frame({ history: new Float32Array([-1]) }));
    expect(columns.length).toBe(1 + ACT_STEPS);
    // The one command there is sits against the seam, leaving the far column
    // empty rather than pretending the rollout had a step before its first.
    expect(columns[0].position).toBe(HISTORY_COLUMNS - 1);
    expect(columns[0].values[columns[0].offset]).toBe(-1);
  });

  it('starts a rollout with no history at all', () => {
    const columns = stripColumns(frame({ history: new Float32Array() }));
    expect(columns.length).toBe(ACT_STEPS);
    expect(columns[0].position).toBe(HISTORY_COLUMNS);
  });

  it('gives the trail to the horizon being integrated and not to the history', () => {
    const trail = new Float32Array(ACT_STEPS);
    const columns = stripColumns(frame({ trail }));
    expect(columns.slice(0, HISTORY_COLUMNS).every(column => column.trail === null)).toBe(true);
    expect(columns.slice(HISTORY_COLUMNS).every(column => column.trail === trail)).toBe(true);
  });

  it('indexes a row of joints per column', () => {
    const columns = stripColumns(frame({
      joints: 2,
      history: new Float32Array([-4, -3, -2, -1]),
      values: new Float32Array([0, 1, 2, 3]),
      actSteps: 2
    }));
    expect(columns.map(column => column.offset)).toEqual([0, 2, 0, 2]);
    expect(columns.map(column => [...column.values.slice(column.offset, column.offset + 2)]))
      .toEqual([[-4, -3], [-2, -1], [0, 1], [2, 3]]);
  });
});
