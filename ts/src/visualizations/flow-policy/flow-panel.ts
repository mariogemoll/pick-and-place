// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

// The policy's output as it is being generated: one row per joint, one column
// per predicted timestep, and in every cell a 1-D track carrying that one
// number. A horizon is 16 steps of 6 joints, so the grid is exactly the tensor
// the U-Net produces.
//
// Sampling starts from a standard normal draw and integrates to a value in
// [-1, 1], so the tracks are drawn over [-3, 3] with the valid band shaded:
// the dots begin scattered across the whole track and collapse into the band.
// These are normalized action-space values, not joint angles -- only the
// finished sample means anything in degrees.

import { ARM_JOINT_NAMES } from '../../ik/kinematics';

export const JOINT_ROW_NAMES = [...ARM_JOINT_NAMES, 'gripper'] as const;

const TRACK_MIN = -3;
const TRACK_MAX = 3;

const LABEL_WIDTH = 78;
const ROW_GAP = 6;
const COLUMN_GAP = 3;
const PADDING = 10;
const HEADER_HEIGHT = 16;

const COLORS = {
  text: '#2f3337',
  muted: '#8d949c',
  track: '#e2e2e7',
  band: '#eef2f8',
  bandEdge: '#d3dbe6',
  dot: '#2f6f9b',
  dotPending: '#b9c6d1',
  trail: 'rgba(47, 111, 155, 0.18)',
  executing: '#2f6f45',
  executingFill: 'rgba(47, 111, 69, 0.08)'
} as const;

export interface FlowPanelFrame {
  /** `steps * joints` values in normalized action space, row-major by step. */
  values: Float32Array;
  /** Optional earlier state of the same horizon, drawn as a faint trail. */
  trail: Float32Array | null;
  steps: number;
  joints: number;
  /** Steps of this horizon that will actually be executed. */
  actSteps: number;
  /** Which predicted step is being executed now, or -1 outside execution. */
  executingStep: number;
  /** How far the integration has run, 0 at the noise draw, 1 at the sample. */
  progress: number;
  /**
   * How visible this horizon is. The grid it lives on is always drawn; the
   * values on it fade out once the horizon has been executed.
   */
  opacity: number;
  /** Which beat of the cycle this frame belongs to. */
  phase: 'sample' | 'flow' | 'execute';
}

export interface FlowPanel {
  canvas: HTMLCanvasElement;
  draw(frame: FlowPanelFrame): void;
  resize(): void;
  destroy(): void;
}

interface Geometry {
  cellWidth: number;
  rowHeight: number;
  plotLeft: number;
  plotTop: number;
}

function geometry(width: number, height: number, steps: number, joints: number): Geometry {
  const plotLeft = PADDING + LABEL_WIDTH;
  const plotTop = PADDING + HEADER_HEIGHT;
  return {
    plotLeft,
    plotTop,
    cellWidth: (width - plotLeft - PADDING) / steps,
    rowHeight: (height - plotTop - PADDING) / joints
  };
}

/** Where a normalized value sits inside a row, top-down in canvas pixels. */
function valueY(value: number, rowTop: number, rowHeight: number): number {
  const usable = rowHeight - ROW_GAP;
  const clamped = Math.min(Math.max(value, TRACK_MIN), TRACK_MAX);
  const fraction = (TRACK_MAX - clamped) / (TRACK_MAX - TRACK_MIN);
  return rowTop + ROW_GAP / 2 + fraction * usable;
}

function drawFrame(
  context: CanvasRenderingContext2D,
  width: number,
  height: number,
  frame: FlowPanelFrame
): void {
  const {
    steps, joints, values, trail, actSteps, executingStep, progress, opacity, phase
  } = frame;
  const { cellWidth, rowHeight, plotLeft, plotTop } = geometry(width, height, steps, joints);

  context.clearRect(0, 0, width, height);
  context.font = '11px ui-monospace, menlo, monospace';
  context.textBaseline = 'middle';

  // Header: which steps get executed, and which are thrown away.
  context.fillStyle = COLORS.muted;
  context.textAlign = 'left';
  context.fillText(`predicted step 1..${steps}`, plotLeft, PADDING + HEADER_HEIGHT / 2);
  context.textAlign = 'right';
  const headline = {
    sample: 'noise draw, t = 0',
    flow: `integrating, t = ${progress.toFixed(2)}`,
    execute: `executing ${actSteps} of ${steps}`
  }[phase];
  context.globalAlpha = opacity;
  context.fillText(headline, width - PADDING, PADDING + HEADER_HEIGHT / 2);
  context.globalAlpha = 1;

  // The executed prefix of the horizon, behind everything else.
  context.fillStyle = COLORS.executingFill;
  context.fillRect(plotLeft, plotTop, cellWidth * actSteps, rowHeight * joints);

  for (let joint = 0; joint < joints; joint++) {
    const rowTop = plotTop + joint * rowHeight;

    context.fillStyle = COLORS.text;
    context.textAlign = 'right';
    context.fillText(
      JOINT_ROW_NAMES[joint] ?? `joint ${joint}`,
      plotLeft - 8,
      rowTop + rowHeight / 2
    );

    // The band the finished sample has to land in.
    const bandTop = valueY(1, rowTop, rowHeight);
    const bandBottom = valueY(-1, rowTop, rowHeight);
    context.fillStyle = COLORS.band;
    context.fillRect(plotLeft, bandTop, cellWidth * steps, bandBottom - bandTop);
    context.strokeStyle = COLORS.bandEdge;
    context.lineWidth = 0.5;
    context.beginPath();
    context.moveTo(plotLeft, bandTop);
    context.lineTo(plotLeft + cellWidth * steps, bandTop);
    context.moveTo(plotLeft, bandBottom);
    context.lineTo(plotLeft + cellWidth * steps, bandBottom);
    context.stroke();

    for (let step = 0; step < steps; step++) {
      const cellX = plotLeft + step * cellWidth;
      const centerX = cellX + cellWidth / 2;
      const executed = step < actSteps;

      // The 1-D track this value lives on.
      context.strokeStyle = COLORS.track;
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(centerX, valueY(TRACK_MAX, rowTop, rowHeight));
      context.lineTo(centerX, valueY(TRACK_MIN, rowTop, rowHeight));
      context.stroke();

      // Everything from here on belongs to the horizon itself, so it fades
      // with it; the grid it is drawn on stays.
      context.globalAlpha = opacity;
      const index = step * joints + joint;
      if (trail !== null) {
        const fromY = valueY(trail[index], rowTop, rowHeight);
        const toY = valueY(values[index], rowTop, rowHeight);
        context.strokeStyle = COLORS.trail;
        context.lineWidth = 2.5;
        context.beginPath();
        context.moveTo(centerX, fromY);
        context.lineTo(centerX, toY);
        context.stroke();
      }

      context.fillStyle = executed ? COLORS.dot : COLORS.dotPending;
      context.beginPath();
      context.arc(
        centerX,
        valueY(values[index], rowTop, rowHeight),
        Math.min(2.6, cellWidth / 2 - COLUMN_GAP / 2),
        0,
        Math.PI * 2
      );
      context.fill();
      context.globalAlpha = 1;
    }
  }

  // The step whose command the arm is following right now. It is gone the
  // moment execution ends rather than dwelling on the last step.
  if (executingStep >= 0 && executingStep < steps) {
    context.globalAlpha = opacity;
    context.strokeStyle = COLORS.executing;
    context.lineWidth = 1.5;
    context.strokeRect(
      plotLeft + executingStep * cellWidth + COLUMN_GAP / 2,
      plotTop,
      cellWidth - COLUMN_GAP,
      rowHeight * joints
    );
    context.globalAlpha = 1;
  }
}

export function createFlowPanel(parent: HTMLElement): FlowPanel {
  const canvas = document.createElement('canvas');
  canvas.className = 'flow-policy-viz-panel';
  parent.appendChild(canvas);
  const context = canvas.getContext('2d');
  if (context === null) { throw new Error('Unable to get a 2d context for the flow panel'); }

  let cssWidth = 0;
  let cssHeight = 0;
  let latest: FlowPanelFrame | null = null;

  const resize = (): void => {
    const ratio = window.devicePixelRatio || 1;
    cssWidth = canvas.clientWidth || parent.clientWidth;
    cssHeight = canvas.clientHeight || parent.clientHeight;
    canvas.width = Math.round(cssWidth * ratio);
    canvas.height = Math.round(cssHeight * ratio);
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    if (latest !== null) { drawFrame(context, cssWidth, cssHeight, latest); }
  };

  resize();

  return {
    canvas,
    draw(frame: FlowPanelFrame): void {
      latest = frame;
      drawFrame(context, cssWidth, cssHeight, frame);
    },
    resize,
    destroy(): void { canvas.remove(); }
  };
}
