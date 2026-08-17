// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

// The policy's output as a strip that scrolls: one row per joint, one column
// per control step, and in every cell a 1-D track carrying that step's value
// for that joint. A seam down the middle of the strip marks now. The two
// columns left of it are the commands the arm has just carried out; the columns
// right of it are the steps it is about to.
//
// A horizon is 16 steps of which only the first 8 are ever executed, so the
// strip shows those 8 and leaves the discarded tail off: everything on screen
// is something the arm will actually do.
//
// Sampling starts from a standard normal draw and integrates to a value in
// [-1, 1], so the tracks are drawn over [-3, 3] with the valid band shaded:
// the dots begin scattered across the whole track and collapse into the band.
// Once they land, a line through them shows the movement they describe. It
// waits for them: a line through a half-integrated horizon would draw a motion
// that does not exist yet. These are normalized action-space values, not joint
// angles -- only the finished sample means anything in degrees.
//
// Executing the horizon then scrolls the strip left, one column per step, so
// the dots cross the seam as the arm carries them out. The two that end up left
// of the seam are the two the next horizon will be predicted from, which is why
// they are the ones kept: the strip settles into the picture the next cycle
// starts from, and no horizon has to be cleared away.

import { ARM_JOINT_NAMES } from '../../ik/kinematics';

const JOINT_ROW_NAMES = [...ARM_JOINT_NAMES, 'gripper'] as const;

// The panel is small enough that a column of full joint names would cost more
// room than the values it labels, and the rows are in a fixed order anyway.
const SHORT_ROW_LABELS: Record<string, string> = {
  shoulder_pan: 'pan',
  shoulder_lift: 'lift',
  elbow_flex: 'elbow',
  wrist_flex: 'w.flex',
  wrist_roll: 'w.roll',
  gripper: 'grip'
};

function rowLabel(joint: number): string {
  const name = JOINT_ROW_NAMES[joint] ?? `joint ${joint}`;
  return SHORT_ROW_LABELS[name] ?? name;
}

/** Executed commands kept left of the seam, which is what the policy is fed. */
export const HISTORY_COLUMNS = 2;

const TRACK_MIN = -3;
const TRACK_MAX = 3;

const LABEL_FONT = '9px ui-monospace, menlo, monospace';
const LABEL_WIDTH = 40;
const COLUMN_GAP = 3;
const PADDING = 7;
const DOT_RADIUS = 2.1;

// How tall a row's track is as a multiple of the distance to the next row.
// Above 1 the rows overlap, which buys the values room to move without the
// panel needing the height to give each row a lane of its own. The bands stay
// clear of each other as long as this is under 3.
const TRACK_OVERSCAN = 1.6;

const COLORS = {
  text: '#2f3337',
  track: '#e2e2e7',
  band: '#eef2f8',
  bandEdge: '#d3dbe6',
  past: '#9fb0bd',
  pastFill: 'rgba(143, 155, 168, 0.07)',
  pastLine: 'rgba(159, 176, 189, 0.85)',
  pending: '#2f6f9b',
  pendingLine: 'rgba(47, 111, 155, 0.5)',
  trail: 'rgba(47, 111, 155, 0.18)',
  seam: '#2f6f45'
} as const;

export interface FlowPanelFrame {
  /**
   * The commands the arm has already executed, oldest first, at most
   * `HISTORY_COLUMNS` of them and fewer at the start of a rollout. Sits right
   * up against the seam, so a short history leaves the leftmost column empty.
   */
  history: Float32Array;
  /** The horizon's executed prefix: `actSteps * joints`, row-major by step. */
  values: Float32Array;
  /** Optional state the values are being integrated from, drawn as a trail. */
  trail: Float32Array | null;
  joints: number;
  /** Steps of the horizon that get executed, which is what the strip shows. */
  actSteps: number;
  /**
   * How far the strip has scrolled left, in columns: 0 while a horizon is being
   * generated, rising to the number of steps executed as the arm carries it out.
   */
  slide: number;
  /** How far the integration has run, 0 at the noise draw, 1 at the sample. */
  progress: number;
}

export interface FlowPanel {
  canvas: HTMLCanvasElement;
  draw(frame: FlowPanelFrame): void;
  resize(): void;
  destroy(): void;
}

export interface PanelGeometry {
  plotLeft: number;
  plotTop: number;
  plotWidth: number;
  plotHeight: number;
  cellWidth: number;
  /** Vertical distance from one row's values to the next row's. */
  rowPitch: number;
  /** How tall one row's track is, which is more than the pitch. */
  trackHeight: number;
  /** Where now sits: the boundary between the executed and the pending columns. */
  seamX: number;
}

export function layoutPanel(
  width: number,
  height: number,
  columns: number,
  joints: number
): PanelGeometry {
  const plotLeft = PADDING + LABEL_WIDTH;
  const plotTop = PADDING;
  const plotWidth = Math.max(width - plotLeft - PADDING, 0);
  const plotHeight = Math.max(height - plotTop - PADDING, 0);
  // Overlapping tracks cover the plot in one pitch per row plus the overhang of
  // the outermost two half-tracks, so the overlap comes out of the rows rather
  // than hanging over the header.
  const rowPitch = plotHeight / (joints + TRACK_OVERSCAN - 1);
  return {
    plotLeft,
    plotTop,
    plotWidth,
    plotHeight,
    cellWidth: plotWidth / columns,
    rowPitch,
    trackHeight: rowPitch * TRACK_OVERSCAN,
    seamX: plotLeft + (plotWidth / columns) * HISTORY_COLUMNS
  };
}

/** Where a row's values are centered, top-down in canvas pixels. */
export function rowCenter(geometry: PanelGeometry, joint: number): number {
  return geometry.plotTop + geometry.trackHeight / 2 + joint * geometry.rowPitch;
}

/**
 * Where a column sits, in pixels. `position` is a column index that has been
 * scrolled, so it is fractional while the strip moves and goes negative once a
 * column has scrolled off the left edge.
 */
export function columnCenter(geometry: PanelGeometry, position: number): number {
  return geometry.plotLeft + (position + 0.5) * geometry.cellWidth;
}

/** One column of the strip: where it is now, and which values it carries. */
export interface StripColumn {
  position: number;
  values: Float32Array;
  /** Where this column's joint values start in `values`. */
  offset: number;
  trail: Float32Array | null;
  /** Whether the arm has executed this column's command. */
  executed: boolean;
}

/**
 * The strip laid out for one frame, history first. Both halves scroll together,
 * so the executed commands slide off the left edge as the pending ones cross
 * the seam to replace them.
 */
export function stripColumns(frame: FlowPanelFrame): StripColumn[] {
  const { history, values, trail, joints, actSteps, slide } = frame;
  const columns: StripColumn[] = [];
  const historyCount = joints > 0 ? history.length / joints : 0;

  const place = (
    column: number,
    source: Float32Array,
    offset: number,
    columnTrail: Float32Array | null
  ): StripColumn => {
    const position = column - slide;
    return {
      position,
      values: source,
      offset,
      trail: columnTrail,
      // The seam sits on a column boundary, half a column right of the center of
      // the last column to have been executed.
      executed: position < HISTORY_COLUMNS - 0.5
    };
  };

  for (let index = 0; index < historyCount; index++) {
    columns.push(place(HISTORY_COLUMNS - historyCount + index, history, index * joints, null));
  }
  for (let step = 0; step < actSteps; step++) {
    columns.push(place(HISTORY_COLUMNS + step, values, step * joints, trail));
  }
  return columns;
}

/** Where a normalized value sits on a row's track, top-down in canvas pixels. */
function valueY(value: number, center: number, trackHeight: number): number {
  const clamped = Math.min(Math.max(value, TRACK_MIN), TRACK_MAX);
  return center - (clamped / (TRACK_MAX - TRACK_MIN)) * trackHeight;
}

// The fixed furniture: row labels, the band a finished sample has to land in,
// one track per column, and the seam. None of it scrolls.
function drawGrid(
  context: CanvasRenderingContext2D,
  geometry: PanelGeometry,
  columns: number,
  joints: number
): void {
  const { plotLeft, plotTop, plotHeight, cellWidth, trackHeight, seamX } = geometry;

  context.fillStyle = COLORS.pastFill;
  context.fillRect(plotLeft, plotTop, seamX - plotLeft, plotHeight);

  for (let joint = 0; joint < joints; joint++) {
    const center = rowCenter(geometry, joint);

    context.fillStyle = COLORS.text;
    context.textAlign = 'right';
    context.fillText(rowLabel(joint), plotLeft - 6, center);

    const bandTop = valueY(1, center, trackHeight);
    const bandBottom = valueY(-1, center, trackHeight);
    context.fillStyle = COLORS.band;
    context.fillRect(plotLeft, bandTop, cellWidth * columns, bandBottom - bandTop);
    context.strokeStyle = COLORS.bandEdge;
    context.lineWidth = 0.5;
    context.beginPath();
    context.moveTo(plotLeft, bandTop);
    context.lineTo(plotLeft + cellWidth * columns, bandTop);
    context.moveTo(plotLeft, bandBottom);
    context.lineTo(plotLeft + cellWidth * columns, bandBottom);
    context.stroke();

    context.strokeStyle = COLORS.track;
    context.lineWidth = 1;
    context.beginPath();
    for (let column = 0; column < columns; column++) {
      const centerX = columnCenter(geometry, column);
      context.moveTo(centerX, valueY(TRACK_MAX, center, trackHeight));
      context.lineTo(centerX, valueY(TRACK_MIN, center, trackHeight));
    }
    context.stroke();
  }

  context.strokeStyle = COLORS.seam;
  context.lineWidth = 1.5;
  context.beginPath();
  context.moveTo(seamX, plotTop);
  context.lineTo(seamX, plotTop + plotHeight);
  context.stroke();
}

/** The last of the integration, over which the pending line comes in. */
const MOVEMENT_FADE = 0.2;

/**
 * How solid the line through the pending steps is.
 *
 * Mid-integration those values are on their way from noise to a sample and are
 * not a trajectory the arm could follow, so the line stays away until they are
 * arriving, and then fades in over the last of the integration rather than
 * popping into place. In continuous mode every horizon is shown finished, so it
 * is always fully in.
 *
 * The steps already executed are not gated on any of this: that movement has
 * happened, and it stays on screen while the next horizon is drawn.
 */
export function movementOpacity(frame: FlowPanelFrame): number {
  // Measured back from the end of the integration, so a finished horizon lands
  // on exactly 1 rather than a rounding error short of it.
  const settling = 1 - (1 - frame.progress) / MOVEMENT_FADE;
  return Math.min(Math.max(settling, 0), 1);
}

// The movement the commands add up to: one line per joint through the value in
// every column, so a horizon reads as a motion the arm will make rather than
// ten separate numbers. It runs straight across the seam, because the movement
// does: the commands left of it are where the joint has just come from.
//
// A segment is the move into the step on its right, so that step's side of the
// seam is what colors it -- the line turns from pending to past one segment at
// a time as the strip scrolls.
function drawMovement(
  context: CanvasRenderingContext2D,
  geometry: PanelGeometry,
  columns: StripColumn[],
  joint: number,
  center: number,
  trackHeight: number,
  pendingOpacity: number
): void {
  const dotY = (column: StripColumn): number =>
    valueY(column.values[column.offset + joint], center, trackHeight);

  context.lineWidth = 1;
  // Batched by color rather than drawn segment by segment, which would restroke
  // every row of every frame a dozen times over.
  for (const executed of [true, false]) {
    const opacity = executed ? 1 : pendingOpacity;
    if (opacity <= 0) { continue; }
    context.globalAlpha = opacity;
    context.strokeStyle = executed ? COLORS.pastLine : COLORS.pendingLine;
    context.beginPath();
    for (let index = 1; index < columns.length; index++) {
      const to = columns[index];
      if (to.executed !== executed) { continue; }
      const from = columns[index - 1];
      context.moveTo(columnCenter(geometry, from.position), dotY(from));
      context.lineTo(columnCenter(geometry, to.position), dotY(to));
    }
    context.stroke();
  }
  context.globalAlpha = 1;
}

function drawFrame(
  context: CanvasRenderingContext2D,
  width: number,
  height: number,
  frame: FlowPanelFrame
): void {
  const { joints, actSteps } = frame;
  const columns = HISTORY_COLUMNS + actSteps;
  const geometry = layoutPanel(width, height, columns, joints);
  const { plotLeft, plotTop, plotWidth, plotHeight, cellWidth, trackHeight } = geometry;

  context.clearRect(0, 0, width, height);
  context.font = LABEL_FONT;
  context.textBaseline = 'middle';

  drawGrid(context, geometry, columns, joints);

  // The strip itself, clipped to the plot so scrolled-out columns disappear at
  // the left edge instead of running out over the row labels.
  const radius = Math.min(DOT_RADIUS, cellWidth / 2 - COLUMN_GAP / 2);
  context.save();
  context.beginPath();
  context.rect(plotLeft, plotTop, plotWidth, plotHeight);
  context.clip();
  const columnStrip = stripColumns(frame);
  const pendingOpacity = movementOpacity(frame);
  for (let joint = 0; joint < joints; joint++) {
    const center = rowCenter(geometry, joint);
    drawMovement(context, geometry, columnStrip, joint, center, trackHeight, pendingOpacity);

    for (const column of columnStrip) {
      const centerX = columnCenter(geometry, column.position);
      const value = column.values[column.offset + joint];

      if (column.trail !== null) {
        context.strokeStyle = COLORS.trail;
        context.lineWidth = 2;
        context.beginPath();
        context.moveTo(centerX, valueY(column.trail[column.offset + joint], center, trackHeight));
        context.lineTo(centerX, valueY(value, center, trackHeight));
        context.stroke();
      }

      context.fillStyle = column.executed ? COLORS.past : COLORS.pending;
      context.beginPath();
      context.arc(centerX, valueY(value, center, trackHeight), radius, 0, Math.PI * 2);
      context.fill();
    }
  }
  context.restore();
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
