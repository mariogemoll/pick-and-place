// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import type { CarryProfilePoint } from './trajectory';

export interface CarryProfilePlot {
  element: HTMLDivElement;
  // Set the carry height-over-time profile to draw.
  setProfile(points: CarryProfilePoint[]): void;
  // Position the playback marker by carry phase [0, 1], or hide it with null.
  setMarker(fraction: number | null): void;
  resize(): void;
  destroy(): void;
}

export type CarryProfileXAxis = 'time' | 'distance';

const PADDING = { top: 10, right: 10, bottom: 22, left: 30 };
// The carry is much wider than it is tall; the axes are scaled independently so
// the arc shape is legible rather than geometrically true. A little vertical
// headroom keeps the apex off the top edge.
const HEIGHT_HEADROOM = 0.003;

// Sample the stored profile at a carry-phase fraction.
function sampleAt(
  points: CarryProfilePoint[], fraction: number
): CarryProfilePoint {
  const clamped = Math.min(1, Math.max(0, fraction));
  const end = points.findIndex(point => point.phase >= clamped);
  const b = points[Math.max(1, end)];
  const a = points[Math.max(0, end - 1)];
  const t = b.phase === a.phase ? 0 : (clamped - a.phase) / (b.phase - a.phase);
  return {
    phase: clamped,
    time: a.time + (b.time - a.time) * t,
    distance: a.distance + (b.distance - a.distance) * t,
    height: a.height + (b.height - a.height) * t
  };
}

export function createCarryProfilePlot(xAxis: CarryProfileXAxis): CarryProfilePlot {
  const element = document.createElement('div');
  element.className =
    `pick-and-place-viz-profile pick-and-place-viz-profile-${xAxis}`;
  element.hidden = true;

  const title = document.createElement('div');
  title.className = 'pick-and-place-viz-profile-title';
  title.textContent = xAxis === 'time'
    ? 'Carry height over time'
    : 'Carry height over distance';
  element.appendChild(title);

  const canvas = document.createElement('canvas');
  canvas.className = 'pick-and-place-viz-profile-canvas';
  element.appendChild(canvas);
  const context = canvas.getContext('2d');

  let profile: CarryProfilePoint[] = [];
  let marker: number | null = null;

  const bounds = (): {
    maxX: number;
    minHeight: number;
    maxHeight: number;
  } => {
    let maxX = 0;
    let minHeight = Infinity;
    let maxHeight = -Infinity;
    for (const point of profile) {
      maxX = Math.max(maxX, point[xAxis]);
      minHeight = Math.min(minHeight, point.height);
      maxHeight = Math.max(maxHeight, point.height);
    }
    if (!Number.isFinite(minHeight)) { minHeight = 0; maxHeight = 0; }
    return {
      maxX: maxX || 1,
      minHeight: minHeight - HEIGHT_HEADROOM,
      maxHeight: maxHeight + HEIGHT_HEADROOM
    };
  };

  const draw = (): void => {
    if (!context) { return; }
    const ratio = window.devicePixelRatio || 1;
    const cssWidth = canvas.clientWidth || 220;
    const cssHeight = canvas.clientHeight || 130;
    if (canvas.width !== Math.round(cssWidth * ratio) ||
      canvas.height !== Math.round(cssHeight * ratio)) {
      canvas.width = Math.round(cssWidth * ratio);
      canvas.height = Math.round(cssHeight * ratio);
    }
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, cssWidth, cssHeight);

    const plotLeft = PADDING.left;
    const plotRight = cssWidth - PADDING.right;
    const plotTop = PADDING.top;
    const plotBottom = cssHeight - PADDING.bottom;
    const plotWidth = plotRight - plotLeft;
    const plotHeight = plotBottom - plotTop;

    const { maxX, minHeight, maxHeight } = bounds();
    const heightSpan = maxHeight - minHeight || 1;
    const toX = (x: number): number =>
      plotLeft + (x / maxX) * plotWidth;
    const toY = (height: number): number =>
      plotBottom - ((height - minHeight) / heightSpan) * plotHeight;

    // Axes.
    context.strokeStyle = 'rgba(148, 163, 184, 0.8)';
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(plotLeft, plotTop);
    context.lineTo(plotLeft, plotBottom);
    context.lineTo(plotRight, plotBottom);
    context.stroke();

    context.fillStyle = 'rgba(71, 85, 105, 0.9)';
    context.font = '10px system-ui, sans-serif';

    // Height ticks (cm) at the spread of the arc.
    context.textAlign = 'right';
    context.textBaseline = 'middle';
    for (const heightCm of heightTicks(minHeight, maxHeight)) {
      const y = toY(heightCm / 100);
      context.strokeStyle = 'rgba(148, 163, 184, 0.25)';
      context.beginPath();
      context.moveTo(plotLeft, y);
      context.lineTo(plotRight, y);
      context.stroke();
      context.fillText(heightCm.toFixed(1), plotLeft - 4, y);
    }

    // X-axis ticks at the ends.
    context.textAlign = 'center';
    context.textBaseline = 'top';
    for (const value of [0, maxX]) {
      const x = toX(value);
      const label = xAxis === 'time'
        ? `${value.toFixed(1)}s`
        : `${(value * 100).toFixed(0)}cm`;
      context.fillText(label, x, plotBottom + 4);
    }

    if (profile.length < 2) { return; }

    // The arc.
    context.strokeStyle = '#ef4444';
    context.lineWidth = 2;
    context.beginPath();
    profile.forEach((point, index) => {
      const x = toX(point[xAxis]);
      const y = toY(point.height);
      if (index === 0) { context.moveTo(x, y); } else { context.lineTo(x, y); }
    });
    context.stroke();

    // Explicit path waypoints.
    context.font = '9px system-ui, sans-serif';
    context.textAlign = 'center';
    context.textBaseline = 'bottom';
    for (const point of profile.filter(
      candidate => candidate.waypoint !== undefined
    )) {
      const x = toX(point[xAxis]);
      const y = toY(point.height);
      context.fillStyle = '#1d4ed8';
      context.beginPath();
      context.arc(x, y, 3, 0, Math.PI * 2);
      context.fill();
      context.fillStyle = '#334155';
      context.fillText(point.waypoint ?? '', x, y - 5);
    }

    // Playback marker.
    if (marker !== null) {
      const point = sampleAt(profile, marker);
      context.fillStyle = '#f97316';
      context.beginPath();
      context.arc(toX(point[xAxis]), toY(point.height), 4, 0, Math.PI * 2);
      context.fill();
    }
  };

  return {
    element,
    setProfile(points: CarryProfilePoint[]): void {
      profile = points;
      draw();
    },
    setMarker(fraction: number | null): void {
      marker = fraction;
      draw();
    },
    resize(): void { draw(); },
    destroy(): void { element.remove(); }
  };
}

// A few evenly spaced height tick values (cm) spanning the arc, rounded to 0.5.
function heightTicks(minHeight: number, maxHeight: number): number[] {
  const minCm = Math.ceil(minHeight * 100 / 0.5) * 0.5;
  const maxCm = Math.floor(maxHeight * 100 / 0.5) * 0.5;
  const ticks: number[] = [];
  for (let cm = minCm; cm <= maxCm + 1e-9; cm += 0.5) {
    ticks.push(Number(cm.toFixed(1)));
  }
  return ticks;
}
