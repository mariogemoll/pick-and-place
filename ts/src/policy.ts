// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

// Entry point for the live policy page.
//
// Its own page rather than another block on the index: the simulator and the
// inference runtime together are an order of magnitude heavier than everything
// the index loads, and nobody should pay for them by scrolling past.

import './style.css';

import { initLivePolicyVisualization } from './visualizations/live-policy';

const container = document.getElementById('live-policy-visualization');
if (container !== null) {
  initLivePolicyVisualization(container).catch((error: unknown) => {
    container.textContent = `Could not start the policy runner: ${String(error)}`;
  });
}
