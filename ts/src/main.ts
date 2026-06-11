// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import './style.css'
import {
  type DummyVisualization,
  initDummyVisualization
} from './visualizations/dummy'

let visualization: DummyVisualization | null = null;

function initialize(): void {
  const panel = document.getElementById('dummy-visualization');
  if (panel) {
    visualization?.destroy();
    visualization = null;
    void initDummyVisualization(panel).then(viz => {
      visualization = viz;
    });
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initialize);
} else {
  initialize();
}
