// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import './style.css';

import {
  type DummyVisualization,
  initDummyVisualization
} from './visualizations/dummy';
import {
  type GripperVisualization,
  initGripperVisualization
} from './visualizations/gripper';

let dummyVisualization: DummyVisualization | null = null;
let gripperVisualization: GripperVisualization | null = null;

function initialize(): void {
  const dummyPanel = document.getElementById('dummy-visualization');
  if (dummyPanel) {
    dummyVisualization?.destroy();
    dummyVisualization = null;

    void initDummyVisualization(dummyPanel).then(viz => {
      dummyVisualization = viz;
    });
  }

  const gripperPanel = document.getElementById('gripper-visualization');
  if (gripperPanel) {
    gripperVisualization?.destroy();
    gripperVisualization = null;

    void initGripperVisualization(gripperPanel).then(viz => {
      gripperVisualization = viz;
    });
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initialize);
} else {
  initialize();
}
