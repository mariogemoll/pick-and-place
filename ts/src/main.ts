// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import './style.css';

import {
  type BodyTreeVisualization,
  initBodyTreeVisualization
} from './visualizations/body-tree';
import {
  type DummyVisualization,
  initDummyVisualization
} from './visualizations/dummy';
import {
  type GripperVisualization,
  initGripperVisualization
} from './visualizations/gripper';
import {
  initPregraspPoseVisualization,
  type PregraspPoseVisualization
} from './visualizations/pregrasp-pose';
import {
  initPregraspPoseBreakdownVisualization,
  type PregraspPoseBreakdownVisualization
} from './visualizations/pregrasp-pose-breakdown';
import {
  initRobotVisualization,
  type RobotVisualization
} from './visualizations/robot';

let dummyVisualization: DummyVisualization | null = null;
let pregraspPoseVisualization: PregraspPoseVisualization | null = null;
let pregraspPoseBreakdownVisualization: PregraspPoseBreakdownVisualization | null = null;
let gripperVisualization: GripperVisualization | null = null;
let robotVisualization: RobotVisualization | null = null;
let bodyTreeVisualization: BodyTreeVisualization | null = null;

function initialize(): void {
  const dummyPanel = document.getElementById('dummy-visualization');
  if (dummyPanel) {
    dummyVisualization?.destroy();
    dummyVisualization = null;

    void initDummyVisualization(dummyPanel).then(viz => {
      dummyVisualization = viz;
    });
  }

  const pregraspPosePanel = document.getElementById('pregrasp-pose-visualization');
  if (pregraspPosePanel) {
    pregraspPoseVisualization?.destroy();
    pregraspPoseVisualization = null;

    void initPregraspPoseVisualization(pregraspPosePanel).then(viz => {
      pregraspPoseVisualization = viz;
    });
  }

  const pregraspPoseBreakdownPanel =
    document.getElementById('pregrasp-pose-breakdown-visualization');
  if (pregraspPoseBreakdownPanel) {
    pregraspPoseBreakdownVisualization?.destroy();
    pregraspPoseBreakdownVisualization = null;

    void initPregraspPoseBreakdownVisualization(pregraspPoseBreakdownPanel).then(viz => {
      pregraspPoseBreakdownVisualization = viz;
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

  const robotPanel = document.getElementById('robot-visualization');
  if (robotPanel) {
    robotVisualization?.destroy();
    robotVisualization = null;

    void initRobotVisualization(robotPanel).then(viz => {
      robotVisualization = viz;
    });
  }

  const bodyTreePanel = document.getElementById('body-tree-visualization');
  if (bodyTreePanel) {
    bodyTreeVisualization?.destroy();
    void initBodyTreeVisualization(bodyTreePanel).then(viz => {
      bodyTreeVisualization = viz;
    });
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initialize);
} else {
  initialize();
}
