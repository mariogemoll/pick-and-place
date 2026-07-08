// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import './style.css';

import {
  type BodyTreeVisualization,
  initBodyTreeVisualization
} from './visualizations/body-tree';
import {
  type CanonicalGraspVisualization,
  initCanonicalGraspVisualization
} from './visualizations/canonical-grasp';
import {
  type EpisodeReplayVisualization,
  initEpisodeReplayVisualization
} from './visualizations/episode-replay';
import {
  type GraspAndLiftVisualization,
  initGraspAndLiftVisualization } from './visualizations/grasp-and-lift';
import {
  type GraspPoseVisualization,
  initGraspPoseVisualization } from './visualizations/grasp-pose';
import {
  type GraspPoseBreakdownVisualization,
  initGraspPoseBreakdownVisualization } from './visualizations/grasp-pose-breakdown';
import {
  type GripperVisualization,
  initGripperVisualization
} from './visualizations/gripper';
import {
  PickAndPlace,
  type PickAndPlaceVisualization
} from './visualizations/pick-and-place';
import {
  initRobotVisualization,
  type RobotVisualization
} from './visualizations/robot';
import {
  initializeRobotGridVisualization,
  type RobotGridVisualization
} from './visualizations/robot-grid';
import {
  initRobotViewerVisualization,
  type RobotViewerConfig,
  type RobotViewerVisualization
} from './visualizations/robot-viewer';
import {
  initSimpleGraspIkVisualization,
  type SimpleGraspIkVisualization
} from './visualizations/simple-grasp-ik';
import {
  initSimpleGraspPoseVisualization,
  type SimpleGraspPoseVisualization
} from './visualizations/simple-grasp-pose';
import {
  initStandardSceneVisualization,
  type StandardSceneVisualization } from './visualizations/standard-scene';

let standardSceneVisualization: StandardSceneVisualization | null = null;
let graspPoseVisualization: GraspPoseVisualization | null = null;
let graspPoseBreakdownVisualization: GraspPoseBreakdownVisualization | null = null;
let graspAndLiftVisualization: GraspAndLiftVisualization | null = null;
let gripperVisualization: GripperVisualization | null = null;
let robotVisualization: RobotVisualization | null = null;
let robotGridVisualization: RobotGridVisualization | null = null;
let bodyTreeVisualization: BodyTreeVisualization | null = null;
let simpleGraspPoseVisualization: SimpleGraspPoseVisualization | null = null;
let simpleGraspIkVisualization: SimpleGraspIkVisualization | null = null;
let canonicalGraspVisualization: CanonicalGraspVisualization | null = null;
let robotViewerVisualizations: RobotViewerVisualization[] = [];
let pickAndPlaceVisualization: PickAndPlaceVisualization | null = null;
let episodeReplayVisualization: EpisodeReplayVisualization | null = null;

function initialize(): void {
  const standardScenePanel = document.getElementById('standard-scene-visualization');
  if (standardScenePanel) {
    standardSceneVisualization?.destroy();
    standardSceneVisualization = null;

    void initStandardSceneVisualization(standardScenePanel).then(viz => {
      standardSceneVisualization = viz;
    });
  }

  const graspPosePanel = document.getElementById('grasp-pose-visualization');
  if (graspPosePanel) {
    graspPoseVisualization?.destroy();
    graspPoseVisualization = null;

    void initGraspPoseVisualization(graspPosePanel).then(viz => {
      graspPoseVisualization = viz;
    });
  }

  const simpleGraspPosePanel =
    document.getElementById('simple-grasp-pose-visualization');
  if (simpleGraspPosePanel) {
    simpleGraspPoseVisualization?.destroy();
    simpleGraspPoseVisualization = null;

    void initSimpleGraspPoseVisualization(simpleGraspPosePanel).then(viz => {
      simpleGraspPoseVisualization = viz;
    });
  }

  const simpleGraspIkPanel =
    document.getElementById('simple-grasp-ik-visualization');
  if (simpleGraspIkPanel) {
    simpleGraspIkVisualization?.destroy();
    simpleGraspIkVisualization = null;

    void initSimpleGraspIkVisualization(simpleGraspIkPanel).then(viz => {
      simpleGraspIkVisualization = viz;
    });
  }

  const canonicalGraspPanel =
    document.getElementById('canonical-grasp-visualization');
  if (canonicalGraspPanel) {
    canonicalGraspVisualization?.destroy();
    canonicalGraspVisualization = null;

    void initCanonicalGraspVisualization(canonicalGraspPanel).then(viz => {
      canonicalGraspVisualization = viz;
    });
  }

  const pickAndPlacePanel = document.getElementById('pick-and-place-visualization');
  if (pickAndPlacePanel) {
    pickAndPlaceVisualization?.destroy();
    pickAndPlaceVisualization = null;

    void PickAndPlace(pickAndPlacePanel, {
      startFromAndReturnToRestPose: true
    }).then(viz => {
      pickAndPlaceVisualization = viz;
    });
  }

  const episodeReplayPanel = document.getElementById('episode-replay-visualization');
  if (episodeReplayPanel) {
    episodeReplayVisualization?.destroy();
    episodeReplayVisualization = null;

    void initEpisodeReplayVisualization(episodeReplayPanel).then(viz => {
      episodeReplayVisualization = viz;
    });
  }

  const graspAndLiftPanel = document.getElementById('grasp-and-lift-visualization');
  if (graspAndLiftPanel) {
    graspAndLiftVisualization?.destroy();
    graspAndLiftVisualization = null;

    void initGraspAndLiftVisualization(graspAndLiftPanel).then(viz => {
      graspAndLiftVisualization = viz;
    });
  }

  const graspPoseBreakdownPanel =
    document.getElementById('grasp-pose-breakdown-visualization');
  if (graspPoseBreakdownPanel) {
    graspPoseBreakdownVisualization?.destroy();
    graspPoseBreakdownVisualization = null;

    void initGraspPoseBreakdownVisualization(graspPoseBreakdownPanel).then(viz => {
      graspPoseBreakdownVisualization = viz;
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

  const robotViewersPanel = document.getElementById('robot-viewers-visualization');
  if (robotViewersPanel) {
    for (const viz of robotViewerVisualizations) { viz.destroy(); }
    robotViewerVisualizations = [];

    const robots: RobotViewerConfig[] = [
      {
        label: 'UR5e',
        modelUrl: '/ur5e.json',
        modelBasePath: '/ur5e_assets',
        defaultJointDegrees: {
          shoulder_pan_joint: 70,
          shoulder_lift_joint: -40,
          elbow_joint: 70,
          wrist_1_joint: 68,
          wrist_2_joint: 26,
          wrist_3_joint: 70,
          gripper_right_driver_joint: 10
        }
      },
      {
        label: 'Panda',
        modelUrl: '/panda.json',
        modelBasePath: '/panda_assets',
        defaultJointDegrees: {
          joint1: 22,
          joint2: 64,
          joint3: -72,
          joint4: -92,
          joint5: 60,
          joint6: 67,
          joint7: 38
        },
        defaultJointMillimeters: {
          finger_joint1: 20
        }
      }
    ];
    for (const robot of robots) {
      void initRobotViewerVisualization(robotViewersPanel, robot).then(viz => {
        robotViewerVisualizations.push(viz);
      });
    }
  }

  const robotGridPanel = document.getElementById('robot-grid-visualization');
  if (robotGridPanel) {
    robotGridVisualization?.destroy();
    robotGridVisualization = null;

    void initializeRobotGridVisualization(robotGridPanel).then(viz => {
      robotGridVisualization = viz;
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
