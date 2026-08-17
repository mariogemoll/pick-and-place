// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import {
  buildWebModel,
  type BuiltWebModel,
  type WebModel
} from '../web-model';

const WORKSPACE_FRAME_BODY = 'workspace_frame_frame';
const ROBOT_BASE_FRAME_PARTS = new Set([
  'workspace_frame_north_03_visual'
]);

export function buildEnvironmentModel(
  environmentModel: WebModel,
  modelBasePath = '/so101_assets'
): BuiltWebModel {
  return buildWebModel(environmentModel, modelBasePath);
}

/**
 * The environment as a replay wants it: the workspace frame and the overhead
 * camera rig, but no floor plane (a grid stands in for it, and the two would
 * z-fight at z = 0) and no `pick_cube` (the replay animates its own cube).
 */
export function buildReplayEnvironmentModel(
  environmentModel: WebModel,
  modelBasePath = '/so101_assets'
): BuiltWebModel {
  return buildEnvironmentModel({
    ...environmentModel,
    bodies: environmentModel.bodies
      .filter(body => body.name !== 'pick_cube')
      .map(body => body.name === 'world' ? { ...body, geometries: [] } : body)
  }, modelBasePath);
}

export function buildWorkspaceFrameBase(
  environmentModel: WebModel,
  modelBasePath = '/so101_assets'
): BuiltWebModel {
  const world = environmentModel.bodies.find(body => body.name === 'world');
  const frame = environmentModel.bodies.find(body => body.name === WORKSPACE_FRAME_BODY);
  if (world === undefined || frame === undefined) {
    throw new Error(`Environment model is missing ${WORKSPACE_FRAME_BODY}`);
  }

  return buildEnvironmentModel({
    ...environmentModel,
    bodies: [
      { ...world, geometries: [] },
      {
        ...frame,
        geometries: frame.geometries.filter(geometry =>
          ROBOT_BASE_FRAME_PARTS.has(geometry.name)
        )
      }
    ]
  }, modelBasePath);
}
