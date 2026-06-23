// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import type { WebModel } from '../web-model';

const ROBOT_BASE_BODY = 'base';

export function robotModelWithBaseOnFloor(model: WebModel): WebModel {
  return {
    ...model,
    bodies: model.bodies.map(body => (
      body.name === ROBOT_BASE_BODY
        ? { ...body, position: [body.position[0], body.position[1], 0] }
        : body
    ))
  };
}
