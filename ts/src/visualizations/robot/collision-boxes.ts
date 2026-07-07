// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

export interface CollisionBoxDefinition {
  name: string;
  position: [number, number, number];
  quaternion: [number, number, number, number];
  size: [number, number, number];
  grip?: boolean;
}

export const SO101_COLLISION_BOXES: Record<string, readonly CollisionBoxDefinition[]> = {
  base: [
    {
      name: 'base_col0',
      position: [0.0108262962, -8.97657e-09, 0.0455999985],
      quaternion: [0.5, 0.5, 0.5, -0.5],
      size: [0.024, 0.042, 0.026]
    },
    {
      name: 'base_col1',
      position: [0.0208262962, -8.97657e-09, 0.00859999845],
      quaternion: [0.5, 0.5, 0.5, -0.5],
      size: [0.011, 0.044, 0.056]
    }
  ],
  shoulder: [
    {
      name: 'shoulder_col0',
      position: [-0.0181991995, 0.000162462614, 0.0188999997],
      quaternion: [0.0, 0.707106781, -0.707106781, 0.0],
      size: [0.025, 0.031, 0.028]
    },
    {
      name: 'shoulder_col1',
      position: [-0.0321991995, 0.00116246261, -0.0381000003],
      quaternion: [0.0, 0.707106781, -0.707106781, 0.0],
      size: [0.028, 0.019, 0.027]
    }
  ],
  upper_arm: [
    {
      name: 'upper_arm_col0',
      position: [-0.039084999, 0.000899999723, 0.0201499995],
      quaternion: [-0.5, -0.5, 0.5, 0.5],
      size: [0.012, 0.034, 0.052]
    },
    {
      name: 'upper_arm_col1',
      position: [-0.112084999, -0.0131000003, 0.0191499995],
      quaternion: [-0.5, -0.5, 0.5, 0.5],
      size: [0.026, 0.025, 0.019]
    }
  ],
  lower_arm: [
    {
      name: 'lower_arm_col0',
      position: [-0.0385499965, -0.000550141265, 0.0201997877],
      quaternion: [0.499974717, 0.500025282, -0.499974717, -0.500025282],
      size: [0.012, 0.032, 0.050]
    },
    {
      name: 'lower_arm_col1',
      position: [-0.117549996, 0.00344985871, 0.0202001922],
      quaternion: [0.499974717, 0.500025282, -0.499974717, -0.500025282],
      size: [0.018, 0.028, 0.027]
    }
  ],
  wrist: [
    {
      name: 'wrist_col0',
      position: [-0.000795616758, -0.00584594182, 0.0221499983],
      quaternion: [-0.00478076322, -0.00478066147, 0.707090645, 0.707090595],
      size: [0.012, 0.032, 0.017]
    },
    {
      name: 'wrist_col1',
      position: [-0.0000924933516, -0.0578411875, 0.028150002],
      quaternion: [-0.00478076322, -0.00478066147, 0.707090645, 0.707090595],
      size: [0.016, 0.026, 0.007]
    },
    {
      name: 'wrist_col2',
      position: [-0.00237626471, -0.0368701509, 0.0221500008],
      quaternion: [-0.00478076322, -0.00478066147, 0.707090645, 0.707090595],
      size: [0.018, 0.032, 0.012]
    }
  ],
  gripper: [
    {
      name: 'gripper_servo_col',
      position: [0.0088, 0.0002, -0.0234],
      quaternion: [0.707, -0.009, 0.707, 0.009],
      size: [0.012, 0.020, 0.012]
    },
    {
      name: 'fixed_jaw_col0a',
      position: [-0.01125, 0.0, -0.0285],
      quaternion: [1.0, 0.0, 0.0, 0.0],
      size: [0.02325, 0.024, 0.012],
      grip: true
    },
    {
      name: 'fixed_jaw_col0b',
      position: [-0.00225, -0.00225, -0.00675],
      quaternion: [1.0, 0.0, 0.0, 0.0],
      size: [0.03225, 0.02625, 0.00825],
      grip: true
    },
    {
      name: 'fixed_jaw_col1',
      position: [-0.0244, 0.0, -0.041025],
      quaternion: [1.0, 0.0, 0.0, 0.0],
      size: [0.00941, 0.016, 0.00412],
      grip: true
    },
    {
      name: 'fixed_jaw_col2',
      position: [-0.02291, 0.0, -0.05485],
      quaternion: [1.0, 0.0, 0.0, 0.0],
      size: [0.00962, 0.01134, 0.00981],
      grip: true
    },
    {
      name: 'fixed_jaw_col3',
      position: [-0.0176, 0.0, -0.0745],
      quaternion: [1.0, 0.0, 0.0, 0.0],
      size: [0.00601, 0.00805, 0.0098],
      grip: true
    },
    {
      name: 'fixed_jaw_col4',
      position: [-0.01492, -0.00018, -0.0893],
      quaternion: [1.0, 0.0, 0.0, 0.0],
      size: [0.00503, 0.00703, 0.005],
      grip: true
    },
    {
      name: 'fixed_jaw_col5',
      position: [-0.01189, -0.00015, -0.099363],
      quaternion: [1.0, 0.0, 0.0, 0.0],
      size: [0.004, 0.00545, 0.005063],
      grip: true
    },
    {
      name: 'camera_mount_col0',
      position: [0.0022, 0.04737, 0.00338],
      quaternion: [0.92388, 0.382683, 0.0, 0.0],
      size: [0.018, 0.014, 0.002]
    },
    {
      name: 'camera_mount_col1',
      position: [0.002414, 0.07091, 0.00478],
      quaternion: [0.976296, -0.21644, 0.0, 0.0],
      size: [0.018, 0.017823, 0.002608]
    },
    {
      name: 'camera_mount_col2',
      position: [0.001, 0.026, -0.016666],
      quaternion: [0.707107, 0.707107, 0.0, 0.0],
      size: [0.012, 0.011174, 0.002]
    },
    {
      name: 'camera_mount_col3',
      position: [0.002414, 0.032, -0.00715],
      quaternion: [1.0, 0.0, 0.0, 0.0],
      size: [0.018, 0.008, 0.002]
    },
    {
      name: 'wrist_camera_board_collision',
      position: [0.0025, 0.073357, 0.007515],
      quaternion: [-0.976296, 0.21644, 0.0, 0.0],
      size: [0.016, 0.016, 0.001]
    },
    {
      name: 'wrist_camera_lens_collision_box',
      position: [0.0025, 0.06870819, -0.00245438],
      quaternion: [-0.976296, 0.21644, 0.0, 0.0],
      size: [0.007, 0.007, 0.010]
    }
  ],
  moving_jaw_so101_v1: [
    {
      name: 'moving_jaw_col0',
      position: [-1.01141632e-06, -0.00600266783, 0.0189],
      quaternion: [1.0, 0.0, 0.0, 0.0],
      size: [0.00999898836, 0.0159973321, 0.0240000002],
      grip: true
    },
    {
      name: 'moving_jaw_col1',
      position: [0.0, -0.0398, 0.01835],
      quaternion: [1.0, 0.0, 0.0, 0.0],
      size: [0.00841, 0.02243, 0.01],
      grip: true
    },
    {
      name: 'moving_jaw_col2',
      position: [-0.004, -0.0669, 0.019],
      quaternion: [1.0, 0.0, 0.0, 0.0],
      size: [0.006, 0.005, 0.007],
      grip: true
    },
    {
      name: 'moving_jaw_col3',
      position: [-0.00695, -0.07695, 0.01902],
      quaternion: [1.0, 0.0, 0.0, 0.0],
      size: [0.005, 0.00505, 0.006],
      grip: true
    }
  ]
} as const;
