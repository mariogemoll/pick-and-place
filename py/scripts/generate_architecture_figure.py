# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Generate the ACT policy architecture diagram."""

from pathlib import Path

from svgfig import Figure, arrow, elbow_arrow, image, rect, text

ACTION_COLORS = ["#f28b82", "#fcad70", "#fdd663", "#81c995", "#78d9ec"]
LAST_ACTION_COLOR = "#d0bcf1"
TOKEN_COLOR = "#cfe2f3"
ENCODER_COLOR = "#e6e6e6"

SQUARE_SIZE = 26
TOKEN_PITCH = 42
TOKEN_Y = 275

CVAE_X = 45
POLICY_ENCODER_X = 443
POLICY_DECODER_X = 932

SCRIPTS_DIRECTORY = Path(__file__).parent
PYTHON_DIRECTORY = SCRIPTS_DIRECTORY.parent
ASSETS_DIRECTORY = SCRIPTS_DIRECTORY / "architecture_figure_assets"
OUTPUT_DIRECTORY = PYTHON_DIRECTORY / "out"


def slot(base: float, index: int) -> float:
    return base + index * TOKEN_PITCH


def center(base: float, index: int) -> float:
    return slot(base, index) + SQUARE_SIZE / 2


def main() -> None:
    figure = Figure(1000, 362, drawing_width=1228, drawing_height=445)
    padding = 5

    cvae_cls_x = center(CVAE_X, 0)

    figure.add(
        rect(15, 15, 383, 350, fill="none", stroke="#808080", dash="4 3"),
        text(206.5, 356, "(training only)", size=12, fill="#808080"),
        rect(slot(CVAE_X, 0), 50, SQUARE_SIZE, SQUARE_SIZE, fill=TOKEN_COLOR),
        arrow(cvae_cls_x, 110, cvae_cls_x, 81),
    )

    figure.add(
        arrow(76, 57, 168, 43),
        arrow(76, 69, 168, 83),
        rect(172, 30, SQUARE_SIZE, SQUARE_SIZE, fill=TOKEN_COLOR),
        rect(172, 70, SQUARE_SIZE, SQUARE_SIZE, fill=TOKEN_COLOR),
        text(185, 47, "μ", size=12),
        text(185, 87, "σ", size=12),
        arrow(202, 43, 296, 57, dash="4 3"),
        arrow(202, 83, 296, 69, dash="4 3"),
        text(250, 67, "sample", size=12),
        rect(300, 50, SQUARE_SIZE, SQUARE_SIZE, fill=TOKEN_COLOR),
        text(313, 94, "z", size=12),
    )

    figure.add(
        rect(30, 110, 358, 140, rx=14, fill=ENCODER_COLOR),
        text(209, 187, "transformer encoder", size=21),
        rect(slot(CVAE_X, 0), TOKEN_Y, SQUARE_SIZE, SQUARE_SIZE, fill="white"),
        text(center(CVAE_X, 0), 328, "[CLS]", size=12),
        rect(
            slot(CVAE_X, 1) - padding,
            TOKEN_Y - padding,
            6 * TOKEN_PITCH - (TOKEN_PITCH - SQUARE_SIZE) + 2 * padding,
            SQUARE_SIZE + 2 * padding,
            rx=8,
        ),
    )
    cvae_centers = [center(CVAE_X, 0)]
    for index, color in enumerate(ACTION_COLORS[:4], start=1):
        figure.add(rect(slot(CVAE_X, index), TOKEN_Y, SQUARE_SIZE, SQUARE_SIZE, fill=color))
        cvae_centers.append(center(CVAE_X, index))
    figure.add(
        text(center(CVAE_X, 5), 294, ". . .", size=11),
        rect(slot(CVAE_X, 6), TOKEN_Y, SQUARE_SIZE, SQUARE_SIZE, fill=LAST_ACTION_COLOR),
        rect(slot(CVAE_X, 7), TOKEN_Y, SQUARE_SIZE, SQUARE_SIZE, fill=TOKEN_COLOR),
    )
    cvae_centers.extend([center(CVAE_X, 6), center(CVAE_X, 7)])
    action_center = (slot(CVAE_X, 1) + slot(CVAE_X, 6) + SQUARE_SIZE) / 2
    figure.add(text(action_center, 328, "action sequence", size=12))
    for token_center in cvae_centers:
        figure.add(arrow(token_center, TOKEN_Y - 2, token_center, 254))

    policy_token_slots = [0, 1, 2, 3, 5, 6, 7, 9]
    policy_dots_slots = [4, 8]
    policy_row_width = 10 * TOKEN_PITCH - (TOKEN_PITCH - SQUARE_SIZE) + 2 * padding
    camera_group_width = 4 * TOKEN_PITCH - (TOKEN_PITCH - SQUARE_SIZE) + 2 * padding

    figure.add(rect(slot(POLICY_ENCODER_X, 0) - padding, 50 - padding, policy_row_width, SQUARE_SIZE + 2 * padding, rx=8))
    for index in [*range(8), 9]:
        figure.add(
            rect(slot(POLICY_ENCODER_X, index), 50, SQUARE_SIZE, SQUARE_SIZE, fill=TOKEN_COLOR),
            arrow(center(POLICY_ENCODER_X, index), 110, center(POLICY_ENCODER_X, index), 89),
        )
    figure.add(
        text(center(POLICY_ENCODER_X, 8), 69, ". . .", size=11),
        rect(428, 110, 434, 140, rx=14, fill=ENCODER_COLOR),
        text(645, 187, "transformer encoder", size=21),
        rect(slot(POLICY_ENCODER_X, 2) - padding, TOKEN_Y - padding, camera_group_width, SQUARE_SIZE + 2 * padding, rx=8),
        rect(slot(POLICY_ENCODER_X, 6) - padding, TOKEN_Y - padding, camera_group_width, SQUARE_SIZE + 2 * padding, rx=8),
    )
    for index in policy_token_slots:
        figure.add(
            rect(slot(POLICY_ENCODER_X, index), TOKEN_Y, SQUARE_SIZE, SQUARE_SIZE, fill=TOKEN_COLOR),
            arrow(center(POLICY_ENCODER_X, index), TOKEN_Y - 2, center(POLICY_ENCODER_X, index), 254),
        )
    for index in policy_dots_slots:
        figure.add(text(center(POLICY_ENCODER_X, index), 294, ". . .", size=11))

    for first_slot, label, filename in [
        (2, "cam 1", "overhead.jpg"),
        (6, "cam 2", "wrist.jpg"),
    ]:
        camera_center = (
            slot(POLICY_ENCODER_X, first_slot)
            + slot(POLICY_ENCODER_X, first_slot + 3)
            + SQUARE_SIZE
        ) / 2
        figure.add(
            arrow(camera_center, 338, camera_center, 314),
            image(ASSETS_DIRECTORY / filename, camera_center - 40, 340, 80, 60),
            text(camera_center, 418, label, size=12),
        )

    cvae_joints_x = center(CVAE_X, 7)
    policy_joints_x = center(POLICY_ENCODER_X, 1)
    joints_x = (cvae_joints_x + policy_joints_x) / 2
    figure.add(
        elbow_arrow([(joints_x, 406), (joints_x, 375), (cvae_joints_x, 375), (cvae_joints_x, TOKEN_Y + SQUARE_SIZE + 5)]),
        elbow_arrow([(joints_x, 406), (joints_x, 375), (policy_joints_x, 375), (policy_joints_x, TOKEN_Y + SQUARE_SIZE + 5)]),
        text(joints_x, 418, "joints", size=12),
        elbow_arrow([(330, 63), (408, 63), (408, 330), (center(POLICY_ENCODER_X, 0), 330), (center(POLICY_ENCODER_X, 0), TOKEN_Y + SQUARE_SIZE + 5)]),
    )

    decoder_slots = [0, 1, 2, 3, 5]
    decoder_span = 5 * TOKEN_PITCH + SQUARE_SIZE
    decoder_center = slot(POLICY_DECODER_X, 0) + decoder_span / 2
    decoder_left = slot(POLICY_DECODER_X, 0) - 30
    figure.add(
        rect(slot(POLICY_DECODER_X, 0) - padding, 50 - padding, decoder_span + 2 * padding, SQUARE_SIZE + 2 * padding, rx=8),
        text(decoder_center, 36, "action sequence", size=12),
    )
    for index, color in zip(decoder_slots, [*ACTION_COLORS[:4], LAST_ACTION_COLOR], strict=True):
        figure.add(
            rect(slot(POLICY_DECODER_X, index), 50, SQUARE_SIZE, SQUARE_SIZE, fill=color),
            arrow(center(POLICY_DECODER_X, index), 110, center(POLICY_DECODER_X, index), 89),
        )
    figure.add(
        text(center(POLICY_DECODER_X, 4), 69, ". . .", size=11),
        rect(decoder_left, 110, decoder_span + 60, 140, rx=14, fill=ENCODER_COLOR),
        text(decoder_center, 187, "transformer decoder", size=21),
    )
    for index in decoder_slots:
        figure.add(
            rect(slot(POLICY_DECODER_X, index), TOKEN_Y, SQUARE_SIZE, SQUARE_SIZE, fill=TOKEN_COLOR),
            arrow(center(POLICY_DECODER_X, index), TOKEN_Y - 2, center(POLICY_DECODER_X, index), 254),
        )
    policy_output_right = slot(POLICY_ENCODER_X, 0) - padding + policy_row_width
    figure.add(
        text(center(POLICY_DECODER_X, 4), 294, ". . .", size=11),
        text(decoder_center, 328, "position embeddings (fixed/learned)", size=12),
        elbow_arrow([(policy_output_right, 63), (882, 63), (882, 180), (decoder_left - 4, 180)], radius=8),
    )

    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    figure.save(OUTPUT_DIRECTORY / "architecture.svg")


if __name__ == "__main__":
    main()
