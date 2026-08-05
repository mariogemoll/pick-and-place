# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Argument groups the command-line scripts compose their parsers from.

Several scripts drive the same subsystems — the same policy, the same rig, the
same scene, the same dataset writer — and each used to declare that subsystem's
flags itself. They agreed by hand, and only approximately: the same flag carried
three different help texts, one was missing its ``type=Path``, and the seven
Diffusion Policy server flags existed in three copies with drifting descriptions.

Here each subsystem declares its own flags once. Where scripts legitimately
differ — a controller that also offers ``scripted``, a recorder whose default
codec differs — the group takes that as a parameter rather than being copied.
"""
