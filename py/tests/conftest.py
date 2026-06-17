# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Global pytest fixtures and configurations."""

import pupil_apriltags

# pupil_apriltags 1.0.4.post1 has a known bug in Python 3.13 where the Detector.__del__
# method causes a segmentation fault during garbage collection. 
# We monkey-patch it here to avoid crashing the pytest runner.
# This leaks a small amount of memory per detector, which is harmless in tests.
pupil_apriltags.Detector.__del__ = lambda self: None
