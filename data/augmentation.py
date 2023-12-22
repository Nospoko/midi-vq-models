import random

import numpy as np


def change_speed(dstart: np.ndarray, duration: np.ndarray, factor: float = None) -> tuple[np.ndarray, np.ndarray]:
    if not factor:
        slow = 0.8
        change_range = 0.4
        factor = slow + random.random() * change_range

    dstart /= factor
    duration /= factor
    return dstart, duration


def pitch_shift(pitch: np.ndarray, shift_threshold: int = 5) -> np.ndarray:
    # No more than given number of steps
    PITCH_LOW = 21
    PITCH_HI = 108
    low_shift = -min(shift_threshold, pitch.min() - PITCH_LOW)
    high_shift = min(shift_threshold, PITCH_HI - pitch.max())

    if low_shift > high_shift:
        shift = 0
    else:
        shift = random.randint(low_shift, high_shift)
    pitch += shift

    return pitch
