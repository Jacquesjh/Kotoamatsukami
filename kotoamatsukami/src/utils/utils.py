
import numpy as np


def normalize_array(values: np.array) -> np.array:
    normalized_array = []

    for v in values:
        norm = v - np.min(values)/(np.max(values) - np.min(values))
        normalized_array.append(norm)

    return np.array(normalized_array)


def normalize_landmarks(landmarks: list) -> list:
    norm_landmarks = []

    xs_array = np.array(landmarks)[:, 0]
    ys_array = np.array(landmarks)[:, 1]

    for x, y in landmarks:
        norm_x = (x - np.min(xs_array))/(np.max(xs_array) - np.min(xs_array))
        norm_y = (y - np.min(ys_array))/(np.max(ys_array) - np.min(ys_array))

        norm_landmarks.append([norm_x, norm_y])

    return norm_landmarks