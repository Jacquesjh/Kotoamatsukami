
from typing import Any, List

import cv2
import mediapipe as mp
from numba import jit
import numpy as np


class HandDetector:


    max_hands    : int
    detection_con: float
    min_track_con: float
    hands        : Any
    tip_ids      : List[int]


    def __init__(self, max_hands: int = 2, detection_con: float = 0.5, min_track_con: float = 0.5) -> None:
        self.max_hands     = max_hands
        self.detection_con = detection_con
        self.min_track_con = min_track_con

        self.hands = mp.solutions.hands.Hands(max_num_hands = max_hands, min_detection_confidence = detection_con,
                                              min_tracking_confidence = min_track_con, model_complexity = 0)

        self.tip_ids = [4, 8, 12, 16, 20]


    def find_hands(self, image: np.ndarray, draw_marks: bool = True, draw_box: bool = True) -> tuple:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        
        height, width, _color = image.shape

        if results.multi_hand_landmarks:
            hands = self._get_hands(results = results, height = height, width = width,
                                    draw_marks = draw_marks, draw_box = draw_box, image = image)
        
        else:
            hands = []

        if draw_marks or draw_box:
            return hands, image

        else:
            return hands


    def _get_hands(self, results, height: int, width: int, draw_marks: bool, draw_box: bool, image: np.ndarray) -> List[dict]:
        hands = []

        for hand_type, hand_landmarks in zip(results.multi_handedness, results.multi_hand_landmarks):
            hand   = dict()
            unpack = self._get_landmarks(hand_landmarks = hand_landmarks, height = height, width = width)
            
            hand["landmarks"] = unpack[0]
            xs = unpack[1]
            ys = unpack[2]
            
            hand["bounding_box"] = self._get_bounding_box(xarray = xs, yarray = ys)
            bbox = hand["bounding_box"]

            center_x = bbox[0] + (bbox[2]//2)
            center_y = bbox[1] + (bbox[3]//2)

            hand["center"] = (center_x, center_y)

            if hand_type.classification[0].label == "Right":
                hand["type"] = "Left"
            
            else:
                hand["type"] = "Right"
            
            hands.append(hand)

            if draw_marks:
                mp.solutions.drawing_utils.draw_landmarks(image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            
            if draw_box:
                self._draw_box(image = image, hand = hand)

        return hands


    def _draw_box(self, image: np.ndarray, hand: dict) -> None:
        bbox  = hand["bounding_box"]
        line1 = bbox[0] - 20, bbox[1] - 20
        line2 = bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20

        cv2.rectangle(image, line1, line2, (255, 0, 255), 2)
        cv2.putText(image, hand["type"], (bbox[0] - 30, bbox[1] - 30),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)


    @staticmethod
    # @jit(nopython = True, fastmath = True)
    def _get_bounding_box(xarray: np.ndarray, yarray: np.ndarray) -> tuple:
        box_width    = xarray.max() - xarray.min()
        box_height   = yarray.max() - yarray.min()
        bounding_box = (xarray.min(), yarray.min(), box_width, box_height)

        return bounding_box


    @staticmethod
    def _get_landmarks(height: int, width: int, hand_landmarks) -> List[list]:        
        my_landmarks = []

        xs = np.empty(shape = len(hand_landmarks.landmark), dtype = np.int32)
        ys = np.empty(shape = len(hand_landmarks.landmark), dtype = np.int32)
        
        for id, landmark in enumerate(hand_landmarks.landmark):
            x, y = int(landmark.x*width), int(landmark.y*height)
            my_landmarks.append([x, y])
            xs[id] = x
            ys[id] = y

        return [my_landmarks, xs, ys]