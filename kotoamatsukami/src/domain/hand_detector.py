import math
from threading import Thread
from typing import List

import numpy as np
import mediapipe as mp
import cv2
from numba import jit


class HandDetector:


    def __init__(self, max_hands: int = 2, detection_con: float = 0.5, min_track_con: float = 0.5) -> None:

        self.max_hands     = max_hands
        self.detection_con = detection_con
        self.min_track_con = min_track_con

        self.hands = mp.solutions.hands.Hands(max_num_hands = max_hands, min_detection_confidence = detection_con,
                                              min_tracking_confidence = min_track_con)

        self.tip_ids = [4, 8, 12, 16, 20]


    def find_hands(self, image: np.ndarray, draw_marks: bool = True, draw_box: bool = True, flip_view: bool = True) -> tuple:
        
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        
        height, width, color = image.shape
        hands = []
        if results.multi_hand_landmarks:
            hands = self._get_hands(results = results, height = height, width = width,
                                    draw_marks = draw_marks, draw_box = draw_box,
                                    flip_view = flip_view, image = image)

        if draw_marks or draw_box:
            return hands, image

        else:
            return hands

    # @jit
    def _get_hands(self, results, height: int, width: int, draw_marks: bool, draw_box: bool, flip_view: bool, image: np.ndarray) -> List[dict]:

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

            if flip_view:
                if hand_type.classification[0].label == "Right":
                    hand["type"] = "Left"
                
                else:
                    hand["type"] = "Right"
            
            else:
                hand["type"] = hand_type.classification[0].label

            hands.append(hand)

            if draw_marks:
                mp.solutions.drawing_utils.draw_landmarks(image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            
            if draw_box:
                self._draw_box(image = image, hand = hand, hand_landmarks = hand_landmarks)

        return hands


    def _draw_box(self, image: np.ndarray, hand: dict, hand_landmarks) -> None:

        bbox  = hand["bounding_box"]
        line1 = bbox[0] - 20, bbox[1] - 20
        line2 = bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20

        cv2.rectangle(image, line1, line2, (255, 0, 255), 2)
        cv2.putText(image, hand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 0, 255), 2)


    @staticmethod
    @jit(nopython = True, fastmath = True)
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


def main():
    
    cap = cv2.VideoCapture(0)

    detector = HandDetector(max_hands = 2, detection_con = 0.8)
    
    while True:
        # Get image frame
        success, img = cap.read()
        # Find the hand and its landmarks
        hands, img = detector.find_hands(img, draw_box = False)  # with draw
        # hands = detector.find_hands(img, draw = False)  # without draw

        if hands:
            # Hand 1
            hand1 = hands[0]
            lmList1 = hand1["landmarks"]  # List of 21 Landmark points
            bbox1 = hand1["bounding_box"]  # Bounding box info x,y,w,h
            centerPoint1 = hand1["center"]  # center of the hand cx,cy
            handType1 = hand1["type"]  # Handtype Left or Right


            if len(hands) == 2:
                # Hand 2
                hand2 = hands[1]
                lmList2 = hand2["landmarks"]  # List of 21 Landmark points
                bbox2 = hand2["bounding_box"]  # Bounding box info x,y,w,h
                centerPoint2 = hand2["center"]  # center of the hand cx,cy
                handType2 = hand2["type"]  # Hand Type "Left" or "Right"
                
        # Display
        cv2.imshow("Image", img)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
