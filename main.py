
import  pickle
from typing import Dict

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from kotoamatsukami import HandDetector



def load_model() -> RandomForestClassifier:
    with open("models/decision_tree", "rb") as file:
        model = pickle.load(file)

    return model


def get_gesture_labes() -> Dict[int, str]:
    labels = {
        0: "click",
        1: "closed",
        2: "down",
        3: "mouse_tracking",
        4: "negative_closed",
        5: "negative_mouse_tracking",
        6: "negative_side",
        7: "negative_up",
        8: "side",
        9: "up"
    }

    return labels


def normalize(z: float, max: float, min: float) -> float:
    norm = (z - min)/(max - min)
    round_norm = round(norm, 3)
    
    return round_norm


def normalize_landmarks(landmarks: list) -> list:
    norm_landmarks = list()

    for landmark in landmarks:
        xs = [landmark[i][0] for i in range(len(landmark))]
        ys = [landmark[i][1] for i in range(len(landmark))]

        xmax = max(xs)
        ymax = max(ys)
        xmin = min(xs)
        ymin = min(ys)

        norm_landmark = list()

        for x, y in zip(xs, ys):
            norm = [normalize(z=x, max=xmax, min=xmin), normalize(z=y, max=ymax, min=ymin)]

            norm_landmark.append(norm)

        norm_landmarks.append(norm_landmark)

    return norm_landmarks


def main() -> None:
    print("Starting!!!")

    stream = cv2.VideoCapture(0)
    labels = get_gesture_labes()

    model = load_model()
    hand_detector = HandDetector(max_hands=2, detection_con=0.9, model_complexity=1, normalize=False)

    while True:
        _grabbed, raw_frame = stream.read()
        hands, image = hand_detector.find_hands(image=raw_frame, draw_box=True, draw_marks=True)

        cv2.imshow("Video", image)

        if hands != []:
            hand = hands[0]
            landmarks = hand["landmarks"]
            norm_landmakrs = normalize_landmarks(landmarks=landmarks)[0]

            prediction = model.predict_proba(np.array(norm_landmakrs).reshape(1, 42))
            best_prediction = prediction.argmax()
            print(f"Caught the gesture: {labels.get(best_prediction)}!")
        
    stream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()