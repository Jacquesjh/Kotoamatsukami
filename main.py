
import cv2

from kotoamatsukami.src.domain.hand_detector import HandDetector
from kotoamatsukami.src.domain.video_get import VideoGet
from kotoamatsukami.src.domain.video_show import VideoShow


def main():
    
    getter = VideoGet().start()
    shower = VideoShow(frame = cv2.flip(getter.frame, 1)).start()

    detector = HandDetector(max_hands = 2, detection_con = 0.8)
    
    while True:
        if getter.stopped or shower.stopped:
            getter.stop()
            shower.stop()
            break

        else:
            image = getter.frame
            image = cv2.flip(image, 1)
            hands, image = detector.find_hands(image, flip_view = False)
            shower.frame = image

    getter.release()
    cv2.destroyAllWindows()

import numpy as np
import cv2
from PIL import ImageGrab as ig
import time

def main():
    last_time = time.time()
    while(True):
        screen = ig.grab(bbox=(50,50,800,640))
        print('Loop took {} seconds',format(time.time()-last_time))
        cv2.imshow("test", np.array(screen))
        last_time = time.time()
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()



# commands = {
#     "RIGHT-two-fingers-togheter": click,
#     "RIGHT-"
# }