
import cv2

from kotoamatsukami.src.domain.hand_detector import HandDetector
from kotoamatsukami.src.domain.video_get import VideoGet
from kotoamatsukami.src.domain.video_show import VideoShow



def main():
    
    getter = VideoGet().start()
    shower = VideoShow(frame = getter.frame).start()

    detector = HandDetector(max_hands = 2, detection_con = 0.8)
    
    while True:
        if getter.stopped or shower.stopped:
            getter.stop()
            shower.stop()
            break

        else:
            image = getter.frame
            hands, image = detector.find_hands(image)
            shower.frame = image

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
