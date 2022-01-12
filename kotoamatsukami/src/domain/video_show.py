
from threading import Thread

import cv2


class VideoShow:


    def __init__(self, frame = None) -> None:
        
        self.frame   = frame
        self.stopped = False

    
    def start(self) -> None:

        Thread(target = self.show, args = ()).start()
        return self


    def show(self) -> None:

        while not self.stopped:
            cv2.imshow("Video", self.frame)

            if cv2.waitKey(1) == ord("q"):
                self.stop()
    

    def stop(self) -> None:
        
        self.stopped = True



def main():
    
    getter = VideoGet().start()
    shower = VideoShow(frame = getter.frame).start()

    while True:
        if shower.stopped or getter.stopped:
            getter.stop()
            shower.stop()
            break

        shower.frame = getter.frame


if __name__ == "__main__":
    main()