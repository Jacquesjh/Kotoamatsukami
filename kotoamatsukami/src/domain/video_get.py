
from threading import Thread

import cv2


class VideoGet:

    def __init__(self, source: int = 0) -> None:

        self.stream  = cv2.VideoCapture(source)
        self.stopped = False
        self.grabbed, self.frame = self.stream.read()


    def start(self):
        
        Thread(target = self.update, args = ()).start()
        return self


    def update(self) -> None:

        while not self.stopped:
            if not self.grabbed:
                self.stop()

            else:
                self.grabbed, self.frame = self.stream.read()


    def stop(self) -> None:

        self.stopped = True

