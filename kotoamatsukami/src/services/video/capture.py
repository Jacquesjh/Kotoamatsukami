import cv2


class VideoCapture:
    def __init__(self, source: int = 0) -> None:
        self.stream = cv2.VideoCapture(source)
        self.stopped = False
        self.grabbed, self.frame = self.stream.read()

    def update(self) -> None:
        while not self.stopped:
            if not self.grabbed:
                self.stop()

            else:
                self.grabbed, self.frame = self.stream.read()

    def stop(self) -> None:
        self.stopped = True

    def release(self) -> None:
        self.stream.release()
