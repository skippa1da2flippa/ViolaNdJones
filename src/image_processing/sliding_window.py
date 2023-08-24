from typing import Generator
import cv2
import imutils
from numpy import ndarray


class SlidingWindow:
    def __init__(self, windowSize: int, scale: float = 1.5, stepSize: int = 8):
        self._windowSize: int = windowSize
        self._minSize: int = windowSize
        self._scale: float = scale
        self._stepSize: int = stepSize

    def _generatePyramid(self, img: ndarray) -> Generator[tuple[int, ndarray], None, None]:
        pyramidHeight: int = 0
        yield pyramidHeight, img

        while img.shape[0] >= self._minSize and img.shape[1] >= self._minSize:
            pyramidHeight += 1
            width: int = int(img.shape[1] / self._scale)
            img = imutils.resize(img, width=width)

            yield pyramidHeight, img

    def _slidingWindow(self, img: ndarray) -> Generator[tuple[float, float, ndarray], None, None]:
        for x in range(0, img.shape[0], self._stepSize):
            for y in range(0, img.shape[1], self._stepSize):
                # yield the current window
                yield y, x, img[x:x + self._windowSize, y:y + self._windowSize]

    def getAllWindows(self, img: ndarray) -> Generator[tuple[float, float, ndarray, int], None, None]:
        for pyramidHeight, resized in self._generatePyramid(img):
            for (x, y, window) in self._slidingWindow(resized):
                if window.shape[0] == self._windowSize and window.shape[1] == self._windowSize:
                    yield x, y, window, pyramidHeight

    # TODO the rectangles get shown, but they are not drawn on the real photo
    # TODO test this scale factor (done because the rectangles is drawn on th real image not a rescale)
    def drawRectangle(self, img: ndarray, coord: tuple[float, float], scaleFactor: float):
        stepLength: float = scaleFactor * self._scale if scaleFactor else 1

        startCoord: tuple[float, float] = (
            int(coord[0] * stepLength),
            int(coord[1] * stepLength)
        )
        endCoord: tuple[float, float] = (
            startCoord[0] + int(self._windowSize * stepLength),
            startCoord[1] + int(self._windowSize * stepLength)
        )

        color: tuple[int, int, int] = (0, 255, 0)  # red
        cv2.rectangle(img, startCoord, endCoord, color=color, thickness=2)
