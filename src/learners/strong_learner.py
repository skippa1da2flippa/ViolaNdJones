import cv2
from numpy import ndarray, full, sum, array, int32, append
from math import log2
from src.haar_features.features_extractor import FeatureExtractor
from src.image_processing.sliding_window import SlidingWindow
from src.learners.weak_learner import WeakLearner


class StrongLearner(object):
    def __init__(self, weakLearners: ndarray[WeakLearner], windowSize: int):
        self._featureExtractor: FeatureExtractor = FeatureExtractor(windowSize)
        self._windowManager: SlidingWindow = SlidingWindow(windowSize)
        self._weakLearners: ndarray[WeakLearner] = weakLearners
        self._alphas: ndarray[float] = full(len(weakLearners), .0, dtype=float)
        self._initializeAlphas()

    def _predictWindow(self, img: ndarray) -> bool:
        weakLearnerPredictions: ndarray[int] = array([], dtype=int32)
        for weakLearner in self._weakLearners:
            specificFeat: float = self._featureExtractor.extractOne(
                img, weakLearner.getFeatType(), weakLearner.getFeatIndex()
            )
            weakLearnerPredictions = append(weakLearnerPredictions, weakLearner.predictOne(specificFeat))

        return self._alphas @ weakLearnerPredictions >= 0.5 * sum(self._alphas)

    def predictImage(self, img: ndarray):
        # TODO you are assuming someone already preprocess the image for you otherwise you need to do it here
        for x, y, window, scaleFactor in self._windowManager.getAllWindows(img):
            if self._predictWindow(window):
                self._windowManager.drawRectangle(img, (x, y), scaleFactor)
                cv2.imshow("Window2", img)
                cv2.waitKey(0)

    def predictImages(self, images: ndarray):
        # TODO same thing here
        for idx in range(0, images.size):
            self.predictImage(images[idx])

    def _initializeAlphas(self):
        for idx, weakLearner in enumerate(self._weakLearners):
            self._alphas[idx] = log2(1 / weakLearner.getBeta())

    def getWeakLearners(self) -> ndarray[WeakLearner]:
        return self._weakLearners
