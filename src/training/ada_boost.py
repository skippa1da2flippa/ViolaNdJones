from typing import Callable
from numpy import ndarray, array, append, ones
from pandas import DataFrame

from src.haar_features.features_extractor import assignRightFeature, FeatureExtractor
from src.learners.strong_learner import StrongLearner
from src.learners.weak_learner import WeakLearner


def getTheBestWeakLearner(weakLearners: ndarray[WeakLearner]) -> WeakLearner:
    minErr: float = weakLearners[0].getErrorRate()
    bestIdx: int = 0
    for idx, weakLearner in enumerate(weakLearners):
        wlErrorRate: float = weakLearner.getErrorRate()
        if wlErrorRate < minErr:
            minErr = wlErrorRate
            bestIdx = idx

    return weakLearners[bestIdx]


"""
    Class wrapping all the functionalities needed to make a training algorithm based 
    on an ensemble approach
"""


class AdaBoost:
    def __init__(
            self, nEras: int, dataset: DataFrame, labelSet: ndarray[int],
            index: dict[str, int], imgSize: int, weakLearnerEpochs: int = 10
    ):
        self._extractedFeatures: DataFrame[float] = dataset
        self._featuresIndex: dict[str, int] = index
        self._weakLearners: ndarray[WeakLearner] = array([])
        self._nEras: int = nEras
        self._nSamples: int = self._extractedFeatures.shape[0]
        self._labelSet: ndarray[int] = labelSet
        self._weights: ndarray[float] = AdaBoost.initializeWeights(self._labelSet)
        self._imgSize: int = imgSize
        self._weakLearnerEpochs: int = weakLearnerEpochs

    @staticmethod
    def initializeWeights(labelSet: ndarray[int]) -> ndarray[float]:
        weights: ndarray[float] = ones(labelSet.size)
        posSamples: float = labelSet.sum()
        negSamples: float = labelSet.size - posSamples

        weights[labelSet == 0] = 1 / (2 * negSamples)
        weights[labelSet == 1] = 1 / (2 * posSamples)

        return weights

    @staticmethod
    def normalizeWeights(weights: ndarray[float]):
        weights /= weights.sum()

    @staticmethod
    def updateWeights(weights: ndarray[float], weakLearnerBeta: float, weakLearnerWeightsMap: ndarray[bool]):
        weights[weakLearnerWeightsMap] *= weakLearnerBeta

    def startGenerator(self, verbose: int = 0) -> Callable[[ndarray[float]], StrongLearner]:
        def detachedStart(weights: ndarray[float]) -> StrongLearner:
            for era in range(self._nEras):
                # normalize the weights
                AdaBoost.normalizeWeights(weights)

                # find the weak learner with the lowest loss
                bestWeakLearner: WeakLearner = self._getBestWeakLearner(verbose)
                AdaBoost.updateWeights(weights, bestWeakLearner.getBeta(), bestWeakLearner.getWeightsMap())

                # store the best weak learner at the t-th era
                self._weakLearners = append(self._weakLearners, [bestWeakLearner])

                if verbose > 1:
                    print(f"\033[31mTime left: {-1}\033[0m")

                return StrongLearner(self._weakLearners, self._imgSize)

        return detachedStart

    def start(self, verbose: int = 0) -> StrongLearner:
        start: Callable[[ndarray[float]], StrongLearner] = self.startGenerator(verbose)
        return start(self._weights)

    def _getBestWeakLearner(self, verbose: int = 0) -> WeakLearner:
        weakLearners: ndarray[WeakLearner] = array([])
        for idx in range(0, self._extractedFeatures.shape[1]):

            if verbose > 1:
                print(
                    f"\033[33mWeak learner number: {idx}. \n\033[0m"
                    f"\033[33mWeak learner left: {self._extractedFeatures.shape[1] - (idx + 1)}\033[0m"
                )

            rightFeat, rightIdx = assignRightFeature(self._featuresIndex, idx)
            weakLearner: WeakLearner = WeakLearner(
                dataset=self._extractedFeatures.iloc[:, idx].values,
                labelSet=self._labelSet, weights=self._weights,
                ftType=rightFeat, ftIndex=rightIdx,
                epochs=self._weakLearnerEpochs, verbose=verbose - 1
            )

            weakLearners = append(weakLearners, [weakLearner])

        return getTheBestWeakLearner(weakLearners)

    def getWeights(self) -> ndarray[float]:
        return self._weights
