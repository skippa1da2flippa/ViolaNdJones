import numpy as np
from numpy import ndarray, full
from torch import tensor, Tensor, float32
from src.learners.base_learner import BaseLearnerSoftMax


class WeakLearner(object):
    def __init__(
            self, dataset: ndarray[float], labelSet: ndarray[int], weights: ndarray[float],
            ftType: str, ftIndex: int, epochs: int = 10, verbose: int = 0
    ):
        self._baseLearner: BaseLearnerSoftMax = BaseLearnerSoftMax()
        self._threshold: float
        self._errorRate: float
        self._beta: float
        self._accuracy: float
        self._dataset: ndarray[float] = dataset
        self._weights: ndarray[float] = weights
        self._labels: ndarray[int] = labelSet
        self._ftType: str = ftType
        self._ftIndex: int = ftIndex
        self._weightsMap: ndarray[bool] = full(self._dataset.size, False, dtype=np.bool)
        self._fit(epochs=epochs, verbose=verbose)

    def _fit(self, epochs: int = 5, verbose: int = 0):
        trainingResult = self._baseLearner.fit(
            self._dataset, self._labels, self._weights,
            verbose=verbose, epochs=epochs
        )
        self._errorRate = trainingResult[1]
        self._beta = self._errorRate / (1 - self._errorRate)
        self._updateWeightsMap(trainingResult[0])

    def _updateWeightsMap(self, data: list[dict[str, Tensor]]):
        for elem in data:
            yPred: ndarray[tuple[float, float]] = elem["yPred"].detach().numpy()
            ids: ndarray[int] = elem["ids"].detach().numpy()

            for index, _id in enumerate(ids):
                actualTup: tuple[float, float] = tuple(yPred[index])
                label: int = actualTup.index(max(actualTup))
                self._weightsMap[_id] = label == self._labels[_id]

    def predictAll(self, samples: ndarray[float]) -> ndarray[int]:
        return self._baseLearner.predictAll(tensor(samples, dtype=float32)).numpy()

    def predictOne(self, sample: float) -> int:
        return self._baseLearner.predict(tensor(sample, dtype=float32)).item()

    def getErrorRate(self) -> float:
        return self._errorRate

    def getBeta(self) -> float:
        return self._beta

    def getWeights(self) -> ndarray[float]:
        return self._weights

    def getWeightsMap(self) -> ndarray[bool]:
        return self._weightsMap

    def getFeatType(self) -> str:
        return self._ftType

    def getFeatIndex(self) -> int:
        return self._ftIndex

