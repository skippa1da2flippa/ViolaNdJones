import torch.nn as nn
from torch import Tensor, inner
from torch import linalg as euclidean


class WeightedEuclideanDistance(nn.Module):
    def __init__(self):
        super().__init__()
        self._errorMap: list[dict[str, Tensor]] = []

    def forward(self, yTrue: Tensor, yPred: Tensor, weights: Tensor,
                ids: Tensor, save: bool = False) -> Tensor:

        distancesTensor: Tensor = yTrue - yPred
        euclideanDistances: Tensor = euclidean.norm(distancesTensor, dim=1)
        weightedDistances: Tensor = inner(weights, euclideanDistances)

        if save:
            self._errorMap.append({
                "yPred": yPred,
                "ids": ids
            })

        return weightedDistances

    def getErrorMap(self) -> list[dict[str, Tensor]]:
        return self._errorMap
