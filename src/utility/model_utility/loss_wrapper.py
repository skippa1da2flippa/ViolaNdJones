import torch.nn as nn
from torch import Tensor
from torch import linalg as euclidean


class WeightedEuclideanDistance(nn.Module):
    def __init__(self):
        super().__init__()
        self._errorMap: list[dict[str, Tensor]] = []

    def forward(self, yTrue: Tensor, yPred: Tensor, weights: Tensor,
                ids: Tensor, save: bool = False) -> Tensor:

        distanceTensor: Tensor = yTrue - yPred
        weightedDistance: Tensor = weights * euclidean.norm(distanceTensor)
        if save:
            self._errorMap.append({
                "yPred": yPred,
                "ids": ids
            })

        return weightedDistance

    def getErrorMap(self) -> list[dict[str, Tensor]]:
        return self._errorMap
