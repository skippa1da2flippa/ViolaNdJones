from torch.utils.data import Dataset
from torch import Tensor, tensor, float32, int32
from numpy import ndarray, array


class MultiModalDataset(Dataset):
    def __init__(self, data: ndarray[float], labels: ndarray[int], weights: ndarray[float], softMax: bool = True):
        self._xTrain: Tensor = tensor(data, dtype=float32, requires_grad=True)

        if softMax:
            self._yTrain: Tensor = tensor(
                array([
                    (0, 1) if elem else (1, 0) for elem in labels
                ]),
                dtype=float32, requires_grad=True
            )
        else:
            self._yTrain: Tensor = tensor(labels, dtype=float32, requires_grad=True)

        self._weights: Tensor = tensor(weights, dtype=float32, requires_grad=True)
        self._ids: Tensor = tensor(array([idx for idx in range(data.size)]), dtype=int32)

    def __len__(self) -> int:
        return self._xTrain.numel()

    def __getitem__(self, idx: int) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        return (self._xTrain[idx], self._ids[idx]), (self._yTrain[idx], self._weights[idx])


