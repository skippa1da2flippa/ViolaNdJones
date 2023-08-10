import torch.nn as nn
import torch as th
from numpy import ndarray
from torch.nn import Linear, Softmax
from torch import Tensor, tensor, argmax, int32
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset
from src.utility.model_utility.dataset_wrapper import MultiModalDataset
from src.utility.model_utility.loss_wrapper import WeightedEuclideanDistance


def prepareDataset(dataset: Dataset, batchSize: int = 1):
    return DataLoader(dataset, batchSize, shuffle=True)


class BaseLearnerSoftMax(nn.Module):

    def __init__(self):
        super().__init__()
        self._weightsApplier: Linear = Linear(in_features=1, out_features=2, bias=False)
        self._softMax: Softmax = Softmax(dim=0)

    def forward(self, batch: Tensor) -> Tensor:
        if not batch.dim():
            batch = batch.unsqueeze(0)  # Convert scalar to 1D tensor with one element

        weightedInput: Tensor = self._weightsApplier(batch)
        return self._softMax(weightedInput)

    def fit(self, xTrain: ndarray[float], yTrain: ndarray[int],
            weights: ndarray[float], batchSize: int = 32, epochs: int = 5,
            verbose: int = 0
            ) -> tuple[list[dict[str, Tensor]], float]:

        optimizer = SGD(self.parameters(), lr=0.05, momentum=0.5)
        weightedLoss: WeightedEuclideanDistance = WeightedEuclideanDistance()
        dataset: MultiModalDataset = MultiModalDataset(
            xTrain, yTrain, weights
        )
        batches: DataLoader = prepareDataset(dataset, batchSize)
        predictionsHolder: Tensor = tensor([])
        trueLoss: float = 0.0

        for epoch in range(epochs):
            trueLoss = 0.0
            for batch in batches:

                if verbose > 2:
                    print(f"\033[37mWeights: {self._weightsApplier.weight}\033[0m")

                (xBatch, ids), (yBatch, weightsBatch) = batch

                for sample in xBatch:
                    prediction: Tensor = self(sample)  # forward pass
                    predictionsHolder = tensorAppend(predictionsHolder, prediction, retainDim=True)

                loss: Tensor = weightedLoss(
                    yBatch, predictionsHolder, weightsBatch, ids,
                    save=True if epoch == epochs - 1 else False
                )

                trueLoss += loss.item()

                optimizer.zero_grad()  # initialize gradient to zero
                loss.backward()  # compute gradient
                optimizer.step()  # backpropagation

                predictionsHolder = tensor([])

            if verbose > 1:
                print(f"\033[32mEpoch:{epoch} loss is {trueLoss}\033[0m")

            if verbose >= 1:
                print(f"\033[34mloss: {trueLoss}\033[0m")

        return weightedLoss.getErrorMap(), trueLoss

    def predict(self, sample: Tensor) -> Tensor:
        with th.no_grad():
            self.eval()  # Set the model to evaluation mode (if applicable)
            return argmax(self(sample))

    def predictAll(self, samples: Tensor) -> Tensor:
        res: Tensor = tensor([], dtype=int32)
        for sample in samples:
            prediction: Tensor = self.predict(sample)
            res = tensorAppend(res, prediction)

        return res


def tensorAppend(lst: Tensor, elem: Tensor, retainDim: bool = False) -> Tensor:
    if not lst.numel():
        return elem if not retainDim else elem.unsqueeze(dim=0)
    else:
        return th.cat((lst, elem)) if not retainDim else th.cat((lst, elem.unsqueeze(dim=0)))


# TODO redo the model with just one neuron and use binary cross entropy as loss
