import os.path
from multiprocessing import Process
from os.path import join
from typing import Any, Union, Callable, Generator
from numpy import ndarray, concatenate, array, append
from pandas import DataFrame
from src.haar_features.features_extractor import FeatureExtractor
from src.learners.strong_learner import StrongLearner
from src.learners.weak_learner import WeakLearner
from src.training.ada_boost import AdaBoost
from src.utility.data_handler import DataHandler
from src.utility.fetch_data import DataManager
from src.utility.model_utility.model_handler import ModelHandler
from src.utility.parallelization_handler import Parallelize


def startAction(nEras: int, dataset: DataFrame, labelSet: ndarray[int], index: dict[str, int],
                imgSize: int, path: str, verbose: int = 0, weakLearnerEpochs: int = 10):
    adaBoostManager: AdaBoost = AdaBoost(
        nEras, dataset, labelSet, index, imgSize, weakLearnerEpochs
    )

    start: Callable[[ndarray[float]], StrongLearner] = adaBoostManager.startGenerator(verbose)

    newModel: WeakLearner = start(adaBoostManager.getWeights()).getWeakLearners()[0]
    ModelHandler.storeModel(newModel, path)


def endAction(candidatesPaths: list[str], outputPath: str):
    model: WeakLearner = ModelHandler.getModel(candidatesPaths[0])
    for idx in range(1, len(candidatesPaths)):
        tempModel: WeakLearner = ModelHandler.getModel(candidatesPaths[idx])
        if tempModel.getErrorRate() < model.getErrorRate():
            model = tempModel

    ModelHandler.storeModel(model, outputPath)


def splitDataframe(data: DataFrame, split: int) -> Generator[ndarray[DataFrame], None, None]:
    for idx in range(0, data.shape[1], split):
        if idx + split >= data.shape[1]:
            indexLst: list[int] = [idj for idj in range(idx, data.shape[1])]
        else:
            indexLst: list[int] = [idj for idj in range(idx, idx + split)]

        yield data[indexLst].copy()


def prepareRightSplit(data: DataFrame) -> Generator[ndarray[DataFrame], None, None]:
    nProcesses: int = Parallelize.getMaxProcessesNumber() // 2
    nSplits: int = data.shape[1] // nProcesses
    return splitDataframe(data, nSplits)


class TrainingHandler:
    def __init__(
            self, inputPath: str, additionalSection: list[str],
            nEras: int, windowSize: int, pathToDataset: str, weakLearnerEpochs: int = 1
    ):
        self._dataset: ndarray[ndarray] = array([])
        self._labels: ndarray[int] = array([])
        self._nEras: int = nEras
        self._windowSize: int = windowSize
        self._pathToDataset: str = pathToDataset
        self._weakLearnerEpochs: int = weakLearnerEpochs
        try:
            self._extractedFeats, self._labels, self._index = DataHandler.getData(
                self._pathToDataset
            )
        except EOFError or Exception:
            self._initializeDataset(inputPath, additionalSection)
            self._extractedFeats, self._index = self._setupDataset()

    def _initializeDataset(self, inputPath: str, additionalSection: list[str]):
        for section in additionalSection:
            tempDataset, tempLabelSet = DataManager.fetchData(
                inputPath + f'/{section}',
                False if section == "non_faces" else True
            )

            self._dataset = tempDataset if not self._dataset.size else concatenate((self._dataset, tempDataset))
            self._labels = concatenate((self._labels, tempLabelSet))

    def _setupDataset(self) -> tuple[DataFrame, dict[str, int]]:
        shuffledDataset: ndarray[ndarray] = array([])
        shuffledLabelSet: ndarray[int] = array([], dtype=int)

        for img, label in DataManager.shuffleDataset(self._dataset, self._labels):
            shuffledDataset = array([img]) if not shuffledDataset.size else append(shuffledDataset, [img], axis=0)
            shuffledLabelSet = append(shuffledLabelSet, int(label))

        self._dataset = shuffledDataset
        self._labels = shuffledLabelSet

        featExtractor = FeatureExtractor(self._windowSize)
        extractedFeats, index = featExtractor.extractAll(self._dataset)
        DataHandler.storeData((extractedFeats, self._labels, index), self._pathToDataset)
        return extractedFeats, index

    def start(self, pathToModelDir: str, parallelize: bool = False, adaBoost: bool = True, verbose: int = 0) \
            -> Union[StrongLearner, Any]:

        if adaBoost:
            strongLearner: StrongLearner
            if parallelize:
                strongLearner = self._multiProcessAdaBoost(pathToModelDir, verbose)

            else:
                adaboostManager: AdaBoost = AdaBoost(
                    self._nEras, self._extractedFeats, self._labels, self._index, self._windowSize,
                    self._weakLearnerEpochs
                )

                strongLearner = adaboostManager.start(verbose)

            ModelHandler.storeModel(strongLearner, join(pathToModelDir, "finalClassifier.pkl"))

            return strongLearner

        else:
            pass  # TODO in case we would like to perform attentional cascade

    def _multiProcessAdaBoost(self, pathToDirectory: str, verbose: int = 0) -> StrongLearner:
        weights: ndarray[float] = AdaBoost.initializeWeights(self._labels)
        weakLearners: ndarray[WeakLearner] = array([])

        for era in range(0, self._nEras):
            args: ndarray[tuple[Any, ...]] = array([(
                    1, dataFrame, self._labels, self._index,
                    self._windowSize, os.path.join(pathToDirectory, f"wl_{idx}.pkl"),
                    verbose, self._weakLearnerEpochs
                ) for idx, dataFrame in enumerate(prepareRightSplit(self._extractedFeats.iloc[:5000, :]))], dtype=object)

            processes: ndarray[Process] = Parallelize.parallelizeOneWithMany(
                startAction, args
            )

            Parallelize.waitProcesses(processes, endAction, (
                [os.path.join(pathToDirectory, f"wl_{idx}.pkl") for idx in range(Parallelize.getMaxProcessesNumber())],
                os.path.join(pathToDirectory, f"epoch:{era}.pkl")
            ))

            tempWeakLearner: WeakLearner = ModelHandler.getModel(os.path.join(pathToDirectory, f"epoch:{era}" + ".pkl"))
            weakLearners = append(weakLearners, [tempWeakLearner])

            # update weights
            AdaBoost.updateWeights(weights, tempWeakLearner.getBeta(), tempWeakLearner.getWeightsMap())

        return StrongLearner(weakLearners, self._windowSize)

    def getDataset(self) -> tuple[DataFrame, ndarray[int], dict[str, int]]:
        return self._extractedFeats, self._labels, self._index
