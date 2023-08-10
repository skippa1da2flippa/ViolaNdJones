import os
from PIL import Image
from numpy import ndarray, array, ones, zeros
from numpy.random import shuffle
from src.haar_features.features_extractor import biDimNumpyAppend


class DataManager:

    @staticmethod
    def fetchData(inputPath: str, pos: bool = True) -> tuple[ndarray[ndarray], ndarray[int]]:
        images: ndarray[ndarray] = array([])
        imgNames: list[str] = os.listdir(inputPath)
        labels: ndarray[int] = ones(len(imgNames)) if pos else zeros(len(imgNames))
        for imageName in imgNames:
            pathToImg: str = os.path.join(inputPath, imageName)
            with Image.open(pathToImg) as image:
                images = biDimNumpyAppend(images, array(image))

        return images, labels

    @staticmethod
    def shuffleDataset(dataset: ndarray[ndarray], labels: ndarray[int]) -> ndarray[tuple[ndarray, int]]:
        fullData: ndarray[tuple[ndarray, int]] = array(list(zip(dataset, labels)), dtype=object)
        shuffle(fullData)
        return fullData

