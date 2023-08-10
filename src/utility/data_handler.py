from typing import TypeVar
import joblib
from pandas import DataFrame, concat

T = TypeVar('T')


class DataHandler:
    @staticmethod
    def storeData(data: T, path: str):
        joblib.dump(data, path)

    @staticmethod
    def getData(path: str) -> T:
        return joblib.load(path)


class DataLink:
    @staticmethod
    def appendDataFrame(data: DataFrame, path: str):
        try:
            oldFeats: DataFrame = DataHandler.getData(path)
            newFeats: DataFrame = concat([oldFeats, data], axis=0)

        except EOFError or Exception:
            newFeats: DataFrame = data

        DataHandler.storeData(newFeats, path)

