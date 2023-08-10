from typing import TypeVar
import joblib

T = TypeVar('T')


class ModelHandler:

    @staticmethod
    def storeModel(model: T, path: str):
        joblib.dump(model, path)

    @staticmethod
    def getModel(path: str) -> T:
        return joblib.load(path)


