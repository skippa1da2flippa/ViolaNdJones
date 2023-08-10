from numpy import ndarray, array, append, float32
from pandas import DataFrame
from skimage.feature import haar_like_feature
from skimage.transform import integral_image

from src.utility.data_handler import DataLink

haarFeaturesType: tuple[str, str, str, str, str] = (
    "type-2-x", "type-2-y", "type-3-x", "type-3-y", "type-4"
)


class FeatureExtractor(object):
    def __init__(self, imgSize: int):
        self._imgSize = imgSize
        self._previousImg: ndarray = array([])
        self._previousIntegralImage: ndarray = array([])
        self._previousBytesImage: bytes = bytes()
        self._availableType: list[str] = []
        self._oldFeatures: dict[str, ndarray] = {}
        self._initializeRepetitionCheck()

    def extractAll(self, dataset: ndarray) -> tuple[DataFrame, dict[str, int]]:
        haarFeatures: ndarray[ndarray[float]] = array([])
        featuresIndex: dict[str, int] = {}

        for idx, img in enumerate(dataset):
            tempImg = integral_image(img)
            tempHaarFeats: ndarray[float] = array([])
            for ftType in haarFeaturesType:
                haarFeat: ndarray = haar_like_feature(
                    tempImg, 0, 0, self._imgSize, self._imgSize, feature_type=ftType
                )
                if not idx:  # TODO modify this thing, ugly as hell
                    featuresIndex[ftType] = haarFeat.size

                tempHaarFeats = append(tempHaarFeats, haarFeat)

            haarFeatures = biDimNumpyAppend(haarFeatures, tempHaarFeats)

        extractedFeatures = DataFrame(
            data=haarFeatures,
            dtype=float
        )

        return extractedFeatures, featuresIndex

    def extractOneFromAll(self, images: ndarray, featureType: str, featureIdx: int) -> ndarray[float]:
        res: ndarray[float] = array([])
        for img in images:
            res = append(res, self.extractOne(img, featureType, featureIdx))

        return res

    def extractOne(self, image: ndarray, featureType: str, featureIdx: int) -> float:
        return self._handleRepetitions(
            image,
            featureType,
            featureIdx
        )

    def _handleRepetitions(self, img: ndarray, featureType: str, featureIdx: int) -> float:
        if not self._previousImg.size:  # no old image
            self._previousImg = img
            self._previousIntegralImage = integral_image(img)
            self._previousBytesImage = img.tobytes()

        elif self._previousBytesImage != img.tobytes():  # old image does not match the new image
            self._previousImg = img
            self._previousBytesImage = img.tobytes()
            self._previousIntegralImage = integral_image(img)
            self._initializeRepetitionCheck()

        if not self._oldFeatures[featureType].size:  # if the features are yet to be computed
            self._oldFeatures[featureType] = haar_like_feature(
                self._previousIntegralImage, 0, 0, self._imgSize, self._imgSize, feature_type=featureType
            )

        return self._oldFeatures[featureType][featureIdx]

    def _initializeRepetitionCheck(self):
        for key in haarFeaturesType:
            self._oldFeatures[key] = array([])


def biDimNumpyAppend(arr: ndarray[ndarray], elem: ndarray) -> ndarray:
    if not arr.size:
        return array([elem])
    else:
        return append(arr, [elem], axis=0)


def assignRightFeature(index: dict[str, int], idx: int) -> tuple[str, int]:
    acc: int = 0
    for key in index:
        acc += index[key]
        if idx <= acc - 1:
            return key, idx - (acc - index[key])

    return "", -1
