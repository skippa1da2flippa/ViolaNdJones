from src.image_processing.base.base_transformer import BaseTransformer


class ImagesManager(BaseTransformer):

    def __init__(self, windowSize: int):
        super().__init__(windowSize)

    def _squarifyOne(self, intputPath: str, outputPath: str):
        self._mapOne(
            intputPath, outputPath,
            actions=["squarify"]
        )

    def _squarifyAll(self, inputPath: str, outputPath: str):
        self._mapAll(
            inputPath, outputPath,
            actions=["squarify"]
        )

    def resizeOne(self, intputPath: str, outputPath: str):
        self._mapOne(
            intputPath, outputPath,
            actions=["resize"]
        )

    def resizeAll(self, inputPath: str, outputPath: str):
        self._mapAll(
            inputPath, outputPath,
            actions=["resize"]
        )

    def normalizeOne(self, intputPath: str, outputPath: str):
        self._mapOne(
            intputPath, outputPath,
            actions=["normalize"]
        )

    def normalizeAll(self, intputPath: str, outputPath: str):
        self._mapAll(
            intputPath, outputPath,
            actions=["normalize"]
        )

    def toGreyScaleOne(self, intputPath: str, outputPath: str):
        self._mapOne(
            intputPath, outputPath,
            actions=["to_grey"]
        )

    def toGreyScaleAll(self, intputPath: str, outputPath: str):
        self._mapAll(
            intputPath, outputPath,
            actions=["to_grey"]
        )

    def mirrorImageOne(self, intputPath: str, outputPath: str):
        self._mapOne(
            intputPath, outputPath,
            actions=["mirror"]
        )

    def mirrorImageAll(self, intputPath: str, outputPath: str):
        self._mapAll(
            intputPath, outputPath,
            actions=["mirror"]
        )

    def preprocessOne(self, intputPath: str, outputPath: str):
        self._mapOne(
            intputPath, outputPath,
            actions=["squarify", "resize", "to_grey", "normalize"]
        )

    def preprocessAll(self, intputPath: str, outputPath: str):
        self._mapAll(
            intputPath, outputPath,
            actions=["squarify", "resize", "to_grey", "normalize"]
        )
