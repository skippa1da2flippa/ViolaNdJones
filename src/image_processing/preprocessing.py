from src.image_processing.base.base_transformer import BaseTransformer


class ImagesManager(BaseTransformer):

    def __init__(self, windowSize: int):
        super().__init__(windowSize)

    def _squarifyOne(self, inputPath: str, outputPath: str):
        self._mapOne(
            inputPath, outputPath,
            actions=["squarify"]
        )

    def _squarifyAll(self, inputPath: str, outputPath: str):
        self._mapAll(
            inputPath, outputPath,
            actions=["squarify"]
        )

    def resizeOne(self, inputPath: str, outputPath: str):
        self._mapOne(
            inputPath, outputPath,
            actions=["resize"]
        )

    def resizeAll(self, inputPath: str, outputPath: str):
        self._mapAll(
            inputPath, outputPath,
            actions=["resize"]
        )

    def normalizeOne(self, inputPath: str, outputPath: str):
        self._mapOne(
            inputPath, outputPath,
            actions=["normalize"]
        )

    def normalizeAll(self, inputPath: str, outputPath: str):
        self._mapAll(
            inputPath, outputPath,
            actions=["normalize"]
        )

    def toGreyScaleOne(self, inputPath: str, outputPath: str):
        self._mapOne(
            inputPath, outputPath,
            actions=["to_grey"]
        )

    def toGreyScaleAll(self, inputPath: str, outputPath: str):
        self._mapAll(
            inputPath, outputPath,
            actions=["to_grey"]
        )

    def mirrorImageOne(self, inputPath: str, outputPath: str):
        self._mapOne(
            inputPath, outputPath,
            actions=["mirror"]
        )

    def mirrorImageAll(self, inputPath: str, outputPath: str):
        self._mapAll(
            inputPath, outputPath,
            actions=["mirror"]
        )

    def preprocessOne(self, inputPath: str, outputPath: str):
        self._mapOne(
            inputPath, outputPath,
            actions=["squarify", "resize", "to_grey", "normalize"]
        )

    def preprocessAll(self, inputPath: str, outputPath: str):
        self._mapAll(
            inputPath, outputPath,
            actions=["squarify", "resize", "to_grey", "normalize"]
        )
