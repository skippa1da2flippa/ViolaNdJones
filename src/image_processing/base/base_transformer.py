import os
from typing import Callable
from PIL import Image
from numpy import array, uint8

possibleActions: tuple[str, str, str, str, str] = (
    "resize", "normalize", "to_grey", "squarify", "mirror"
)


class BaseTransformer:
    def __init__(self, windowSize: int):
        self._windowSize: int = windowSize
        self._transformers: dict[str, Callable] = {}
        self._initializeTransformers()

    def _initializeTransformers(self):
        for key in possibleActions:
            match key:
                case "resize":
                    def resize(image):
                        image.thumbnail((self._windowSize, self._windowSize))
                        return image

                    self._transformers[key] = resize

                case "normalize":
                    def normalize(image):
                        image_array = array(image)
                        normalized_image_array = image_array / 255.0
                        return Image.fromarray((normalized_image_array * 255).astype(uint8))

                    self._transformers[key] = normalize

                case "to_grey":
                    self._transformers[key] = lambda img: img.convert("L")

                case "squarify":

                    def squarify(image):
                        width, height = image.size

                        if width != height:
                            # Choose the size of the square image (largest dimension as side length)
                            sideLength = max(width, height)

                            # Create a new square image with the specified background color
                            squareImg = Image.new('RGB', (sideLength, sideLength), (255, 255, 255))

                            # Calculate the position to center the original image within the square image
                            xOffset = (sideLength - width) // 2
                            yOffset = (sideLength - height) // 2

                            # Paste the original image onto the square image
                            squareImg.paste(image, (xOffset, yOffset))

                            return squareImg

                        return image

                    self._transformers[key] = squarify

                case "mirror":
                    self._transformers[key] = lambda img: img.transpose(Image.FLIP_LEFT_RIGHT)

    def _mapOne(self, inputPath: str, outputPath: str, actions: list[str]):
        with Image.open(inputPath) as image:
            mappedImg = image
            for action in actions:
                mappedImg = self._transformers[action](mappedImg)

            mappedImg.save(outputPath)

    def _mapAll(self, inputPath: str, outputPath: str, actions: list[str]):
        for imageName in os.listdir(inputPath):
            pathToImg: str = os.path.join(inputPath, imageName)
            pathToMappedImg: str = os.path.join(outputPath, imageName)
            self._mapOne(pathToImg, pathToMappedImg, actions)
