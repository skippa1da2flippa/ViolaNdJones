import os
from time import time
import joblib
import torch.nn as nn
import numpy as np
import torch
from numpy import ones, array, column_stack, concatenate
from tensorflow import cast
from torch import tensor
from torch.nn import Linear
from torchvision.transforms import ToTensor
from torchvision import transforms
from src.haar_features.features_extractor import FeatureExtractor
from src.image_processing.preprocessing import ImagesManager
from src.learners.base_learner import BaseLearnerSoftMax
from src.learners.weak_learner import WeakLearner
from src.training.ada_boost import AdaBoost
from src.utility.model_utility.model_handler import ModelHandler
from src.utility.parallelization_handler import Parallelize
from src.utility.training_handler import TrainingHandler
import tensorflow as tf
from torchvision import datasets

"""
    image = cv2.imread("dataset/full_images/0--Parade/0_Parade_Parade_0_4.jpg")
    windowHandler = SlidingWindow(24, stepSize=30)
    image = array(image)
    for x, y, img, scaleFactor in windowHandler.getAllWindows(image):
        windowHandler.drawRectangle(image, (x, y), scaleFactor)
        cv2.waitKey(1)
        time.sleep(0.0000000000001)
"""

"""
    ModelHandler.storeModel(Fraction(0.45), 'dataset/' + "test")
    print(ModelHandler.getModel('dataset/' + "test"))
"""

"""
    manager = ImagesManager(24)
    manager.resizeOne("dataset/training/frontal_faces/faces/169508.jpg", "dataset/Capture2.PNG")
"""

"""
     for imageName in os.listdir("dataset/training/non_faces"):
        if imageName.endswith("pgm"):
            pathToImg: str = os.path.join("dataset/training/non_faces", imageName)
            output: str = os.path.join("dataset/training/non_faces", "_Nor_" + imageName)
            manager.normalizeOne(pathToImg, output)
            os.remove(pathToImg)
"""

"""
    print("Number of faces: ", len(os.listdir("dataset/training/frontal_faces")))
    print("Number of non faces: ", len(os.listdir("dataset/training/non_faces")))
"""

"""
    th = TrainingHandler(
        "dataset/training", ["non_faces"], 12, 24,
        r'dataset/training/extracted_dataset\extracted_dataset.pkl'
    )
    dataset, labelSet = np.random.rand(1000), concatenate((np.ones(500), np.zeros(500)))
    learner = BaseLearnerSoftMax()

    learner.fit(dataset, labelSet, ones(labelSet.size)/1.75, batchSize=30, verbose=2)
"""

"""
    dataset, labelSet = np.random.rand(1000), concatenate((np.ones(500), np.zeros(500)))
    weakLearner = WeakLearner(dataset, labelSet, ones(labelSet.size)/1.75, "type-2-x", 0, verbose=2)
    print(weakLearner.getErrorRate())
"""

"""
     th = TrainingHandler(
        "dataset/training", ["non_faces"], 12, 24,
        r'dataset/training/extracted_dataset\extracted_dataset.pkl'
    )
    dataset, labelSet, index = th.getDataset()

    adaBoost = AdaBoost(1, dataset, labelSet, index, 24)
    adaBoost.start(2)
"""

"""
    th = TrainingHandler(
        "dataset/training", ["frontal_faces", "non_faces"], 12, 24,
        r'dataset/training/extracted_dataset\extracted_dataset.pkl'
    )

    data, labels, index = th.getDataset()

    learner = BaseLearnerSoftMax()

    start = time()

    learner.fit(data.iloc[:, 33].values, labels, ones(labels.size) / 0.5, epochs=1, verbose=2)

    end = time()

    print((end - start))
"""

"""
    th = TrainingHandler(
        "dataset/training", ["non_faces"], 2, 24,
        r'dataset/training/extracted_dataset\extracted_dataset.pkl'
    )

    th.start(pathToModelDir="models", parallelize=True, adaBoost=True, verbose=2)
"""

"""
    torch1 = tensor([[2.4, 5.4], [45.8, 66.4]])
    torch2 = torch.linalg.norm(torch1, dim=1)
    print(torch2)
    print(torch.linalg.norm(tensor([2.4, 5.4])))
    print(torch.linalg.norm(tensor([45.8, 66.4])))
"""

if __name__ == '__main__':
    weak: WeakLearner = ModelHandler.getModel(os.path.join("models", "epoch_zero.pkl"))
    weak.getErrorRate()

    th = TrainingHandler(
        "dataset/training", ["non_faces"], 2, 24,
        r'dataset/training/extracted_dataset\extracted_dataset.pkl'
    )

    AdaBoost.updateWeights(weak.getWeights()[:5000], weak.getBeta(), weak.getWeightsMap())

    th.start(
        pathToModelDir="models", parallelize=True, adaBoost=True,
        oldWeights=weak.getWeights()[:5000],
        verbose=2
    )
