import glob
import os
from typing import Tuple

import numpy as np
from PIL import Image


def compute_mean_and_std(dir_name: str) -> Tuple[float, float]:
    """Compute the mean and the standard deviation of all images present within the directory.

    Note: convert the image in grayscale and then in [0,1] before computing mean
    and standard deviation

    Mean = (1/n) * \sum_{i=1}^n x_i
    Variance = (1 / (n-1)) * \sum_{i=1}^n (x_i - mean)^2
    Standard Deviation = sqrt( Variance )

    Args:
        dir_name: the path of the root dir

    Returns:
        mean: mean value of the gray color channel for the dataset
        std: standard deviation of the gray color channel for the dataset
    """
    mean = None
    std = None

    ############################################################################
    # Student code begin
    ############################################################################

    path = dir_name + "**/*.jpg"
    files = glob.glob(path, recursive=True)
    numSamples = len(files)

    samples = []

    for i in range(0, numSamples):
        img = Image.open(files[i]).convert("L")
        img = np.asarray(img) / 255
        img = np.reshape(img, [-1, 1]).flatten()
        samples.append(img)
    
    samples = np.hstack(samples)
    
    mean = np.mean(samples)
    std = np.std(samples)
    ############################################################################
    # Student code end
    ############################################################################
    return mean, std
