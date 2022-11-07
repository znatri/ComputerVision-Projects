#!/usr/bin/python3

import numpy as np


def compute_normalized_patch_descriptors(
    image_bw: np.ndarray, X: np.ndarray, Y: np.ndarray, feature_width: int
) -> np.ndarray:
    """Create local features using normalized patches.

    Normalize image intensities in a local window centered at keypoint to a
    feature vector with unit norm. This local feature is simple to code and
    works OK.

    Choose the top-left option of the 4 possible choices for center of a square
    window.

    Args:
        image_bw: array of shape (M,N) representing grayscale image
        X: array of shape (K,) representing x-coordinate of keypoints
        Y: array of shape (K,) representing y-coordinate of keypoints
        feature_width: size of the square window

    Returns:
        fvs: array of shape (K,D) representing feature descriptors
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    # Init FVS with shape
    fvs = np.zeros(shape=(len(X), feature_width**2))
    
    # Set decrement and increment for window
    decr = np.int(np.floor(feature_width / 2)) - 1
    incr = feature_width - decr

    # Loop and shift window while normalizing local intensities
    for idx in range(0, len(X)):
        i = X[idx]
        j = Y[idx]

        window = image_bw[j - decr : j + incr, i - decr : i + incr]
        if window.shape != (feature_width, feature_width):
            continue
        else:
            window = np.reshape(window, (1, feature_width**2))
            fvs[idx,:] = (window / np.linalg.norm(window)).squeeze()

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return fvs
