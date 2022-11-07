import math

import numpy as np
from vision.part2_fundamental_matrix import estimate_fundamental_matrix


def calculate_num_ransac_iterations(
    prob_success: float, sample_size: int, ind_prob_correct: float
) -> int:
    """
    Calculates the number of RANSAC iterations needed for a given guarantee of
    success.

    Args:
        prob_success: float representing the desired guarantee of success
        sample_size: int the number of samples included in each RANSAC
            iteration
        ind_prob_success: float representing the probability that each element
            in a sample is correct

    Returns:
        num_samples: int the number of RANSAC iterations needed

    """
    num_samples = None
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    p = prob_success
    s = sample_size
    e = ind_prob_correct

    num_samples = math.log(1-p)/math.log(1 - math.pow(e, s))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return int(num_samples)


def ransac_fundamental_matrix(
    matches_a: np.ndarray, matches_b: np.ndarray
) -> np.ndarray:
    """
    For this section, use RANSAC to find the best fundamental matrix by
    randomly sampling interest points. You would reuse
    estimate_fundamental_matrix() from part 2 of this assignment and
    calculate_num_ransac_iterations().

    If you are trying to produce an uncluttered visualization of epipolar
    lines, you may want to return no more than 30 points for either left or
    right images.

    Tips:
        0. You will need to determine your prob_success, sample_size, and
            ind_prob_success values. What is an acceptable rate of success? How
            many points do you want to sample? What is your estimate of the
            correspondence accuracy in your dataset?
        1. A potentially useful function is numpy.random.choice for creating
            your random samples.
        2. You will also need to choose an error threshold to separate your
            inliers from your outliers. We suggest a threshold of 0.1.

    Args:
        matches_a: A numpy array of shape (N, 2) representing the coordinates
            of possibly matching points from image A
        matches_b: A numpy array of shape (N, 2) representing the coordinates
            of possibly matching points from image B
    Each row is a correspondence (e.g. row 42 of matches_a is a point that
    corresponds to row 42 of matches_b)

    Returns:
        best_F: A numpy array of shape (3, 3) representing the best fundamental
            matrix estimation
        inliers_a: A numpy array of shape (M, 2) representing the subset of
            corresponding points from image A that are inliers with respect to
            best_F
        inliers_b: A numpy array of shape (M, 2) representing the subset of
            corresponding points from image B that are inliers with respect to
            best_F
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    p = 0.95
    s = 13
    e = 0.495

    n = matches_a.shape[0]
    
    num_iter = calculate_num_ransac_iterations(p, s, e)
    print(num_iter)

    threshold = 0.5
    max_inlier = 0

    best_F = np.zeros((3, 3))

    matches_a_reshaped = np.hstack([matches_a, np.ones([n, 1])])
    matches_b_reshaped = np.hstack([matches_b, np.ones([n, 1])])

    A = np.multiply(np.tile(matches_a_reshaped, 3), matches_b_reshaped.repeat(3, axis=1))

    for i in range(num_iter):
        random_index = np.random.randint(n, size=s)

        pts_a = matches_a[random_index, :]
        pts_b = matches_b[random_index, :]

        F = estimate_fundamental_matrix(pts_a, pts_b)

        score = np.abs(np.matmul(A, F.reshape((-1))))

        inlier = np.sum(score <= threshold)

        if inlier > max_inlier:
            best_F = F.copy()
            max_inlier = inlier

    score = np.abs(np.matmul(A, best_F.reshape((-1))))

    # Return no more than 30 inliers for right
    inliers_a = matches_a[np.argsort(score)[:29]]
    inliers_b = matches_b[np.argsort(score)[:29]]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return best_F, inliers_a, inliers_b
